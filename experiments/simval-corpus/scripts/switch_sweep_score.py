#!/usr/bin/env python
"""Score decode-knob sweep tags with eval_suite.eval_sample on either corpus, without touching
the eval_suite registry: SNP+RC / all-sites error, genome coverage, founder-pair changes vs the
panel-ceiling reference path (switch ratio) and breakpoint precision / recall within 100 kb.

  switch_sweep_score.py reference --corpus calibration --classes OUT-INBRED ... [--workers 4]
  switch_sweep_score.py score --corpus calibration --depth 0.1 --tags TAG1 TAG2 ... --out FILE.tsv
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import eval_suite as es  # noqa: E402
import fast_snprc_score as fss  # noqa: E402

MANIFESTS = {
    "validation": "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv",
    "calibration": "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_calibration/manifest.tsv",
}
COLS = ["snprc_err", "all_err", "genome_covered", "pair_changes", "ref_pair_changes",
        "bp_precision_100kb", "bp_recall_100kb", "ceiling_snprc_err", "router_hybrid"]


def rows(corpus, classes, depth=None):
    return [r for r in csv.DictReader(open(MANIFESTS[corpus]), delimiter="\t")
            if r["dataset_id"] in classes and (depth is None or r["coverage"] == depth)]


_REFC = {}


def _light(job):
    """SNP+RC / all-sites error from fast_eval_ckpt's JSON; coverage, founder-pair changes and
    breakpoint precision / recall (100 kb) vs the reference path from the BED. No allele panel."""
    import json
    r, tag, lam = job
    bed, js = es.sample_dirs(tag, r["dataset_id"], r["individual"], r["coverage"])
    if not bed.is_dir() or not js.exists():
        return None
    P = _light_panel()
    res = json.loads(js.read_text())
    f1, f2 = fss.pair_from_bed(P, bed)
    called = (f1 >= 0) & (f2 >= 0)
    key = (r["truth_h1"], r["truth_h2"])
    if key not in _REFC:
        (g1, _), (g2, _) = es.ref_path(P, r["truth_h1"], lam), es.ref_path(P, r["truth_h2"], lam)
        _REFC[key] = es.pair_changes(P, g1, g2, np.ones(P.pos.size, bool))
    bp_r = _REFC[key]
    bp_p = es.pair_changes(P, f1, f2, called)
    hyb = "hybrid=True" in (res.get("homo_scale_log") or "")
    gf = {}
    if _GAPFILL:
        g1f, g2f = gapfill(P, f1, f2, called)
        sc = fss.score_arrays(P, fss.load_truth(r["truth_h1"]), fss.load_truth(r["truth_h2"]), g1f, g2f)
        gf = dict(snprc_err_gapfill=sc["snprc_error_rate"], all_err_gapfill=sc["error_rate"])
    return dict(**gf,dataset=r["dataset_id"], individual=r["individual"], depth=r["coverage"], tag=tag,
                snprc_err=res["snprc_error_rate"], all_err=res["error_rate"],
                genome_covered=es.bed_span(bed), pair_changes=len(bp_p), ref_pair_changes=len(bp_r),
                bp_precision_100kb=es.bp_match(P, bp_p, bp_r, 1e5), bp_recall_100kb=es.bp_match(P, bp_r, bp_p, 1e5),
                ceiling_snprc_err=None, router_hybrid=hyb)


_LP = None
_GAPFILL = False


def gapfill(P, f1, f2, called):
    """Fill every uncalled record with the call of the nearest called record (bp) on its chromosome:
    the BED leaves the stretch between the last row of one segment and the first row of the next
    uncalled, so decodes that switch more (or at wider row gaps) lose coverage at switch points."""
    g1, g2 = f1.copy(), f2.copy()
    for c, (lo, hi) in P.bounds.items():
        ci = lo + np.flatnonzero(called[lo:hi])
        if ci.size == 0:
            continue
        pos = P.pos[lo:hi]
        j = np.searchsorted(ci, np.arange(lo, hi))
        left = ci[np.clip(j - 1, 0, ci.size - 1)]
        right = ci[np.clip(j, 0, ci.size - 1)]
        take_right = np.abs(P.pos[right] - pos) < np.abs(pos - P.pos[left])
        src = np.where(take_right, right, left)
        g1[lo:hi], g2[lo:hi] = f1[src], f2[src]
    return g1, g2


def _light_panel():
    global _LP
    if _LP is None:
        _LP = fss.Panel()
        _LP.offset = np.zeros(_LP.pos.size, np.int64)
        for c, (lo, hi) in _LP.bounds.items():
            _LP.offset[lo:hi] = c * 10 ** 10
    return _LP


def _one(job):
    r, tag, lam = job
    o = es.eval_sample(r, tag, lam)
    return None if o is None else {k: o.get(k) for k in ["dataset", "individual", "depth", "tag"] + COLS}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("reference", "score"):
        p = sub.add_parser(n)
        p.add_argument("--corpus", choices=sorted(MANIFESTS), default="calibration")
        p.add_argument("--classes", nargs="+", default=es.CLASSES)
        p.add_argument("--lam", type=float, default=50.0)
        p.add_argument("--workers", type=int, default=4)
    s = sub.choices["score"]
    s.add_argument("--depth", default="0.1")
    s.add_argument("--tags", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--light", action="store_true", help="error from the fast_eval JSON; no per-record rescoring")
    s.add_argument("--gapfill", action="store_true", help="with --light: also score with BED gaps filled")
    a = ap.parse_args()
    global _GAPFILL
    _GAPFILL = getattr(a, "gapfill", False)
    if a.cmd == "reference":
        tps = sorted({r[k] for r in rows(a.corpus, a.classes) for k in ("truth_h1", "truth_h2")})
        todo = [tp for tp in tps if not (es.REF / f"ceiling_lam{a.lam:g}_{fss.truth_key(tp)}.npz").exists()]
        print(f"{len(tps)} truth haplotypes, {len(todo)} to compute", flush=True)
        with ProcessPoolExecutor(a.workers) as ex:
            for tp, n, sec in ex.map(es._ref_one, [(tp, a.lam) for tp in todo]):
                print(f"  {fss.truth_key(tp)}: {n:,} switches ({sec}s)", flush=True)
        return
    jobs = [(r, t, a.lam) for t in a.tags for r in rows(a.corpus, a.classes, a.depth)]
    with ProcessPoolExecutor(a.workers) as ex:
        res = [o for o in ex.map(_light if a.light else _one, jobs, chunksize=4) if o is not None]
    keys = ["dataset", "individual", "depth", "tag"] + COLS + (["snprc_err_gapfill", "all_err_gapfill"] if _GAPFILL else [])
    with open(a.out, "w") as f:
        f.write("\t".join(keys) + "\n")
        for o in res:
            f.write("\t".join("" if o[k] is None else str(o[k]) for k in keys) + "\n")
    import pandas as pd
    df = pd.read_csv(a.out, sep="\t")
    df["switch_ratio"] = df.pair_changes / df.ref_pair_changes.clip(lower=1)
    cols = ["snprc_err", "all_err", "genome_covered", "switch_ratio", "pair_changes", "bp_precision_100kb",
            "bp_recall_100kb"] + (["snprc_err_gapfill", "all_err_gapfill"] if _GAPFILL else [])
    g = df.groupby(["tag", "dataset"])[cols].mean()
    print((g * np.array([100, 100, 100, 1, 1, 1, 1] + [100, 100] * _GAPFILL)).round(3).to_string())


if __name__ == "__main__":
    main()
