#!/usr/bin/env python
"""Check the inbred/hybrid router on the evaluation and calibration corpora, and compare its pairwise
`disj` statistic with a physical-window version.

pairwise disj (route_real_sample): among consecutive informative rows in the same refmap bin, the
fraction whose matching-founder sets are disjoint.
window disj (here): all pairs of informative rows whose positions fall in the same physical window of
W bp on the same contig; the fraction of those pairs with disjoint founder sets. Positions come from
row_bin (contig << 40 | position), which the decoder already has, so no new input is needed.

Writes one TSV row per sample and prints, per corpus and statistic, the separation between the
highest non-hybrid and the lowest hybrid value.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import simval_eval_indel as sei  # noqa: E402
# the model code must be imported before heldout_assembly_eval, which puts another checkout's src/ first
sys.path.insert(0, sei.CRF_SRC)
from python.crf.train_diploid_indel import ROUTE_DISJ_HYBRID, route_real_sample  # noqa: E402
import heldout_assembly_eval as hae  # noqa: E402

WORK = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
CORPORA = {
    "eval": ("/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv",
             WORK / "scratch/lift_s200_local20k", "eval", ["0.1", "1.0"]),
    "calib": ("/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_calibration/manifest.tsv",
              WORK / "scratch/calib_s200_local20k", "calib", ["0.1"]),
}
K = 25
INFORMATIVE_MAX = 12


def window_disj(M, row_bin, w):
    """Fraction of informative-row pairs within the same w-bp window whose founder sets are disjoint."""
    Mf = M.reshape(-1, K)
    b = row_bin.reshape(-1)
    nm = Mf.sum(1)
    keep = (nm >= 1) & (nm <= INFORMATIVE_MAX)
    Mf, b = Mf[keep], b[keep]
    contig, pos = b >> 40, b & ((1 << 40) - 1)
    win = contig * (1 << 34) + pos // w
    order = np.argsort(win, kind="stable")
    win, Mf = win[order], Mf[order].astype(np.float32)
    starts = np.flatnonzero(np.r_[True, win[1:] != win[:-1]])
    ends = np.r_[starts[1:], len(win)]
    disjoint = total = 0
    for s, e in zip(starts, ends):
        n = e - s
        if n < 2:
            continue
        if n > 400:                       # cap pathological pile-ups (repeats) to keep this cheap
            idx = np.linspace(s, e - 1, 400).astype(int)
            G = Mf[idx]; n = 400
        else:
            G = Mf[s:e]
        ov = G @ G.T
        iu = np.triu_indices(n, 1)
        disjoint += int((ov[iu] == 0).sum())
        total += len(iu[0])
    return disjoint / total if total else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", nargs="+", type=int, default=[1000, 5000, 20000])
    a = ap.parse_args()
    cols = ["corpus", "depth", "dataset", "individual", "truth_hybrid", "disj_pair", "p90", "router_hybrid"] + \
           [f"disj_w{w}" for w in a.windows]
    out = open(a.out, "w")
    out.write("\t".join(cols) + "\n")
    for corpus, (manifest, root, prefix, depths) in CORPORA.items():
        rows = [r for r in csv.DictReader(open(manifest), delimiter="\t") if r["coverage"] in depths]
        for r in rows:
            ds, ind, dep = r["dataset_id"], r["individual"], r["coverage"]
            src = root / f"{prefix}__{ds}__{ind}__{dep}x"
            if not (src / "windowed_k25native_wcount.npy").exists():
                print("missing", src, flush=True)
                continue
            data = np.load(src / "windowed_k25native_wcount.npy", mmap_mode="r")
            T = data.shape[1]
            layout = hae.load_contig_layout(src / "raw.npy.bins.tsv", T, bin_size=256)
            row_bin = np.concatenate([np.asarray(p, np.int64) + (ci << 40) for ci, (_c, p, _n) in enumerate(layout)]
                                     ).reshape(data.shape[0], T)
            M = np.asarray(data[..., :K]) == 1
            rr = route_real_sample(M, row_bin)
            vals = [corpus, dep, ds, ind, ds.endswith("-HYB"), f"{rr['disj']:.4f}", f"{rr['p90']:.3f}", rr["hybrid"]]
            vals += [f"{window_disj(M, row_bin, w):.4f}" for w in a.windows]
            out.write("\t".join(map(str, vals)) + "\n")
            out.flush()
            print(corpus, dep, ds, ind, vals[5:], flush=True)
    out.close()

    import pandas as pd
    df = pd.read_csv(a.out, sep="\t")
    print(f"\nrouter threshold disj_pair > {ROUTE_DISJ_HYBRID}")
    for (corpus, dep), g in df.groupby(["corpus", "depth"]):
        wrong = g[g.truth_hybrid != g.router_hybrid]
        print(f"\n{corpus} {dep}x: router correct {len(g) - len(wrong)}/{len(g)}"
              + (f"  WRONG: {list(zip(wrong.dataset, wrong.individual))}" if len(wrong) else ""))
        for c in ["disj_pair"] + [f"disj_w{w}" for w in a.windows]:
            hi_non = g[~g.truth_hybrid][c].max()
            lo_hyb = g[g.truth_hybrid][c].min()
            print(f"  {c:<12} non-hybrid max {hi_non:.3f} | hybrid min {lo_hyb:.3f} | gap {lo_hyb - hi_non:+.3f} "
                  f"(relative {(lo_hyb - hi_non) / max(lo_hyb, 1e-9):+.0%})")


if __name__ == "__main__":
    main()
