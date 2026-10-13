#!/usr/bin/env python
"""How models behave inside old v6's null regions, for out-of-panel inbreds: het-call rate, founder
switch density, agreement with the panel-ceiling founder, and the error split by het vs homozygous calls.

Per sample: old v6's BED defines null records; models are read from their BEDs (pair_from_bed); the
panel ceiling path is recomputed (panel_ceiling.best_path, --lam). Scoring uses fast_snprc_score's
SNP+RefCall rule, so "error" matches every other table.
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fast_snprc_score as fss  # noqa: E402
from panel_ceiling import best_path  # noqa: E402
from region_split import MANIFEST, TAGS, bed_dir  # noqa: E402


def sites_error(panel, t1, t2, f1, f2, mask):
    g1, g2 = f1.copy(), f2.copy()
    g1[~mask] = -1
    g2[~mask] = -1
    r = fss.score_arrays(panel, t1, t2, g1, g2)
    return r["snprc_error_rate"], r["snprc_compared_sites"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["OUT-INBRED"])
    ap.add_argument("--lam", type=float, default=50.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    panel = fss.Panel()
    panel.fa = np.asarray(panel.fa)
    chrom_of = np.empty(panel.pos.size, np.int16)
    for c, (lo, hi) in panel.bounds.items():
        chrom_of[lo:hi] = c
    rows = [r for r in csv.DictReader(open(MANIFEST), delimiter="\t")
            if r["coverage"] in TAGS and r["dataset_id"] in a.classes]
    rows.sort(key=lambda r: (r["individual"], r["coverage"]))
    out = open(a.out, "w")
    out.write("depth\tdataset\tindividual\tmodel\tnull_frac\thet_call_frac\tswitches_per_mb\t"
              "allele_agree_ceiling\terr_null\terr_null_homcalls\terr_null_hetcalls\thet_err_share\n")
    paths = {}
    for r in rows:
        dep, ds, ind = r["coverage"], r["dataset_id"], r["individual"]
        dirs = {m: bed_dir(t, ds, ind, dep) for m, t in TAGS[dep].items()}
        if not all(dirs.values()):
            print("skip", dep, ds, ind, flush=True)
            continue
        t1, t2 = fss.load_truth(r["truth_h1"]), fss.load_truth(r["truth_h2"])
        for tp, t in ((r["truth_h1"], t1), (r["truth_h2"], t2)):
            if tp not in paths:
                paths[tp] = best_path(panel, t["code"], a.lam)[0]
        pred = {m: fss.pair_from_bed(panel, d) for m, d in dirs.items()}
        pred["ceiling"] = (paths[r["truth_h1"]], paths[r["truth_h2"]])
        o1, o2 = pred["old_v6"]
        null = ~((o1 >= 0) & (o2 >= 0))
        c1 = pred["ceiling"][0]
        idx = np.nonzero(null)[0]
        for m in ("v6_nonull", "v6cfullfe", "tern", "da", "ceiling"):
            f1, f2 = pred[m]
            called = null & (f1 >= 0) & (f2 >= 0)
            ci = np.nonzero(called)[0]
            het = called & (f1 != f2)
            # pair changes between consecutive called null records on the same chromosome, per Mb spanned
            same_chr = chrom_of[ci[1:]] == chrom_of[ci[:-1]]
            changed = ((f1[ci[1:]] != f1[ci[:-1]]) | (f2[ci[1:]] != f2[ci[:-1]])) & same_chr
            span_mb = np.sum(np.where(same_chr, panel.pos[ci[1:]] - panel.pos[ci[:-1]], 0)) / 1e6
            # does the model's first founder carry the same panel allele as the ceiling's founder?
            fa1 = panel.fa[ci, np.clip(f1[ci], 0, None)]
            fac = panel.fa[ci, np.clip(c1[ci], 0, None)]
            ok = (fa1 >= 0) & (fac >= 0)
            agree = float(np.mean(fa1[ok] == fac[ok])) if ok.any() else float("nan")
            e_all, n_all = sites_error(panel, t1, t2, f1, f2, called)
            e_hom, n_hom = sites_error(panel, t1, t2, f1, f2, called & ~het)
            e_het, n_het = sites_error(panel, t1, t2, f1, f2, het)
            share = (e_het or 0) * n_het / max((e_all or 0) * n_all, 1e-9)
            out.write(f"{dep}\t{ds}\t{ind}\t{m}\t{null.mean():.4f}\t{het.sum() / max(called.sum(), 1):.4f}\t"
                      f"{changed.sum() / max(span_mb, 1e-9):.2f}\t{agree:.4f}\t{e_all}\t{e_hom}\t{e_het}\t{share:.4f}\n")
        out.flush()
        print("done", dep, ds, ind, flush=True)


if __name__ == "__main__":
    main()
