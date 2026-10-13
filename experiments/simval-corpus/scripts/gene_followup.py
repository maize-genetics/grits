#!/usr/bin/env python
"""Follow-up to gene_carry_check.py: is the genic error excess a wrong founder that only *shows* in genes?

Per sample and feature group (genic / TE / intergenic, as in gene_carry_check.py):
  wrong_founder_frac   SNP+RC sites where the model's founder pair differs from the reference path's
  hidden_by_sharing    of those, the share where the model's genotype is still right (the wrong founder
                       carries the same alleles as the truth)
  err_given_wrong / err_given_right   SNP+RC error when the founder pair is wrong / right
  ceil_bp_per_mb, model_bp_per_mb     founder-pair changes per Mb of sequence in the group, where a
                       group's span is the sum of gaps between consecutive panel records both in the group
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import eval_suite as es  # noqa: E402
import fast_snprc_score as fss  # noqa: E402
from gene_carry_check import FEAT, GROUPS, pair_code  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["0.1=v6cfullfe_homo", "1.0=v6cfullfe_homo_1x"])
    ap.add_argument("--classes", nargs="+", default=es.CLASSES)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = es.panel()
    prim = np.load(FEAT / "panel_feature_annotation.npz")["primary"]
    grp = np.full(len(prim), -1, np.int8)
    for gi, (_, s) in enumerate(GROUPS.items()):
        grp[np.isin(prim, list(s))] = gi
    same_chr = np.r_[P.chrom[1:] == P.chrom[:-1], False]
    gap = np.r_[np.diff(P.pos), 0].astype(np.int64)
    span_mb = {gi: gap[same_chr & (grp == gi) & np.r_[grp[1:] == gi, False]].sum() / 1e6 for gi in range(len(GROUPS))}
    print({g: round(span_mb[i], 1) for i, g in enumerate(GROUPS)}, "Mb spanned", flush=True)
    tags = dict(t.split("=") for t in a.tags)
    man = [r for r in csv.DictReader(open(es.MANIFEST), delimiter="\t") if r["dataset_id"] in a.classes]
    cols = ["depth", "dataset", "individual"]
    for g in GROUPS:
        cols += [f"{g}_wrong_founder_frac", f"{g}_hidden_by_sharing", f"{g}_err_given_wrong", f"{g}_err_given_right",
                 f"{g}_ceil_bp_per_mb", f"{g}_model_bp_per_mb"]
    out = open(a.out, "w")
    out.write("\t".join(cols) + "\n")
    for dep, tag in tags.items():
        for r in [r for r in man if r["coverage"] == dep]:
            ds, ind = r["dataset_id"], r["individual"]
            bed, _ = es.sample_dirs(tag, ds, ind, dep)
            if not bed.is_dir():
                continue
            t1, t2 = fss.load_truth(r["truth_h1"]), fss.load_truth(r["truth_h2"])
            f1, f2 = fss.pair_from_bed(P, bed)
            (c1, _), (c2, _) = es.ref_path(P, r["truth_h1"], 50.0), es.ref_path(P, r["truth_h2"], 50.0)
            pr = es.per_record(P, t1, t2, f1, f2)
            rows, m, sn = pr["rows"], pr["match"], pr["snprc"]
            wrong = pair_code(f1, f2)[rows] != pair_code(c1, c2)[rows]
            bp_c = es.pair_changes(P, c1, c2, np.ones(len(c1), bool))
            bp_m = es.pair_changes(P, f1, f2, (f1 >= 0) & (f2 >= 0))
            vals = [dep, ds, ind]
            for gi in range(len(GROUPS)):
                s = sn & (grp[rows] == gi)
                w = s & wrong
                vals += [w.sum() / max(s.sum(), 1), m[w].mean() if w.any() else None,
                         1 - m[w].mean() if w.any() else None, 1 - m[s & ~wrong].mean() if (s & ~wrong).any() else None,
                         (grp[bp_c] == gi).sum() / span_mb[gi], (grp[bp_m] == gi).sum() / span_mb[gi]]
            out.write("\t".join("" if v is None else f"{float(v):.5g}" if not isinstance(v, str) else v for v in vals) + "\n")
            out.flush()
            print(dep, ds, ind, "done", flush=True)


if __name__ == "__main__":
    main()
