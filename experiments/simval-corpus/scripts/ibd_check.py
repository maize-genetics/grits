#!/usr/bin/env python
"""Inside stretches where a panel founder is IBD with the sample, do SNPs match?

For out-of-panel inbreds (one haplotype path): the reference (panel-ceiling) founder is compared with the
truth at every SNP+RC record. A record is "IBD-like" when the ceiling founder matches the truth at every
compared record within +-W records on the same chromosome (W = --window); "near" with 1-2 mismatches;
"divergent" otherwise. Within each state and feature group (genic / TE / intergenic) report the share of
records, the ceiling's error (should be ~0 in IBD-like stretches, else something is inconsistent between
truth and panel), the model's error, and how often the model's founder differs from the ceiling's there.
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

STATES = ("IBD-like", "near", "divergent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v6cfullfe_homo")
    ap.add_argument("--depth", default="0.1")
    ap.add_argument("--classes", nargs="+", default=["OUT-INBRED"])
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = es.panel()
    prim = np.load(FEAT / "panel_feature_annotation.npz")["primary"]
    grp = np.full(len(prim), -1, np.int8)
    for gi, (_, s) in enumerate(GROUPS.items()):
        grp[np.isin(prim, list(s))] = gi
    man = [r for r in csv.DictReader(open(es.MANIFEST), delimiter="\t")
           if r["dataset_id"] in a.classes and r["coverage"] == a.depth]
    out = open(a.out, "w")
    out.write("dataset\tindividual\tstate\tgroup\tsites\tsite_share\tceiling_err\tmodel_err\tmodel_founder_differs\n")
    for r in man:
        ds, ind = r["dataset_id"], r["individual"]
        bed, _ = es.sample_dirs(a.tag, ds, ind, a.depth)
        t1, t2 = fss.load_truth(r["truth_h1"]), fss.load_truth(r["truth_h2"])
        (c1, _), (c2, _) = es.ref_path(P, r["truth_h1"], 50.0), es.ref_path(P, r["truth_h2"], 50.0)
        cr = es.per_record(P, t1, t2, c1, c2)
        rows = cr["rows"][cr["snprc"]]
        cmatch = cr["match"][cr["snprc"]]
        # mismatches of the ceiling founder within +-W compared records, same chromosome
        mis = (~cmatch).astype(np.int32)
        state = np.empty(len(rows), np.int8)
        chrom = P.chrom[rows]
        for c in np.unique(chrom):
            s = np.flatnonzero(chrom == c)
            cs = np.concatenate([[0], np.cumsum(mis[s])])
            lo = np.clip(np.arange(len(s)) - a.window, 0, len(s))
            hi = np.clip(np.arange(len(s)) + a.window + 1, 0, len(s))
            k = cs[hi] - cs[lo]
            state[s] = np.where(k == 0, 0, np.where(k <= 2, 1, 2))
        f1, f2 = fss.pair_from_bed(P, bed)
        mr = es.per_record(P, t1, t2, f1, f2)
        mm = np.full(P.pos.size, -1, np.int8)
        mm[mr["rows"][mr["snprc"]]] = mr["match"][mr["snprc"]]
        differs = pair_code(f1, f2)[rows] != pair_code(c1, c2)[rows]
        for si, sname in enumerate(STATES):
            for gi, gname in enumerate(GROUPS):
                sel = (state == si) & (grp[rows] == gi)
                n = int(sel.sum())
                mv = mm[rows[sel]]
                mv = mv[mv >= 0]
                out.write(f"{ds}\t{ind}\t{sname}\t{gname}\t{n}\t{n / len(rows):.4f}\t"
                          f"{(1 - cmatch[sel].mean()) if n else float('nan'):.5f}\t"
                          f"{(1 - mv.mean()) if len(mv) else float('nan'):.5f}\t"
                          f"{differs[sel].mean() if n else float('nan'):.4f}\n")
        out.flush()
        print(ds, ind, "done", flush=True)


if __name__ == "__main__":
    main()
