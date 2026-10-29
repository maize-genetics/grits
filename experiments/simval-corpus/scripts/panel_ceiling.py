#!/usr/bin/env python
"""
Panel ceiling: the best genotype error ANY founder-path model could reach on a
sample, given this 25-founder panel -- scored with the exact simval
SNP+RefCall metric (fast_snprc_score, validated identical to the full
bed-to-vcf + compare pipeline).

Per truth haplotype, an exact haploid Viterbi over all autosomal panel records
(per chromosome) chooses one founder per record: cost 1 when the founder's
panel allele != the truth allele (a truth allele absent from the panel costs 1
for every founder -- irreducible), 0 when it matches or when truth has no
comparable call there; each founder switch costs `lam`. The two haplotype
paths form the predicted pair per record, scored with score_arrays exactly
like a model's BED. Founders with no panel call ('.') cost 0.5: choosing them
drops the site from the comparison (not an error, not a match).

lam=0 is the per-site floor (free switching); larger lam forces sparser,
more realistic paths. Reports error and switches per haplotype per sample.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).parent))
import fast_snprc_score as fss  # noqa: E402

MANIFEST = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv"


@njit(cache=True)
def _viterbi(fa, tcode, lam):
    n, K = fa.shape
    cost = np.zeros(K)
    back_min = np.empty(n, np.int8)        # argmin of previous costs at each step
    stay = np.empty((n, K), np.bool_)      # True if state k at step i came from k itself
    for i in range(n):
        t = tcode[i]
        best = 0
        for k in range(1, K):
            if cost[k] < cost[best]:
                best = k
        back_min[i] = best
        switch_cost = cost[best] + lam
        for k in range(K):
            if cost[k] <= switch_cost:
                stay[i, k] = True
                c = cost[k]
            else:
                stay[i, k] = False
                c = switch_cost
            a = fa[i, k]
            if t == -1 or t == -3:          # no comparable truth call
                e = 0.0
            elif a < 0:                      # founder has no panel call -> site dropped
                e = 0.5
            elif t == -2:                    # truth allele absent from panel
                e = 1.0
            else:
                e = 0.0 if a == t else 1.0
            cost[k] = c + e
    path = np.empty(n, np.int16)
    k = 0
    for j in range(1, K):
        if cost[j] < cost[k]:
            k = j
    n_sw = 0
    for i in range(n - 1, -1, -1):
        path[i] = k
        if not stay[i, k]:
            k = back_min[i]
            n_sw += 1
    return path, n_sw


def best_path(panel, tcode, lam):
    path = np.empty(panel.pos.size, np.int16)
    switches = 0
    for c, (lo, hi) in panel.bounds.items():
        if hi > lo:
            p, s = _viterbi(np.ascontiguousarray(panel.fa[lo:hi]), tcode[lo:hi], lam)
            path[lo:hi] = p
            switches += s
    return path, switches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["OUT-INBRED", "OUT-HYB", "OUT-RIL2",
                                                     "IDX-INBRED", "IDX-HYB", "IDX-RIL2"])
    ap.add_argument("--lams", nargs="+", type=float, default=[0, 2, 10, 50, 200])
    ap.add_argument("--out", default="/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/panel_ceiling_0.1x.json")
    args = ap.parse_args()
    panel = fss.Panel()
    panel.fa = np.asarray(panel.fa)
    rows = [r for r in csv.DictReader(open(MANIFEST), delimiter="\t")
            if r["coverage"] == "0.1" and r["dataset_id"] in args.classes]
    out = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    cache = {}
    for r in rows:
        t1, t2 = fss.load_truth(r["truth_h1"]), fss.load_truth(r["truth_h2"])
        for lam in args.lams:
            key = f"{r['dataset_id']}__{r['individual']}__lam{lam:g}"
            if key in out:
                continue
            paths = []
            for tp, t in ((r["truth_h1"], t1), (r["truth_h2"], t2)):
                ck = (tp, lam)
                if ck not in cache:
                    cache[ck] = best_path(panel, t["code"], lam)
                paths.append(cache[ck])
            res = fss.score_arrays(panel, t1, t2, paths[0][0], paths[1][0])
            res["switches_h1"], res["switches_h2"] = paths[0][1], paths[1][1]
            out[key] = res
            Path(args.out).write_text(json.dumps(out, indent=1))
            print(f"{r['dataset_id']:<11} {r['individual']:<14} lam={lam:<5g} err={100 * res['error_rate']:6.2f}%  "
                  f"snprc={100 * res['snprc_error_rate']:6.2f}%  switches/hap={paths[0][1]:,}/{paths[1][1]:,}",
                  flush=True)


if __name__ == "__main__":
    main()
