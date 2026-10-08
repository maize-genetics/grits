#!/usr/bin/env python
"""Why are genes harder than TEs for out-of-panel samples? Two checks on existing BEDs.

1 carried-through founders. For every SNP+RC error at a genic record (CDS, UTR, intron, 2 kb flank),
  compare the model's founder pair with the reference (panel-ceiling) path's pair at that record and
  at the nearest non-genic record on each side of the gene block:
    same_as_ceiling   model pair == ceiling pair at the record (error intrinsic to the panel)
    carried_through   model pair != ceiling pair here, == the ceiling pair in a flank, and the ceiling
                      switches inside the gene block (ceiling pair here differs from both flanks)
    wrong_regionally  model pair != ceiling here and != the ceiling pair in either flank
    other             anything else (e.g. model matches a flank but the ceiling does not switch)
  plus, for all genic vs TE records, how often the ceiling pair differs from both flanking pairs.
2 switch density. Ceiling and model founder-pair changes per 1,000 records by feature group, and the
  share of ceiling changes the model has within 10 kb, by feature group.

Feature labels: results/genomic_feature_errors/panel_feature_annotation.npz (primary category).
Reference paths: eval_suite's cache (results/eval_suite/_reference, lambda 50).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import eval_suite as es  # noqa: E402
import fast_snprc_score as fss  # noqa: E402

FEAT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/genomic_feature_errors")
PRIMARY = ['intergenic_nonTE', 'gene_flank_2kb', 'CDS', 'UTR', 'intron', 'LTR_Gypsy', 'LTR_Copia', 'LTR_unknown',
           'TIR_DTA_hAT', 'TIR_DTC_CACTA', 'TIR_DTH_PIF', 'TIR_DTM_Mutator', 'TIR_DTT_Tc1', 'Helitron', 'LINE',
           'knob', 'centromeric_CentC_CRM', 'rDNA_spacer', 'subtelomere', 'other_repeat']
GENIC = {PRIMARY.index(n) for n in ("gene_flank_2kb", "CDS", "UTR", "intron")}
TE = {PRIMARY.index(n) for n in PRIMARY[5:14]}
GROUPS = {"genic": GENIC, "TE": TE, "intergenic": {0}}


def flank_index(is_in, chrom):
    """For each record: index of the nearest record NOT in the block set before / after it on the same
    chromosome (-1 if none)."""
    n = len(is_in)
    idx = np.arange(n)
    prev = np.where(~is_in, idx, -1)
    prev = np.maximum.accumulate(prev)
    nxt = np.where(~is_in, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    prev = np.where((prev >= 0) & (chrom[np.clip(prev, 0, n - 1)] == chrom), prev, -1)
    ok = nxt < n
    nxt = np.where(ok & (chrom[np.clip(nxt, 0, n - 1)] == chrom), nxt, -1)
    return prev, nxt


def pair_code(a, b):
    return np.minimum(a, b).astype(np.int32) * 64 + np.maximum(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["0.1=v6cfullfe_homo", "1.0=v6cfullfe_homo_1x"])
    ap.add_argument("--classes", nargs="+", default=es.CLASSES)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = es.panel()
    prim = np.load(FEAT / "panel_feature_annotation.npz")["primary"]
    grp = np.full(len(prim), -1, np.int8)
    for gi, (gname, s) in enumerate(GROUPS.items()):
        grp[np.isin(prim, list(s))] = gi
    genic = grp == 0
    gprev, gnext = flank_index(genic, P.chrom)
    te = grp == 1
    tprev, tnext = flank_index(te, P.chrom)
    tags = dict(t.split("=") for t in a.tags)
    man = [r for r in csv.DictReader(open(es.MANIFEST), delimiter="\t") if r["dataset_id"] in a.classes]
    out = open(a.out, "w")
    cols = ["depth", "dataset", "individual", "genic_errors", "same_as_ceiling", "carried_through", "wrong_regionally",
            "other", "ceil_switch_in_block_genic", "ceil_switch_in_block_TE"]
    for g in GROUPS:
        cols += [f"ceil_bp_per1k_{g}", f"model_bp_per1k_{g}", f"model_recall10kb_{g}"]
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
            mp, cp = pair_code(f1, f2), pair_code(c1, c2)
            # check 1
            err = rows[sn & ~m]
            err = err[genic[err] & (gprev[err] >= 0) & (gnext[err] >= 0)]
            L, R = cp[gprev[err]], cp[gnext[err]]
            here_m, here_c = mp[err], cp[err]
            same = here_m == here_c
            in_flank = (here_m == L) | (here_m == R)
            ceil_switch = (here_c != L) & (here_c != R)
            carried = ~same & in_flank & ceil_switch
            regional = ~same & ~in_flank
            vals = [dep, ds, ind, len(err), same.mean(), carried.mean(), regional.mean(),
                    (~same & ~carried & ~regional).mean()]
            for blk, pv, nx in ((genic, gprev, gnext), (te, tprev, tnext)):
                ii = np.flatnonzero(blk & (pv >= 0) & (nx >= 0))
                vals.append(float(((cp[ii] != cp[pv[ii]]) & (cp[ii] != cp[nx[ii]])).mean()))
            # check 2
            called = (f1 >= 0) & (f2 >= 0)
            bp_c = es.pair_changes(P, c1, c2, np.ones(len(c1), bool))
            bp_m = es.pair_changes(P, f1, f2, called)
            for gi, gname in enumerate(GROUPS):
                nrec = max(int((grp == gi).sum()), 1)
                bc, bm = bp_c[grp[bp_c] == gi], bp_m[grp[bp_m] == gi]
                vals += [1000 * len(bc) / nrec, 1000 * len(bm) / nrec, es.bp_match(P, bc, bp_m, 1e4)]
            out.write("\t".join("" if v is None else (f"{v:.5g}" if isinstance(v, float) else str(v)) for v in vals) + "\n")
            out.flush()
            print(dep, ds, ind, f"genic errors {len(err):,}: same {same.mean():.2f} carried {carried.mean():.2f} "
                  f"regional {regional.mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
