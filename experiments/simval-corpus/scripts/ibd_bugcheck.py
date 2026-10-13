#!/usr/bin/env python
"""Are the isolated truth/panel mismatches inside near-IBD stretches real variants or representation bugs?

Out-of-panel inbreds (one haplotype path). As in ibd_check.py, a SNP+RC record is "near-IBD" when the
reference (panel-ceiling) founder mismatches the truth at 1-2 of the compared SNP+RC records within
+-W records. For every mismatch in those stretches:
  novel             truth allele is not any panel allele at the record (T_NOVEL)
  in_panel_ref_alt  truth REF, IBD founder carries an ALT allele
  in_panel_alt_ref  truth ALT (in panel), IBD founder carries REF
  in_panel_alt_alt  truth ALT and founder a different ALT
  near_indel        within --indel-bp of a compared record where the truth or the founder is INS/DEL
  clustered         another near-IBD mismatch within 100 bp
near_indel is compared with the same rate at MATCHING records in the same stretches (background).
Also reports how many panel records share a position with the mismatch (split multi-allelic records).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import eval_suite as es  # noqa: E402
import fast_snprc_score as fss  # noqa: E402


def near_mask(pos, chrom, query_idx, flag_idx, d):
    """For records query_idx: is any record in flag_idx (same chromosome) within d bp? (sorted inputs)"""
    if len(flag_idx) == 0:
        return np.zeros(len(query_idx), bool)
    key = chrom.astype(np.int64) * 10 ** 10
    fq = key[query_idx] + pos[query_idx]
    ff = np.sort(key[flag_idx] + pos[flag_idx])
    j = np.searchsorted(ff, fq)
    lo = ff[np.clip(j - 1, 0, len(ff) - 1)]
    hi = ff[np.clip(j, 0, len(ff) - 1)]
    return (np.abs(fq - lo) <= d) | (np.abs(hi - fq) <= d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["OUT-INBRED"])
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--indel-bp", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = es.panel()
    man = [r for r in csv.DictReader(open(es.MANIFEST), delimiter="\t")
           if r["dataset_id"] in a.classes and r["coverage"] == "0.1"]
    out = open(a.out, "w")
    cols = ["dataset", "individual", "near_ibd_sites", "mismatches", "mismatch_rate", "novel", "in_panel_ref_alt",
            "in_panel_alt_ref", "in_panel_alt_alt", "near_indel", "near_indel_background", "clustered",
            "shared_position", "shared_position_background"]
    out.write("\t".join(cols) + "\n")
    same_pos = np.zeros(P.pos.size, bool)
    dup = (P.pos[1:] == P.pos[:-1]) & (P.chrom[1:] == P.chrom[:-1])
    same_pos[1:] |= dup
    same_pos[:-1] |= dup
    for r in man:
        ds, ind = r["dataset_id"], r["individual"]
        t1, t2 = fss.load_truth(r["truth_h1"]), fss.load_truth(r["truth_h2"])
        (c1, _), (c2, _) = es.ref_path(P, r["truth_h1"], 50.0), es.ref_path(P, r["truth_h2"], 50.0)
        # all compared records with their truth / founder allele classes
        n = P.pos.size
        idx = np.arange(n)
        f = P.fa[idx, c1]
        ok = (t1["code"] != fss.T_NOINFO) & (t1["code"] != fss.T_MISSING) & (f >= 0)
        rows = idx[ok]
        tc = t1["code"][rows]
        t_cls = np.where(tc == fss.T_NOVEL, t1["ncls"][rows], P.acls[rows, np.clip(tc, 0, None)])
        f_cls = P.acls[rows, f[rows]]
        indel = (t_cls == 2) | (t_cls == 3) | (f_cls == 2) | (f_cls == 3)
        snprc = (t_cls <= 1) & (f_cls <= 1)
        same = np.where(tc == fss.T_NOVEL, False, tc == f[rows])
        srow, smatch = rows[snprc], same[snprc]
        mis = (~smatch).astype(np.int32)
        state = np.zeros(len(srow), np.int8)
        chrom = P.chrom[srow]
        for c in np.unique(chrom):
            s = np.flatnonzero(chrom == c)
            cs = np.concatenate([[0], np.cumsum(mis[s])])
            lo = np.clip(np.arange(len(s)) - a.window, 0, len(s))
            hi = np.clip(np.arange(len(s)) + a.window + 1, 0, len(s))
            k = cs[hi] - cs[lo]
            state[s] = np.where((k >= 1) & (k <= 2), 1, 0)
        near = state == 1
        mm = srow[near & ~smatch]
        bg = srow[near & smatch]
        tcm = t1["code"][mm]
        fm = P.fa[mm, c1[mm]]
        indel_rows = rows[indel]
        ni = near_mask(P.pos, P.chrom, mm, indel_rows, a.indel_bp)
        nb = near_mask(P.pos, P.chrom, bg, indel_rows, a.indel_bp)
        # clustered: another near-IBD mismatch within 100 bp (excluding itself)
        key = P.chrom[mm].astype(np.int64) * 10 ** 10 + P.pos[mm]
        o = np.sort(key)
        dprev = np.r_[np.inf, np.diff(o)]
        dnext = np.r_[np.diff(o), np.inf]
        clustered = float(np.mean((dprev <= 100) | (dnext <= 100))) if len(o) else float("nan")
        m = max(len(mm), 1)
        vals = [ds, ind, int(near.sum()), len(mm), len(mm) / max(int(near.sum()), 1),
                np.mean(tcm == fss.T_NOVEL), np.mean((tcm == 0) & (fm > 0)), np.mean((tcm > 0) & (fm == 0)),
                np.mean((tcm > 0) & (fm > 0)), ni.mean(), nb.mean(), clustered,
                same_pos[mm].mean(), same_pos[bg].mean()]
        out.write("\t".join(v if isinstance(v, str) else f"{float(v):.4g}" for v in vals) + "\n")
        out.flush()
        print(ds, ind, "done", flush=True)


if __name__ == "__main__":
    main()
