#!/usr/bin/env python
"""
Where do a decoder's genotype errors come from? For homozygous samples
(INBRED / RIL2), compares a decoder's per-record founder (from its BED) with
the panel-ceiling best path (panel_ceiling.best_path, one path per truth
haplotype) and classifies every wrong compared site by
  * same_as_ceiling: the decoder chose the ceiling's founder (the error is one
    the best possible path also makes -- panel-limited), vs a different one;
  * distance (bp) to the nearest ceiling-path founder switch.
Scoring rules are fast_snprc_score's (exact to the full pipeline).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fast_snprc_score as fss  # noqa: E402
import panel_ceiling as pc  # noqa: E402

MANIFEST = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv"
BINS = [(0, 1e4, "<10kb"), (1e4, 1e5, "10-100kb"), (1e5, np.inf, ">100kb")]


def dist_to_switch(panel, path):
    d = np.full(path.size, np.inf)
    for c, (lo, hi) in panel.bounds.items():
        if hi <= lo:
            continue
        pos = panel.pos[lo:hi]
        sw = np.flatnonzero(np.diff(path[lo:hi]) != 0)
        if sw.size == 0:
            continue
        bpos = (pos[sw] + pos[sw + 1]) / 2
        i = np.searchsorted(bpos, pos)
        left = np.abs(pos - bpos[np.clip(i - 1, 0, bpos.size - 1)])
        right = np.abs(bpos[np.clip(i, 0, bpos.size - 1)] - pos)
        d[lo:hi] = np.minimum(left, right)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", nargs=2, action="append", required=True, metavar=("NAME", "BED_GLOB"),
                    help="NAME and a glob with {ds} {ind} placeholders for the BED dir")
    ap.add_argument("--lam", type=float, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    panel = fss.Panel()
    panel.fa = np.asarray(panel.fa)
    rows = [r for r in csv.DictReader(open(MANIFEST), delimiter="\t")
            if r["coverage"] == "0.1" and r["dataset_id"] in ("OUT-INBRED", "OUT-RIL2")]
    out = {}
    for r in rows:
        ds, ind = r["dataset_id"], r["individual"]
        t = fss.load_truth(r["truth_h1"])
        path, _ = pc.best_path(panel, t["code"], args.lam)
        dsw = dist_to_switch(panel, path)
        code = t["code"]
        truth_ok = (code != fss.T_NOINFO) & (code != fss.T_MISSING)
        for name, pattern in args.decoder:
            import glob
            beds = glob.glob(pattern.format(ds=ds, ind=ind))
            if not beds:
                continue
            f1, f2 = fss.pair_from_bed(panel, beds[0])
            m = truth_ok & (f1 >= 0)
            idx = np.flatnonzero(m)
            a = panel.fa[idx, f1[idx]]
            ok = a >= 0
            idx, a = idx[ok], a[ok]
            wrong = a != np.where(code[idx] == fss.T_NOVEL, -99, code[idx])
            same = f1[idx] == path[idx]
            rec = {"compared": int(idx.size), "err": float(wrong.mean()),
                   "err_same_as_ceiling": float((wrong & same).sum() / idx.size),
                   "err_diff_founder": float((wrong & ~same).sum() / idx.size),
                   "frac_sites_diff_founder": float((~same).mean())}
            for lo, hi, lab in BINS:
                b = (dsw[idx] >= lo) & (dsw[idx] < hi)
                rec[f"err_{lab}"] = float((wrong & b).sum() / idx.size)
                rec[f"sites_{lab}"] = float(b.mean())
                rec[f"errrate_within_{lab}"] = float(wrong[b].mean()) if b.any() else None
            out[f"{name}__{ds}__{ind}"] = rec
            print(f"{name:<14} {ds} {ind:<14} err={100 * rec['err']:5.2f}%  "
                  f"[same founder as ceiling {100 * rec['err_same_as_ceiling']:5.2f} | different founder "
                  f"{100 * rec['err_diff_founder']:5.2f}]  by distance to ceiling switch: "
                  + "  ".join(f"{lab}={100 * rec[f'err_{lab}']:5.2f}" for _, _, lab in BINS), flush=True)
            Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
