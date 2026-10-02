#!/usr/bin/env python
"""Within-row structure of refmap's per-founder anchor-distance feature, on the CALIBRATION
corpus only, for simulate_alleles --dist-structure.

refmap's distance is per (256 bp bin, founder): the bp from the bin to that founder's nearest
lift anchor. Founders collinear around the same anchor gap share one value, so a row is a
shared modal distance plus a few shared off-mode levels, not K independent draws (the old
--obs-table model). Measured in the simulator's log code space (code = round(8 log2(1+bp))):

  mode_code_pmf           modal code over the non-reference founders of a row
  mode_code_pmf_by_del_frac  the same by the row's fraction of -1 founders (deleted-heavy rows
                          sit in large anchor gaps, so their mode is high)
  split[truth|state]      P(at mode / below / above) for a founder with that panel truth
                          (del / present, as ternary_obs_stats) and observed ternary state
  split_by_mode_band      the same, separately for rows whose mode is <= / > the 2kb code
                          (lo / hi): the mode is not an allowed -1 code in lo rows, nor 0 in hi
  delta_pmf[truth|state|side]  |code - mode code| for below / above founders (index = |delta|)
  levels[state|side]      mean distinct off-mode codes per row / off-mode founders (pool fit)
  validation              per state: at-mode/below/above share and distinct/n (all truths)

Rows sharing a bin have identical distances; reads in different bins almost never do, so the
simulator redraws per row except rows at the same site, which copy with P(same bin)."""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_real_stats as mrs  # noqa: E402

K = mrs.K
NCODE = 128
STATES = (1, 0, -1)


def code_of(bp):
    return np.clip(np.rint(8.0 * np.log2(1.0 + np.maximum(bp, 0))), 0, NCODE - 1).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=mrs.CALIB_MANIFEST)
    ap.add_argument("--align-root", required=True)
    ap.add_argument("--coverage", default="0.1")
    ap.add_argument("--rows-per-sample", type=int, default=60000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.manifest), delimiter="\t") if r["coverage"] == a.coverage]
    dirs = []
    for r in rows:
        if mrs.lines_of(r["individual"]) & mrs.EVAL_LINES:
            raise SystemExit(f"refusing evaluation-corpus line {r['individual']}")
        d = glob.glob(f"{a.align_root}/*__{r['dataset_id']}__{r['individual']}__{a.coverage}x")
        if d and (Path(d[0]) / "raw.npy").exists():
            dirs.append(Path(d[0]))
    print(f"{len(dirs)} calibration samples", flush=True)

    P = mrs.fss.Panel()
    fa_all, ac_all = np.asarray(P.fa), np.asarray(P.acls)
    mode_hist = np.zeros(NCODE, np.int64)
    DEL_EDGES = [0.0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.01]               # row fraction of -1 founders
    mode_by_del = np.zeros((len(DEL_EDGES) - 1, NCODE), np.int64)
    keys = [f"{tr}|{s}" for tr in ("present", "del") for s in STATES]
    split = {k: np.zeros(3, np.int64) for k in keys}
    THRESH = int(code_of(np.array([2000]))[0])                   # refmap -1 rule: anchor > 2kb
    split_band = {f"{b}|{k}": np.zeros(3, np.int64) for b in ("lo", "hi") for k in keys}
    delta = {f"{k}|{side}": np.zeros(NCODE, np.int64) for k in keys for side in ("below", "above")}
    lev = {f"{s}|{side}": np.zeros(2) for s in STATES for side in ("below", "above")}
    val = {s: np.zeros(4) for s in STATES}
    dn = {s: [] for s in STATES}
    ref_col = None
    srcs = []
    for d in dirs:
        raw = np.load(d / "raw.npy", mmap_mode="r")
        bins = pd.read_csv(d / "raw.npy.bins.tsv", sep="\t")
        gam = pd.read_csv(d / "raw.npy.gametes.tsv", sep="\t").sort_values("gameteIndex").sampleName.tolist()
        idx = np.sort(np.random.default_rng(0).choice(len(raw), min(len(raw), a.rows_per_sample), replace=False))
        srcs.append((np.asarray(raw[idx]), bins.iloc[idx].reset_index(drop=True), [P.fidx[g] for g in gam]))
    for ci, c in enumerate(mrs.fss.AUTOSOMES):
        lo, hi = P.bounds[ci]
        pos = P.pos[lo:hi]
        fa, ac = fa_all[lo:hi], ac_all[lo:hi]
        valid = fa >= 0
        dele = valid & (np.take_along_axis(ac, np.clip(fa, 0, None), 1) == 3)
        cv = np.vstack([np.zeros((1, K), np.int32), np.cumsum(valid, 0, dtype=np.int32)])
        cd = np.vstack([np.zeros((1, K), np.int32), np.cumsum(dele, 0, dtype=np.int32)])
        for X, bins, col in srcs:
            sel = np.where(bins.contig.values == c)[0]
            if not len(sel):
                continue
            st = bins.bin.values[sel] * mrs.BIN_BP
            lo_, hi_ = np.searchsorted(pos, st, "left"), np.searchsorted(pos, st + mrs.BIN_BP, "left")
            nv, nd = (cv[hi_] - cv[lo_])[:, col], (cd[hi_] - cd[lo_])[:, col]
            tern = X[sel, K:2 * K]
            dist = X[sel, 2 * K:3 * K]
            if ref_col is None:
                ref_col = int(np.argmax((X[:, 2 * K:3 * K] == 0).mean(0)))
                print("reference column", ref_col, P.founders[col[ref_col]] if hasattr(P, "founders") else "", flush=True)
            keep = np.arange(K) != ref_col
            tern, dist, nv, nd = tern[:, keep], dist[:, keep], nv[:, keep], nd[:, keep]
            live = (dist >= 0).all(1)
            tern, dist, nv, nd = tern[live], dist[live], nv[live], nd[live]
            code = code_of(dist)
            # row mode (lowest code among ties)
            cnt = np.zeros((len(code), NCODE), np.int32)
            np.add.at(cnt, (np.repeat(np.arange(len(code)), code.shape[1]), code.ravel()), 1)
            m = cnt.argmax(1)
            np.add.at(mode_hist, m, 1)
            db = np.clip(np.searchsorted(DEL_EDGES, (tern == -1).mean(1), "right") - 1, 0, len(DEL_EDGES) - 2)
            np.add.at(mode_by_del, (db, m), 1)
            side = np.sign(code - m[:, None])                       # -1 below, 0 at, 1 above
            truth = np.where(nd >= 0.5 * np.maximum(nv, 1), "del", "present")
            ok = (nv > 0) & ((nd == 0) | (nd >= 0.5 * nv))
            for s in STATES:
                ms = tern == s
                for tr in ("present", "del"):
                    mm = ms & ok & (truth == tr)
                    k = f"{tr}|{s}"
                    split[k] += [(mm & (side == 0)).sum(), (mm & (side < 0)).sum(), (mm & (side > 0)).sum()]
                    for b, bm in (("lo", m <= THRESH), ("hi", m > THRESH)):
                        mb = mm & bm[:, None]
                        split_band[f"{b}|{k}"] += [(mb & (side == 0)).sum(), (mb & (side < 0)).sum(),
                                                   (mb & (side > 0)).sum()]
                    delta[f"{k}|below"] += np.bincount(np.abs(code - m[:, None])[mm & (side < 0)], minlength=NCODE)[:NCODE]
                    delta[f"{k}|above"] += np.bincount((code - m[:, None])[mm & (side > 0)], minlength=NCODE)[:NCODE]
                val[s] += [(ms & (side == 0)).sum(), (ms & (side < 0)).sum(), (ms & (side > 0)).sum(), ms.sum()]
                for sd, sg in (("below", -1), ("above", 1)):
                    mo = ms & (side == sg)
                    nrow = mo.sum(1)
                    rr = np.where(nrow >= 2)[0][:4000]
                    if len(rr):
                        lev[f"{s}|{sd}"] += [sum(len(np.unique(code[i][mo[i]])) for i in rr), nrow[rr].sum()]
                for i in np.where(ms.sum(1) >= 3)[0][:3000]:
                    dn[s].append(len(np.unique(code[i][ms[i]])) / ms[i].sum())
        print(f"  {c}", flush=True)

    norm = lambda h: (h / max(h.sum(), 1)).tolist()
    out = {"source": "calibration corpus", "coverage": a.coverage, "n_samples": len(dirs),
           "bin_bp": mrs.BIN_BP,
           "mode_code_pmf": norm(mode_hist),
           "mode_del_frac_edges": DEL_EDGES,
           "mode_code_pmf_by_del_frac": [norm(h) for h in mode_by_del],
           "rows_by_del_frac": mode_by_del.sum(1).tolist(),
           "split": {k: norm(v) for k, v in split.items()},
           "split_counts": {k: v.tolist() for k, v in split.items()},
           "thresh_code": THRESH,
           "split_by_mode_band": {k: norm(v) for k, v in split_band.items()},
           "split_by_mode_band_counts": {k: v.tolist() for k, v in split_band.items()},
           "delta_pmf": {k: norm(v) for k, v in delta.items()},
           "levels_per_offmode_founder": {k: float(v[0] / max(v[1], 1)) for k, v in lev.items()},
           "validation": {str(s): {"at_mode": float(v[0] / v[3]), "below": float(v[1] / v[3]),
                                   "above": float(v[2] / v[3]), "distinct_per_n": float(np.mean(dn[s]))}
                          for s, v in val.items()}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("split", "levels_per_offmode_founder", "validation")}, indent=1))


if __name__ == "__main__":
    main()
