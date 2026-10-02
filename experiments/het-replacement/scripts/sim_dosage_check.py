#!/usr/bin/env python
"""
Stage 2.5 of experiments/het-replacement/PLAN.md: can a checkpoint tell 0 / 1 / 2
deleted haplotypes apart on held-out SIMULATED individuals?

Decodes each individual with the canonical infer_real_founder_pairs, using the
individual's true kind (homozygous if h1 == h2 at every labelled row) for
homo_scale, and reports per individual kind:
  * founder-pair accuracy;
  * a truth-vs-predicted table of #haplotypes whose founder lacks the B73
    sequence at each row (0 / 1 / 2), the same table as the real-data diagnosis;
  * the read mix at each true dosage: collinear rows (no matched founder is
    deleted at the row's site) vs replacement rows (every matched founder is).

A founder "lacks the B73 sequence" at a row when its anchor-distance code
exceeds --del-dist-code (0 for simulate_alleles' default anchor_thresh=0).

Input: an unsliced simulate_alleles output [n_ind, L, 3K+2] (e.g. a *_inb0.0.npy
half) or a sliced one with --windows-per-individual.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, infer_real_founder_pairs  # noqa: E402

T = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--num-parents", type=int, default=25)
    ap.add_argument("--n-ind", type=int, default=40)
    ap.add_argument("--first-ind", type=int, default=0, help="skip individuals used for training")
    ap.add_argument("--windows-per-individual", type=int, default=0,
                    help="set for sliced [N,512,C] input; 0 = unsliced [n_ind, L, C]")
    ap.add_argument("--del-dist-code", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    K = args.num_parents

    model = GRITSCRFDiploidIndel.load_from_checkpoint(args.ckpt, map_location=args.device,
                                                      strict=False).eval().to(args.device)
    a = np.load(args.data, mmap_mode="r")
    G = args.windows_per_individual
    n_total = a.shape[0] // G if G else a.shape[0]
    stats = {}
    for i in range(args.first_ind, min(n_total, args.first_ind + args.n_ind)):
        if G:
            x = np.asarray(a[i * G:(i + 1) * G])
        else:
            g = a.shape[1] // T
            x = np.asarray(a[i, :g * T]).reshape(g, T, -1)
        h1, h2 = x[..., K].astype(int), x[..., K + 1].astype(int)
        ok = (h1 >= 0) & (h2 >= 0)
        homo = bool((h1[ok] == h2[ok]).all())
        lo, hi = infer_real_founder_pairs(model, x, K, device=args.device,
                                          homo_scale=0.0 if homo else 1.0)
        tern, dist = x[..., :K], x[..., K + 2:2 * K + 2]
        dele = dist > args.del_dist_code
        gd = lambda f: np.take_along_axis(dele, np.clip(f, 0, K - 1)[..., None], 2)[..., 0]
        tdos, pdos = gd(h1).astype(int) + gd(h2), gd(lo).astype(int) + gd(hi)
        match = tern == 1
        nm = match.sum(-1)
        nmd = (match & dele).sum(-1)
        kind = "homozygous" if homo else "hybrid"
        s = stats.setdefault(kind, {"conf": np.zeros((3, 3)), "pair_ok": 0, "n": 0,
                                    "rows": np.zeros(3), "colin": np.zeros(3), "repl": np.zeros(3)})
        np.add.at(s["conf"], (tdos[ok], pdos[ok]), 1)
        tl, th = np.minimum(h1, h2), np.maximum(h1, h2)
        s["pair_ok"] += int(((lo == tl) & (hi == th))[ok].sum())
        s["n"] += int(ok.sum())
        for d in range(3):
            m = ok & (tdos == d) & (nm > 0)
            s["rows"][d] += m.sum()
            s["colin"][d] += (m & (nmd == 0)).sum()
            s["repl"][d] += (m & (nmd == nm)).sum()
    out = {}
    for kind, s in stats.items():
        c = s["conf"]
        rows = {str(d): {"share": float(c[d].sum() / c.sum()),
                         "pred_0_1_2": (c[d] / c[d].sum()).tolist() if c[d].sum() else None,
                         "collinear_row_share": float(s["colin"][d] / s["rows"][d]) if s["rows"][d] else None,
                         "replacement_row_share": float(s["repl"][d] / s["rows"][d]) if s["rows"][d] else None}
                for d in range(3)}
        out[kind] = {"pair_acc": s["pair_ok"] / max(s["n"], 1), "rows": s["n"], "by_true_dosage": rows}
        print(f"\n== {kind}: {s['n']:,} rows, pair accuracy {100 * out[kind]['pair_acc']:.1f}%")
        for d in range(3):
            r = rows[str(d)]
            if r["pred_0_1_2"] is None:
                print(f"   true {d} deleted: share 0.0% (never occurs)")
                continue
            p = r["pred_0_1_2"]
            print(f"   true {d} deleted: share {100 * r['share']:5.1f}%  pred 0/1/2 = "
                  f"{100 * p[0]:5.1f}/{100 * p[1]:5.1f}/{100 * p[2]:5.1f}   rows collinear "
                  f"{100 * (r['collinear_row_share'] or 0):5.1f}% / replacement {100 * (r['replacement_row_share'] or 0):5.1f}%")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
