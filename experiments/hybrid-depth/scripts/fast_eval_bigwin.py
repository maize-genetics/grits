#!/usr/bin/env python
"""fast_eval_ckpt for a model trained on longer windows (e.g. T=4096), from the existing 512-row inputs.

Per contig, the 512-row windows of windowed_k25native_wcount.npy are concatenated and cut into
--window-rows windows; the last one is end-aligned (it overlaps the previous one) so the rows the
converter kept are all decoded and coverage matches the 512-row model. Overlapping rows keep the earlier
window's call. Contigs with fewer rows than one long window are left uncalled. Calls are mapped back onto
the 512-row layout and written with heldout_assembly_eval.write_imputed_bed (gap-filling), then scored
with fast_snprc_score exactly as fast_eval_ckpt does. Writes results/<tag>_<scope>_fast/ like fast_eval_ckpt.
"""
import argparse
import contextlib
import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/sh-distfix-wt/experiments/simval-corpus/scripts")
sys.path.insert(0, str(SCRIPTS))
import fast_eval_ckpt as fe  # noqa: E402  (sets up sys.path / imports as in production)

hae, fss, sei = fe.hae, fe.fss, fe.sei
infer_real_founder_pairs, GRITSCRFDiploidIndel = fe.infer_real_founder_pairs, fe.GRITSCRFDiploidIndel


def long_windows(layout, T0, T):
    """[(row_start, row_end_exclusive)] in flat 512-layout row coordinates, end-aligned last window."""
    out, base = [], 0
    for _c, _pos, n in layout:
        rows = n * T0
        if rows >= T:
            starts = list(range(0, rows - T + 1, T))
            if starts[-1] + T < rows:
                starts.append(rows - T)
            out += [(base + s, base + s + T) for s in starts]
        base += rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scope", nargs="+", choices=sorted(fe.SCOPES), default=["out", "idx"])
    ap.add_argument("--window-rows", type=int, default=4096)
    ap.add_argument("--knockout", default=None)
    ap.add_argument("--depth", default="0.1")
    ap.add_argument("--manifest", default=fe.MANIFEST)
    ap.add_argument("--input-glob", default="scratch/lift_s200_local20k")
    ap.add_argument("--route", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these individuals")
    ap.add_argument("--decode", choices=["viterbi", "wholechrom"], default="viterbi")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GRITSCRFDiploidIndel.load_from_checkpoint(args.ckpt, map_location=device, strict=False).eval().to(device)
    panel = fss.Panel()
    rows = [r for r in csv.DictReader(open(args.manifest), delimiter="\t") if r["coverage"] == args.depth]
    T = args.window_rows
    for scope in args.scope:
        out_root = fe.WORK / "scratch" / f"{args.tag}_{scope}_fast"
        res_dir = fe.WORK / "results" / f"{args.tag}_{scope}_fast"
        res_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            ds, ind = r["dataset_id"], r["individual"]
            if ds not in fe.SCOPES[scope] or (args.only and ind not in args.only):
                continue
            t0 = time.time()
            try:
                src = fe.source_dir(ds, ind, args.depth, args.input_glob)
            except FileNotFoundError as e:
                print(f"SKIP {e}", flush=True)
                continue
            key = f"{args.tag}__{ds}__{ind}__{args.depth}x"
            data = np.load(src / "windowed_k25native_wcount.npy")
            nw0, T0, F = data.shape
            assert T % T0 == 0, (T, T0)
            layout = hae.load_contig_layout(src / "raw.npy.bins.tsv", T0, bin_size=256)
            flat = data.reshape(nw0 * T0, F)
            wins = long_windows(layout, T0, T)
            big = np.stack([flat[a:b] for a, b in wins])
            row_bin = None
            if args.route:
                rb = np.concatenate([np.asarray(pos, np.int64) + (ci << 40) for ci, (_c, pos, _n) in enumerate(layout)])
                row_bin = np.stack([rb[a:b] for a, b in wins])
            own = np.ones((len(wins), T), bool)
            for i in range(1, len(wins)):
                (pa, pb), (a, b) = wins[i - 1], wins[i]
                if a < pb and b > pb:              # end-aligned last window overlaps the previous one
                    own[i, :pb - a] = False
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                lo, hi = infer_real_founder_pairs(model, big, sei.K, device=device, route=args.route,
                                                  row_bin=row_bin, knockout=args.knockout, decode=args.decode,
                                                  own=own if args.decode == "wholechrom" else None,
                                                  batch_size=max(1, 256 * T0 // T))   # same rows per batch as production
                n_g = len(hae.load_gamete_names(src / "raw.npy.gametes.tsv"))
                plo = np.full(nw0 * T0, n_g, np.int64)
                phi = np.full(nw0 * T0, n_g, np.int64)
                done = np.zeros(nw0 * T0, bool)
                for (a, b), l, h in zip(wins, np.asarray(lo), np.asarray(hi)):
                    m = ~done[a:b]
                    plo[a:b][m], phi[a:b][m] = l[m], h[m]
                    done[a:b] = True
                bed_dir = out_root / key / "bed"
                hae.write_imputed_bed(ind, plo.reshape(nw0, T0), phi.reshape(nw0, T0), None,
                                      hae.load_gamete_names(src / "raw.npy.gametes.tsv"),
                                      src / "raw.npy.bins.tsv", bed_dir, bin_size=256)
            chosen = [l for l in buf.getvalue().splitlines() if "homo_scale=" in l or "route:" in l]
            f1, f2 = fss.pair_from_bed(panel, bed_dir)
            res = fss.score_arrays(panel, fss.load_truth(r["truth_h1"]), fss.load_truth(r["truth_h2"]), f1, f2)
            res.update(row_key=key, status="ok", source_input=str(src), window_rows=T,
                       rows_decoded_frac=round(float(done.mean()), 5),
                       homo_scale_log=" | ".join(c.strip() for c in chosen) if chosen else None,
                       wall_seconds=round(time.time() - t0, 1))
            (res_dir / f"{key}.json").write_text(json.dumps(res, indent=1))
            print(f"{key:<55} err={100 * res['error_rate']:6.2f}%  snprc={100 * res['snprc_error_rate']:6.2f}%  "
                  f"decoded={100 * done.mean():.2f}%  {res['homo_scale_log'] or ''}  ({res['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
