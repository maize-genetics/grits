"""
Convert a ropebwt3-phg PS4G/numpy export (`refmap --npy=... --ps4g=... --label-bed=...`,
branch `worktree-refmap-ps4g-numpy` of maize-genetics/ropebwt3-phg) into the standard
grits training-matrix contract consumed by `crf/eval.py` / `crf/train_haploid.py` /
`crf/train_diploid.py` (`make_splits` / `make_diploid_splits`).

The two formats differ in three load-bearing ways (not just a dtype cast):
  1. dtype: ropebwt3-phg writes int32; grits expects int8 (read-counts clipped to [0,127],
     matching the convention already used by `cross/build_training_data.py`).
  2. shape: ropebwt3-phg's `.npy` is FLAT — one row per (contig, bin, gameteSet), with a
     companion `<npy>.bins.tsv` (row -> contig, bin) and `<npy>.gametes.tsv` (column -> founder
     name). grits expects PRE-WINDOWED `[N, window_size, K+2]` blocks. This script groups rows
     by contig (bins.tsv is already sorted contig-then-bin) and slices consecutive
     `window_size`-row chunks, matching the same non-collapsed, row-not-position windowing
     convention the external `extract_windows_single.py` scripts use for the maize/cassava
     training sets, dropping a trailing partial window per contig.
  3. unknown-label sentinel: ropebwt3-phg writes `-1` for unlabeled bins (a standard,
     tool-agnostic missing-value convention). grits' `PreWindowedHaploidDataset`/`make_splits`
     silently mis-handles `-1` as founder-index 0 (`np.clip(labels, 0, K)`), NOT as the CRF's
     real "unknown" state (index `K`) — so this script remaps `-1 -> K` explicitly.

This fix is deliberately kept downstream (here), not pushed into ropebwt3-phg's exporter: the
windowing size and the unknown-state encoding are grits-CRF-specific choices, and even grits
itself keeps windowing external to its own PS4G parser.

Optional: `--target-num-parents` trims a real panel down to a checkpoint's trained `num_parents`
(e.g. a real K=25 panel -> K=24 to match a checkpoint trained on 24 simulated founders, no
retraining needed). Drops the least-covered founder(s) genome-wide — "least covered" = fewest
rows with any nonzero read support for that founder — until the panel reaches the target size.
Any true label that pointed at a dropped founder becomes unknown (state `K`), same as an
unlabeled bin; those instances are genuinely unrepresentable in the smaller panel.

Usage:
    python src/python/crf/ropebwt_npy_to_matrix.py \
        --npy ropebwt_refMap/bench_ps4g_npy/results_fixed/full_1M.npy \
        --bins ropebwt_refMap/bench_ps4g_npy/results_fixed/full_1M.npy.bins.tsv \
        --gametes ropebwt_refMap/bench_ps4g_npy/results_fixed/full_1M.npy.gametes.tsv \
        --num-parents 25 --window-size 512 \
        --out grits_workdir/data/real/ropebwt_oh43_1M.npy
        [--target-num-parents 24]

To point at new reads later, only --npy/--bins/--gametes (and --out) need to change.

--- --anchor-dist-npy mode -------------------------------------------------
`refmap --anchor-dist-npy` (branch `lift-ridx-ternary-dist-map`, requires `--lift`)
widens the export from `[n_rows, K+2]` (read counts) to `[n_rows, 3K+2]`: counts,
then a per-founder ternary read-sharing block (match=1/diverged=0/deletion=-1,
`rb3_lift_ternary_state`), then a per-founder bp-distance-to-nearest-lift-anchor
block (-1 = no anchor at all on this reference sequence), then gA/gB — see the
`<npy>.layout.tsv` sidecar it writes. Pass `--anchor-dist-npy` here to read that
layout instead and emit the `[N, window_size, 2K+2]` ternary+distance training
contract `simulate_alleles.py --simulate-indels` also produces (`[ternary(K) | H1 |
H2 | distance(K)]`), so real and simulated indel-aware data share one format.

Distance is log-coded exactly like the simulator's `_encode_dist`
(`DIST_LOG_SCALE=8.0`): `code = clip(round(8*log2(1+d)), 0, 126)`, `127` reserved
for values that saturate the code (not used by real bp distances at this scale,
kept for contract parity with the simulator); refmap's `-1` (no anchor) maps to
`DIST_PAD=-1`, matching the simulator's own sentinel. `--target-num-parents` is
not supported in this mode (real-data comparison/characterization only, not yet
wired for training-panel trimming).

Usage:
    python src/python/crf/ropebwt_npy_to_matrix.py \
        --anchor-dist-npy \
        --npy scratch/simval_eval/.../raw.npy \
        --bins scratch/simval_eval/.../raw.npy.bins.tsv \
        --gametes scratch/simval_eval/.../raw.npy.gametes.tsv \
        --num-parents 25 --window-size 512 \
        --out grits_workdir/data/real/oh43xil14h_0.1x_ternary.npy
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="ropebwt3-phg PS4G/npy -> grits training matrix")
    p.add_argument("--npy", required=True,
                   help="ropebwt3-phg --npy export (int32, [n_rows, K+2]).")
    p.add_argument("--bins", required=True,
                   help="Companion <npy>.bins.tsv (row, contig, bin).")
    p.add_argument("--gametes", default=None,
                   help="Companion <npy>.gametes.tsv (gameteIndex, sampleName). Optional; "
                        "used only to sanity-check --num-parents and log founder order.")
    p.add_argument("--num-parents", type=int, default=None,
                   help="K. Inferred from --gametes or npy.shape[1]-2 if not given.")
    p.add_argument("--window-size", type=int, default=512,
                   help="Rows per window (matches grits' T=512 site-window convention).")
    p.add_argument("--step-size", type=int, default=None,
                   help="Window stride; defaults to --window-size (non-overlapping).")
    p.add_argument("--target-num-parents", type=int, default=None,
                   help="If given and < the source panel size, drop the least-covered "
                        "founder(s) genome-wide (fewest rows with any nonzero read support) "
                        "until the panel reaches this size. Labels pointing at a dropped "
                        "founder become unknown. Default: off, panel size unchanged.")
    p.add_argument("--out", required=True, help="Output .npy path.")
    p.add_argument("--anchor-dist-npy", action="store_true",
                   help="Source .npy is a `refmap --anchor-dist-npy` export "
                        "([n_rows, 3K+2]: counts | ternary | distance-bp | gA,gB), "
                        "not the plain [n_rows, K+2] read-count layout. Emits the "
                        "[N, window_size, 2K+2] ternary+distance training contract "
                        "instead of [N, window_size, K+2].")
    p.add_argument("--dist-log-scale", type=float, default=8.0,
                   help="Log-scale factor for distance-to-anchor coding, matching "
                        "simulate_alleles.py's DIST_LOG_SCALE default.")
    return p.parse_args()


def main():
    args = parse_args()
    window_size = args.window_size
    step_size = args.step_size or window_size

    arr = np.load(args.npy, mmap_mode="r")
    bins_df = pd.read_csv(args.bins, sep="\t")
    assert list(bins_df.columns) == ["row", "contig", "bin"], \
        f"unexpected bins.tsv columns: {list(bins_df.columns)}"
    assert len(bins_df) == arr.shape[0], (
        f"row-count mismatch: npy has {arr.shape[0]} rows, bins.tsv has {len(bins_df)}")
    assert np.array_equal(bins_df["row"].to_numpy(), np.arange(len(bins_df))), \
        "bins.tsv 'row' column is not a plain 0..N-1 sequence — cannot align by position"

    gametes = None
    if args.gametes:
        gametes = pd.read_csv(args.gametes, sep="\t")
        assert list(gametes.columns) == ["gameteIndex", "sampleName"], \
            f"unexpected gametes.tsv columns: {list(gametes.columns)}"

    K = args.num_parents
    if K is None:
        divisor = 3 if args.anchor_dist_npy else 1
        K = (len(gametes) if gametes is not None
             else (arr.shape[1] - 2) // divisor)
    if gametes is not None:
        assert len(gametes) == K, (
            f"--num-parents {K} disagrees with {len(gametes)} founders in --gametes")

    if args.anchor_dist_npy:
        if args.target_num_parents is not None:
            raise SystemExit("--target-num-parents is not supported with "
                              "--anchor-dist-npy (see module docstring)")
        assert arr.shape[1] == 3 * K + 2, (
            f"npy has {arr.shape[1]} cols; expected 3K+2={3 * K + 2} for K={K} "
            f"(counts | ternary | distance | gA,gB — did you mean plain mode?)")
        arr = np.asarray(arr)                                  # materialize the mmap
        tern = arr[:, K:2 * K].astype(np.int8)                 # already -1/0/1
        dist_bp = arr[:, 2 * K:3 * K].astype(np.int64)
        gA = arr[:, 3 * K].astype(np.int64)
        gB = arr[:, 3 * K + 1].astype(np.int64)
        gA[gA < 0] = K
        gB[gB < 0] = K

        DIST_PAD, DIST_SAT = -1, 127
        has_anchor = dist_bp >= 0
        code = np.rint(args.dist_log_scale * np.log2(1.0 + np.maximum(dist_bp, 0).astype(np.float64)))
        code = np.clip(code, 0, DIST_SAT - 1).astype(np.int8)
        dist = np.where(has_anchor, code, DIST_PAD).astype(np.int8)

        labels = np.stack([gA, gB], axis=1).astype(np.int8)    # [n_rows, 2]
        rows = np.concatenate([tern, labels, dist], axis=1)    # [n_rows, 2K+2] int8, matches
                                                                 # simulate_alleles.py's [ternary(K)|H1|H2|distance(K)]
    else:
        assert arr.shape[1] == K + 2, (
            f"npy has {arr.shape[1]} cols; expected K+2={K + 2} for K={K}")

        arr = np.asarray(arr)                                  # materialize the mmap
        feats = np.clip(arr[:, :K], 0, 127).astype(np.int8)
        gA = arr[:, K].astype(np.int64)
        gB = arr[:, K + 1].astype(np.int64)
        gA[gA < 0] = K                                          # -1 -> real "unknown" state
        gB[gB < 0] = K

        if args.target_num_parents is not None:
            target = args.target_num_parents
            if target > K:
                raise ValueError(f"--target-num-parents {target} > source panel size {K}")
            if target < K:
                hits = (feats != 0).sum(axis=0)                # [K] rows-with-coverage, genome-wide
                n_drop = K - target
                drop_idx = np.argsort(hits, kind="stable")[:n_drop]
                keep_idx = np.setdiff1d(np.arange(K), drop_idx)  # sorted ascending
                names = (gametes.sort_values("gameteIndex")["sampleName"].to_numpy()
                         if gametes is not None else np.array([str(i) for i in range(K)]))
                print(f"Dropping {n_drop} least-covered founder(s) (target K={target}, source K={K}):")
                for i in np.sort(drop_idx):
                    print(f"  {names[i]:<12} idx={i:<3} hits={hits[i]:,}")
                print(f"  lowest kept: {names[keep_idx[np.argmin(hits[keep_idx])]]}  "
                      f"hits={hits[keep_idx].min():,}")

                remap = np.full(K + 1, target, dtype=np.int64)  # default: dropped + old unknown -> new unknown
                remap[keep_idx] = np.arange(len(keep_idx))      # kept founders -> compacted index
                gA = remap[gA]
                gB = remap[gB]
                feats = feats[:, keep_idx]
                K = target
            else:
                print(f"--target-num-parents {target} == source panel size; nothing to drop")

        labels = np.stack([gA, gB], axis=1).astype(np.int8)    # [n_rows, 2]
        rows = np.concatenate([feats, labels], axis=1)         # [n_rows, K+2] int8

    windows = []
    contig_window_counts = {}
    for contig, idx in bins_df.groupby("contig", sort=False).indices.items():
        idx = np.sort(idx)                                     # preserve on-disk order
        n_windows = 0
        start = 0
        while start + window_size <= len(idx):
            sel = idx[start:start + window_size]
            windows.append(rows[sel])
            n_windows += 1
            start += step_size
        contig_window_counts[contig] = n_windows

    if not windows:
        raise ValueError("no complete windows produced — window_size too large for the data")
    out = np.stack(windows, axis=0)                            # [N, window_size, K+2] int8

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)

    het = (out[:, :, K] != out[:, :, K + 1])
    labeled = (out[:, :, K] != K) | (out[:, :, K + 1] != K)
    print(f"\nWrote {out_path}")
    print(f"  shape={out.shape}  dtype={out.dtype}  size={out.nbytes / 1e9:.3f} GB")
    print(f"  source rows={arr.shape[0]:,}  windows={out.shape[0]:,}  "
          f"window_size={window_size}  step_size={step_size}")
    nonzero = {c: n for c, n in contig_window_counts.items() if n > 0}
    n_empty = len(contig_window_counts) - len(nonzero)
    print(f"  contigs             : {len(contig_window_counts)} seen, "
          f"{len(nonzero)} contributed windows, {n_empty} too sparse (<{window_size} rows)")
    print(f"    " + ", ".join(f"{c}={n}" for c, n in sorted(nonzero.items())))
    print(f"  label coverage      : {labeled.mean()*100:.1f}%  "
          f"(unknown state = index {K})")
    print(f"  het (gA != gB)      : {het.mean()*100:.1f}%")
    if args.anchor_dist_npy:
        tern_out = out[:, :, :K]
        dist_out = out[:, :, K + 2:]
        either_covered = (tern_out == 1).any(axis=2).mean()
        print(f"  ternary match/div/del: "
              f"{(tern_out==1).mean()*100:.1f}% / {(tern_out==0).mean()*100:.1f}% / "
              f"{(tern_out==-1).mean()*100:.1f}%")
        print(f"  either-founder-covered (>=1 match per site): {either_covered*100:.1f}%")
        print(f"  distance code: pad(no anchor)={ (dist_out==-1).mean()*100:.2f}%  "
              f"sat(127)={(dist_out==127).mean()*100:.2f}%  "
              f"mean(non-pad)={dist_out[dist_out>=0].mean():.1f}")


if __name__ == "__main__":
    main()
