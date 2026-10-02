#!/usr/bin/env python
"""
Reusable position-jitter transform for the 100kbp position-randomization
probe -- see /home/zrm22/.claude/plans/wondrous-discovering-octopus.md
("RIL2 indel-density diagnostics"). The CRF model has NO explicit position
feature (GRITSCRFDiploid.forward()'s transition cost `c` comes from
self.encoder(X_pad, ...), a function of the read-support matrix only) --
position affects the model only indirectly, through (a) row order (windows
are 512 consecutive on-disk rows, row-count-based not position-based) and
(b) reported BED coordinates. jitter_positions() perturbs both at once by
reassigning each row's bin to a random value drawn uniformly within its own
100kbp window, then re-sorting all rows (per contig) by the new bin -- a
direct, cheap test of whether the model/pipeline is sensitive to local
read-order/position structure. Per explicit user decision, the jittered
position is used for scoring too, not just for reordering model input.
"""
import numpy as np
import pandas as pd


def jitter_positions(bins_df, arr, window_bp=100_000, bin_size=256, seed=None):
    """Randomly reassign each row's genomic bin within its own window_bp-wide
    window, then re-sort (per contig, preserving original first-appearance
    contig order) by the new bin. Returns (new_bins_df, new_arr), same shape
    and dtype as the inputs -- a pure reorder, nothing dropped or added.

    bins_df: DataFrame with columns ["row", "contig", "bin"] (ropebwt3-phg's
    <npy>.bins.tsv schema). arr: the companion raw.npy array, arr.shape[0]
    == len(bins_df).

    Every jittered bp is guaranteed to land within [window_floor,
    window_floor + window_bp) AND within [0, that contig's own max observed
    bp] -- the latter clip handles the rare tail case where a true position
    falls in a chromosome's last, truncated 100kbp window (window_floor +
    window_bp could otherwise exceed the contig's real extent)."""
    assert list(bins_df.columns) == ["row", "contig", "bin"], \
        f"unexpected bins.tsv columns: {list(bins_df.columns)}"
    assert len(bins_df) == arr.shape[0], (
        f"row-count mismatch: bins.tsv has {len(bins_df)} rows, arr has {arr.shape[0]}")

    rng = np.random.default_rng(seed)
    contigs = bins_df["contig"].to_numpy()
    bp = bins_df["bin"].to_numpy(dtype=np.int64) * bin_size

    window_floor = (bp // window_bp) * window_bp
    offset = rng.integers(0, window_bp, size=len(bp))
    new_bp = window_floor + offset

    # Clip to this contig's own max observed bp so a jittered position never
    # falls beyond the real genome this data actually covers (only matters
    # for reads in a chromosome's final, possibly-truncated 100kbp window).
    contig_max_bp = pd.Series(bp).groupby(contigs).transform("max").to_numpy()
    new_bp = np.minimum(new_bp, contig_max_bp)
    new_bin = new_bp // bin_size

    # Preserve original first-appearance contig order (matches
    # ropebwt_npy_to_matrix.py's own `groupby("contig", sort=False)`
    # windowing convention) -- sort primarily by that, then by the new bin
    # within each contig, with original row index as the final, stable
    # tiebreak so re-running with the same seed is fully reproducible.
    _, first_seen = np.unique(contigs, return_index=True)
    contig_rank_by_name = {c: i for i, c in enumerate(contigs[np.sort(first_seen)])}
    contig_rank = np.array([contig_rank_by_name[c] for c in contigs])
    orig_idx = np.arange(len(bins_df))
    order = np.lexsort((orig_idx, new_bin, contig_rank))  # primary key = last arg

    new_bins_df = pd.DataFrame({
        "row": np.arange(len(order)),
        "contig": contigs[order],
        "bin": new_bin[order],
    })
    new_arr = arr[order]
    return new_bins_df, new_arr


def jitter_mosaic_positions(mosaic, window_bp=100_000, seed=None):
    """Jitter-noise-floor control: apply the same within-100kbp reassignment
    directly to a TRUE mosaic's own breakpoints (no model, no refmap) --
    quantifies how much founder-path error a PERFECT decode would still show
    once its reported boundaries are snapped to this jittered grid, so real
    posrand100kb results can be read against that floor rather than assumed
    to be model error. mosaic: {chrom: [(start, end, label), ...]}. Returns a
    same-shaped mosaic with (start, end) replaced by jittered breakpoints
    (start of segment i+1 == end of segment i, so the founder-path scorer's
    breakpoint list stays a clean partition)."""
    rng = np.random.default_rng(seed)
    out = {}
    for chrom, segs in mosaic.items():
        # Jitter every interior breakpoint (the boundary between segs[i] and
        # segs[i+1]); the first start and last end are genome edges, left
        # untouched.
        starts = [s for s, _, _ in segs]
        new_breaks = []
        for b in starts[1:]:
            window_floor = (b // window_bp) * window_bp
            new_breaks.append(window_floor + int(rng.integers(0, window_bp)))
        new_breaks = sorted(new_breaks)
        new_starts = [segs[0][0]] + new_breaks
        new_ends = new_breaks + [segs[-1][1]]
        out[chrom] = [(s, e, lab) for s, e, (_, _, lab) in zip(new_starts, new_ends, segs)]
    return out
