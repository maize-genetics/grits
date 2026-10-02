#!/usr/bin/env python
"""
Per-sample adaptive founder-drop selection -- factored out of
ropebwt_npy_to_matrix.py's own `--target-num-parents` logic (`hits = (feats
!= 0).sum(axis=0)`, drop the argmin) so driver scripts can compute and pin
ONE drop_idx per (sample, coverage) from that row's own raw.npy, instead of
reusing a single index hardcoded from an unrelated pilot row.

User-flagged bug (2026-08-24): every RIL2 driver script this session
hardcoded drop_idx=23 (P39), pinned once from "pilot row 3" and reused for
every sample since. For Oh43xIl14H specifically, P39 is NOT low-hit -- it
ranks 23rd of 25 FROM THE BOTTOM (i.e. 3rd from the top) at every coverage
checked; the sample's own genuinely lowest-hit founder is CML277. Hit
counts are computed from `(feats != 0).sum(axis=0)` over ALL rows of the
raw K25 npy, so they don't depend on row ORDER -- the same drop_idx applies
unchanged whether the npy is in position order or jitter-reordered, so it's
safe to compute once from the raw (pre-jitter) npy and reuse across every
jitter-window arm for a given (sample, coverage).
"""
import numpy as np


def adaptive_drop_idx(raw_npy_path, gamete_names, num_parents=25):
    """Returns (drop_idx, dropped_name, hits_array) -- the lowest-hit
    founder's index into gamete_names, for THIS sample's own raw.npy."""
    arr = np.load(raw_npy_path, mmap_mode="r")
    feats = arr[:, :num_parents]
    hits = (feats != 0).sum(axis=0)
    drop_idx = int(np.argmin(hits))
    return drop_idx, gamete_names[drop_idx], hits
