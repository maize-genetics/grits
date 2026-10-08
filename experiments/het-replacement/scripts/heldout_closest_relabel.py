#!/usr/bin/env python
"""Replace held-out augment v2's null labels with the closest visible founder.

heldout_augment_v2.py labels a row whose true founder is hidden with a visible lineage-mate, or
LABEL_PAD (-> the CRF's null founder) when none is visible. The null label teaches the CRF to
abstain wherever the panel fits poorly, so on real out-of-panel samples whole regions get no call.
Here every such cell gets the visible founder whose sequence the hidden haplotype's own reads match
best nearby: over clean rows from the same haplotype whose true founder is hidden (provenance kind
and .ood bit), within +-W rows (whole window if fewer than --min-evidence informative calls),
score = (matches + 1) / (matches + mismatches + 2) on the ternary input; argmax, lowest index on ties.
Under --tie-aware-loss the founder's lineage-mates at the row are allowed too.

In:  <in>_fullscale_sliced.{npy,lin.npy,prov.npy,ood.npy} from heldout_augment_v2.py (3K+2 columns)
Out: <out>_fullscale_sliced.npy with relabelled K/K+1 columns; lin/prov/ood are symlinked."""
import argparse
import os
from pathlib import Path

import numpy as np

PROV_KIND_MASK, PROV_OFF_SITE, PROV_BAD_SITE = 7, 8, 16


def window_sum(a, w):
    """Sum of a[:, t-w : t+w+1] along axis 1 (zero-padded), a [B,T,K]."""
    c = np.cumsum(np.pad(a, ((0, 0), (1, 0), (0, 0))), axis=1)
    T = a.shape[1]
    hi = np.minimum(np.arange(T) + w + 1, T)
    lo = np.maximum(np.arange(T) - w, 0)
    return c[:, hi] - c[:, lo]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-prefix", required=True, help="path prefix, without _fullscale_sliced.npy")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--num-parents", type=int, default=25)
    ap.add_argument("--half-window", type=int, default=64, help="rows on each side")
    ap.add_argument("--min-evidence", type=int, default=8, help="informative calls needed locally")
    ap.add_argument("--chunk", type=int, default=2000)
    a = ap.parse_args()
    K = a.num_parents
    src = a.in_prefix + "_fullscale_sliced"
    dst = a.out_prefix + "_fullscale_sliced"
    X = np.load(src + ".npy", mmap_mode="r")
    P = np.load(src + ".prov.npy", mmap_mode="r")
    O = np.load(src + ".ood.npy", mmap_mode="r")
    N, T, _ = X.shape
    Y = np.lib.format.open_memmap(dst + ".npy", mode="w+", dtype=X.dtype, shape=X.shape)
    n_null = n_fixed = n_global = 0
    for s in range(0, N, a.chunk):
        x = np.array(X[s:s + a.chunk])
        prov = np.asarray(P[s:s + a.chunk]).astype(np.int16)
        ood = np.asarray(O[s:s + a.chunk]).astype(np.int16)
        tern = np.rint(x[..., :K]).astype(np.int8)
        clean = (prov >= 0) & ((prov & (PROV_OFF_SITE | PROV_BAD_SITE)) == 0)
        hap = (prov & PROV_KIND_MASK) % 2
        for c in range(2):
            lab = x[..., K + c]
            hidden = ((ood >> c) & 1).astype(bool)
            target = hidden & (lab < 0)
            n_null += int(target.sum())
            if not target.any():
                continue
            ev = (clean & (hap == c) & hidden)[..., None]
            m = ((tern == 1) & ev).astype(np.int32)
            mm = ((tern == 0) & ev).astype(np.int32)
            ml, mml = window_sum(m, a.half_window), window_sum(mm, a.half_window)
            mg, mmg = m.sum(1, keepdims=True), mm.sum(1, keepdims=True)
            local_ok = (ml + mml).max(-1) >= a.min_evidence                  # [B,T]
            use_m = np.where(local_ok[..., None], ml, mg)
            use_mm = np.where(local_ok[..., None], mml, mmg)
            score = (use_m + 1.0) / (use_m + use_mm + 2.0)
            best = score.argmax(-1)
            informative = (use_m + use_mm).max(-1) > 0
            fix = target & informative
            lab[fix] = best[fix]
            x[..., K + c] = lab
            n_fixed += int(fix.sum())
            n_global += int((fix & ~local_ok).sum())
        Y[s:s + a.chunk] = x
        print(f"{min(s + a.chunk, N)}/{N} windows | null cells {n_null:,} relabelled {n_fixed:,} "
              f"(whole-window fallback {n_global:,})", flush=True)
    Y.flush()
    for ext in (".lin.npy", ".prov.npy", ".ood.npy"):
        if not os.path.exists(dst + ext):
            os.symlink(os.path.abspath(src + ext), dst + ext)
    print(f"done: {n_fixed:,}/{n_null:,} null cells relabelled; {n_null - n_fixed:,} left null "
          f"(no informative read from that haplotype in the window)")


if __name__ == "__main__":
    main()
