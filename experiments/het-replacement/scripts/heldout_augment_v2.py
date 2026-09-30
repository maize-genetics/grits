#!/usr/bin/env python
"""Held-out augmentation for tie-aware training: turn a set simulated with K+2 founders into a
K-founder set where a fraction of individuals are out-of-panel, like the real OUT/MIX samples.

Per individual, two founder columns are dropped (the model always sees K):
  in-panel  (1 - frac):  two founders the individual does not carry
  one hidden (frac * p_one): one of its own founders + one it does not carry  (MIX / OUT-inbred-like)
  both hidden (the rest of frac, outbred only): both of its founders          (OUT-hybrid-like)
Rows whose true founder is hidden are relabelled to a VISIBLE founder sharing its ancestral lineage
at the row (lowest index). Under --tie-aware-loss every lineage-mate is an allowed answer, so the
pick carries no information -- unlike the first held-out augment, whose per-site random pick was
label noise. The allowed set changes wherever the hidden founder's lineage changes, which is the
real "closest panel founder changes" signal. Rows with no visible lineage-mate get LABEL_PAD.
Supervised-head targets (het, lineage-aware switch, support gate, affinity) are all
lineage-based, so they are unaffected by which mate is picked.

In:  <in>_fullscale_sliced.{npy,lin.npy,prov.npy} generated with founders K+2 (3(K+2)+2 columns)
Out: <out>_fullscale_sliced.{npy,lin.npy,prov.npy} with 3K+2 columns, + <out>.heldout.npy [n_ind,3]
     (mode 0/1/2, hidden founder a, hidden founder b)."""
import argparse
from pathlib import Path

import numpy as np

LABEL_PAD = -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-prefix", required=True, help="path prefix, without _fullscale_sliced.npy")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--num-parents", type=int, default=25, help="K visible founders (input has K+2)")
    ap.add_argument("--windows-per-individual", type=int, default=117)
    ap.add_argument("--frac", type=float, default=0.3, help="share of individuals with a hidden own founder")
    ap.add_argument("--p-one", type=float, default=0.5, help="outbred: share of hidden cases hiding only one parent")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    K, G = a.num_parents, a.windows_per_individual
    K2 = K + 2
    X = np.load(a.in_prefix + "_fullscale_sliced.npy", mmap_mode="r")
    L = np.load(a.in_prefix + "_fullscale_sliced.lin.npy", mmap_mode="r")
    P = np.load(a.in_prefix + "_fullscale_sliced.prov.npy", mmap_mode="r")
    if X.shape[2] != 3 * K2 + 2:
        raise SystemExit(f"expected {3 * K2 + 2} columns (K+2={K2} founders), got {X.shape[2]}")
    N, T = X.shape[:2]
    n_ind = N // G
    rng = np.random.default_rng(a.seed)
    Xo = np.lib.format.open_memmap(a.out_prefix + "_fullscale_sliced.npy", mode="w+", dtype=X.dtype, shape=(N, T, 3 * K + 2))
    Lo = np.lib.format.open_memmap(a.out_prefix + "_fullscale_sliced.lin.npy", mode="w+", dtype=L.dtype, shape=(N, T, K))
    np.save(a.out_prefix + "_fullscale_sliced.prov.npy", np.asarray(P))
    meta = np.zeros((n_ind, 3), np.int16)
    stats = np.zeros(4)
    for i in range(n_ind):
        sl = slice(i * G, (i + 1) * G)
        x = np.asarray(X[sl]); lin = np.asarray(L[sl]).astype(np.int64)
        lab = x[:, :, K2:K2 + 2].astype(np.int64)
        carried = np.unique(lab[lab >= 0])
        others = np.setdiff1d(np.arange(K2), carried)
        # outbred = the two haplotypes ever differ (a simulated inbred carries several founders
        # along the genome through crossovers, so "carries more than one founder" is not the test)
        outbred = bool(((lab[..., 0] != lab[..., 1]) & (lab[..., 0] >= 0) & (lab[..., 1] >= 0)).any())
        mode = 0
        if rng.random() < a.frac and len(carried):
            mode = 2 if (outbred and len(carried) > 1 and rng.random() >= a.p_one) else 1
        if mode == 0:
            hid = rng.choice(others, 2, replace=False)
        elif mode == 1:
            hid = np.array([rng.choice(carried), rng.choice(others)])
        else:
            hid = rng.choice(carried, 2, replace=False)
        meta[i] = (mode, hid[0], hid[1])
        keep = np.setdiff1d(np.arange(K2), hid)                       # visible founders, K
        newidx = np.full(K2, -1); newidx[keep] = np.arange(K)
        for c in range(2):
            h = lab[..., c]
            hc = np.clip(h, 0, K2 - 1)
            is_hid = (h >= 0) & np.isin(h, hid)
            lh = np.take_along_axis(lin, hc[..., None], 2)             # [G,T,1] true founder's lineage
            mates = (lin[..., keep] == lh)                             # [G,T,K]
            has = mates.any(-1)
            first = mates.argmax(-1)
            h_new = np.where(h < 0, LABEL_PAD, newidx[hc])
            h_new = np.where(is_hid, np.where(has, first, LABEL_PAD), h_new)
            lab[..., c] = h_new
            stats += [is_hid.sum(), (is_hid & ~has).sum(), 0, 0]
        stats[2] += (lab >= 0).sum(); stats[3] += lab.size
        out = np.empty((G, T, 3 * K + 2), x.dtype)
        out[:, :, :K] = x[:, :, keep]
        out[:, :, K:K + 2] = lab.astype(x.dtype)
        out[:, :, K + 2:2 * K + 2] = x[:, :, K2 + 2 + keep]
        out[:, :, 2 * K + 2:3 * K + 2] = x[:, :, 2 * K2 + 2 + keep]
        Xo[sl] = out
        Lo[sl] = lin[..., keep].astype(L.dtype)
    Xo.flush(); Lo.flush()
    np.save(a.out_prefix + ".heldout.npy", meta)
    m = meta[:, 0]
    print(f"{n_ind} individuals: in-panel {np.mean(m == 0):.3f}, one hidden {np.mean(m == 1):.3f}, "
          f"both hidden {np.mean(m == 2):.3f} | hidden-founder label cells {int(stats[0])}, "
          f"of which no visible lineage-mate (-> pad) {stats[1] / max(stats[0], 1):.3%} | labelled {stats[2] / stats[3]:.3%}")


if __name__ == "__main__":
    main()
