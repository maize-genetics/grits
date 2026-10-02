#!/usr/bin/env python
"""Per-row lineage labels for --tie-aware-loss: <prefix>_fullscale_sliced.lin.npy [N,512,K] int8,
row-aligned with <prefix>_fullscale_sliced.npy. For each output row, the ancestral lineage of
every founder at the row's generation site (simulate()'s ibd [n,R,K] gathered by refpos [n,L]);
founders sharing a lineage at a site are IBD there and indistinguishable in the input."""
import argparse
from pathlib import Path

import numpy as np

T = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    parts = []
    for inb in ("1.0", "0.0"):
        ibd = np.load(d / f"{args.prefix}_inb{inb}.ibd.npy", mmap_mode="r")
        ref = np.load(d / f"{args.prefix}_inb{inb}.refpos.npy")
        out = np.empty(ref.shape + (ibd.shape[2],), dtype=np.int8)
        for i in range(ref.shape[0]):
            r = np.clip(ref[i], 0, ibd.shape[1] - 1)
            out[i] = np.asarray(ibd[i])[r]
        parts.append(out)
    comb = np.concatenate(parts, axis=0)
    G = comb.shape[1] // T
    dst = d / f"{args.prefix}_fullscale_sliced.lin.npy"
    np.save(dst, comb[:, :G * T].reshape(-1, T, comb.shape[2]))
    print(f"wrote {dst}  {comb.shape[0] * G} windows")


if __name__ == "__main__":
    main()
