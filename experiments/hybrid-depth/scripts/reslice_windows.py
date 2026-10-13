#!/usr/bin/env python
"""Re-slice a <prefix>_fullscale_sliced set into longer windows.

gen_training_data.py simulates each individual as one long row sequence and slices it into
G consecutive T-row windows (individual i = windows i*G .. i*G+G-1), so the G*T rows of an
individual are one genome path. This concatenates them back and cuts G2 = (G*T)//T2 windows of
T2 rows, dropping the remainder (from the end of each individual). Applied identically to the
data, .lin, .prov and .ood sidecars. Never touches the inputs.

usage: reslice_windows.py IN_PREFIX OUT_PREFIX --G 117 --T2 4096
  (prefixes without _fullscale_sliced.npy)
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("in_prefix")
ap.add_argument("out_prefix")
ap.add_argument("--G", type=int, default=117)
ap.add_argument("--T2", type=int, default=4096)
ap.add_argument("--chunk-ind", type=int, default=100)
a = ap.parse_args()
for suf in (".npy", ".lin.npy", ".prov.npy", ".ood.npy"):
    src = os.path.realpath(a.in_prefix + "_fullscale_sliced" + suf)
    dst = a.out_prefix + "_fullscale_sliced" + suf
    if not os.path.exists(src):
        print("skip (missing)", src)
        continue
    X = np.load(src, mmap_mode="r")
    N, T = X.shape[:2]
    G = a.G
    assert N % G == 0, (N, G)
    n_ind = N // G
    G2 = (G * T) // a.T2
    rest = X.shape[2:]
    Y = np.lib.format.open_memmap(dst, mode="w+", dtype=X.dtype, shape=(n_ind * G2, a.T2) + rest)
    for s in range(0, n_ind, a.chunk_ind):
        e = min(n_ind, s + a.chunk_ind)
        x = np.asarray(X[s * G:e * G]).reshape((e - s, G * T) + rest)
        Y[s * G2:e * G2] = x[:, :G2 * a.T2].reshape(((e - s) * G2, a.T2) + rest)
    Y.flush()
    print(f"{dst}: {X.shape} -> {Y.shape}  (--windows-per-individual {G2}, dropped "
          f"{G * T - G2 * a.T2} rows/individual)", flush=True)
