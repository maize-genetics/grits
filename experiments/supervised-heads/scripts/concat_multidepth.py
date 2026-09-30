"""Concatenate <prefix>_d{0.1,0.5,1,2}_fullscale_sliced.{npy,lin.npy,prov.npy} into
<prefix>_multidepth_fullscale_sliced.* (same depth order as maize_v3prov_multidepth)."""
import sys
import numpy as np
D, prefix = sys.argv[1], sys.argv[2]
parts = [f"{D}/{prefix}_d{d}" for d in ("0.1", "0.5", "1", "2")]
import os
SUFS = ["_fullscale_sliced.npy", "_fullscale_sliced.lin.npy", "_fullscale_sliced.prov.npy"]
if all(os.path.exists(p + "_fullscale_sliced.ood.npy") for p in parts):
    SUFS.append("_fullscale_sliced.ood.npy")            # held-out augment sidecar
for suf in SUFS:
    arrs = [np.load(p + suf, mmap_mode="r") for p in parts]
    n = sum(len(a) for a in arrs)
    out = np.lib.format.open_memmap(f"{D}/{prefix}_multidepth{suf}", mode="w+", dtype=arrs[0].dtype,
                                    shape=(n,) + arrs[0].shape[1:])
    o = 0
    for a in arrs:
        for s in range(0, len(a), 20000):
            b = np.asarray(a[s:s + 20000]); out[o:o + len(b)] = b; o += len(b)
    out.flush()
    print(suf, out.shape, flush=True)
