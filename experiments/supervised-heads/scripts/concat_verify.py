"""Verify the provenance rebuilds are byte-identical to the original training sets (only the
.prov.npy sidecar is new), then concatenate 0.1x+0.5x+1x+2x into maize_v3prov_multidepth
(data, .lin.npy, .prov.npy) in the same order as maize_v3_multidepth."""
import hashlib, sys
import numpy as np
D = "/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/data/training/"
NEW = ["maize_v3prov_d0.1", "maize_v3prov_d0.5", "maize_v3prov_d1", "maize_v3prov_d2"]
OLD = ["maize_v3_hetrepl_v3_lineage", "maize_v3_lineage_d0.5", "maize_v3_lineage_d1", "maize_v3_lineage_d2"]


def same(a, b):
    x, y = np.load(a, mmap_mode="r"), np.load(b, mmap_mode="r")
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    for s in range(0, len(x), 20000):
        if not np.array_equal(np.asarray(x[s:s + 20000]), np.asarray(y[s:s + 20000])):
            return False
    return True


ok = True
for n, o in zip(NEW, OLD):
    for suf in ("_fullscale_sliced.npy", "_fullscale_sliced.lin.npy"):
        r = same(D + n + suf, D + o + suf)
        ok &= r
        print(f"{n+suf:<50} == {o+suf}: {r}", flush=True)
if not ok:
    sys.exit("MISMATCH: provenance rebuild is not byte-identical to the original sets")
for suf in ("_fullscale_sliced.npy", "_fullscale_sliced.lin.npy", "_fullscale_sliced.prov.npy"):
    arrs = [np.load(D + p + suf, mmap_mode="r") for p in NEW]
    n = sum(len(a) for a in arrs)
    out = np.lib.format.open_memmap(D + "maize_v3prov_multidepth" + suf, mode="w+",
                                    dtype=arrs[0].dtype, shape=(n,) + arrs[0].shape[1:])
    o = 0
    for a in arrs:
        for s in range(0, len(a), 20000):
            b = np.asarray(a[s:s + 20000]); out[o:o + len(b)] = b; o += len(b)
    out.flush()
    print(suf, out.shape, flush=True)
# the concatenated data must equal the existing multidepth set too
print("multidepth data == maize_v3_multidepth:",
      same(D + "maize_v3prov_multidepth_fullscale_sliced.npy", D + "maize_v3_multidepth_fullscale_sliced.npy"))
p = np.load(D + "maize_v3prov_multidepth_fullscale_sliced.prov.npy", mmap_mode="r")
from python.crf.simulate_alleles import prov_is_clean, PROV_KIND_MASK, PROV_OFF_SITE, PROV_BAD_SITE
per = len(p) // 4
for i, d in enumerate(("0.1x", "0.5x", "1x", "2x")):
    q = np.asarray(p[i * per:(i + 1) * per:7]).astype(int)
    k = q & PROV_KIND_MASK
    print(d, "kind frac", [round(float((k == j).mean()), 4) for j in range(6)],
          "pad", round(float((q < 0).mean()), 5), "bad", round(float(((q & PROV_BAD_SITE) > 0).mean()), 4),
          "off-site", round(float(((q & PROV_OFF_SITE) > 0).mean()), 4),
          "clean", round(float(prov_is_clean(q).mean()), 4), flush=True)
