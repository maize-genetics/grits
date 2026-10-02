import sys
sys.path.insert(0, "src")
import numpy as np
from python.crf.simulate_alleles import simulate

K = 25
recipe = dict(
    windows=500, sites=60000, founders=K,
    min_cross=2, max_cross=10,
    allele_sharing=0.2, bad_frac=0.05,
    sharing_model="coalescent", ancestors=6, sharing_theta=4.0,
    simulate_indels=True, indel_model="overlay",
    indel_coverage=2.0, emit_read_counts=True, collapse_rows=True,
    indel_region_mult=2,
)

for label, inbreeding, seed in [("inb1.0", 1.0, 401), ("inb0.0", 0.0, 402)]:
    out, ibd, ind, panel, het, cls, refpos, short, true_cov = simulate(
        np.random.default_rng(seed), inbreeding=inbreeding, **recipe)
    lab1 = out[:, :, K]
    lab2 = out[:, :, K + 1]
    n = out.shape[0]
    switches = []
    het_fracs = []
    for i in range(n):
        r = refpos[i]
        valid = r >= 0
        l1v = lab1[i][valid]
        l2v = lab2[i][valid]
        sw = int((l1v[1:] != l1v[:-1]).sum())
        switches.append(sw)
        het_fracs.append(float((l1v != l2v).mean()))
    switches = np.array(switches)
    het_fracs = np.array(het_fracs)
    short_pct = 100.0 * short.mean()
    print(f"{label}: short%={short_pct:.3f}% (n_short={int(short.sum())}/{n})  "
          f"mean_crossovers/individual={switches.mean():.3f}  "
          f"mean_het_frac={het_fracs.mean():.4f}", flush=True)
