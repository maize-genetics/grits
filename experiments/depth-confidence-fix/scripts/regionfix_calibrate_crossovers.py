import sys
sys.path.insert(0, "src")
import numpy as np
from python.crf.simulate_alleles import simulate

K = 25
recipe = dict(
    windows=500, sites=60000, founders=K,
    allele_sharing=0.2, bad_frac=0.05,
    sharing_model="coalescent", ancestors=6, sharing_theta=4.0,
    simulate_indels=True, indel_model="overlay",
    indel_coverage=2.0, emit_read_counts=True, collapse_rows=True,
    indel_region_mult=2,
)

for min_c, max_c in [(6, 25), (8, 23)]:
    out, ibd, ind, panel, het, cls, refpos, short, true_cov = simulate(
        np.random.default_rng(401), inbreeding=1.0,
        min_cross=min_c, max_cross=max_c, **recipe)
    lab1 = out[:, :, K]
    n = out.shape[0]
    switches = []
    for i in range(n):
        r = refpos[i]
        valid = r >= 0
        l1v = lab1[i][valid]
        sw = int((l1v[1:] != l1v[:-1]).sum())
        switches.append(sw)
    switches = np.array(switches)
    short_pct = 100.0 * short.mean()
    print(f"min={min_c} max={max_c} (nominal_mean={(min_c+max_c)/2}): "
          f"short%={short_pct:.3f}% (n_short={int(short.sum())}/{n})  "
          f"mean_crossovers/individual={switches.mean():.3f}", flush=True)
