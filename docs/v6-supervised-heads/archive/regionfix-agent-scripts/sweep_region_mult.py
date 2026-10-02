import sys
sys.path.insert(0, "src")
import numpy as np
from python.crf.simulate_alleles import simulate

K = 25
recipe = dict(
    windows=200, sites=60000, founders=K,
    min_cross=2, max_cross=10,
    allele_sharing=0.2, bad_frac=0.05,
    sharing_model="coalescent", ancestors=6, sharing_theta=4.0,
    simulate_indels=True, indel_model="overlay",
    indel_coverage=2.0, emit_read_counts=True, collapse_rows=True,
)

for region_mult in [1, 2, 3, 4]:
    out, ibd, ind, panel, het, cls, refpos, short, true_cov = simulate(
        np.random.default_rng(777), inbreeding=1.0,
        indel_region_mult=region_mult, **recipe)
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
    print(f"region_mult={region_mult}  short%={short_pct:.3f}%  "
          f"mean_crossovers/individual={switches.mean():.3f}  "
          f"(n={n}, R={region_mult*60000})", flush=True)
