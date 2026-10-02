"""Slice long per-individual contigs (one row = one real sampled read, real
rows a clean prefix per individual per _indel_chunk's design) into
non-overlapping T=512 training windows, matching ropebwt_npy_to_matrix.py's
real convention exactly: `while start + window_size <= n_real: window; start
+= window_size`, dropping the trailing short remainder -- no padding.

Truncates every individual to the SAME number of sub-windows (the minimum
across all individuals) so IndelDiploidAffinityDataset's `len(data) % G == 0`
assumption holds and --windows-per-individual G is exactly the number of
sub-windows contributed per individual, with no accidental cross-individual
grouping.
"""
import sys
import numpy as np

IN_PATH = sys.argv[1]
OUT_PATH = sys.argv[2]
T = int(sys.argv[3]) if len(sys.argv) > 3 else 512

LABEL_PAD = -1

data = np.load(IN_PATH, mmap_mode="r")
N, S, W = data.shape
K = (W - 2) // 2
print(f"loaded {IN_PATH}: shape={data.shape} K={K}")

lab1 = np.asarray(data[:, :, K])  # [N,S] int8, LABEL_PAD=-1 marks the padded tail
n_real = np.zeros(N, dtype=np.int64)
for i in range(N):
    valid = lab1[i] != LABEL_PAD
    # real rows are a clean prefix by design -- verify, don't just assume
    if valid.any():
        last_real = np.flatnonzero(valid).max()
        n_real_i = last_real + 1
        assert valid[:n_real_i].all(), f"individual {i}: real rows are NOT a clean prefix -- design assumption violated"
    else:
        n_real_i = 0
    n_real[i] = n_real_i

sub_per_ind = n_real // T
G = int(sub_per_ind.min())
print(f"real rows per individual: min={n_real.min()} max={n_real.max()} mean={n_real.mean():.0f}")
print(f"sub-windows per individual (T={T}): min={sub_per_ind.min()} max={sub_per_ind.max()} mean={sub_per_ind.mean():.1f}")
print(f"truncating every individual to G={G} sub-windows (the minimum) for uniform affinity grouping")

if G == 0:
    raise SystemExit(f"at least one individual produced 0 real rows >= T={T} -- cannot proceed, check --sites/coverage settings")

out = np.empty((N * G, T, W), dtype=data.dtype)
refpos_col_present = False  # main array only, refpos sidecar sliced separately if present
for i in range(N):
    for g in range(G):
        s = g * T
        out[i * G + g] = data[i, s:s + T]

print(f"sliced output shape: {out.shape}  ({N} individuals x {G} sub-windows)")

# Sanity check: verify genuine continuity across a slice boundary within one
# individual (h1 label should show real recombination-path structure, and
# consecutive sub-windows from the SAME individual should NOT look like two
# independently-drawn windows -- spot check refpos monotonicity if available).
refpos_path = IN_PATH.replace(".npy", ".refpos.npy")
try:
    refpos = np.load(refpos_path, mmap_mode="r")
    rp0 = refpos[0]
    boundary_ok = True
    for g in range(min(G - 1, 3)):
        tail = rp0[g * T:(g + 1) * T]
        head = rp0[(g + 1) * T:(g + 2) * T]
        if not (head[0] >= tail[-1]):
            boundary_ok = False
            print(f"  WARNING: refpos not monotonic across boundary at sub-window {g}->{g+1}: "
                  f"tail_end={tail[-1]} head_start={head[0]}")
    if boundary_ok:
        print(f"  continuity check (individual 0, first {min(G-1,3)} boundaries): "
              f"refpos monotonically increasing across sub-window boundaries -- PASS")
except FileNotFoundError:
    print(f"  (no refpos sidecar found at {refpos_path}, skipping continuity check)")

np.save(OUT_PATH, out)
print(f"wrote {OUT_PATH}")
print(f"G (windows-per-individual for training) = {G}")
