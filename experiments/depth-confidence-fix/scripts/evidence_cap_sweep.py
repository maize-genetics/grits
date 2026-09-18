"""
Fix #1 screen: cap effective real-evidence density per window at inference
time. The already-windowed ternary+distance real npy doesn't carry raw
per-site read counts (only match/diverged/deletion STATE), so this
approximates "fewer reads at this depth" by randomly masking a fraction of
currently-populated real sites (any site with >=1 founder in a non-PAD
ternary state) back to PAD, at increasing thinning strengths, and
re-scoring. If evidence density genuinely drives the depth-confidence
effect, thinning 2.0x data toward 0.1x's real site-density should recover
some of 0.1x's accuracy.
"""
import sys
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/grits_workdir/indel-v3-simulator-wt/src")
import numpy as np
import pandas as pd
import torch

from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, infer_real_founder_pairs, score_ibd_adjusted_accuracy
from python.crf.train_diploid_indel import TERN_PAD, DIST_PAD

CKPT = "/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt"
K = 25
REALDIR = "/workdir/zrm22/HackathonJun2026/grits_workdir/data/real"
SCRATCH_ROOT = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/simval_eval_fixed"
IDX = {"B73": 0, "B97": 1, "CML103": 2, "Il14H": 11, "Oh43": 21}
KEEP_FRACS = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]  # 1.0 = untouched (baseline)
SEED = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GRITSCRFDiploidIndel.load_from_checkpoint(CKPT, map_location=device).eval().to(device)


def build_row_idx(outdir, window_size=512):
    bins_df = pd.read_csv(f"{outdir}/raw.npy.bins.tsv", sep="\t")
    row_idx = []
    for _contig, idx in bins_df.groupby("contig", sort=False).indices.items():
        idx = np.sort(idx)
        start = 0
        while start + window_size <= len(idx):
            row_idx.append(idx[start:start + window_size])
            start += window_size
    return row_idx


def ril2_truth(outdir, window_size=512):
    bins_df = pd.read_csv(f"{outdir}/raw.npy.bins.tsv", sep="\t")
    truth = np.load(f"{outdir}/truth_labels.npy")
    windows = []
    for _contig, idx in bins_df.groupby("contig", sort=False).indices.items():
        idx = np.sort(idx)
        start = 0
        while start + window_size <= len(idx):
            windows.append(truth[idx[start:start + window_size]])
            start += window_size
    tw = np.stack(windows, axis=0)
    valid = (tw[:, :, 0] >= 0) & (tw[:, :, 1] >= 0)
    return np.minimum(tw[:, :, 0], tw[:, :, 1]), np.maximum(tw[:, :, 0], tw[:, :, 1]), valid


def thin_evidence(data, keep_frac, k, rng):
    """Randomly mask a (1-keep_frac) share of currently-populated real
    sites (any founder non-PAD) back to full-PAD (no evidence), leaving
    untouched sites exactly as-is. keep_frac=1.0 is a no-op copy."""
    data = data.copy()
    tern = data[:, :, :k]
    populated = (tern != TERN_PAD).any(axis=2)  # [N,T]
    if keep_frac >= 1.0:
        return data, float(populated.mean())
    drop_mask = populated & (rng.random(populated.shape) > keep_frac)
    data[drop_mask, :k] = TERN_PAD
    data[drop_mask, k + 2:2 * k + 2] = DIST_PAD
    new_density = ((data[:, :, :k] != TERN_PAD).any(axis=2)).mean()
    return data, float(new_density)


SAMPLES = [("IDX-HYB", "Oh43xIl14H", ("Oh43", "Il14H")), ("IDX-RIL2", "Oh43xIl14H", None)]
DEPTHS = ["0.1x", "2.0x"]

# first, report real site density at both depths for context
print("=== real site density (fraction of T=512 slots with >=1 founder non-PAD) ===")
for ds, ind, _ in SAMPLES:
    for depth in DEPTHS:
        data = np.load(f"{REALDIR}/{ind}_{ds}_{depth}_ternary_k25native.npy")
        tern = data[:, :, :K]
        density = ((tern != TERN_PAD).any(axis=2)).mean()
        print(f"{ds:<12}{ind:<12}{depth:<6}density={density:.4f}")
print()

results = []
for ds, ind, truth_pair in SAMPLES:
    outdir_2x = f"{SCRATCH_ROOT}/{ds}__{ind}__2.0x"
    row_idx_2x = build_row_idx(outdir_2x)
    raw_2x = np.load(f"{outdir_2x}/raw.npy", mmap_mode="r")

    if ds == "IDX-RIL2":
        true_lo, true_hi, valid = ril2_truth(outdir_2x)
    else:
        p1, p2 = truth_pair
        tlo, thi = min(IDX[p1], IDX[p2]), max(IDX[p1], IDX[p2])
        data0 = np.load(f"{REALDIR}/{ind}_{ds}_2.0x_ternary_k25native.npy")
        N = data0.shape[0]
        true_lo = np.full((N, 512), tlo); true_hi = np.full((N, 512), thi); valid = np.ones((N, 512), bool)

    data_2x = np.load(f"{REALDIR}/{ind}_{ds}_2.0x_ternary_k25native.npy")
    rng = np.random.default_rng(SEED)
    for keep_frac in KEEP_FRACS:
        thinned, density = thin_evidence(data_2x, keep_frac, K, rng)
        pa, ia, cred, chk = score_ibd_adjusted_accuracy(
            *infer_real_founder_pairs(model, thinned, K, device=device),
            true_lo, true_hi, valid, raw_2x, row_idx_2x, K)
        results.append((ds, ind, keep_frac, density, pa, ia))
        print(f"{ds:<12}{ind:<12}keep_frac={keep_frac:<5.2f}density={density:.4f}  "
              f"pair_acc={pa*100:6.2f}%  ibd_adj={ia*100:6.2f}%")
    print()

print("\n=== SUMMARY: pair_acc at 2.0x by evidence keep_frac (1.0=untouched baseline) ===")
print(f"{'sample':<20}" + "".join(f"{kf:>9.2f}" for kf in KEEP_FRACS))
for ds, ind, _ in SAMPLES:
    row = [r for r in results if r[0] == ds and r[1] == ind]
    label = f"{ind}"
    print(f"{label:<20}" + "".join(f"{100*r[4]:>8.2f}%" for r in row))
