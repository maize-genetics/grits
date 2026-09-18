"""
Fix #2 screen: sweep stay_bonus (the CRF's within-window-consistency
reinforcement term) at inference time -- a learned nn.Parameter, freely
overridable without retraining -- to see whether reducing it counteracts
the depth-confidence mechanism (more real reads -> more decode timesteps
per window -> stay_bonus compounds confidence on an already-ambiguous
mapping-support signal, without becoming more correct).

Tests HYB Oh43xIl14H and RIL2 Oh43xIl14H at 0.1x and 2.0x -- the pair/
depths already shown (via genome-wide SNP+RefCall rescoring) to have this
problem most clearly. See experiments/depth-confidence-fix/results/
depth_confidence_fix_2026-09-18.md for the full writeup.
"""
import sys
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/grits_workdir/indel-v3-simulator-wt/src")
import numpy as np
import pandas as pd
import torch

from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, infer_real_founder_pairs, score_ibd_adjusted_accuracy

CKPT = "/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt"
K = 25
REALDIR = "/workdir/zrm22/HackathonJun2026/grits_workdir/data/real"
SCRATCH_ROOT = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/simval_eval_fixed"
IDX = {"B73": 0, "B97": 1, "CML103": 2, "Il14H": 11, "Oh43": 21}
DEPTHS = ["0.1x", "2.0x"]
STAY_BONUS_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # 2.0 is the trained default

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GRITSCRFDiploidIndel.load_from_checkpoint(CKPT, map_location=device).eval().to(device)
default_stay_bonus = float(model.stay_bonus.item())
print(f"trained default stay_bonus = {default_stay_bonus}")


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


SAMPLES = [("IDX-HYB", "Oh43xIl14H", ("Oh43", "Il14H")), ("IDX-RIL2", "Oh43xIl14H", None)]

results = []
for ds, ind, truth_pair in SAMPLES:
    for depth in DEPTHS:
        outdir = f"{SCRATCH_ROOT}/{ds}__{ind}__{depth}"
        data_path = f"{REALDIR}/{ind}_{ds}_{depth}_ternary_k25native.npy"
        data = np.load(data_path)
        row_idx = build_row_idx(outdir)
        raw = np.load(f"{outdir}/raw.npy", mmap_mode="r")

        if ds == "IDX-RIL2":
            true_lo, true_hi, valid = ril2_truth(outdir)
        else:
            p1, p2 = truth_pair
            tlo, thi = min(IDX[p1], IDX[p2]), max(IDX[p1], IDX[p2])
            N = data.shape[0]
            true_lo = np.full((N, 512), tlo); true_hi = np.full((N, 512), thi); valid = np.ones((N, 512), bool)

        for sb in STAY_BONUS_VALUES:
            model.stay_bonus.data = torch.tensor(sb, device=device)
            pa, ia, cred, chk = score_ibd_adjusted_accuracy(
                *infer_real_founder_pairs(model, data, K, device=device),
                true_lo, true_hi, valid, raw, row_idx, K)
            results.append((ds, ind, depth, sb, pa, ia))
            print(f"{ds:<12}{ind:<12}{depth:<6}stay_bonus={sb:<5.1f}pair_acc={pa*100:6.2f}%  ibd_adj={ia*100:6.2f}%")
        print()

model.stay_bonus.data = torch.tensor(default_stay_bonus, device=device)  # restore

print("\n=== SUMMARY: pair_acc by stay_bonus ===")
print(f"{'sample/depth':<24}" + "".join(f"{sb:>9.1f}" for sb in STAY_BONUS_VALUES))
seen = set()
for ds, ind, depth, sb, pa, ia in results:
    key = (ds, ind, depth)
    if key in seen:
        continue
    seen.add(key)
    row = [r for r in results if r[0] == ds and r[1] == ind and r[2] == depth]
    label = f"{ind}/{depth}"
    print(f"{label:<24}" + "".join(f"{100*r[4]:>8.2f}%" for r in row))
