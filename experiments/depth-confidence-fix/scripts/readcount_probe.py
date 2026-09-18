"""
Lab-meeting suggestion (colleague, via user, 2026-09-18): add a per-cell
read-SUPPORT-COUNT channel alongside the existing {-1,0,1} ternary state --
"the matrix is still {-1,0,1} but there is a count of how many reads had
that exact distribution." Different from the density_probe.py per-ROW
gap/stack test (a property of POSITION, already shown not to predict
per-row correctness within a fixed depth) -- this is a per-CELL confidence
signal: for the TRUE founder specifically, how many reads actually voted
for the call the model sees there.

Cheap, no-new-pipeline-code test using data already cached this session:
raw.npy's counts block (already used by score_ibd_adjusted_accuracy) gives
real per-founder read counts per row. Pull the true founder's count at
every real row (for HYB, the true founder is fixed; for RIL2, per-row from
truth_labels.npy) and check whether it separates correct from incorrect
predictions -- the same before-you-build-it validation Phase 0 used.
"""
import sys
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/grits_workdir/indel-v3-simulator-wt/src")
import numpy as np
import pandas as pd
import torch

from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, infer_real_founder_pairs

CKPT = "/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt"
K = 25
REALDIR = "/workdir/zrm22/HackathonJun2026/grits_workdir/data/real"
SCRATCH_ROOT = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/simval_eval_fixed"
IDX = {"B73": 0, "B97": 1, "CML103": 2, "Il14H": 11, "Oh43": 21}
WINDOW = 512

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GRITSCRFDiploidIndel.load_from_checkpoint(
    CKPT, map_location=device, strict=False).eval().to(device)
with torch.no_grad():
    model.encoder.density_proj.weight.zero_()
    model.encoder.density_proj.bias.zero_()


def build_row_idx(outdir, window_size=WINDOW):
    bins_df = pd.read_csv(f"{outdir}/raw.npy.bins.tsv", sep="\t")
    row_idx = []
    for _contig, idx in bins_df.groupby("contig", sort=False).indices.items():
        idx = np.sort(idx)
        start = 0
        while start + window_size <= len(idx):
            row_idx.append(idx[start:start + window_size])
            start += window_size
    return row_idx


def ril2_truth(outdir, row_idx_list):
    truth = np.load(f"{outdir}/truth_labels.npy")
    windows = [truth[sel] for sel in row_idx_list]
    tw = np.stack(windows, axis=0)
    valid = (tw[:, :, 0] >= 0) & (tw[:, :, 1] >= 0)
    return np.minimum(tw[:, :, 0], tw[:, :, 1]), np.maximum(tw[:, :, 0], tw[:, :, 1]), valid


SAMPLES = [("IDX-HYB", "Oh43xIl14H", ("Oh43", "Il14H")), ("IDX-RIL2", "Oh43xIl14H", None)]
DEPTHS = ["0.1x", "2.0x"]

all_rows = []  # (ds, depth, true_count_lo, true_count_hi, correct)

for ds, ind, truth_pair in SAMPLES:
    for depth in DEPTHS:
        outdir = f"{SCRATCH_ROOT}/{ds}__{ind}__{depth}"
        data = np.load(f"{REALDIR}/{ind}_{ds}_{depth}_ternary_k25native.npy")
        raw = np.asarray(np.load(f"{outdir}/raw.npy", mmap_mode="r"))
        row_idx = build_row_idx(outdir)
        assert len(row_idx) == data.shape[0]

        pred_lo, pred_hi = infer_real_founder_pairs(model, data, K, device=device)

        if ds == "IDX-RIL2":
            true_lo, true_hi, valid = ril2_truth(outdir, row_idx)
        else:
            p1, p2 = truth_pair
            tlo, thi = min(IDX[p1], IDX[p2]), max(IDX[p1], IDX[p2])
            true_lo = np.full_like(pred_lo, tlo)
            true_hi = np.full_like(pred_hi, thi)
            valid = np.ones_like(pred_lo, dtype=bool)

        correct = (pred_lo == true_lo) & (pred_hi == true_hi) & valid

        # raw counts block: first K columns of raw.npy, per real row
        counts = raw[:, :K].astype(np.float64)
        # per-window rows -> flat real-row index via row_idx, then true founder's count
        n_windows, T = data.shape[0], data.shape[1]
        cnt_lo = np.empty((n_windows, T)); cnt_hi = np.empty((n_windows, T))
        for w in range(n_windows):
            sel = row_idx[w]
            cnt_lo[w] = counts[sel, true_lo[w]]
            cnt_hi[w] = counts[sel, true_hi[w]]
        true_support = np.minimum(cnt_lo, cnt_hi)  # weaker of the two true founders' support

        v = valid.ravel()
        s = true_support.ravel()[v]
        c = correct.ravel()[v]
        all_rows.append((ds, depth, s, c))

        err = 1.0 - c.mean()
        print(f"{ds:<12}{depth:<6} n={v.sum():<9} err={err*100:5.2f}%  "
              f"mean_true_support[correct]={s[c].mean():6.3f}  "
              f"mean_true_support[wrong]={s[~c].mean() if (~c).any() else float('nan'):6.3f}")

print("\n=== error rate by true-founder read-support bucket (pooled across HYB+RIL2 @ 2.0x) ===")
s_all, c_all = [], []
for ds, depth, s, c in all_rows:
    if depth == "2.0x":
        s_all.append(s); c_all.append(c)
s_all, c_all = np.concatenate(s_all), np.concatenate(c_all)
edges = [0, 1, 2, 3, 5, 10, np.inf]
prev = -1
for e in edges:
    m = (s_all > prev) & (s_all <= e)
    n = m.sum()
    if n == 0:
        prev = e
        continue
    err = 1.0 - c_all[m].mean()
    print(f"  {prev}<support<={e:<6} n={n:<10} err_rate={err*100:5.2f}%")
    prev = e

print("\n=== same, at 0.1x ===")
s01, c01 = [], []
for ds, depth, s, c in all_rows:
    if depth == "0.1x":
        s01.append(s); c01.append(c)
s01, c01 = np.concatenate(s01), np.concatenate(c01)
prev = -1
for e in edges:
    m = (s01 > prev) & (s01 <= e)
    n = m.sum()
    if n == 0:
        prev = e
        continue
    err = 1.0 - c01[m].mean()
    print(f"  {prev}<support<={e:<6} n={n:<10} err_rate={err*100:5.2f}%")
    prev = e
