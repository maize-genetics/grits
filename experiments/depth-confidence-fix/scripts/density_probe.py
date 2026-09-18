"""
Plan Phase 0 (go/no-go): before building the density-feature channels and
spending a retrain, check on real data whether "how redundant is this row's
evidence" actually separates windows/rows the CURRENT v3-K25 model gets
wrong from ones it gets right. If it doesn't, the whole premise (see
experiments/depth-confidence-fix/results/depth_confidence_fix_2026-09-18.md)
is wrong and the plan should be cancelled before any implementation.

Two per-ROW density signals, matching the plan's design table:
  - gap:   bin-units since the previous row in this window (0 = same bin as
           the previous row -- maximally redundant; large = fresh genomic
           ground)
  - stack: how many total rows in the whole sample share this row's exact
           (contig, bin) -- read-stacking at one position

For each of HYB and RIL2 Oh43xIl14H at 2.0x (the depth where the problem is
worst), predictions come from infer_real_founder_pairs (the one canonical
real-data inference path) so this probe is on the actual deployed model, not
a proxy. Per-row correctness is checked against truth, and error rate is
reported bucketed by gap and by stack, plus a simple rank-correlation-style
check (mean gap/stack for correct vs incorrect rows).
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
model = GRITSCRFDiploidIndel.load_from_checkpoint(CKPT, map_location=device).eval().to(device)


def build_row_idx_and_gap_stack(bins_df, window=WINDOW):
    """Per-contig: covered-row indices grouped into consecutive windows of
    `window` rows (today's exact windowing convention), plus per-row gap
    (bin units since the previous row IN THIS WINDOW, 0 for the first row)
    and per-row stack (count of rows sharing this row's exact bin, computed
    genome-wide per contig, not just within the window)."""
    row_idx_all, gap_all, stack_all = [], [], []
    for _contig, idx in bins_df.groupby("contig", sort=False).indices.items():
        idx = np.sort(idx)
        binpos = bins_df["bin"].values[idx]
        _, inv, counts = np.unique(binpos, return_inverse=True, return_counts=True)
        stack = counts[inv]  # [len(idx)] stack depth per row, genome-wide within contig
        start = 0
        while start + window <= len(idx):
            sel = idx[start:start + window]
            pos = binpos[start:start + window]
            gap = np.empty(window, dtype=np.float64)
            gap[0] = 0.0
            gap[1:] = pos[1:] - pos[:-1]
            row_idx_all.append(sel)
            gap_all.append(gap)
            stack_all.append(stack[start:start + window])
            start += window
    return row_idx_all, np.stack(gap_all, axis=0), np.stack(stack_all, axis=0)


def ril2_truth(outdir, row_idx_list):
    truth = np.load(f"{outdir}/truth_labels.npy")
    windows = [truth[sel] for sel in row_idx_list]
    tw = np.stack(windows, axis=0)
    valid = (tw[:, :, 0] >= 0) & (tw[:, :, 1] >= 0)
    return np.minimum(tw[:, :, 0], tw[:, :, 1]), np.maximum(tw[:, :, 0], tw[:, :, 1]), valid


SAMPLES = [("IDX-HYB", "Oh43xIl14H", ("Oh43", "Il14H")), ("IDX-RIL2", "Oh43xIl14H", None)]
DEPTHS = ["0.1x", "2.0x"]

all_rows = []  # (ds, ind, depth, gap, stack, correct) flattened rows for the summary table

for ds, ind, truth_pair in SAMPLES:
    for depth in DEPTHS:
        outdir = f"{SCRATCH_ROOT}/{ds}__{ind}__{depth}"
        data = np.load(f"{REALDIR}/{ind}_{ds}_{depth}_ternary_k25native.npy")
        bins_df = pd.read_csv(f"{outdir}/raw.npy.bins.tsv", sep="\t")

        row_idx, gap, stack = build_row_idx_and_gap_stack(bins_df)
        assert len(row_idx) == data.shape[0], (len(row_idx), data.shape[0])

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
        v = valid.ravel()
        g = gap.ravel()[v]
        s = stack.ravel()[v]
        c = correct.ravel()[v]

        all_rows.append((ds, ind, depth, g, s, c))

        err_rate = 1.0 - c.mean()
        mean_gap_correct = g[c].mean() if c.any() else float("nan")
        mean_gap_wrong = g[~c].mean() if (~c).any() else float("nan")
        mean_stack_correct = s[c].mean() if c.any() else float("nan")
        mean_stack_wrong = s[~c].mean() if (~c).any() else float("nan")
        print(f"{ds:<12}{ind:<12}{depth:<6} n_rows={v.sum():<9} err_rate={err_rate*100:5.2f}%  "
              f"mean_gap[correct]={mean_gap_correct:6.3f}  mean_gap[wrong]={mean_gap_wrong:6.3f}  "
              f"mean_stack[correct]={mean_stack_correct:6.3f}  mean_stack[wrong]={mean_stack_wrong:6.3f}")

print("\n=== error rate by STACK-DEPTH bucket (pooled across HYB+RIL2 @ 2.0x, the depth where stack varies most) ===")
g_all, s_all, c_all = [], [], []
for ds, ind, depth, g, s, c in all_rows:
    if depth == "2.0x":
        g_all.append(g); s_all.append(s); c_all.append(c)
g_all, s_all, c_all = np.concatenate(g_all), np.concatenate(s_all), np.concatenate(c_all)

stack_bins = [1, 2, 3, 5, 10, 1000]
edges = [1, 2, 3, 5, 10, np.inf]
prev = 0
for e in edges:
    if e == 1:
        m = s_all == 1
        label = "stack=1"
    else:
        m = (s_all > prev) & (s_all <= e)
        label = f"{prev}<stack<={e}"
    n = m.sum()
    if n == 0:
        prev = e
        continue
    err = 1.0 - c_all[m].mean()
    print(f"  {label:<16} n={n:<10} err_rate={err*100:5.2f}%")
    prev = e

print("\n=== error rate by GAP bucket (pooled across HYB+RIL2 @ 0.1x, the depth where gap varies most) ===")
g01, s01, c01 = [], [], []
for ds, ind, depth, g, s, c in all_rows:
    if depth == "0.1x":
        g01.append(g); s01.append(s); c01.append(c)
g01, s01, c01 = np.concatenate(g01), np.concatenate(s01), np.concatenate(c01)
gap_edges = [0, 1, 2, 3, 5, 10, np.inf]
prev = -1
for e in gap_edges:
    m = (g01 > prev) & (g01 <= e)
    n = m.sum()
    if n == 0:
        prev = e
        continue
    err = 1.0 - c01[m].mean()
    print(f"  {prev}<gap<={e:<6} n={n:<10} err_rate={err*100:5.2f}%")
    prev = e
