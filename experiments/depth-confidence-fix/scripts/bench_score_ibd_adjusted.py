"""
Speed investigation (user request, while waiting on the full-scale control
retrain): score_ibd_adjusted_accuracy has the same class of bug already
fixed once this session in homo_scale_from_affinity -- an O(N) Python loop
over every window, with a per-window mmap fancy-index into raw_counts
inside the loop. This defines a vectorized replacement, verifies BIT-EXACT
equivalence against the original on real cached data (not just "close"),
then benchmarks both on a large real sample.
"""
import sys
import time
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/grits_workdir/indel-v3-simulator-wt/src")
import numpy as np
import pandas as pd
import torch

from python.crf.train_diploid_indel import (
    GRITSCRFDiploidIndel, infer_real_founder_pairs, score_ibd_adjusted_accuracy as score_orig,
    _window_error_slots,
)

CKPT = "/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt"
K = 25
REALDIR = "/workdir/zrm22/HackathonJun2026/grits_workdir/data/real"
SCRATCH_ROOT = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/simval_eval_fixed"
IDX = {"B73": 0, "B97": 1, "CML103": 2, "Il14H": 11, "Oh43": 21}


def score_fast(pred_lo, pred_hi, true_lo, true_hi, valid, raw_counts, row_idx, num_parents, ibd_thresh=0.8):
    pair_acc = float(((pred_lo == true_lo) & (pred_hi == true_hi))[valid].mean())

    N, T = pred_lo.shape
    correct = (pred_lo == true_lo) & (pred_hi == true_hi)
    has_err = (~correct & valid).any(axis=1)
    has_switch = (true_lo != true_lo[:, :1]).any(axis=1) | (true_hi != true_hi[:, :1]).any(axis=1)
    candidates = np.flatnonzero(has_err & ~has_switch)

    credited = np.zeros((N, T), dtype=bool)
    n_checked, n_credited_windows = 0, 0

    if candidates.size:
        idx2d = np.stack([row_idx[w] for w in candidates], axis=0)              # [C,T]
        mean_support_all = np.asarray(raw_counts[idx2d, :num_parents]).astype(np.float64).mean(axis=1)  # [C,K]

        for i, w in enumerate(candidates):
            err_w = (~correct[w]) & valid[w]
            pairs, counts = np.unique(np.stack([pred_lo[w], pred_hi[w]], axis=1), axis=0, return_counts=True)
            maj_lo, maj_hi = pairs[np.argmax(counts)]
            slots = _window_error_slots(true_lo[w, 0], true_hi[w, 0], maj_lo, maj_hi)
            if not slots:
                continue
            mean_support = mean_support_all[i]
            n_checked += 1
            ok = all(mean_support[tf] > 1e-9 and mean_support[pf] / mean_support[tf] >= ibd_thresh
                     for tf, pf in slots)
            if ok:
                credited[w] = err_w
                n_credited_windows += 1

    adjusted_correct = ((pred_lo == true_lo) & (pred_hi == true_hi)) | credited
    ibd_adj_acc = float(adjusted_correct[valid].mean())
    return pair_acc, ibd_adj_acc, n_credited_windows, n_checked


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


device = "cuda" if torch.cuda.is_available() else "cpu"
model = GRITSCRFDiploidIndel.load_from_checkpoint(CKPT, map_location=device, strict=False).eval().to(device)
with torch.no_grad():
    model.encoder.count_proj.weight.zero_()
    model.encoder.count_proj.bias.zero_()

# Two samples: HYB Oh43xIl14H 2.0x (large, mostly-correct baseline -- few
# candidates) and RIL2 B73xOh43 0.1x (smaller, higher error rate -- more
# candidates) to exercise both regimes.
CASES = [
    ("IDX-HYB", "Oh43xIl14H", "2.0x", ("Oh43", "Il14H")),
    ("IDX-RIL2", "B73xOh43", "0.1x", None),
]

for ds, ind, depth, truth_pair in CASES:
    data = np.load(f"{REALDIR}/{ind}_{ds}_{depth}_ternary_k25native_wcount.npy")
    outdir = f"{SCRATCH_ROOT}/{ds}__{ind}__{depth}"
    raw = np.load(f"{outdir}/raw.npy", mmap_mode="r")
    row_idx = build_row_idx(outdir)

    if ds == "IDX-RIL2":
        true_lo, true_hi, valid = ril2_truth(outdir)
    else:
        p1, p2 = truth_pair
        tlo, thi = min(IDX[p1], IDX[p2]), max(IDX[p1], IDX[p2])
        N = data.shape[0]
        true_lo = np.full((N, 512), tlo); true_hi = np.full((N, 512), thi); valid = np.ones((N, 512), bool)

    pred_lo, pred_hi = infer_real_founder_pairs(model, data, K, device=device)

    t0 = time.perf_counter()
    orig = score_orig(pred_lo, pred_hi, true_lo, true_hi, valid, raw, row_idx, K)
    t_orig = time.perf_counter() - t0

    t0 = time.perf_counter()
    fast = score_fast(pred_lo, pred_hi, true_lo, true_hi, valid, raw, row_idx, K)
    t_fast = time.perf_counter() - t0

    match = orig == fast
    print(f"{ds}/{ind}/{depth}  N={data.shape[0]:,}  orig={orig}  fast={fast}  "
          f"MATCH={match}  t_orig={t_orig:.3f}s  t_fast={t_fast:.3f}s  speedup={t_orig/t_fast:.1f}x")
