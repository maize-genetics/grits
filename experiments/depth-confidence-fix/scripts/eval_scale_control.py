"""
Ablation: is the HYB regression seen with the read-count feature actually
caused by the count feature, or just by retraining at small scale (200
individuals vs v3-K25's original ~1000)? diploid-indel-v3-scale-control-
nocount was warm-started and trained on the EXACT SAME isolated-variable
data/individuals as the readcount checkpoint, with the count block simply
stripped (2K+2, not 3K+2) -- count_proj never fires during this training
(no "count" key ever appears in the batch), so its weights stay exactly
zero throughout, a true "same everything except count" control.
"""
import sys
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/grits_workdir/indel-v3-simulator-wt/src")
import numpy as np
import pandas as pd
import torch

from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, infer_real_founder_pairs, score_ibd_adjusted_accuracy

CKPT = ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
        "diploid-indel-v3-scale-control-nocount/d-epoch=01-val_pair_acc=0.2445.ckpt")
K = 25
REALDIR = "/workdir/zrm22/HackathonJun2026/grits_workdir/data/real"
SCRATCH_ROOT = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/simval_eval_fixed"
IDX = {"B73": 0, "B97": 1, "CML103": 2, "Il14H": 11, "Oh43": 21}
DEPTHS = ["0.01x", "0.1x", "0.5x", "1.0x", "2.0x"]
SAMPLES = [("IDX-HYB", "Oh43xIl14H", ("Oh43", "Il14H")), ("IDX-RIL2", "Oh43xIl14H", None)]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GRITSCRFDiploidIndel.load_from_checkpoint(
    CKPT, map_location=device, strict=False).eval().to(device)


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


print("=== scale-control (no count) ===")
for ds, ind, truth_pair in SAMPLES:
    for depth in DEPTHS:
        data_path = f"{REALDIR}/{ind}_{ds}_{depth}_ternary_k25native.npy"
        outdir = f"{SCRATCH_ROOT}/{ds}__{ind}__{depth}"
        data = np.load(data_path)
        raw = np.load(f"{outdir}/raw.npy", mmap_mode="r")
        row_idx = build_row_idx(outdir)

        if ds == "IDX-RIL2":
            true_lo, true_hi, valid = ril2_truth(outdir)
        else:
            p1, p2 = truth_pair
            tlo, thi = min(IDX[p1], IDX[p2]), max(IDX[p1], IDX[p2])
            N = data.shape[0]
            true_lo = np.full((N, 512), tlo); true_hi = np.full((N, 512), thi); valid = np.ones((N, 512), bool)

        pa, ia, cred, chk = score_ibd_adjusted_accuracy(
            *infer_real_founder_pairs(model, data, K, device=device),
            true_lo, true_hi, valid, raw, row_idx, K)
        print(f"scale-control (no count)   {ds:<12}{ind:<12}{depth:<7}pair_acc={pa*100:6.2f}%  ibd_adj={ia*100:6.2f}%")
