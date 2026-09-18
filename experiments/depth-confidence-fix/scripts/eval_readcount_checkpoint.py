"""
Isolated-variable read-count-feature evaluation on real data. Unlike the
window_density round's eval, `count` is genuinely non-null here for BOTH
checkpoints -- real HYB/RIL2 Oh43xIl14H data was regenerated at all 5
depths with --emit-read-counts (Oh43xIl14H_{ds}_{depth}_ternary_k25native
_wcount.npy). The baseline checkpoint's count_proj is zero-initialized
(strict=False load, never trained), so it correctly IGNORES the count
signal regardless of whether it's fed count data -- feeding both
checkpoints the same _wcount arrays is safe and lets this be a clean
apples-to-apples comparison.
"""
import sys
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/grits_workdir/indel-v3-simulator-wt/src")
import numpy as np
import pandas as pd
import torch

from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, infer_real_founder_pairs, score_ibd_adjusted_accuracy

CKPTS = {
    "v3-K25 (baseline)": ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
                          "diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt"),
    "readcount-isolated (new)": ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
                                 "diploid-indel-v3-readcount-isolated/d-epoch=03-val_pair_acc=0.2527.ckpt"),
}
K = 25
REALDIR = "/workdir/zrm22/HackathonJun2026/grits_workdir/data/real"
SCRATCH_ROOT = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/simval_eval_fixed"
IDX = {"B73": 0, "B97": 1, "CML103": 2, "Il14H": 11, "Oh43": 21}
DEPTHS = ["0.01x", "0.1x", "0.5x", "1.0x", "2.0x"]
SAMPLES = [("IDX-HYB", "Oh43xIl14H", ("Oh43", "Il14H")), ("IDX-RIL2", "Oh43xIl14H", None)]

device = "cuda" if torch.cuda.is_available() else "cpu"


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


results = {}
for label, ckpt_path in CKPTS.items():
    model = GRITSCRFDiploidIndel.load_from_checkpoint(
        ckpt_path, map_location=device, strict=False).eval().to(device)
    if label.startswith("v3-K25"):
        with torch.no_grad():
            model.encoder.count_proj.weight.zero_()
            model.encoder.count_proj.bias.zero_()

    for ds, ind, truth_pair in SAMPLES:
        for depth in DEPTHS:
            data_path = f"{REALDIR}/{ind}_{ds}_{depth}_ternary_k25native_wcount.npy"
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
            results[(label, ds, ind, depth)] = (pa, ia)
            print(f"{label:<28}{ds:<12}{ind:<12}{depth:<7}pair_acc={pa*100:6.2f}%  ibd_adj={ia*100:6.2f}%")
    print()

print("\n=== SUMMARY: pair_acc, baseline vs new, by depth ===")
print(f"{'sample':<25}" + "".join(f"{d:>10}" for d in DEPTHS))
for ds, ind, _ in SAMPLES:
    for label in CKPTS:
        row = [results.get((label, ds, ind, d), (float("nan"), float("nan")))[0] for d in DEPTHS]
        print(f"{ind+'/'+label[:15]:<25}" + "".join(f"{100*r:>9.2f}% " for r in row))
