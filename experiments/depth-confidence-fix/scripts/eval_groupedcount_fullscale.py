"""
Full-scale grouped-count checkpoint evaluation (branch
indel-readcount-grouped-count): ALL real IDX-INBRED/IDX-HYB/IDX-RIL2
samples (5 each, 15 total), all 5 depths, founder pair_acc AND
ibd_adjusted_accuracy -- v3-K25 baseline vs the full-scale retrain using
the CORRECTED count formula (group by (window,site,own exact ternary
vector), not the old per-founder-MATCH-tally-pooled-across-kinds design
that broke real-data accuracy broadly). Reuses the same *_wcount.npy
real-data conversions as the earlier (broken) readcount-fullscale round
unchanged -- ropebwt_npy_to_matrix.py's real-data producer was never the
bug and needed no changes for this fix (real data is already one row per
site with genuine per-founder counts; only the simulator's synthetic
count derivation was wrong). Reports everything, not just B73/HYB pairs,
matching the comprehensive-coverage precedent from the last round.
"""
import glob
import sys
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/grits_workdir/indel-v3-simulator-wt/src")
import numpy as np
import pandas as pd
import torch

from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, infer_real_founder_pairs, score_ibd_adjusted_accuracy

GROUPEDCOUNT_CKPTS = sorted(glob.glob(
    "/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
    "diploid-indel-v3-groupedcount-fullscale/d-epoch=*.ckpt"))
assert GROUPEDCOUNT_CKPTS, "no diploid-indel-v3-groupedcount-fullscale checkpoint found yet"

CKPTS = {
    "baseline": ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
                "diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt"),
    "groupedcount-fullscale": GROUPEDCOUNT_CKPTS[-1],  # latest epoch
}
K = 25
REALDIR = "/workdir/zrm22/HackathonJun2026/grits_workdir/data/real"
SCRATCH_ROOT = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/simval_eval_fixed"
IDX = {"B73": 0, "B97": 1, "CML103": 2, "Il14H": 11, "Oh43": 21}
DEPTHS = ["0.01x", "0.1x", "0.5x", "1.0x", "2.0x"]

INBRED = ["B73", "Oh43", "Il14H", "B97", "CML103"]
HYB_PAIRS = ["B73xOh43", "B73xCML103", "Oh43xIl14H", "B97xCML103", "Il14HxB97"]
RIL2_PAIRS = ["B73xCML103", "B73xOh43", "B97xCML103", "Il14HxB97", "Oh43xIl14H"]

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


def eval_one(model, ds, ind, depth):
    data_path = f"{REALDIR}/{ind}_{ds}_{depth}_ternary_k25native_wcount.npy"
    outdir = f"{SCRATCH_ROOT}/{ds}__{ind}__{depth}"
    data = np.load(data_path)
    raw = np.load(f"{outdir}/raw.npy", mmap_mode="r")
    row_idx = build_row_idx(outdir)

    if ds == "IDX-RIL2":
        true_lo, true_hi, valid = ril2_truth(outdir)
    elif ds == "IDX-INBRED":
        tlo = thi = IDX[ind]
        N = data.shape[0]
        true_lo = np.full((N, 512), tlo); true_hi = np.full((N, 512), thi); valid = np.ones((N, 512), bool)
    else:  # IDX-HYB
        p1, p2 = ind.split("x")
        tlo, thi = min(IDX[p1], IDX[p2]), max(IDX[p1], IDX[p2])
        N = data.shape[0]
        true_lo = np.full((N, 512), tlo); true_hi = np.full((N, 512), thi); valid = np.ones((N, 512), bool)

    return score_ibd_adjusted_accuracy(
        *infer_real_founder_pairs(model, data, K, device=device),
        true_lo, true_hi, valid, raw, row_idx, K)


results = {}
for label, ckpt_path in CKPTS.items():
    print(f"loading {label}: {ckpt_path}", flush=True)
    model = GRITSCRFDiploidIndel.load_from_checkpoint(
        ckpt_path, map_location=device, strict=False).eval().to(device)
    if label == "baseline":
        with torch.no_grad():
            model.encoder.count_proj.weight.zero_()
            model.encoder.count_proj.bias.zero_()

    all_samples = ([("IDX-INBRED", ind) for ind in INBRED] +
                   [("IDX-HYB", ind) for ind in HYB_PAIRS] +
                   [("IDX-RIL2", ind) for ind in RIL2_PAIRS])
    for ds, ind in all_samples:
        for depth in DEPTHS:
            pa, ia, cred, chk = eval_one(model, ds, ind, depth)
            results[(label, ds, ind, depth)] = (pa, ia)
            print(f"{label:<24}{ds:<12}{ind:<12}{depth:<7}pair_acc={pa*100:6.2f}%  ibd_adj={ia*100:6.2f}%",
                  flush=True)
    print()

print("\n=== SUMMARY: mean pair_acc / ibd_adj by kind and depth, baseline vs groupedcount-fullscale ===")
for kind, sample_list in [("IDX-INBRED", INBRED), ("IDX-HYB", HYB_PAIRS), ("IDX-RIL2", RIL2_PAIRS)]:
    print(f"\n-- {kind} --")
    print(f"{'metric/label':<28}" + "".join(f"{d:>10}" for d in DEPTHS))
    for label in CKPTS:
        for metric_idx, metric_name in [(0, "pair_acc"), (1, "ibd_adj")]:
            cells = []
            for depth in DEPTHS:
                vals = [results[(label, kind, ind, depth)][metric_idx] for ind in sample_list]
                cells.append(sum(vals) / len(vals))
            print(f"{label+'/'+metric_name:<28}" + "".join(f"{100*c:>9.3f}% " for c in cells))
