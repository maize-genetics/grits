"""
Fix #1, corrected design: earlier per-site/per-founder PAD-density checks
proved the depth-confidence mechanism is NOT about evidence sparsity within
a window (ternary fill_rate=1.0000 at every founder/site/depth tested --
see evidence_cap_sweep.py and the per-founder-fill follow-up). The real
difference is genomic SPAN: a fixed T=512-row window is built from
CONSECUTIVE covered rows (ropebwt_npy_to_matrix.py), so at high depth (more
covered bins genome-wide) the same 512-row window spans a much SMALLER
genomic region -- 2.0x HYB/RIL2 Oh43xIl14H windows span a median ~280-330
bin units vs 0.1x's ~3,700-3,800 (~13x more compressed). A geographically
tiny window means the 512 "independent" CRF timesteps are actually mostly
redundant re-observations of the SAME local mapping-ambiguity signal
(nearby bp share the same underlying reads), which the CRF's stay_bonus/
emission-score accumulation reads as high confidence without it being more
correct.

This script re-derives the exact same per-row [tern|labels|dist] content
ropebwt_npy_to_matrix.py computes (verified byte-identical on window 0
below) directly from the cached pre-windowing raw.npy, then re-windows the
SAME real 2.0x rows -- no synthetic masking, no depth change -- by
subsampling every Nth covered row so a 512-row window spans ~0.1x's real
genomic footprint. If genomic compression (not evidence density) is the
real lever, this should recover 0.1x-like accuracy from 2.0x's own reads.
"""
import sys
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/grits_workdir/indel-v3-simulator-wt/src")
import numpy as np
import pandas as pd
import torch

from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, infer_real_founder_pairs, score_ibd_adjusted_accuracy

CKPT = "/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt"
K = 25
DIST_LOG_SCALE = 8.0
REALDIR = "/workdir/zrm22/HackathonJun2026/grits_workdir/data/real"
SCRATCH_ROOT = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/simval_eval_fixed"
IDX = {"B73": 0, "B97": 1, "CML103": 2, "Il14H": 11, "Oh43": 21}
WINDOW = 512
STRIDES = [1, 2, 4, 8, 13, 20]  # 1 = untouched baseline (consecutive rows, today's behavior)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GRITSCRFDiploidIndel.load_from_checkpoint(CKPT, map_location=device).eval().to(device)


def rows_from_raw(raw):
    """Reproduce ropebwt_npy_to_matrix.py's --anchor-dist-npy row transform exactly."""
    tern = raw[:, K:2 * K].astype(np.int8)
    dist_bp = raw[:, 2 * K:3 * K].astype(np.int64)
    gA = raw[:, 3 * K].astype(np.int64)
    gB = raw[:, 3 * K + 1].astype(np.int64)
    gA[gA < 0] = K
    gB[gB < 0] = K
    DIST_PAD, DIST_SAT = -1, 127
    has_anchor = dist_bp >= 0
    code = np.rint(DIST_LOG_SCALE * np.log2(1.0 + np.maximum(dist_bp, 0).astype(np.float64)))
    code = np.clip(code, 0, DIST_SAT - 1).astype(np.int8)
    dist = np.where(has_anchor, code, DIST_PAD).astype(np.int8)
    labels = np.stack([gA, gB], axis=1).astype(np.int8)
    return np.concatenate([tern, labels, dist], axis=1)  # [n_rows, 2K+2]


def windows_with_stride(rows, bins_df, stride, window=WINDOW):
    out, spans = [], []
    for _contig, idx in bins_df.groupby("contig", sort=False).indices.items():
        idx = np.sort(idx)
        binpos = bins_df["bin"].values[idx]
        sub_idx = idx[::stride]
        sub_pos = binpos[::stride]
        start = 0
        while start + window <= len(sub_idx):
            sel = sub_idx[start:start + window]
            out.append(rows[sel])
            spans.append(sub_pos[start + window - 1] - sub_pos[start])
            start += window
    if not out:
        return None, None
    return np.stack(out, axis=0), np.array(spans)


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


def ril2_truth(outdir, window_size=WINDOW):
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


def truth_for_stride(ds, ind, outdir, truth_pair, n_windows):
    if ds == "IDX-RIL2":
        # truth is keyed by original row index, not stride-subsampled positions --
        # re-derive with the same stride-subsampled windowing on the label array.
        bins_df = pd.read_csv(f"{outdir}/raw.npy.bins.tsv", sep="\t")
        truth = np.load(f"{outdir}/truth_labels.npy")
        return None  # handled inline below (needs stride, passed separately)
    else:
        p1, p2 = truth_pair
        tlo, thi = min(IDX[p1], IDX[p2]), max(IDX[p1], IDX[p2])
        true_lo = np.full((n_windows, WINDOW), tlo)
        true_hi = np.full((n_windows, WINDOW), thi)
        valid = np.ones((n_windows, WINDOW), bool)
        return true_lo, true_hi, valid


def ril2_truth_strided(outdir, bins_df, stride, window=WINDOW):
    truth = np.load(f"{outdir}/truth_labels.npy")
    windows = []
    for _contig, idx in bins_df.groupby("contig", sort=False).indices.items():
        idx = np.sort(idx)
        sub_idx = idx[::stride]
        start = 0
        while start + window <= len(sub_idx):
            sel = sub_idx[start:start + window]
            windows.append(truth[sel])
            start += window
    tw = np.stack(windows, axis=0)
    valid = (tw[:, :, 0] >= 0) & (tw[:, :, 1] >= 0)
    return np.minimum(tw[:, :, 0], tw[:, :, 1]), np.maximum(tw[:, :, 0], tw[:, :, 1]), valid


SAMPLES = [("IDX-HYB", "Oh43xIl14H", ("Oh43", "Il14H")), ("IDX-RIL2", "Oh43xIl14H", None)]

# sanity check: stride=1 must reproduce the cached ternary_k25native.npy exactly on window 0
print("=== sanity check: stride=1 reproduces cached windowed data ===")
ds, ind, _ = SAMPLES[0]
outdir = f"{SCRATCH_ROOT}/{ds}__{ind}__2.0x"
raw = np.asarray(np.load(f"{outdir}/raw.npy", mmap_mode="r"))
bins_df = pd.read_csv(f"{outdir}/raw.npy.bins.tsv", sep="\t")
rows = rows_from_raw(raw)
cached = np.load(f"{REALDIR}/{ind}_{ds}_2.0x_ternary_k25native.npy")
mine, spans1 = windows_with_stride(rows, bins_df, stride=1)
match = np.array_equal(mine[0], cached[0]) and mine.shape == cached.shape
print(f"shapes: mine={mine.shape} cached={cached.shape}  window0 identical={match}")
print()

results = []
for ds, ind, truth_pair in SAMPLES:
    outdir = f"{SCRATCH_ROOT}/{ds}__{ind}__2.0x"
    raw = np.asarray(np.load(f"{outdir}/raw.npy", mmap_mode="r"))
    bins_df = pd.read_csv(f"{outdir}/raw.npy.bins.tsv", sep="\t")
    rows = rows_from_raw(raw)

    for stride in STRIDES:
        data, spans = windows_with_stride(rows, bins_df, stride)
        if data is None:
            print(f"{ds:<12}{ind:<12}stride={stride:<4} no complete windows, skipping")
            continue
        n = data.shape[0]
        if ds == "IDX-RIL2":
            true_lo, true_hi, valid = ril2_truth_strided(outdir, bins_df, stride)
        else:
            p1, p2 = truth_pair
            tlo, thi = min(IDX[p1], IDX[p2]), max(IDX[p1], IDX[p2])
            true_lo = np.full((n, WINDOW), tlo); true_hi = np.full((n, WINDOW), thi); valid = np.ones((n, WINDOW), bool)

        # row_idx into raw/raw.npy for the ibd-adjusted read-support check must match
        # the SAME stride-subsampled row selection used to build `data`
        row_idx = []
        for _contig, idx in bins_df.groupby("contig", sort=False).indices.items():
            idx = np.sort(idx)
            sub_idx = idx[::stride]
            start = 0
            while start + WINDOW <= len(sub_idx):
                row_idx.append(sub_idx[start:start + WINDOW])
                start += WINDOW

        pa, ia, cred, chk = score_ibd_adjusted_accuracy(
            *infer_real_founder_pairs(model, data, K, device=device),
            true_lo, true_hi, valid, raw, row_idx, K)
        results.append((ds, ind, stride, n, spans.mean(), pa, ia))
        print(f"{ds:<12}{ind:<12}stride={stride:<4} n_windows={n:<6} mean_span_bin_units={spans.mean():<10.1f} "
              f"pair_acc={pa*100:6.2f}%  ibd_adj={ia*100:6.2f}%")
    print()

print("\n=== SUMMARY: pair_acc at 2.0x by row stride (1=untouched consecutive rows) ===")
print(f"{'sample':<20}" + "".join(f"stride={s:<10}" for s in STRIDES))
for ds, ind, _ in SAMPLES:
    row = [r for r in results if r[0] == ds and r[1] == ind]
    label = f"{ind}({ds})"
    cells = "".join(f"{100*r[5]:>9.2f}% " for r in row)
    print(f"{label:<20}{cells}")
