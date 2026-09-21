"""
Shared helpers for the depth-confidence-fix 15-sample x 5-depth real-data
eval scripts (eval_readcount_fullscale.py, eval_groupedcount_fullscale.py,
eval_collapse_fullscale.py, ...). Pulled out after noticing each of those
scripts independently reloaded the SAME baseline checkpoint
(diploid-indel-v3-k25-overlay-affinity) and reran inference on the SAME 75
real (dataset, individual, depth) rows -- deterministic, unchanged results,
recomputed from scratch three separate times this session for zero new
information (~half of each script's runtime). CACHE_PATH persists that
75-row baseline result once; every eval script should load it via
get_baseline_results() instead of recomputing.

If BASELINE_CKPT or the real *_wcount.npy data ever change, delete
CACHE_PATH (or pass force=True) to invalidate it -- there is no automatic
staleness check beyond that.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/grits_workdir/indel-v3-simulator-wt/src")
import numpy as np
import pandas as pd
import torch

from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, infer_real_founder_pairs, score_ibd_adjusted_accuracy

BASELINE_CKPT = ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
                  "diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt")
CACHE_PATH = Path(__file__).parent.parent / "results" / "baseline_v3k25_15x5_wcount_results.json"

K = 25
REALDIR = "/workdir/zrm22/HackathonJun2026/grits_workdir/data/real"
SCRATCH_ROOT = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/simval_eval_fixed"
IDX = {"B73": 0, "B97": 1, "CML103": 2, "Il14H": 11, "Oh43": 21}
DEPTHS = ["0.01x", "0.1x", "0.5x", "1.0x", "2.0x"]

INBRED = ["B73", "Oh43", "Il14H", "B97", "CML103"]
HYB_PAIRS = ["B73xOh43", "B73xCML103", "Oh43xIl14H", "B97xCML103", "Il14HxB97"]
RIL2_PAIRS = ["B73xCML103", "B73xOh43", "B97xCML103", "Il14HxB97", "Oh43xIl14H"]

ALL_SAMPLES = ([("IDX-INBRED", ind) for ind in INBRED] +
               [("IDX-HYB", ind) for ind in HYB_PAIRS] +
               [("IDX-RIL2", ind) for ind in RIL2_PAIRS])

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


def load_baseline_model():
    model = GRITSCRFDiploidIndel.load_from_checkpoint(
        BASELINE_CKPT, map_location=device, strict=False).eval().to(device)
    with torch.no_grad():
        model.encoder.count_proj.weight.zero_()
        model.encoder.count_proj.bias.zero_()
    return model


def get_baseline_results(force=False, verbose=True):
    """Returns {(ds, ind, depth): (pair_acc, ibd_adj)} for all 75
    (dataset, individual, depth) combos, from CACHE_PATH if present,
    else computed once and persisted there."""
    if not force and CACHE_PATH.exists():
        if verbose:
            print(f"loading cached baseline results from {CACHE_PATH}", flush=True)
        raw = json.loads(CACHE_PATH.read_text())
        return {tuple(k.split("|")): tuple(v) for k, v in raw.items()}

    if verbose:
        print(f"no cache at {CACHE_PATH} (or force=True) -- computing baseline "
              f"75-row pass once", flush=True)
    model = load_baseline_model()
    results = {}
    for ds, ind in ALL_SAMPLES:
        for depth in DEPTHS:
            pa, ia, cred, chk = eval_one(model, ds, ind, depth)
            results[(ds, ind, depth)] = (pa, ia)
            if verbose:
                print(f"baseline{'':<16}{ds:<12}{ind:<12}{depth:<7}"
                      f"pair_acc={pa*100:6.2f}%  ibd_adj={ia*100:6.2f}%", flush=True)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps({"|".join(k): v for k, v in results.items()}, indent=1))
    if verbose:
        print(f"cached baseline results -> {CACHE_PATH}", flush=True)
    return results


def print_summary(results, other_label):
    """results: {(label, ds, ind, depth): (pair_acc, ibd_adj)}, with
    "baseline" and other_label as the two labels present."""
    print(f"\n=== SUMMARY: mean pair_acc / ibd_adj by kind and depth, baseline vs {other_label} ===")
    for kind, sample_list in [("IDX-INBRED", INBRED), ("IDX-HYB", HYB_PAIRS), ("IDX-RIL2", RIL2_PAIRS)]:
        print(f"\n-- {kind} --")
        print(f"{'metric/label':<28}" + "".join(f"{d:>10}" for d in DEPTHS))
        for label in ("baseline", other_label):
            for metric_idx, metric_name in [(0, "pair_acc"), (1, "ibd_adj")]:
                cells = []
                for depth in DEPTHS:
                    vals = [results[(label, kind, ind, depth)][metric_idx] for ind in sample_list]
                    cells.append(sum(vals) / len(vals))
                print(f"{label + '/' + metric_name:<28}" + "".join(f"{100*c:>9.3f}% " for c in cells))
