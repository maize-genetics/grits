"""
Full-scale collapse-rows checkpoint evaluation (branch
indel-readcount-row-collapse): ALL real IDX-INBRED/IDX-HYB/IDX-RIL2
samples (5 each, 15 total), all 5 depths, founder pair_acc AND
ibd_adjusted_accuracy -- v3-K25 baseline vs the full-scale retrain using
the collapse-rows design (same-kind row stacks physically merged into
one row BEFORE sampling, so windows span more distinct reference sites
for the same T-row budget, instead of the sibling grouped-count design's
post-hoc count-value-only fix). Reuses the same *_wcount.npy real-data
conversions as every prior round unchanged -- real data is already one
row per site with genuine per-founder counts, so neither of this
session's simulator-side changes require any real-data producer change.
Reports everything, not just B73/HYB pairs, matching the
comprehensive-coverage precedent from every prior round.

Baseline's 75-row pass is loaded from eval_common's cache instead of
recomputed -- it's the SAME checkpoint/data every eval script in this
directory uses, deterministic, and was independently recomputed three
times this session before this was noticed.
"""
import glob
import sys
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/grits_workdir/indel-v3-simulator-wt/src")
import torch

from python.crf.train_diploid_indel import GRITSCRFDiploidIndel

import eval_common as ec

COLLAPSE_CKPTS = sorted(glob.glob(
    "/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
    "diploid-indel-v3-collapse-fullscale/d-epoch=*.ckpt"))
assert COLLAPSE_CKPTS, "no diploid-indel-v3-collapse-fullscale checkpoint found yet"
COLLAPSE_CKPT = COLLAPSE_CKPTS[-1]  # latest epoch
LABEL = "collapse-fullscale"

results = {}
for (ds, ind, depth), v in ec.get_baseline_results().items():
    results[("baseline", ds, ind, depth)] = v

print(f"loading {LABEL}: {COLLAPSE_CKPT}", flush=True)
model = GRITSCRFDiploidIndel.load_from_checkpoint(
    COLLAPSE_CKPT, map_location=ec.device, strict=False).eval().to(ec.device)
for ds, ind in ec.ALL_SAMPLES:
    for depth in ec.DEPTHS:
        pa, ia, cred, chk = ec.eval_one(model, ds, ind, depth)
        results[(LABEL, ds, ind, depth)] = (pa, ia)
        print(f"{LABEL:<24}{ds:<12}{ind:<12}{depth:<7}pair_acc={pa*100:6.2f}%  ibd_adj={ia*100:6.2f}%",
              flush=True)

ec.print_summary(results, LABEL)
