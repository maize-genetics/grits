"""
Bigger-model collapse-rows checkpoint evaluation (branch
indel-collapse-bigger-model, off indel-readcount-row-collapse): ALL real
IDX-INBRED/IDX-HYB/IDX-RIL2 samples (5 each, 15 total), all 5 depths,
founder pair_acc AND ibd_adjusted_accuracy -- v3-K25 baseline vs a
from-scratch retrain at d_model=384/n_heads=8/n_layers=8 (~3x params vs
the deployed 256/8/6 architecture) on the EXACT SAME collapse-rows
training data Branch B (diploid-indel-v3-collapse-fullscale) used --
architecture-capacity experiment only, not a data or recipe change.
Reuses the same *_wcount.npy real-data conversions as every prior round
unchanged.

Baseline's 75-row pass is loaded from eval_common's cache instead of
recomputed -- it's the SAME checkpoint/data every eval script in this
directory uses, deterministic, and was independently recomputed three
times this session before that cache existed.
"""
import sys
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.claude/worktrees/agent-a793b2b6a2dd9fdc6/src")
import torch

from python.crf.train_diploid_indel import GRITSCRFDiploidIndel

import eval_common as ec

# NOT "latest epoch" -- across 10 total epochs (5 from-scratch + 5 more
# continued from the from-scratch run's own best checkpoint, per the
# user's convergence-adequacy check), the GLOBAL BEST never moved past
# the from-scratch run's own epoch 2. Trajectory (run1/run2):
#   0.4862, 0.5393, 0.6377(*), 0.5817, 0.5890 | 0.5786, 0.5858, 0.6180, 0.6086, 0.5970
# Real training instability (rising spike-skip counts, 6->6->11 then
# 18->27->28->30), not under-convergence -- extending training did not
# find anything better. Using the actual best checkpoint found.
COLLAPSE_CKPT = ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
                  "diploid-indel-v3-collapse-bigmodel/d-epoch=02-val_pair_acc=0.6377.ckpt")
LABEL = "collapse-bigmodel"

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
