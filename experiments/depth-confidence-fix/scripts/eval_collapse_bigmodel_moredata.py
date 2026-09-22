"""
Bigger-model + more-data collapse-rows checkpoint evaluation (branch
indel-collapse-bigmodel-moredata, off indel-collapse-bigger-model): ALL
real IDX-INBRED/IDX-HYB/IDX-RIL2 samples (5 each, 15 total), all 5
depths, founder pair_acc AND ibd_adjusted_accuracy -- v3-K25 baseline vs
a from-scratch retrain at d_model=384/n_heads=8/n_layers=8 (~14.9M
params, same architecture as the sibling collapse-bigmodel experiment)
on 2x the training data (234,000 windows, 1000+1000 individuals instead
of 500+500) vs both the bigmodel-only run (117,000 windows) and Branch B
(diploid-indel-v3-collapse-fullscale, 5.07M params, 117,000 windows).

Question this answers: does 2x data fix the training instability found
with more capacity alone? Answer, from the training trajectory: partially.
Epoch-by-epoch val_pair_acc: 0.6028, [epoch1 lost to an environmental
crash -- unrelated DataLoader OOM from a concurrent sibling job], 0.6519,
0.6677(*best*), 0.636. Monotonically increasing through epoch 3 (unlike
the bigmodel-only run's non-monotonic epoch0-2 peak-then-drop), but still
drops at epoch 4 -- the instability is delayed and dampened by more data,
not eliminated. Using the ModelCheckpoint callback's own best checkpoint
(epoch 3, val_pair_acc=0.6677), same convention as every other checkpoint
choice this session.

Reuses the same *_wcount.npy real-data conversions as every prior round
unchanged. Baseline's 75-row pass is loaded from eval_common's cache.
"""
import sys
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.claude/worktrees/agent-af19f5a0eaeb11150/src")
import torch

from python.crf.train_diploid_indel import GRITSCRFDiploidIndel

import eval_common as ec

COLLAPSE_CKPT = ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
                  "diploid-indel-v3-collapse-bigmodel-moredata/d-epoch=03-val_pair_acc=0.6677.ckpt")
LABEL = "collapse-bigmodel-moredata"

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
        print(f"{LABEL:<26}{ds:<12}{ind:<12}{depth:<7}pair_acc={pa*100:6.2f}%  ibd_adj={ia*100:6.2f}%",
              flush=True)

ec.print_summary(results, LABEL)
