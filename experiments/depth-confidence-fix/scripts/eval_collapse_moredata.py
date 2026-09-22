"""
Full-scale collapse-rows-moredata checkpoint evaluation (branch
indel-collapse-moredata): tests whether doubling training data (1000+1000
individuals vs Branch B's 500+500, same --collapse-rows --emit-read-counts
recipe otherwise, same warm-start) improves further on Branch B's
already-strong real-data result. Baseline's 75-row pass loaded from
eval_common's cache, not recomputed.
"""
import glob
import sys
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/grits_workdir/indel-v3-simulator-wt/src")
import torch

from python.crf.train_diploid_indel import GRITSCRFDiploidIndel

import eval_common as ec

MOREDATA_CKPTS = sorted(glob.glob(
    "/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
    "diploid-indel-v3-collapse-moredata/d-epoch=*.ckpt"))
assert MOREDATA_CKPTS, "no diploid-indel-v3-collapse-moredata checkpoint found yet"
MOREDATA_CKPT = MOREDATA_CKPTS[-1]  # latest epoch (by filename sort; epoch=03 is best per training log)
LABEL = "collapse-moredata"

results = {}
for (ds, ind, depth), v in ec.get_baseline_results().items():
    results[("baseline", ds, ind, depth)] = v

print(f"loading {LABEL}: {MOREDATA_CKPT}", flush=True)
model = GRITSCRFDiploidIndel.load_from_checkpoint(
    MOREDATA_CKPT, map_location=ec.device, strict=False).eval().to(ec.device)
for ds, ind in ec.ALL_SAMPLES:
    for depth in ec.DEPTHS:
        pa, ia, cred, chk = ec.eval_one(model, ds, ind, depth)
        results[(LABEL, ds, ind, depth)] = (pa, ia)
        print(f"{LABEL:<24}{ds:<12}{ind:<12}{depth:<7}pair_acc={pa*100:6.2f}%  ibd_adj={ia*100:6.2f}%",
              flush=True)

ec.print_summary(results, LABEL)
