"""
Full-scale region-buffer-fix checkpoint evaluation (branch
indel-region-buffer-fix): ALL real IDX-INBRED/IDX-HYB/IDX-RIL2 samples
(5 each, 15 total), all 5 depths, founder pair_acc AND
ibd_adjusted_accuracy -- v3-K25 baseline AND Branch B
(diploid-indel-v3-collapse-fullscale) vs the region-buffer-fix retrain
(indel_region_mult 4->2, --min/max-crossovers 2/10->8/23, recovering the
recipe's intended ~6 crossovers/individual instead of the ~1.16 actually
observed under the old buffering waste).

Baseline's 75-row pass is loaded from eval_common's cache. Branch B's own
numbers are read from its own checkpoint fresh (not cached anywhere) so
this script gives a genuine three-way comparison in one place.
"""
import glob
import sys
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/grits_workdir/indel-v3-simulator-wt/src")

from python.crf.train_diploid_indel import GRITSCRFDiploidIndel

import eval_common as ec

REGIONFIX_CKPTS = sorted(glob.glob(
    "/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
    "diploid-indel-v3-regionfix-fullscale/d-epoch=*.ckpt"))
assert REGIONFIX_CKPTS, "no diploid-indel-v3-regionfix-fullscale checkpoint found yet"
REGIONFIX_CKPT = REGIONFIX_CKPTS[-1]  # latest epoch
LABEL = "regionfix-fullscale"

BRANCHB_CKPT = ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
                "diploid-indel-v3-collapse-fullscale/d-epoch=04-val_pair_acc=0.6542.ckpt")

results = {}
for (ds, ind, depth), v in ec.get_baseline_results().items():
    results[("baseline", ds, ind, depth)] = v

print(f"loading branchB: {BRANCHB_CKPT}", flush=True)
model_b = GRITSCRFDiploidIndel.load_from_checkpoint(
    BRANCHB_CKPT, map_location=ec.device, strict=False).eval().to(ec.device)
for ds, ind in ec.ALL_SAMPLES:
    for depth in ec.DEPTHS:
        pa, ia, cred, chk = ec.eval_one(model_b, ds, ind, depth)
        results[("branchB", ds, ind, depth)] = (pa, ia)
        print(f"branchB                 {ds:<12}{ind:<12}{depth:<7}pair_acc={pa*100:6.2f}%  ibd_adj={ia*100:6.2f}%",
              flush=True)
del model_b

print(f"loading {LABEL}: {REGIONFIX_CKPT}", flush=True)
model = GRITSCRFDiploidIndel.load_from_checkpoint(
    REGIONFIX_CKPT, map_location=ec.device, strict=False).eval().to(ec.device)
for ds, ind in ec.ALL_SAMPLES:
    for depth in ec.DEPTHS:
        pa, ia, cred, chk = ec.eval_one(model, ds, ind, depth)
        results[(LABEL, ds, ind, depth)] = (pa, ia)
        print(f"{LABEL:<24}{ds:<12}{ind:<12}{depth:<7}pair_acc={pa*100:6.2f}%  ibd_adj={ia*100:6.2f}%",
              flush=True)

ec.print_summary(results, LABEL)

print("\n=== SUMMARY 2: mean pair_acc / ibd_adj by kind and depth, baseline vs branchB vs regionfix ===")
for kind, sample_list in [("IDX-INBRED", ec.INBRED), ("IDX-HYB", ec.HYB_PAIRS), ("IDX-RIL2", ec.RIL2_PAIRS)]:
    print(f"\n-- {kind} --")
    print(f"{'metric/label':<28}" + "".join(f"{d:>10}" for d in ec.DEPTHS))
    for label in ("baseline", "branchB", LABEL):
        for metric_idx, metric_name in [(0, "pair_acc"), (1, "ibd_adj")]:
            cells = []
            for depth in ec.DEPTHS:
                vals = [results[(label, kind, ind, depth)][metric_idx] for ind in sample_list]
                cells.append(sum(vals) / len(vals))
            print(f"{label + '/' + metric_name:<28}" + "".join(f"{100*c:>9.3f}% " for c in cells))
