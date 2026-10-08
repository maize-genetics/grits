#!/bin/bash
# Frozen-emission fit B on closest-founder labels with no null (--no-null --freeze-emission), on two
# encoders, both starting from v6 fit B's read log-likelihoods (-4.24 / -3.67 / +0.39):
#   v6fe      : the v6 stage-1 encoder (warm start v6-stage2B/last.ckpt)            GPU 0
#   v6cfullfe : the closest-founder retrained encoder (v6cfull-stage1/A_v6emis.ckpt)   GPU 1
# Only the switch scale, founder prior, dosage and out-of-panel weights are fitted. Then 0.1x + 1x scoring.
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/sh-distfix-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python; export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src
D=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline; CK=$D/checkpoints
J=/home/zrm22/.claude/jobs/v6fe; mkdir -p $J
DATA=$D/data/training/maize_v6hoc_multidepth_fullscale_sliced.npy
log() { echo "$(date +%m-%d_%H:%M) $*" >> $J/done.txt; }
BASE="--data $DATA --workdir $D --num-parents 25 --time-local-emis --warmup-steps 500 --homo-penalty 3.0 --spike-skip --founder-affinity --windows-per-individual 117 --batch-size 64 --precision bf16-mixed --tie-aware-loss --pair-emission mixture --het-prior off --gate-target support --aff-source reads --emission likelihood --no-distance --extra-heads --no-null --freeze-emission"
run() {   # GPU NAME START_CKPT
  local g=$1 name=$2
  cd $WT
  CUDA_VISIBLE_DEVICES=$g $PY src/python/crf/train_diploid_indel.py $BASE --run-name $name-stage2B --supervised-heads stage2 --max-epochs 1 --limit-train-batches 0.25 --warmup-steps 100 --lr 1e-2 --warm-start-ckpt $3 > $J/stage2B_$name.log 2>&1
  [ -f $CK/$name-stage2B/last.ckpt ] || { log "$name stage2 B FAILED (no checkpoint)"; return 1; }
  $PY - $CK/$name-stage2B/last.ckpt >> $J/done.txt <<'PYEOF'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
print("  fitted:", {k: [round(v, 2) for v in sd[k].float().flatten().tolist()] if sd[k].numel() > 1 else round(float(sd[k]), 3)
                    for k in ("tern_loglik", "log_xo_scale", "aff_w", "dos_w", "ood_w", "aff_ood_s") if k in sd})
PYEOF
  log "$name stage2 B done"
  cd $WT/experiments/simval-corpus/scripts
  for t in "${name}_B 0.1" "${name}_B_1x 1.0"; do set -- $t
    for sc in idx out mix; do
      CUDA_VISIBLE_DEVICES=$g $PY fast_eval_ckpt.py --tag $1 --ckpt $CK/$name-stage2B/last.ckpt --scope $sc --route --depth $2 --input-glob scratch/lift_s200_local20k > $J/eval_$1_$sc.log 2>&1 &
    done; wait; log "$1 eval done"
  done
  bash $WT/experiments/simval-corpus/scripts/report_card.sh ${name}_B ${name}_B ${name}_B_1x > $J/report_card_${name}_B.log 2>&1 && log "$name report card done" || log "$name report card FAILED"
}
run 0 v6fe $CK/v6-stage2B/last.ckpt &
run 1 v6cfullfe $CK/v6cfull-stage1/A_v6emis.ckpt &
wait
log "V6FE DONE"
