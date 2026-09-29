#!/bin/bash
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/supervised-heads-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python; export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src
J=/home/zrm22/.claude/jobs/sup_heads; CK=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints
log() { echo "$(date +%H:%M) $*" >> $J/done.txt; }
ev() {  # GPU CKPT TAG DEPTH
  cd $WT/experiments/simval-corpus/scripts
  CUDA_VISIBLE_DEVICES=$1 $PY fast_eval_ckpt.py --tag $3 --ckpt $2 --scope idx out mix --route --depth $4 --input-glob scratch/lift_s200_local20k > $J/eval_$3.log 2>&1; log "$3 eval exit=$?"
}
( while kill -0 1367112 2>/dev/null; do sleep 20; done; log "sh3_Aplace eval finished (orphaned run)"
  ev 1 $CK/sh3-stage1/A_place.ckpt sh3_Aplace_1x 1.0 ) &
( until grep -q "sh3_B_1x eval exit" $J/done.txt; do sleep 20; done
  ev 0 $CK/sh3-stage2B_place/last.ckpt sh3_Bplace 0.1
  ev 0 $CK/sh3-stage2B_place/last.ckpt sh3_Bplace_1x 1.0 ) &
wait
log "REBALANCE DONE"
