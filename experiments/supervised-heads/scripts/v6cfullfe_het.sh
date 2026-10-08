#!/bin/bash
# Het-rate retest on v6cfullfe (best full-coverage model): once-per-segment het prior from the pooled
# het head at sample (global), chromosome (chrom) and +-5-window (region) scale, 0.1x + 1x.
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/sh-distfix-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python; export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src
CKPT=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/v6cfullfe-stage2B/last.ckpt
J=/home/zrm22/.claude/jobs/v6fe_het; mkdir -p $J
log() { echo "$(date +%m-%d_%H:%M) $*" >> $J/done.txt; }
cd $WT/experiments/simval-corpus/scripts
ev() {   # GPU SCALE DEPTH
  local t=v6cfullfe_het$2; [ $3 = 1.0 ] && t=${t}_1x
  for sc in idx out mix; do
    CUDA_VISIBLE_DEVICES=$1 $PY fast_eval_ckpt.py --tag $t --ckpt $CKPT --scope $sc --route --depth $3 --input-glob scratch/lift_s200_local20k --het-scale $2 > $J/eval_${t}_$sc.log 2>&1 &
  done; wait; log "$t done"
}
( ev 0 global 0.1; ev 0 chrom 0.1; ev 0 region 0.1 ) &
( ev 1 global 1.0; ev 1 chrom 1.0; ev 1 region 1.0 ) &
wait
log "HET RETEST DONE"
