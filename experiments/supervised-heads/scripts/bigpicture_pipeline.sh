#!/bin/bash
# Big-picture comparison over every corpus depth: align the depths not yet on the -s 200 pipeline
# (0.01x, 0.5x, 2x), then score v6 fit B, ternary-baseline and diploid-affinity at each depth.
# Scoring: one process per scope, at most 3 processes per GPU at a time.
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # this checkout
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python; export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src
CK=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints
V6B=$CK/v6-stage2B/last.ckpt
TERN=$CK/diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt
J=/home/zrm22/.claude/jobs/bigpic
log() { echo "$(date +%m-%d_%H:%M) $*" >> $J/done.txt; }
OUT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/lift_s200_local20k
DS="IDX-INBRED IDX-HYB IDX-RIL2 OUT-INBRED OUT-HYB OUT-RIL2 MIX-HYB MIX-RIL2"
cd $WT/experiments/het-replacement/scripts
if [ "$1" != "--score-only" ]; then
  $PY align_calibration.py --corpus evaluation --coverages 0.01 0.5 --out-root $OUT --datasets $DS --workers 4 --gpu 0 > $J/align_low.log 2>&1 &
  $PY align_calibration.py --corpus evaluation --coverages 2.0 --out-root $OUT --datasets $DS --workers 4 --gpu 1 > $J/align_2x.log 2>&1 &
  wait
else
  while pgrep -f "align_calibration.py --corpus evaluation" > /dev/null; do sleep 120; done
fi
log "alignments done: $(ls $OUT | sed 's/.*__//' | sort | uniq -c | tr '\n' ' ')"
cd $WT/experiments/simval-corpus/scripts
score() {   # GPU DEPTH SUFFIX
  local g=$1 dep=$2 suf=$3
  for sc in idx out mix; do CUDA_VISIBLE_DEVICES=$g $PY fast_eval_ckpt.py --tag v6_B$suf --ckpt $V6B --scope $sc --route --depth $dep --input-glob scratch/lift_s200_local20k > $J/v6B${suf}_$sc.log 2>&1 & done; wait
  for sc in idx out mix; do CUDA_VISIBLE_DEVICES=$g $PY fast_eval_ckpt.py --tag tern$suf --ckpt $TERN --scope $sc --route --depth $dep --input-glob scratch/lift_s200_local20k > $J/tern${suf}_$sc.log 2>&1 & done; wait
  for sc in idx out mix; do CUDA_VISIBLE_DEVICES=$g $PY fast_eval_diploid_affinity.py --tag diploid_affinity_s200$suf --scope $sc --depth $dep --input-glob scratch/lift_s200_local20k > $J/da${suf}_$sc.log 2>&1 & done; wait
  log "depth $dep scored"
}
( score 0 0.01 _0.01x; score 0 2.0 _2x ) &
( score 1 0.5 _0.5x ) &
wait
log "BIGPICTURE DONE"
