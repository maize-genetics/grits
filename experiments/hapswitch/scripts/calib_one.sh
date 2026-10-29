#!/bin/bash
# calib_one.sh NAME DEPTH ONE TWO : emE decode of the calibration HYB rows with hap_switch(one,two) via fast_eval_ckpt
set -u
N=$1; D=$2; O1=$3; O2=$4
REPO=$(cd "$(dirname "$0")/../../.." && pwd)  # this checkout
W=/local/workdir/zrm22/HackathonJun2026/grits_workdir; S=$W/scratch/hapswitch
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=${T:-4} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SUF=01; [ "$D" = 1.0 ] && SUF=1x
TAG=hsC_${N}_${SUF}
[ -f $S/logs/$TAG.done ] && exit 0
PR=${5:-placedfix_hetdel}; HS=""; [ "$N" != off ] && [ "$O1" != - ] && HS="--hap-switch {\"one\":$O1,\"two\":$O2}"
$W/scripts/gpu_slot.sh -m 32 $E/bin/python -W ignore $REPO/experiments/simval-corpus/scripts/fast_eval_ckpt.py \
  --tag $TAG --ckpt $W/indel_baseline/checkpoints/v6cfullfe-stage2B/last.ckpt --scope idx mix out --route \
  --knockout homoroute --preset $PR --depth $D \
  --manifest /workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_calibration/manifest.tsv \
  --input-glob scratch/hapswitch/calib $HS > $S/logs/$TAG.log 2>&1 || exit 1
touch $S/logs/$TAG.done
echo "$(date +%m-%d_%H:%M) calib $TAG" >> $S/status.txt
