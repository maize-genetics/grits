#!/bin/bash
# eval_one.sh NAME DEPTH ONE TWO [PRESET] : (ONE=- : no --hap-switch, the preset decides) emE decode of the EVAL-corpus hybrid rows (IDX/MIX/OUT-HYB) on emE inputs with
# hap_switch(one,two) (NAME=off: no hap_switch, must reproduce emE), then make hsE_NAME_SUF a full-corpus composite
# (non-hybrid rows = emE symlinks, bit-identical by construction).
set -u
N=$1; D=$2; O1=$3; O2=$4; PR=${5:-placedfix_hetdel}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)  # this checkout
W=${GRITS_WORKDIR:?set GRITS_WORKDIR to the GRITS working directory}; S=$W/scratch/hapswitch
E=${GRITS_PYTHON_ENV:?set GRITS_PYTHON_ENV to the python env prefix (bin/python, lib)}
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=${T:-4} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SUF=01; G=rows_0.1; [ "$D" = 1.0 ] && SUF=1x && G=rows_1.0
TAG=hsE_${N}_${SUF}; B=emE_$SUF
[ -f $S/logs/$TAG.done ] && exit 0
HS=""; [ "$N" != off ] && [ "$O1" != - ] && HS="--hap-switch {\"one\":$O1,\"two\":$O2}"
$W/scripts/gpu_slot.sh -m 32 $E/bin/python -W ignore $REPO/experiments/simval-corpus/scripts/fast_eval_ckpt.py \
  --tag $TAG --ckpt $W/indel_baseline/checkpoints/v6cfullfe-stage2B/last.ckpt --scope idx mix out --route \
  --knockout homoroute --preset $PR --depth $D --manifest $S/jobs/manifest_eval_hyb.tsv \
  --input-glob scratch/editmatch/model_eval_E/$G $HS > $S/logs/$TAG.log 2>&1 || exit 1
for sc in idx mix out; do
  for f in $W/results/${B}_${sc}_fast/${B}__*.json; do
    rest=$(basename $f); rest=${rest#${B}__}
    case $rest in *-HYB__*) continue;; esac
    ln -sfn $f $W/results/${TAG}_${sc}_fast/${TAG}__$rest
    ln -sfn $W/scratch/${B}_${sc}_fast/${B}__${rest%.json} $W/scratch/${TAG}_${sc}_fast/${TAG}__${rest%.json}
  done
done
n=$(ls $W/results/${TAG}_{idx,mix,out}_fast | grep -c json)
touch $S/logs/$TAG.done
echo "$(date +%m-%d_%H:%M) eval $TAG ($n jsons)" >> $S/status.txt
