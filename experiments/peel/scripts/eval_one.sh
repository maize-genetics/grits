#!/bin/bash
# eval_one.sh NAME DEPTH PEELJSON : promoted decode (placedfix_hetdel_hs) + --peel PEELJSON of the EVAL-corpus
# hybrid rows (IDX/MIX/OUT-HYB, emE inputs); then pE_NAME_SUF becomes a full-corpus composite whose non-hybrid
# rows are symlinks to hapswitch_hs (hsE_hs_SUF), bit-identical by construction. PEELJSON=- reproduces hsE_hs.
set -u
N=$1; D=$2; PJ=$3
REPO=$(cd "$(dirname "$0")/../../.." && pwd)  # this checkout
W=/local/workdir/zrm22/HackathonJun2026/grits_workdir; S=$W/scratch/peel
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=${T:-4} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SUF=01; G=rows_0.1; [ "$D" = 1.0 ] && SUF=1x && G=rows_1.0
TAG=pE_${N}_${SUF}; B=hsE_hs_$SUF
[ -f $S/logs/$TAG.done ] && exit 0
PL=(); [ "$PJ" != - ] && PL=(--peel "$PJ")
$W/scripts/gpu_slot.sh -m 32 $E/bin/python -W ignore $REPO/experiments/simval-corpus/scripts/fast_eval_ckpt.py \
  --tag $TAG --ckpt $W/indel_baseline/checkpoints/v6cfullfe-stage2B/last.ckpt --scope idx mix out --route \
  --knockout homoroute --preset placedfix_hetdel_hs --depth $D --manifest $W/scratch/hapswitch/jobs/manifest_eval_hyb.tsv \
  --input-glob scratch/editmatch/model_eval_E/$G "${PL[@]}" > $S/logs/$TAG.log 2>&1 || exit 1
for sc in idx mix out; do
  for f in $W/results/${B}_${sc}_fast/${B}__*.json; do
    rest=$(basename $f); rest=${rest#${B}__}
    case $rest in *-HYB__*) continue;; esac
    ln -sfn $(readlink -f $f) $W/results/${TAG}_${sc}_fast/${TAG}__$rest
    ln -sfn $(readlink -f $W/scratch/${B}_${sc}_fast/${B}__${rest%.json}) $W/scratch/${TAG}_${sc}_fast/${TAG}__${rest%.json}
  done
done
n=$(ls $W/results/${TAG}_{idx,mix,out}_fast | grep -c json)
touch $S/logs/$TAG.done
echo "$(date +%m-%d_%H:%M) eval $TAG ($n jsons)" >> $S/status.txt
