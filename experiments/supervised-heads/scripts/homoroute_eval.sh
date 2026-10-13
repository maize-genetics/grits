#!/bin/bash
# Homozygous restriction from the router (--knockout homoroute): samples routed as not hybrid decode
# with homozygous founder pairs only. v6cfullfe (no_null in the checkpoint) on GPU 0; v6 fit B with
# null banned (nonull,homoroute) on GPU 1 once the het-rate retest has finished there. 0.1x + 1x.
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/sh-distfix-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python; export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src
CK=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints
J=/home/zrm22/.claude/jobs/homoroute; mkdir -p $J
log() { echo "$(date +%m-%d_%H:%M) $*" >> $J/done.txt; }
cd $WT/experiments/simval-corpus/scripts
ev() {   # GPU TAG CKPT DEPTH KNOCKOUT
  for sc in idx out mix; do
    CUDA_VISIBLE_DEVICES=$1 $PY fast_eval_ckpt.py --tag $2 --ckpt $3 --scope $sc --route --depth $4 --input-glob scratch/lift_s200_local20k --knockout $5 > $J/eval_$2_$sc.log 2>&1 &
  done; wait; log "$2 done"
}
( ev 0 v6cfullfe_homo $CK/v6cfullfe-stage2B/last.ckpt 0.1 homoroute
  ev 0 v6cfullfe_homo_1x $CK/v6cfullfe-stage2B/last.ckpt 1.0 homoroute ) &
( until grep -q "HET RETEST DONE" /home/zrm22/.claude/jobs/v6fe_het/done.txt; do sleep 30; done
  ev 1 v6nonull_homo $CK/v6-stage2B/last.ckpt 0.1 nonull,homoroute
  ev 1 v6nonull_homo_1x $CK/v6-stage2B/last.ckpt 1.0 nonull,homoroute ) &
wait
for m in v6cfullfe_homo v6nonull_homo; do
  bash $WT/experiments/simval-corpus/scripts/report_card.sh $m $m ${m}_1x > $J/report_card_$m.log 2>&1 && log "$m report card done" || log "$m report card FAILED"
done
log "HOMOROUTE DONE"
