#!/bin/bash
# v6cfullfe-t4096 with whole-chromosome decode (fast_eval_bigwin --decode wholechrom), all depths, then report card.
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/sh-distfix-wt
HD=/local/workdir/zrm22/HackathonJun2026/grits_workdir/hybrid-depth-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python; export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
CK=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/v6cfullfe-t4096-stage2B/last.ckpt
J=/home/zrm22/.claude/jobs/t4096; log() { echo "$(date +%m-%d_%H:%M) $*" >> $J/done.txt; }
cd /local/workdir/zrm22/HackathonJun2026/grits_workdir
ev() {  # GPU TAG DEPTH
  for sc in idx out mix; do
    CUDA_VISIBLE_DEVICES=$1 $PY $HD/experiments/hybrid-depth/scripts/fast_eval_bigwin.py --tag $2 --ckpt $CK --scope $sc \
      --route --knockout homoroute --depth $3 --window-rows 4096 --decode wholechrom > $J/eval_$2_$sc.log 2>&1
  done; log "$2 eval done"
}
B=v6cfullfe_homo_gapfill_t4096_wc
( ev 0 $B 0.1; ev 0 ${B}_0p5x 0.5 ) &
( ev 1 ${B}_1x 1.0; ev 1 ${B}_2x 2.0 ) &
wait
GENEWINDOW=1 bash $WT/experiments/simval-corpus/scripts/report_card.sh $B $B ${B}_1x "v6cfullfe-t4096 + whole-chromosome decode" \
  > $J/report_card_wc.log 2>&1 && log "wc report card done" || log "wc report card FAILED"
log "T4096 WC EVAL DONE"
