#!/bin/bash
# Evaluate v6cfullfe-t4096 like v6cfullfe_homo_gapfill (route + homoroute, gap-filling BEDs) with 4096-row
# windows (fast_eval_bigwin.py: end-aligned last window per contig, so coverage matches the 512 model).
# Waits for the training pipeline. 0.1x and 1x all scopes -> report card + dashboard; 0.5x / 2x HYB scopes.
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/sh-distfix-wt
HD=/local/workdir/zrm22/HackathonJun2026/grits_workdir/hybrid-depth-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python; export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CK=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/v6cfullfe-t4096-stage2B/last.ckpt
J=/home/zrm22/.claude/jobs/t4096
log() { echo "$(date +%m-%d_%H:%M) $*" >> $J/done.txt; }


cd /local/workdir/zrm22/HackathonJun2026/grits_workdir
ev() {  # GPU TAG DEPTH SCOPES...
  local g=$1 tag=$2 dep=$3; shift 3
  for sc in "$@"; do
    CUDA_VISIBLE_DEVICES=$g $PY $HD/experiments/hybrid-depth/scripts/fast_eval_bigwin.py --tag $tag --ckpt $CK --scope $sc \
      --route --knockout homoroute --depth $dep --window-rows 4096 > $J/eval_${tag}_$sc.log 2>&1 &
  done; wait; log "$tag eval done"
}
ev 0 v6cfullfe_homo_gapfill_t4096 0.1 idx out mix &
ev 1 v6cfullfe_homo_gapfill_t4096_1x 1.0 idx out mix &
wait
GENEWINDOW=1 bash $WT/experiments/simval-corpus/scripts/report_card.sh v6cfullfe_homo_gapfill_t4096 \
  v6cfullfe_homo_gapfill_t4096 v6cfullfe_homo_gapfill_t4096_1x "v6cfullfe recipe retrained at T=4096 rows/window" \
  > $J/report_card.log 2>&1 && log "report card done" || log "report card FAILED"
ev 0 v6cfullfe_homo_gapfill_t4096_0p5x 0.5 idx out mix &
ev 1 v6cfullfe_homo_gapfill_t4096_2x 2.0 idx out mix &
wait
log "T4096 EVAL DONE"
