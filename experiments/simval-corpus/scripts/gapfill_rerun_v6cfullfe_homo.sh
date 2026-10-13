#!/bin/bash
# Re-decode v6cfullfe_homo with the gap-filling BED writer (commit a5a1eb8) into new tags.
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/sh-distfix-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python; export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CK=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/v6cfullfe-stage2B/last.ckpt
J=/home/zrm22/.claude/jobs/gapfill
cd $WT/experiments/simval-corpus/scripts
ev() {  # GPU TAG DEPTH
  for sc in idx out mix; do
    CUDA_VISIBLE_DEVICES=$1 $PY $WT/experiments/simval-corpus/scripts/fast_eval_ckpt_wtbed.py --tag $2 --ckpt $CK --scope $sc --route --depth $3 \
      --input-glob scratch/lift_s200_local20k --knockout homoroute > $J/eval_$2_$sc.log 2>&1 &
  done; wait; echo "$(date +%m-%d_%H:%M) $2 done" >> $J/done.txt
}
ev 0 v6cfullfe_homo_gapfill 0.1 &
ev 1 v6cfullfe_homo_gapfill_1x 1.0 &
wait
echo "$(date +%m-%d_%H:%M) DECODE DONE" >> $J/done.txt
