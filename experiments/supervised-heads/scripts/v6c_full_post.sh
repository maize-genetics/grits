#!/bin/bash
# v6c full, after stage 1: A (closest-founder emission table), stage-2 B with the null founder dropped
# from the CRF (--no-null), then 0.1x + 1x scoring. Waits for the running stage-1 process (pid $1).
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/sh-distfix-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python; export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src
D=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline; CK=$D/checkpoints
J=/home/zrm22/.claude/jobs/v6c; S=$WT/experiments/supervised-heads/scripts; R=$WT/experiments/supervised-heads/results
DATA=$D/data/training/maize_v6hoc_multidepth_fullscale_sliced.npy
LIK=$R/lik_table_v6c.json; G=${GPU:-1}; name=v6cfull
log() { echo "$(date +%m-%d_%H:%M) $*" >> $J/done.txt; }
BASE="--data $DATA --workdir $D --num-parents 25 --time-local-emis --warmup-steps 500 --homo-penalty 3.0 --spike-skip --founder-affinity --windows-per-individual 117 --batch-size 64 --precision bf16-mixed --tie-aware-loss --pair-emission mixture --het-prior off --gate-target support --aff-source reads --emission likelihood --no-distance --extra-heads --no-null"
while kill -0 $1 2>/dev/null; do sleep 60; done
S1=$(ls $CK/$name-stage1/d-epoch=*.ckpt | sort -t= -k3 -g | head -1)
log "$name stage1 done $S1"
cd $WT
$PY $S/make_ckpt_A.py $S1 $LIK $CK/$name-stage1/A.ckpt > $J/make_A_$name.log 2>&1 || { log "$name make A FAILED"; exit 1; }
CUDA_VISIBLE_DEVICES=$G $PY src/python/crf/train_diploid_indel.py $BASE --run-name $name-stage2B --supervised-heads stage2 --max-epochs 1 --limit-train-batches 0.25 --warmup-steps 100 --lr 1e-2 --warm-start-ckpt $CK/$name-stage1/A.ckpt > $J/stage2B_$name.log 2>&1
[ -f $CK/$name-stage2B/last.ckpt ] || { log "$name stage2 B FAILED (no checkpoint)"; exit 1; }
log "$name stage2 B done"
cd $WT/experiments/simval-corpus/scripts
for t in "${name}_B 0.1" "${name}_B_1x 1.0"; do set -- $t
  for sc in idx out mix; do
    CUDA_VISIBLE_DEVICES=$G $PY fast_eval_ckpt.py --tag $1 --ckpt $CK/$name-stage2B/last.ckpt --scope $sc --route --depth $2 --input-glob scratch/lift_s200_local20k > $J/eval_$1_$sc.log 2>&1 &
  done; wait; log "$1 eval done"
done
bash $WT/experiments/simval-corpus/scripts/report_card.sh ${name}_B ${name}_B ${name}_B_1x > $J/report_card_${name}_B.log 2>&1 && log "report card done" || log "report card FAILED"
log "V6C FULL DONE"
