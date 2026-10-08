#!/bin/bash
# v6c full: stage-1 retrain with held-out null labels replaced by the closest visible founder
# (heldout_closest_relabel.py), its own emission table, then A, stage-2 B, and 0.1x + 1x scoring
# as decoded and with the null founder banned. Same recipe as v6 (v6_train.sh, --extra-heads).
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/sh-distfix-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python; export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src
D=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline; CK=$D/checkpoints
J=/home/zrm22/.claude/jobs/v6c; mkdir -p $J
S=$WT/experiments/supervised-heads/scripts; R=$WT/experiments/supervised-heads/results
DATA=$D/data/training/maize_v6hoc_multidepth_fullscale_sliced.npy
LIK=$R/lik_table_v6c.json
TERN=$CK/diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt
G=${GPU:-1}; name=v6cfull
log() { echo "$(date +%m-%d_%H:%M) $*" >> $J/done.txt; }
BASE="--data $DATA --workdir $D --num-parents 25 --time-local-emis --warmup-steps 500 --homo-penalty 3.0 --spike-skip --founder-affinity --windows-per-individual 117 --batch-size 64 --precision bf16-mixed --tie-aware-loss --pair-emission mixture --het-prior off --gate-target support --aff-source reads --emission likelihood --no-distance --extra-heads"
cd $WT
$PY $S/lik_table.py --data $DATA --rows all --out $LIK > $J/lik_table.log 2>&1 || { log "$name lik table FAILED"; exit 1; }
log "$name lik table done"
CUDA_VISIBLE_DEVICES=$G $PY src/python/crf/train_diploid_indel.py $BASE --run-name $name-stage1 --supervised-heads stage1 --max-epochs 15 --patience 3 --warm-start-ckpt $TERN > $J/stage1_$name.log 2>&1 || { log "$name stage1 FAILED"; exit 1; }
S1=$(ls $CK/$name-stage1/d-epoch=*.ckpt | sort -t= -k3 -g | head -1)
log "$name stage1 done $S1"
$PY $S/make_ckpt_A.py $S1 $LIK $CK/$name-stage1/A.ckpt > $J/make_A_$name.log 2>&1 || { log "$name make A FAILED"; exit 1; }
CUDA_VISIBLE_DEVICES=$G $PY src/python/crf/train_diploid_indel.py $BASE --run-name $name-stage2B --supervised-heads stage2 --max-epochs 1 --limit-train-batches 0.25 --warmup-steps 100 --lr 1e-2 --warm-start-ckpt $CK/$name-stage1/A.ckpt > $J/stage2B_$name.log 2>&1 || { log "$name stage2 B FAILED"; exit 1; }
log "$name stage2 B done"
cd $WT/experiments/simval-corpus/scripts
evals() {   # TAG DEPTH KNOCKOUT
  for sc in idx out mix; do
    CUDA_VISIBLE_DEVICES=$G $PY fast_eval_ckpt.py --tag $1 --ckpt $CK/$name-stage2B/last.ckpt --scope $sc --route --depth $2 --input-glob scratch/lift_s200_local20k ${3:+--knockout $3} > $J/eval_$1_$sc.log 2>&1 &
  done; wait; log "$1 eval done"
}
evals ${name}_B_nonull 0.1 nonull; evals ${name}_B 0.1
evals ${name}_B_nonull_1x 1.0 nonull; evals ${name}_B_1x 1.0
log "V6C FULL DONE"
