#!/bin/bash
# sh5 variant: no affinity head/loss in stage 1; CRF founder prior = the sample's genome-wide
# read match rate (diploid-affinity's _founder_affinity), same data/recipe as sh5 otherwise.
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/sh-distfix-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python
export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src
D=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline
TD=$D/data/training; CK=$D/checkpoints
J=/home/zrm22/.claude/jobs/sh5
S=$WT/experiments/supervised-heads/scripts
R=$WT/experiments/supervised-heads/results
DATA=$TD/maize_v3read_multidepth_fullscale_sliced.npy
TERN=$D/checkpoints/diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt
log() { echo "$(date +%m-%d_%H:%M) $*" >> $J/done.txt; }
until grep -q "concat ok" $J/done.txt 2>/dev/null && [ -s $R/lik_table_sh5.json ]; do
  grep -q "FAILED" $J/done.txt 2>/dev/null && exit 1; sleep 60; done
name=sh5na; gpu=${GPU:-1}
RECIPE="--data $DATA --workdir $D --num-parents 25 --time-local-emis --warmup-steps 500 --homo-penalty 3.0 --spike-skip --founder-affinity --windows-per-individual 117 --batch-size 64 --precision bf16-mixed --tie-aware-loss --pair-emission mixture --het-prior off --emission likelihood_dist --aff-source reads"
cd $WT
CUDA_VISIBLE_DEVICES=$gpu $PY src/python/crf/train_diploid_indel.py $RECIPE --run-name $name-stage1 --supervised-heads stage1 --max-epochs 15 --patience 3 --warm-start-ckpt $TERN > $J/stage1_$name.log 2>&1 || { log "$name stage1 FAILED"; exit 1; }
S1=$(ls $CK/$name-stage1/d-epoch=*.ckpt | sort -t= -k3 -g | head -1)
log "$name stage1 done $S1"
$PY $S/make_ckpt_A.py $S1 $R/lik_table_sh5.json $CK/$name-stage1/A.ckpt > $J/make_A_$name.log 2>&1 || { log "$name make A FAILED"; exit 1; }
CUDA_VISIBLE_DEVICES=$gpu $PY $S/het_real_check.py $CK/$name-stage1/A.ckpt IDX-INBRED__B73 IDX-INBRED__Oh43 IDX-HYB__B73xOh43 OUT-INBRED__Ia453 OUT-HYB__EP1xIa453 > $R/het_real_check_$name.txt 2>$J/het_real_check_$name.err; log "$name het real check exit=$?"
CUDA_VISIBLE_DEVICES=$gpu $PY src/python/crf/train_diploid_indel.py $RECIPE --run-name $name-stage2B --supervised-heads stage2 --max-epochs 1 --limit-train-batches 0.25 --warmup-steps 100 --lr 1e-2 --warm-start-ckpt $CK/$name-stage1/A.ckpt > $J/stage2B_$name.log 2>&1 && log "$name stage2 B done" || log "$name stage2 B FAILED"
cd $WT/experiments/simval-corpus/scripts
for t in A B; do
  c=$CK/$name-stage1/A.ckpt; [ $t = B ] && c=$CK/$name-stage2B/last.ckpt
  CUDA_VISIBLE_DEVICES=$gpu $PY fast_eval_ckpt.py --tag ${name}_$t --ckpt $c --scope idx out mix --route --input-glob scratch/lift_s200_local20k > $J/eval_${name}_$t.log 2>&1; log "${name}_$t eval exit=$?"
  CUDA_VISIBLE_DEVICES=$gpu $PY fast_eval_ckpt.py --tag ${name}_${t}_1x --ckpt $c --scope idx out mix --route --depth 1.0 --input-glob scratch/lift_s200_local20k > $J/eval_${name}_${t}_1x.log 2>&1; log "${name}_${t}_1x eval exit=$?"
done
log "$name DONE"
