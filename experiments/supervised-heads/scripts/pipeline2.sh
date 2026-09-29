#!/bin/bash
# supervised heads v2 (window crossover head): lik table || stage1 (GPU0) -> pool -> A, A_place
#  GPU1 queue: hq A, evals A, hq A_place, evals A_place, [wait B_place] evals B_place
#  GPU0 queue: fit B (from A), fit B_place (from A_place), hq B, hq B_place, evals B
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/supervised-heads-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python
export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src
D=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline
J=/home/zrm22/.claude/jobs/sup_heads
S=$WT/experiments/supervised-heads/scripts
R=$WT/experiments/supervised-heads/results; mkdir -p $R
DATA=$D/data/training/maize_v3prov_multidepth_fullscale_sliced.npy
AFF=$D/data/training/maize_v3prov_multidepth.affpred_sh2.npy
TERN=$D/checkpoints/diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt
CK=$D/checkpoints
log() { echo "$(date +%H:%M) $*" >> $J/done.txt; }
cd $WT
$PY $S/lik_table.py --data $DATA --out $R/lik_table_A.json > $J/lik_table.log 2>&1 &
LT=$!
RECIPE="--data $DATA --workdir $D --num-parents 25 --time-local-emis --warmup-steps 500 --homo-penalty 3.0 --spike-skip --founder-affinity --windows-per-individual 117 --batch-size 64 --precision bf16-mixed --tie-aware-loss --emission likelihood_dist --pair-emission mixture"
CUDA_VISIBLE_DEVICES=0 $PY src/python/crf/train_diploid_indel.py $RECIPE --run-name sh2-stage1 --supervised-heads stage1 --max-epochs 3 --warm-start-ckpt $TERN > $J/stage1.log 2>&1 || { log "stage1 FAILED"; exit 1; }
S1=$(ls $CK/sh2-stage1/d-epoch=*.ckpt | sort -t= -k3 -g | head -1)
log "stage1 done best(min val_loss) $S1"
CUDA_VISIBLE_DEVICES=0 $PY $S/pool_affinity.py $S1 $DATA $AFF > $J/pool.log 2>&1 || { log "pool FAILED"; exit 1; }
wait $LT || { log "lik table FAILED"; exit 1; }
$PY $S/make_ckpt_A.py $S1 $R/lik_table_A.json $CK/sh2-stage1/A.ckpt > $J/make_A.log 2>&1 || { log "make A FAILED"; exit 1; }
$PY $S/make_ckpt_A.py $S1 $R/lik_table_A.json $CK/sh2-stage1/A_place.ckpt --placement >> $J/make_A.log 2>&1 || { log "make A_place FAILED"; exit 1; }
log "A ckpts built"
evals() {   # GPU CKPT TAG
  cd $WT/experiments/simval-corpus/scripts
  CUDA_VISIBLE_DEVICES=$1 $PY fast_eval_ckpt.py --tag $3 --ckpt $2 --scope idx out mix --route --input-glob scratch/lift_s200_local20k > $J/eval_$3.log 2>&1; log "$3 eval exit=$?"
  CUDA_VISIBLE_DEVICES=$1 $PY fast_eval_ckpt.py --tag ${3}_1x --ckpt $2 --scope idx out mix --route --depth 1.0 --input-glob scratch/lift_s200_local20k > $J/eval_${3}_1x.log 2>&1; log "${3}_1x eval exit=$?"
  cd $WT
}
hq() {      # GPU CKPT NAME
  CUDA_VISIBLE_DEVICES=$1 $PY $S/head_quality.py --ckpt $2 --data $DATA --decode --out $R/head_quality_$3.json > $J/hq_$3.log 2>&1; log "hq $3 exit=$?"
}
( hq 1 $CK/sh2-stage1/A.ckpt A; evals 1 $CK/sh2-stage1/A.ckpt sh2_A
  hq 1 $CK/sh2-stage1/A_place.ckpt A_place; evals 1 $CK/sh2-stage1/A_place.ckpt sh2_Aplace
  until grep -q "stage2 B_place done" $J/done.txt; do sleep 60; done
  evals 1 $CK/sh2-stage2B_place/last.ckpt sh2_Bplace ) &
fit() {     # FROM NAME [extra]
  CUDA_VISIBLE_DEVICES=0 $PY src/python/crf/train_diploid_indel.py $RECIPE --run-name sh2-stage2$2 --supervised-heads stage2 --max-epochs 1 --limit-train-batches 0.25 --warmup-steps 100 --lr 1e-2 --warm-start-ckpt $1 --aff-pred $AFF "${@:3}" > $J/stage2$2.log 2>&1 && log "stage2 $2 done" || log "stage2 $2 FAILED"
}
fit $CK/sh2-stage1/A.ckpt B
fit $CK/sh2-stage1/A_place.ckpt B_place --xo-placement
hq 0 $CK/sh2-stage2B/last.ckpt B
hq 0 $CK/sh2-stage2B_place/last.ckpt B_place
evals 0 $CK/sh2-stage2B/last.ckpt sh2_B
wait
log "PIPELINE DONE"
