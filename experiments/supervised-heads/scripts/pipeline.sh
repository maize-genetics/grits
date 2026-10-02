#!/bin/bash
# supervised heads: verify+concat -> lik table -> stage1 (GPU0) -> pool affinity -> (A) ckpt
#   -> [GPU1: head quality A + evals A 0.1x/1x]  [GPU0: stage2 (B) -> head quality B -> evals B 0.1x/1x]
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/supervised-heads-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python
export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src
D=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline
J=/home/zrm22/.claude/jobs/sup_heads
S=$WT/experiments/supervised-heads/scripts
R=$WT/experiments/supervised-heads/results; mkdir -p $R
DATA=$D/data/training/maize_v3prov_multidepth_fullscale_sliced.npy
TERN=$D/checkpoints/diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt
log() { echo "$(date +%H:%M) $*" >> $J/done.txt; }
cd $WT
$PY $S/concat_verify.py > $J/concat_verify.log 2>&1 || { log "concat/verify FAILED"; exit 1; }
log "concat+verify ok"
$PY $S/lik_table.py --data $DATA --out $R/lik_table_A.json > $J/lik_table.log 2>&1 &
LT=$!
RECIPE="--data $DATA --workdir $D --num-parents 25 --time-local-emis --warmup-steps 500 --homo-penalty 3.0 --spike-skip --founder-affinity --windows-per-individual 117 --batch-size 64 --precision bf16-mixed --tie-aware-loss --emission likelihood_dist --pair-emission mixture"
CUDA_VISIBLE_DEVICES=0 $PY src/python/crf/train_diploid_indel.py $RECIPE --run-name sh-stage1 --supervised-heads stage1 --max-epochs 3 --warm-start-ckpt $TERN > $J/stage1.log 2>&1 || { log "stage1 FAILED"; exit 1; }
S1=$(ls $D/checkpoints/sh-stage1/d-epoch=*.ckpt | sort -t= -k3 -g | head -1)
log "stage1 done best(min val_loss) $S1"
CUDA_VISIBLE_DEVICES=0 $PY $S/pool_affinity.py $S1 $DATA $D/data/training/maize_v3prov_multidepth.affpred_sh.npy > $J/pool.log 2>&1 || { log "pool FAILED"; exit 1; }
wait $LT || { log "lik table FAILED"; exit 1; }
$PY $S/make_ckpt_A.py $S1 $R/lik_table_A.json $D/checkpoints/sh-stage1/A.ckpt > $J/make_A.log 2>&1 || { log "make A FAILED"; exit 1; }
log "A ckpt built"
A=$D/checkpoints/sh-stage1/A.ckpt
evals() {   # GPU CKPT TAG
  cd $WT/experiments/simval-corpus/scripts
  CUDA_VISIBLE_DEVICES=$1 $PY fast_eval_ckpt.py --tag $3 --ckpt $2 --scope idx out mix --route --input-glob scratch/lift_s200_local20k > $J/eval_$3.log 2>&1; log "$3 eval exit=$?"
  CUDA_VISIBLE_DEVICES=$1 $PY fast_eval_ckpt.py --tag ${3}_1x --ckpt $2 --scope idx out mix --route --depth 1.0 --input-glob scratch/lift_s200_local20k > $J/eval_${3}_1x.log 2>&1; log "${3}_1x eval exit=$?"
  cd $WT
}
( CUDA_VISIBLE_DEVICES=1 $PY $S/head_quality.py --ckpt $A --data $DATA --decode --out $R/head_quality_A.json > $J/hq_A.log 2>&1; log "hq A exit=$?"
  evals 1 $A sh_A ) &
CUDA_VISIBLE_DEVICES=0 $PY src/python/crf/train_diploid_indel.py $RECIPE --run-name sh-stage2B --supervised-heads stage2 --max-epochs 1 --limit-train-batches 0.25 --warmup-steps 100 --lr 1e-2 --warm-start-ckpt $A --aff-pred $D/data/training/maize_v3prov_multidepth.affpred_sh.npy > $J/stage2B.log 2>&1 || { log "stage2B FAILED"; wait; exit 1; }
B=$D/checkpoints/sh-stage2B/last.ckpt
log "stage2B done"
CUDA_VISIBLE_DEVICES=0 $PY $S/head_quality.py --ckpt $B --data $DATA --decode --out $R/head_quality_B.json > $J/hq_B.log 2>&1; log "hq B exit=$?"
evals 0 $B sh_B
wait
log "PIPELINE DONE"
