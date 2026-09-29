#!/bin/bash
# --dist-structure rebuild of the 4-depth supervised-heads training set, then the sh3 recipe
# unchanged: stage 1 -> (A) ckpt (+ het_w=0 variant) -> head checks -> real 0.1x/1x scoring,
# and the (B) stage-2 fit. Only the simulated anchor-distance feature differs from sh3.
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/sh-distfix-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python
export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src
D=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline
TD=$D/data/training
J=/home/zrm22/.claude/jobs/distfix; mkdir -p $J
S=$WT/experiments/supervised-heads/scripts
H=$WT/experiments/het-replacement
R=$WT/experiments/supervised-heads/results; mkdir -p $R
P=maize_v3dist
DATA=$TD/${P}_multidepth_fullscale_sliced.npy
AFF=$TD/${P}_multidepth.affpred_sh4.npy
TERN=$D/checkpoints/diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt
CK=$D/checkpoints
GS=${GPU_STAGE1:-1}; GE=${GPU_EVAL:-0}
log() { echo "$(date +%m-%d_%H:%M) $*" >> $J/done.txt; }

if [ ! -f $DATA ]; then
  cd $H/scripts
  G="$PY gen_training_data.py --params ../results/repl_params_v3.json --dist-structure ../results/calib_dist_structure_s200.json --emit-row-provenance --out-dir $TD"
  $G --prefix ${P}_d0.1 > $J/gen_d0.1.log 2>&1 &
  $G --depth-factor 5 --seeds 703 704 --prefix ${P}_d0.5 > $J/gen_d0.5.log 2>&1 &
  $G --depth-factor 10 --seeds 705 706 --prefix ${P}_d1 > $J/gen_d1.log 2>&1 &
  $G --depth-factor 20 --seeds 707 708 --prefix ${P}_d2 > $J/gen_d2.log 2>&1 &
  wait
  for d in 0.1 0.5 1 2; do [ -f $TD/${P}_d${d}_fullscale_sliced.lin.npy ] || { log "build d$d FAILED"; exit 1; }; done
  log "builds done"
  $PY $S/concat_multidepth.py $TD $P > $J/concat.log 2>&1 || { log "concat FAILED"; exit 1; }
  log "concat ok"
fi
cd $WT
$PY $S/lik_table.py --data $DATA --out $R/lik_table_sh4.json > $J/lik_table.log 2>&1 &
LT=$!
RECIPE="--data $DATA --workdir $D --num-parents 25 --time-local-emis --warmup-steps 500 --homo-penalty 3.0 --spike-skip --founder-affinity --windows-per-individual 117 --batch-size 64 --precision bf16-mixed --tie-aware-loss --emission likelihood_dist --pair-emission mixture"
CUDA_VISIBLE_DEVICES=$GS $PY src/python/crf/train_diploid_indel.py $RECIPE --run-name sh4-stage1 --supervised-heads stage1 --max-epochs 15 --patience 3 --warm-start-ckpt $TERN > $J/stage1.log 2>&1 || { log "stage1 FAILED"; exit 1; }
S1=$(ls $CK/sh4-stage1/d-epoch=*.ckpt | sort -t= -k3 -g | head -1)
log "stage1 done $S1"
wait $LT || { log "lik table FAILED"; exit 1; }
$PY $S/make_ckpt_A.py $S1 $R/lik_table_sh4.json $CK/sh4-stage1/A.ckpt > $J/make_A.log 2>&1 || { log "make A FAILED"; exit 1; }
$PY - $CK/sh4-stage1/A.ckpt <<'PYEOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ck["state_dict"]["het_w"] = torch.zeros_like(ck["state_dict"]["het_w"])
torch.save(ck, sys.argv[1].replace(".ckpt", "_hw0.ckpt"))
PYEOF
log "A ckpts built"
CUDA_VISIBLE_DEVICES=$GS $PY $S/het_real_check.py $CK/sh4-stage1/A.ckpt IDX-INBRED__B73 IDX-INBRED__Oh43 IDX-INBRED__CML103 IDX-HYB__B73xOh43 OUT-INBRED__Ia453 > $R/het_real_check_sh4.txt 2>$J/het_real_check.err; log "het real check exit=$?"
CUDA_VISIBLE_DEVICES=$GS $PY $S/pool_affinity.py $S1 $DATA $AFF > $J/pool.log 2>&1 || log "pool FAILED"
evals() {   # GPU CKPT TAG [depths]
  cd $WT/experiments/simval-corpus/scripts
  for dep in ${4:-0.1}; do
    t=$3; [ "$dep" != "0.1" ] && t=${3}_${dep%.0}x
    CUDA_VISIBLE_DEVICES=$1 $PY fast_eval_ckpt.py --tag $t --ckpt $2 --scope idx out mix --route --depth $dep --input-glob scratch/lift_s200_local20k > $J/eval_$t.log 2>&1; log "$t eval exit=$?"
  done
  cd $WT
}
hq() { CUDA_VISIBLE_DEVICES=$1 $PY $S/head_quality.py --ckpt $2 --data $DATA --decode --out $R/head_quality_$3.json > $J/hq_$3.log 2>&1; log "hq $3 exit=$?"; }
( evals $GE $CK/sh4-stage1/A.ckpt sh4_A "0.1 1.0"; evals $GE $CK/sh4-stage1/A_hw0.ckpt sh4_A_hw0 ) &
( hq $GS $CK/sh4-stage1/A.ckpt sh4_A
  CUDA_VISIBLE_DEVICES=$GS $PY src/python/crf/train_diploid_indel.py $RECIPE --run-name sh4-stage2B --supervised-heads stage2 --max-epochs 1 --limit-train-batches 0.25 --warmup-steps 100 --lr 1e-2 --warm-start-ckpt $CK/sh4-stage1/A.ckpt --aff-pred $AFF > $J/stage2B.log 2>&1 && log "stage2 B done" || log "stage2 B FAILED"
  evals $GS $CK/sh4-stage2B/last.ckpt sh4_B ) &
wait
log "PIPELINE DONE"
