#!/bin/bash
# per-read-SNP + dist-structure rebuild, then two stage-1 runs in parallel with the sh3 recipe:
# sh5 (ternary + distance, likelihood_dist emission) and sh5nd (--no-distance, ternary only,
# likelihood emission). Het prior off in decoding. Each: A, real het check, head quality,
# stage-2 B, 0.1x + 1x scoring of A and B.
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/sh-distfix-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python
export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src
D=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline
TD=$D/data/training; CK=$D/checkpoints
J=/home/zrm22/.claude/jobs/sh5; mkdir -p $J
S=$WT/experiments/supervised-heads/scripts
H=$WT/experiments/het-replacement
R=$WT/experiments/supervised-heads/results
P=maize_v3read
DATA=$TD/${P}_multidepth_fullscale_sliced.npy
TERN=$D/checkpoints/diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt
log() { echo "$(date +%m-%d_%H:%M) $*" >> $J/done.txt; }

if [ ! -f $DATA ]; then
  cd $H/scripts
  G="$PY gen_training_data.py --params ../results/repl_params_v3.json --dist-structure ../results/calib_dist_structure_s200.json --per-read-snps --emit-row-provenance --out-dir $TD"
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
$PY $S/lik_table.py --data $DATA --out $R/lik_table_sh5.json > $J/lik_table.log 2>&1 || { log "lik table FAILED"; exit 1; }
BASE="--data $DATA --workdir $D --num-parents 25 --time-local-emis --warmup-steps 500 --homo-penalty 3.0 --spike-skip --founder-affinity --windows-per-individual 117 --batch-size 64 --precision bf16-mixed --tie-aware-loss --pair-emission mixture --het-prior off"

evals() {   # GPU CKPT TAG
  cd $WT/experiments/simval-corpus/scripts
  CUDA_VISIBLE_DEVICES=$1 $PY fast_eval_ckpt.py --tag $3 --ckpt $2 --scope idx out mix --route --input-glob scratch/lift_s200_local20k > $J/eval_$3.log 2>&1; log "$3 eval exit=$?"
  CUDA_VISIBLE_DEVICES=$1 $PY fast_eval_ckpt.py --tag ${3}_1x --ckpt $2 --scope idx out mix --route --depth 1.0 --input-glob scratch/lift_s200_local20k > $J/eval_${3}_1x.log 2>&1; log "${3}_1x eval exit=$?"
  cd $WT
}
run() {     # GPU NAME EXTRA...
  local gpu=$1 name=$2; shift 2
  local RECIPE="$BASE $*"
  local AFF=$TD/${P}_multidepth.affpred_$name.npy
  CUDA_VISIBLE_DEVICES=$gpu $PY src/python/crf/train_diploid_indel.py $RECIPE --run-name $name-stage1 --supervised-heads stage1 --max-epochs 15 --patience 3 --warm-start-ckpt $TERN > $J/stage1_$name.log 2>&1 || { log "$name stage1 FAILED"; return 1; }
  local S1=$(ls $CK/$name-stage1/d-epoch=*.ckpt | sort -t= -k3 -g | head -1)
  log "$name stage1 done $S1"
  $PY $S/make_ckpt_A.py $S1 $R/lik_table_sh5.json $CK/$name-stage1/A.ckpt > $J/make_A_$name.log 2>&1 || { log "$name make A FAILED"; return 1; }
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/het_real_check.py $CK/$name-stage1/A.ckpt IDX-INBRED__B73 IDX-INBRED__Oh43 IDX-INBRED__CML103 IDX-RIL2__B73xOh43 IDX-HYB__B73xOh43 IDX-HYB__B97xCML103 OUT-INBRED__Ia453 OUT-INBRED__Tx303 OUT-HYB__EP1xIa453 > $R/het_real_check_$name.txt 2>$J/het_real_check_$name.err; log "$name het real check exit=$?"
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/head_quality.py --ckpt $CK/$name-stage1/A.ckpt --data $DATA --decode --out $R/head_quality_$name.json > $J/hq_$name.log 2>&1; log "$name hq exit=$?"
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/pool_affinity.py $S1 $DATA $AFF > $J/pool_$name.log 2>&1 || log "$name pool FAILED"
  CUDA_VISIBLE_DEVICES=$gpu $PY src/python/crf/train_diploid_indel.py $RECIPE --run-name $name-stage2B --supervised-heads stage2 --max-epochs 1 --limit-train-batches 0.25 --warmup-steps 100 --lr 1e-2 --warm-start-ckpt $CK/$name-stage1/A.ckpt --aff-pred $AFF > $J/stage2B_$name.log 2>&1 && log "$name stage2 B done" || log "$name stage2 B FAILED"
  evals $gpu $CK/$name-stage1/A.ckpt ${name}_A
  evals $gpu $CK/$name-stage2B/last.ckpt ${name}_B
}
run 1 sh5 --emission likelihood_dist &
run 0 sh5nd --emission likelihood --no-distance &
wait
log "PIPELINE DONE"
