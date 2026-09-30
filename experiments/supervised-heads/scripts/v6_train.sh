#!/bin/bash
# v6: waits for v6_build.sh, then two stage-1 runs in parallel on the same data (sh3 recipe):
#   v6      : ternary only (--no-distance), affinity prior from the read match rate (no head), het
#             prior off, support gate, emission table from all rows, + dosage/out-of-panel heads
#   v6base  : the same without the two new heads (isolates their effect)
# Each: A (lik_table --rows all), real het/heads check, head quality, stage-2 B, 0.1x + 1x scoring.
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/sh-distfix-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python; export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src
D=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline; CK=$D/checkpoints
J=/home/zrm22/.claude/jobs/v6; S=$WT/experiments/supervised-heads/scripts; R=$WT/experiments/supervised-heads/results
DATA=$D/data/training/maize_v6ho_multidepth_fullscale_sliced.npy
LIK=$R/lik_table_v6.json
TERN=$CK/diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt
log() { echo "$(date +%m-%d_%H:%M) $*" >> $J/done.txt; }
until grep -q "BUILD DONE" $J/done.txt 2>/dev/null; do grep -q "FAILED" $J/done.txt 2>/dev/null && exit 1; sleep 60; done
BASE="--data $DATA --workdir $D --num-parents 25 --time-local-emis --warmup-steps 500 --homo-penalty 3.0 --spike-skip --founder-affinity --windows-per-individual 117 --batch-size 64 --precision bf16-mixed --tie-aware-loss --pair-emission mixture --het-prior off --gate-target support --aff-source reads --emission likelihood --no-distance"
evals() {   # GPU CKPT TAG
  cd $WT/experiments/simval-corpus/scripts
  for dep in 0.1 1.0; do
    t=$3; [ $dep = 1.0 ] && t=${3}_1x
    CUDA_VISIBLE_DEVICES=$1 $PY fast_eval_ckpt.py --tag $t --ckpt $2 --scope idx out mix --route --depth $dep --input-glob scratch/lift_s200_local20k > $J/eval_$t.log 2>&1; log "$t eval exit=$?"
  done
  cd $WT
}
run() {     # GPU NAME EXTRA...
  local gpu=$1 name=$2; shift 2
  local RECIPE="$BASE $*"
  cd $WT
  CUDA_VISIBLE_DEVICES=$gpu $PY src/python/crf/train_diploid_indel.py $RECIPE --run-name $name-stage1 --supervised-heads stage1 --max-epochs 15 --patience 3 --warm-start-ckpt $TERN > $J/stage1_$name.log 2>&1 || { log "$name stage1 FAILED"; return 1; }
  local S1=$(ls $CK/$name-stage1/d-epoch=*.ckpt | sort -t= -k3 -g | head -1)
  log "$name stage1 done $S1"
  $PY $S/make_ckpt_A.py $S1 $LIK $CK/$name-stage1/A.ckpt > $J/make_A_$name.log 2>&1 || { log "$name make A FAILED"; return 1; }
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/het_real_check.py $CK/$name-stage1/A.ckpt IDX-INBRED__B73 IDX-INBRED__Oh43 IDX-INBRED__CML103 IDX-RIL2__B73xOh43 IDX-HYB__B73xOh43 IDX-HYB__B97xCML103 OUT-INBRED__Ia453 OUT-INBRED__Tx303 OUT-HYB__EP1xIa453 MIX-HYB__B73xTx303 > $R/het_real_check_$name.txt 2>$J/het_real_check_$name.err; log "$name real heads check exit=$?"
  CUDA_VISIBLE_DEVICES=$gpu $PY src/python/crf/train_diploid_indel.py $RECIPE --run-name $name-stage2B --supervised-heads stage2 --max-epochs 1 --limit-train-batches 0.25 --warmup-steps 100 --lr 1e-2 --warm-start-ckpt $CK/$name-stage1/A.ckpt > $J/stage2B_$name.log 2>&1 && log "$name stage2 B done" || log "$name stage2 B FAILED"
  $PY - $CK/$name-stage2B/last.ckpt >> $J/done.txt <<'PYEOF'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
print("  fitted B:", {k: round(float(sd[k]), 3) for k in ("log_xo_scale", "aff_w", "dos_w", "ood_w") if k in sd})
PYEOF
  evals $gpu $CK/$name-stage1/A.ckpt ${name}_A
  evals $gpu $CK/$name-stage2B/last.ckpt ${name}_B
}
run 1 v6 --extra-heads &
run 0 v6base &
wait
log "V6 PIPELINE DONE"
