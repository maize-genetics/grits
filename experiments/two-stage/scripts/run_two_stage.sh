#!/bin/bash
# usage: run.sh GPU MODE   (MODE = twostage | auxw)
G=$1; MODE=$2
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/two-stage-wt
cd $WT
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python
D=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline
J=/home/zrm22/.claude/jobs/two_stage
TERN=$D/checkpoints/diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt
RECIPE="--data $D/data/training/maize_v3_hetrepl_v3_lineage_fullscale_sliced.npy --workdir $D --num-parents 25 --time-local-emis --warmup-steps 500 --homo-penalty 3.0 --spike-skip --founder-affinity --windows-per-individual 117 --batch-size 64 --precision bf16-mixed --tie-aware-loss"
run() { PYTHONPATH=src CUDA_VISIBLE_DEVICES=$G $PY src/python/crf/train_diploid_indel.py $RECIPE "$@"; }
if [ $MODE = twostage ]; then
  run --run-name ts-v3-stage1 --stage1-aux-only --max-epochs 1 --warm-start-ckpt $TERN > $J/stage1.log 2>&1
  echo "stage1 exit=$?" >> $J/done.txt
  run --run-name ts-v3-stage2 --max-epochs 4 --warm-start-ckpt $D/checkpoints/ts-v3-stage1/last.ckpt > $J/stage2.log 2>&1
  echo "stage2 exit=$?" >> $J/done.txt
  N=ts-v3-stage2; TAG=ts_v3_twostage
else
  run --run-name ts-v3-auxw1 --max-epochs 5 --aux-loss-weight 1.0 --warm-start-ckpt $TERN > $J/auxw1.log 2>&1
  echo "auxw1 exit=$?" >> $J/done.txt
  N=ts-v3-auxw1; TAG=ts_v3_auxw1
fi
C=$(ls $D/checkpoints/$N/d-epoch=*.ckpt | sort -t= -k3 -g | tail -1)
echo "$TAG best ckpt $C" >> $J/done.txt
cd experiments/simval-corpus/scripts
CUDA_VISIBLE_DEVICES=$G $PY fast_eval_ckpt.py --tag $TAG --ckpt $C --scope idx out mix --route --input-glob scratch/lift_s200_local20k > $J/eval_$TAG.log 2>&1
echo "$TAG eval exit=$?" >> $J/done.txt
cd $WT
PYTHONPATH=src CUDA_VISIBLE_DEVICES=$G $PY experiments/two-stage/scripts/val_error_anatomy.py --ckpt $C --data $D/data/training/maize_v3_hetrepl_v3_lineage_fullscale_sliced.npy --out experiments/two-stage/results/anatomy_${TAG}.json > $J/anatomy_$TAG.log 2>&1
echo "$TAG anatomy exit=$?" >> $J/done.txt
