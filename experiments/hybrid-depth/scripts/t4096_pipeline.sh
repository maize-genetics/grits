#!/bin/bash
# v6cfullfe recipe at T=4096 rows per window (v6cfullfe-t4096): re-slice v6ho into 14 x 4096-row windows per
# individual, closest-founder relabel (as v6hoc), lik table, stage 1 (warm start v3-k25), A + v6 read table
# (A_v6emis), stage-2 B with --no-null --freeze-emission. Batch 8 keeps rows/step and steps/epoch identical to
# the 512-row recipe (batch 64). Originals are never modified.
set -u
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/sh-distfix-wt
HD=/local/workdir/zrm22/HackathonJun2026/grits_workdir/hybrid-depth-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python; export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src OMP_NUM_THREADS=8
D=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline; CK=$D/checkpoints; TD=$D/data/training
J=/home/zrm22/.claude/jobs/t4096; mkdir -p $J
S=$WT/experiments/supervised-heads/scripts; R=$HD/experiments/hybrid-depth/results; mkdir -p $R
TERN=$CK/diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt
V6=$CK/v6-stage2B/last.ckpt
G=${GPU:-0}; name=v6cfull-t4096; fe=v6cfullfe-t4096
RS=$TD/maize_v6ho_t4096_multidepth; C=$TD/maize_v6hoc_t4096_multidepth; DATA=${C}_fullscale_sliced.npy
LIK=$R/lik_table_v6c_t4096.json
log() { echo "$(date +%m-%d_%H:%M) $*" >> $J/done.txt; }
BASE="--data $DATA --workdir $D --num-parents 25 --time-local-emis --warmup-steps 500 --homo-penalty 3.0 --spike-skip --founder-affinity --windows-per-individual 14 --batch-size 8 --precision bf16-mixed --tie-aware-loss --pair-emission mixture --het-prior off --gate-target support --aff-source reads --emission likelihood --no-distance --extra-heads"

if [ ! -f ${RS}_fullscale_sliced.ood.npy ]; then
  $PY $HD/experiments/hybrid-depth/scripts/reslice_windows.py $TD/maize_v6ho_multidepth $RS --G 117 --T2 4096 > $J/reslice.log 2>&1 \
    || { log "reslice FAILED"; exit 1; }
fi
log "reslice done"
if [ ! -f $DATA ]; then
  $PY $WT/experiments/het-replacement/scripts/heldout_closest_relabel.py --in-prefix $RS --out-prefix $C --chunk 250 > $J/relabel.log 2>&1 \
    || { log "relabel FAILED"; exit 1; }
fi
log "relabel done"
[ -f $LIK ] || $PY $S/lik_table.py --data $DATA --rows all --out $LIK > $J/lik_table.log 2>&1 || { log "lik table FAILED"; exit 1; }
log "lik table done"
cd $WT
CUDA_VISIBLE_DEVICES=$G $PY src/python/crf/train_diploid_indel.py $BASE --run-name $name-stage1 --supervised-heads stage1 \
  --max-epochs 15 --patience 3 --warm-start-ckpt $TERN > $J/stage1.log 2>&1 || { log "stage1 FAILED"; exit 1; }
S1=$(ls $CK/$name-stage1/d-epoch=*.ckpt | sort -t= -k3 -g | head -1)
log "stage1 done $S1"
$PY $S/make_ckpt_A.py $S1 $LIK $CK/$name-stage1/A.ckpt > $J/make_A.log 2>&1 || { log "make A FAILED"; exit 1; }
$PY - $CK/$name-stage1/A.ckpt $V6 $CK/$name-stage1/A_v6emis.ckpt <<'PYEOF' || { log "A_v6emis FAILED"; exit 1; }
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
v = torch.load(sys.argv[2], map_location="cpu", weights_only=False)["state_dict"]
for k in ("tern_loglik", "tern_dist_loglik"):
    a["state_dict"][k] = v[k].clone()
torch.save(a, sys.argv[3])
PYEOF
log "A_v6emis done"
CUDA_VISIBLE_DEVICES=$G $PY src/python/crf/train_diploid_indel.py $BASE --no-null --freeze-emission --run-name $fe-stage2B \
  --supervised-heads stage2 --max-epochs 1 --limit-train-batches 0.25 --warmup-steps 100 --lr 1e-2 \
  --warm-start-ckpt $CK/$name-stage1/A_v6emis.ckpt > $J/stage2B.log 2>&1
[ -f $CK/$fe-stage2B/last.ckpt ] || { log "stage2 B FAILED (no checkpoint)"; exit 1; }
log "stage2 B done $CK/$fe-stage2B/last.ckpt"
log "T4096 TRAINING DONE"
