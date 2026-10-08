#!/bin/bash
# v6cnn: refit stage 2 (fit B), null founder dropped from the CRF (--no-null), on the frozen v6 stage-1 encoder, with held-out null labels replaced by
# the closest visible founder (heldout_closest_relabel.py), then score 0.1x + 1x as decoded and with
# the null founder banned (--knockout nonull).
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/sh-distfix-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python; export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src
D=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline; CK=$D/checkpoints
J=/home/zrm22/.claude/jobs/v6c; mkdir -p $J
DATA=$D/data/training/maize_v6hoc_multidepth_fullscale_sliced.npy
log() { echo "$(date +%m-%d_%H:%M) $*" >> $J/done.txt; }
BASE="--data $DATA --workdir $D --num-parents 25 --time-local-emis --warmup-steps 500 --homo-penalty 3.0 --spike-skip --founder-affinity --windows-per-individual 117 --batch-size 64 --precision bf16-mixed --tie-aware-loss --pair-emission mixture --het-prior off --gate-target support --aff-source reads --emission likelihood --no-distance --extra-heads --no-null"
cd $WT
CUDA_VISIBLE_DEVICES=${GPU:-0} $PY src/python/crf/train_diploid_indel.py $BASE --run-name v6cnn-stage2B --supervised-heads stage2 --max-epochs 1 --limit-train-batches 0.25 --warmup-steps 100 --lr 1e-2 --warm-start-ckpt $CK/v6-stage1/A.ckpt > $J/stage2B.log 2>&1 || { log "v6cnn stage2 B FAILED"; exit 1; }
log "v6cnn stage2 B done"
$PY - $CK/v6-stage2B/last.ckpt $CK/v6cnn-stage2B/last.ckpt >> $J/done.txt <<'PYEOF'
import sys, torch
for p in sys.argv[1:]:
    sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
    print("  ", p.split("/")[-2], {k: [round(v, 2) for v in sd[k].float().flatten().tolist()] if sd[k].numel() > 1 else round(float(sd[k]), 3)
                                   for k in sd if k in ("log_xo_scale", "aff_w", "dos_w", "ood_w", "aff_ood_s") or "loglik" in k})
PYEOF
cd $WT/experiments/simval-corpus/scripts
evals() {   # GPU TAG DEPTH KNOCKOUT
  for sc in idx out mix; do
    CUDA_VISIBLE_DEVICES=$1 $PY fast_eval_ckpt.py --tag $2 --ckpt $CK/v6cnn-stage2B/last.ckpt --scope $sc --route --depth $3 --input-glob scratch/lift_s200_local20k ${4:+--knockout $4} > $J/eval_$2_$sc.log 2>&1 &
  done; wait; log "$2 eval done"
}
# no_null is saved in the checkpoint, so plain decoding never calls the null founder
evals 0 v6cnn_B 0.1; evals 0 v6cnn_B_1x 1.0
bash $WT/experiments/simval-corpus/scripts/report_card.sh v6cnn_B v6cnn_B v6cnn_B_1x > $J/report_card_v6cnn_B.log 2>&1 && log "report card done" || log "report card FAILED"
log "V6CNN DONE"
