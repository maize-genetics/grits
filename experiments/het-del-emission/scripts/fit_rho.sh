#!/bin/bash
# Stage-2 refit on SIMULATED data (v6cfullfe recipe, frozen encoder + frozen read table) with the
# presence-aware pair emission: fits log_del_rho together with the other CRF scalars.
WT=$(cd "$(dirname "$0")/../../.." && pwd)  # this checkout
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python; export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src OMP_NUM_THREADS=8
D=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline; CK=$D/checkpoints
J=/home/zrm22/.claude/jobs/hetdel
DATA=$D/data/training/maize_v6hoc_multidepth_fullscale_sliced.npy
BASE="--data $DATA --workdir $D --num-parents 25 --time-local-emis --warmup-steps 100 --homo-penalty 3.0 --spike-skip --founder-affinity --windows-per-individual 117 --batch-size 64 --precision bf16-mixed --tie-aware-loss --pair-emission mixture --het-prior off --gate-target support --aff-source reads --emission likelihood --no-distance --extra-heads --no-null --freeze-emission"
cd $WT
CUDA_VISIBLE_DEVICES=0 $PY src/python/crf/train_diploid_indel.py $BASE --run-name v6cfullfe-presence-stage2B --supervised-heads stage2 \
  --del-rho-fit 0.5 --del-win 16 --max-epochs 1 --limit-train-batches 0.25 --lr 1e-2 \
  --warm-start-ckpt $CK/v6cfullfe-stage2B/last.ckpt > $J/fit_rho.log 2>&1
$PY - $CK/v6cfullfe-presence-stage2B/last.ckpt >> $J/done.txt <<'PYEOF'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
print("  fitted:", {k: round(float(sd[k].exp() if k == "log_del_rho" else sd[k]), 4) for k in ("log_del_rho", "log_xo_scale", "aff_w", "dos_w", "ood_w", "aff_ood_s") if k in sd})
PYEOF
echo "$(date +%m-%d_%H:%M) fit_rho done" >> $J/done.txt
