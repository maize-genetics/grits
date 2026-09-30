#!/bin/bash
# v6 training data: every realism fix + held-out augment v2 + deletion-dosage bits.
#   per-read SNPs, per-read bad reads at the calibrated 1.23% true-founder-miss rate, dist structure,
#   more founder sharing (read_snps 5 / derived_sfs 0.15), 27 founders -> 25 with 2 hidden per
#   individual (30% held out), ancestral switching 16/individual (OUT churn calibration)
WT=/local/workdir/zrm22/HackathonJun2026/grits_workdir/sh-distfix-wt
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY=$E/bin/python; export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src
TD=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/data/training
J=/home/zrm22/.claude/jobs/v6; S=$WT/experiments/supervised-heads/scripts; H=$WT/experiments/het-replacement/scripts
log() { echo "$(date +%m-%d_%H:%M) $*" >> $J/done.txt; }
cd $H
G="$PY gen_training_data.py --params ../results/repl_params_v3.json --dist-structure ../results/calib_dist_structure_s200.json --per-read-snps --per-read-bad --bad-frac 0.031 --read-snps 5 --derived-sfs 0.15 --founders 27 --ancestor-crossovers 16 --emit-row-provenance --prov-deletion-bits --out-dir $TD"
$G --prefix maize_v6_d0.1 > $J/gen_d0.1.log 2>&1 &
$G --depth-factor 5 --seeds 703 704 --prefix maize_v6_d0.5 > $J/gen_d0.5.log 2>&1 &
$G --depth-factor 10 --seeds 705 706 --prefix maize_v6_d1 > $J/gen_d1.log 2>&1 &
$G --depth-factor 20 --seeds 707 708 --prefix maize_v6_d2 > $J/gen_d2.log 2>&1 &
wait
for d in 0.1 0.5 1 2; do [ -f $TD/maize_v6_d${d}_fullscale_sliced.lin.npy ] || { log "build d$d FAILED"; exit 1; }; done
log "builds done"
i=0
for d in 0.1 0.5 1 2; do
  $PY heldout_augment_v2.py --in-prefix $TD/maize_v6_d$d --out-prefix $TD/maize_v6ho_d$d --frac 0.3 --seed $i > $J/ho_d$d.log 2>&1 || { log "heldout d$d FAILED"; exit 1; }
  i=$((i+1))
done
log "heldout ok: $(tail -n 1 $J/ho_d0.1.log)"
$PY $S/concat_multidepth.py $TD maize_v6ho > $J/concat.log 2>&1 || { log "concat FAILED"; exit 1; }
log "concat ok"
cd $WT
$PY $S/lik_table.py --data $TD/maize_v6ho_multidepth_fullscale_sliced.npy --rows all --out $WT/experiments/supervised-heads/results/lik_table_v6.json > $J/lik_table.log 2>&1 || { log "lik table FAILED"; exit 1; }
log "BUILD DONE"
