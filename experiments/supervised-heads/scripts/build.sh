#!/bin/bash
cd /local/workdir/zrm22/HackathonJun2026/grits_workdir/supervised-heads-wt/experiments/het-replacement/scripts
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
D=/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/data/training
J=/home/zrm22/.claude/jobs/sup_heads
export LD_LIBRARY_PATH=$E/lib
$E/bin/python gen_training_data.py --params ../results/repl_params_v3.json --emit-row-provenance --out-dir $D --prefix maize_v3prov_d0.1 > $J/gen_d0.1.log 2>&1 &
$E/bin/python gen_training_data.py --params ../results/repl_params_v3.json --emit-row-provenance --depth-factor 5 --seeds 703 704 --out-dir $D --prefix maize_v3prov_d0.5 > $J/gen_d0.5.log 2>&1 &
$E/bin/python gen_training_data.py --params ../results/repl_params_v3.json --emit-row-provenance --depth-factor 10 --seeds 705 706 --out-dir $D --prefix maize_v3prov_d1 > $J/gen_d1.log 2>&1 &
$E/bin/python gen_training_data.py --params ../results/repl_params_v3.json --emit-row-provenance --depth-factor 20 --seeds 707 708 --out-dir $D --prefix maize_v3prov_d2 > $J/gen_d2.log 2>&1 &
wait
echo "builds done" >> $J/done.txt
