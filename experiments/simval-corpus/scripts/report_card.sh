#!/bin/bash
# Report card for one model: run eval_suite.py on its 0.1x and 1x results (BEDs + result JSONs already
# written by fast_eval_ckpt.py / fast_eval_diploid_affinity.py), including the gene-window section, then
# rebuild results/eval_suite/dashboard.html (primary view) and report.html (detailed tables).
# Called at the end of every train / eval pipeline.
#   report_card.sh MODEL TAG_0.1x TAG_1x ["note"]
# Either tag may be "-" to skip that depth.
# Env: GENEWINDOW=0 skips the gene-window section (~5-15 min, <= GW_THREADS threads, default 8);
#      SUITE_WORKERS (default 4) sets the core section's processes.
set -e
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WT=$(cd "$D/../../.." && pwd)
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src
tags=()
[ "$2" != "-" ] && tags+=("0.1=$2")
[ "$3" != "-" ] && tags+=("1.0=$3")
sections=(core)
[ "${GENEWINDOW:-1}" != "0" ] && sections+=(genewindow)
cd "$D"
$E/bin/python -W ignore eval_suite.py run --model "$1" --tags "${tags[@]}" ${4:+--note "$4"} \
  --sections "${sections[@]}" --workers "${SUITE_WORKERS:-4}" --gw-threads "${GW_THREADS:-8}"
$E/bin/python -W ignore eval_suite.py dashboard
