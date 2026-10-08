#!/bin/bash
# Report card for one model: run eval_suite.py on its 0.1x and 1x results (BEDs + result JSONs already
# written by fast_eval_ckpt.py / fast_eval_diploid_affinity.py), then rebuild the comparison page
# results/eval_suite/report.html. Called at the end of every train / eval pipeline.
#   report_card.sh MODEL TAG_0.1x TAG_1x ["note"]
# Either tag may be "-" to skip that depth.
set -e
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WT=$(cd "$D/../../.." && pwd)
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$WT/src
tags=()
[ "$2" != "-" ] && tags+=("0.1=$2")
[ "$3" != "-" ] && tags+=("1.0=$3")
cd "$D"
$E/bin/python eval_suite.py run --model "$1" --tags "${tags[@]}" ${4:+--note "$4"} --workers "${SUITE_WORKERS:-4}"
$E/bin/python eval_suite_report.py
