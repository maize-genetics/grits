#!/bin/bash
# run_suite.sh MODEL TAGPREFIX NOTE : eval_suite core + gene-window v2 for composite tags TAGPREFIX_{01,1x}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)  # this checkout
W=/local/workdir/zrm22/HackathonJun2026/grits_workdir; S=$W/scratch/hapswitch
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$REPO/src
cd $REPO/experiments/simval-corpus/scripts
$E/bin/python -W ignore eval_suite.py run --model $1 --tags 0.1=$2_01 1.0=$2_1x --note "$3" --sections core genewindow \
  --workers ${WK:-12} --gw-threads 16 --gw-jobs 2 > $S/logs/suite_$1.log 2>&1
echo "$(date +%m-%d_%H:%M) suite $1 rc=$?" >> $S/status.txt
