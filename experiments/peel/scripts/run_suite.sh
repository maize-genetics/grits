#!/bin/bash
# run_suite.sh MODEL TAGPREFIX NOTE : eval_suite core + gene-window v2 for composite tags TAGPREFIX_{01,1x}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)  # this checkout
W=${GRITS_WORKDIR:?set GRITS_WORKDIR to the GRITS working directory}; S=$W/scratch/peel
E=${GRITS_PYTHON_ENV:?set GRITS_PYTHON_ENV to the python env prefix (bin/python, lib)}
export LD_LIBRARY_PATH=$E/lib PYTHONPATH=$REPO/src
cd $REPO/experiments/simval-corpus/scripts
$E/bin/python -W ignore eval_suite.py run --model $1 --tags 0.1=$2_01 1.0=$2_1x --note "$3" --sections core genewindow \
  --workers ${WK:-12} --gw-threads 16 --gw-jobs 2 > $S/logs/suite_$1.log 2>&1
echo "$(date +%m-%d_%H:%M) suite $1 rc=$?" >> $S/status.txt
