#!/bin/bash
E=${GRITS_PYTHON_ENV:?set GRITS_PYTHON_ENV to the python env prefix (bin/python, lib)}
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1
cd "$(dirname "$0")"
J=${GRITS_JOBDIR:-$GRITS_WORKDIR/scratch/jobs/hetdel}
run() { # depth suffix
  $E/bin/python hetdel_eval.py --corpus eval --scope idx out mix --datasets IDX-HYB OUT-HYB MIX-HYB --depth $1 \
    --config hdv_base$2: hdv_r01w0$2:del_rho=0.1,del_win=0 hdv_r03w0$2:del_rho=0.3,del_win=0 >> $J/val_$1.log 2>&1
  $E/bin/python hetdel_eval.py --corpus eval --scope idx out mix --datasets IDX-INBRED IDX-RIL2 OUT-INBRED OUT-RIL2 MIX-RIL2 --depth $1 \
    --config hdv_r01w0$2:del_rho=0.1,del_win=0 >> $J/val_$1.log 2>&1
  echo "$(date +%m-%d_%H:%M) val $1 done" >> $J/done.txt
}
run 0.1 "" &
run 1.0 _1x &
wait
