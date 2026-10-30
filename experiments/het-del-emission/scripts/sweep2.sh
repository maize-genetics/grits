#!/bin/bash
E=${GRITS_PYTHON_ENV:?set GRITS_PYTHON_ENV to the python env prefix (bin/python, lib)}
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1
cd "$(dirname "$0")"
J=${GRITS_JOBDIR:-$GRITS_WORKDIR/scratch/jobs/hetdel}
run() { # depth suffix
  $E/bin/python hetdel_eval.py --corpus calib --scope idx out mix --datasets IDX-HYB OUT-HYB MIX-HYB --depth $1 \
    --config hdc_base$2: hdc_r03w0$2:del_rho=0.3,del_win=0 hdc_r01w0$2:del_rho=0.1,del_win=0 hdc_r05w0$2:del_rho=0.5,del_win=0 hdc_r03$2:del_rho=0.3 \
    >> $J/sweep2_$1.log 2>&1
  echo "$(date +%m-%d_%H:%M) sweep2 $1 done" >> $J/done.txt
}
run 0.1 "" &
run 1.0 _1x &
wait
