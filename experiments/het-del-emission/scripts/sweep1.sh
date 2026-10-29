#!/bin/bash
E=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1
cd "$(dirname "$0")"
J=/home/zrm22/.claude/jobs/hetdel
run() { # depth suffix
  $E/bin/python hetdel_eval.py --corpus calib --scope idx out mix --datasets IDX-HYB OUT-HYB MIX-HYB --depth $1 \
    --config hdc_base$2: hdc_r05$2:del_rho=0.5 hdc_r03$2:del_rho=0.3 hdc_r01$2:del_rho=0.1 hdc_r003$2:del_rho=0.03 \
             hdc_r03w0$2:del_rho=0.3,del_win=0 hdc_r03w64$2:del_rho=0.3,del_win=64 > $J/sweep1_$1.log 2>&1
  echo "$(date +%m-%d_%H:%M) sweep1 $1 done" >> $J/done.txt
}
run 0.1 "" &
run 1.0 _1x &
wait
