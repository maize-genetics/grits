#!/bin/bash
# ropebwt3-sw pilot for the gene-window eval (truth lines Oh43, Tx303), timed per step into
# $P/resources.tsv. Usage: run_rb3_pilot.sh LINE THREADS   (indexes built by rb3_pilot/build.sh)
set -euo pipefail
GW=/local/workdir/zrm22/HackathonJun2026/grits_workdir
X=$GW/sh-distfix-wt/experiments/seq-extract
V2=$GW/scratch/gene_window_eval/v2
P=$GW/scratch/gene_window_eval/rb3_pilot
ENV=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY="env LD_LIBRARY_PATH=$ENV/lib PYTHONDONTWRITEBYTECODE=1 $ENV/bin/python -W ignore"
L=$1; T=${2:-3}
cd $P; mkdir -p time
RES=$P/resources.tsv
[ -s "$RES" ] || echo -e "step\tn_items\twall_s\tcpu_s\tmax_rss_GB\tthreads" > "$RES"
timed() {
  local step=$1 n=$2 th=$3; shift 3
  /usr/bin/time -v -o "time/$step.txt" "$@"
  awk -F': ' -v s="$step" -v n="$n" -v th="$th" '
    /Elapsed \(wall clock\)/ {k=split($2,a,":"); w=(k==3)?a[1]*3600+a[2]*60+a[3]:a[1]*60+a[2]}
    /User time/ {u=$2} /System time/ {y=$2} /Maximum resident/ {m=$2}
    END {printf "%s\t%s\t%.1f\t%.1f\t%.3f\t%s\n", s, n, w, u+y, m/1048576, th}' "time/$step.txt" >> "$RES"
}
for cfg in "def:" "b4:-A1 -B4 -O6 -E1"; do
  name=${cfg%%:*}; args=${cfg#*:}
  for set in model panel; do
    grep -q "^rb3sw_${name}_${set}_$L	" "$RES" && continue
    timed rb3sw_${name}_${set}_$L 0 $T $PY $X/gw_align_rb3.py --stitch $V2/stitch --out-dir align_$name --idx-dir $P \
      --lines $L --set $set --sw-args "$args" --threads $T
  done
done
