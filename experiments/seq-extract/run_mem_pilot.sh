#!/bin/bash
# MEM-chain pilot (ropebwt3_v5 mem -> chain -> gap DP) for Oh43/Tx303, min MEM length 31 (main), 19, 51,
# plus the minimap2 v2 rescore with the supplementary-block collinearity check. Timed into
# $P/resources_mem.tsv.
set -euo pipefail
GW=/local/workdir/zrm22/HackathonJun2026/grits_workdir
X=$GW/sh-distfix-wt/experiments/seq-extract
V2=$GW/scratch/gene_window_eval/v2
P=$GW/scratch/gene_window_eval/rb3_pilot
ENV=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY="env LD_LIBRARY_PATH=$ENV/lib PYTHONDONTWRITEBYTECODE=1 $ENV/bin/python -W ignore"
T=${T:-3}
cd $P; mkdir -p time
RES=$P/resources_mem.tsv
[ -s "$RES" ] || echo -e "step\tn_items\twall_s\tcpu_s\tmax_rss_GB\tthreads" > "$RES"
timed() {
  local step=$1 n=$2 th=$3; shift 3
  grep -q "^$step	" "$RES" && return 0
  /usr/bin/time -v -o "time/$step.txt" "$@"
  awk -F': ' -v s="$step" -v n="$n" -v th="$th" '
    /Elapsed \(wall clock\)/ {k=split($2,a,":"); w=(k==3)?a[1]*3600+a[2]*60+a[3]:a[1]*60+a[2]}
    /User time/ {u=$2} /System time/ {y=$2} /Maximum resident/ {m=$2}
    END {printf "%s\t%s\t%.1f\t%.1f\t%.3f\t%s\n", s, n, w, u+y, m/1048576, th}' "time/$step.txt" >> "$RES"
}
for l in 31 19 51; do
  for L in Oh43 Tx303; do
    for set in model panel; do
      timed mem_l${l}_${set}_$L 0 $T $PY $X/gw_align_mem.py --stitch $V2/stitch --out-dir align_mem_l$l --idx-dir $P \
        --lines $L --set $set --min-len $l --threads $T
    done
  done
done
# minimap2 v2 PAFs rescored with the collinearity check (all lines)
mkdir -p $V2/align_col
for f in $V2/align/*.paf; do ln -sf ../align/$(basename $f) $V2/align_col/; done
for set in model panel; do
  timed mm2_rescore_collinear_$set 0 1 $PY $X/gw_align.py --stitch $V2/stitch --out-dir $V2/align_col --set $set --rescore-only
done
