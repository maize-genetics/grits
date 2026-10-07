#!/bin/bash
# Gene-window eval, v2 (2026-10-07): ropebwt3_v3 with snapping ON and the built-in projected-span check
# (--max-span-excess 50000 --max-span-ratio 3, defaults); python.seq_extract SpanCheckProvider on the
# stitch side (expected no-op); trimmed-agreement metric. Every step is wrapped in /usr/bin/time -v and
# recorded in $W/resources.tsv (step, n_items, wall_s, cpu_s, max_rss_GB, threads).
# Reuses the minimap2 indexes, windows.tsv and ril_truth.tsv of the v1 run in $V1. <= 4 threads total.
set -euo pipefail
GW=/local/workdir/zrm22/HackathonJun2026/grits_workdir
X=$GW/sh-distfix-wt/experiments/seq-extract
V1=$GW/scratch/gene_window_eval
W=${W:-$V1/v2}
ENV=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY="env LD_LIBRARY_PATH=$ENV/lib PYTHONDONTWRITEBYTECODE=1 $ENV/bin/python -W ignore"
RB3=$GW/scratch/seq_extract_ropebwt/ropebwt3_v3
IDX=/workdir/zrm22/HackathonJun2026/ropebwt_refMap/rope_bwt_index_v2
FMD=$IDX/maizeFastaIndex_SampleContig_v2.fmd
LIFT=$IDX/maizeFastaIndex_SampleContig_v2_s200.lift
MM2=$GW/scratch/tools/minimap2-2.28_x64-linux/minimap2
T=3
LINES="B73 B97 CML103 Il14H Oh43 A188 CML459 EP1 Ia453 Tx303"
mkdir -p "$W/time"
cd "$W"
RES=$W/resources.tsv
[ -s "$RES" ] || echo -e "step\tn_items\twall_s\tcpu_s\tmax_rss_GB\tthreads" > "$RES"

timed() {   # timed STEP N_ITEMS THREADS cmd...
  local step=$1 n=$2 th=$3; shift 3
  /usr/bin/time -v -o "time/$step.txt" "$@"
  awk -F': ' -v s="$step" -v n="$n" -v th="$th" '
    /Elapsed \(wall clock\)/ {k=split($2,a,":"); w=(k==3)?a[1]*3600+a[2]*60+a[3]:a[1]*60+a[2]}
    /User time/ {u=$2} /System time/ {y=$2} /Maximum resident/ {m=$2}
    END {printf "%s\t%s\t%.1f\t%.1f\t%.3f\t%s\n", s, n, w, u+y, m/1048576, th}' "time/$step.txt" >> "$RES"
}
count() { tail -n +2 "$1" | wc -l; }

cp $V1/windows.tsv $V1/ril_truth.tsv .
timed isa_build 1 1 $RB3 isa -o maize_v2_b12.isa $FMD
timed queries 80 1 $PY $X/gw_build.py queries --windows windows.tsv --depths 0.1 1.0 --out queries.tsv
NQ=$(count queries.tsv)
timed project_snap $NQ $T sh -c "$RB3 extract --lift=$LIFT --ref-prefix=B73 --project-only -t $T -I maize_v2_b12.isa -f queries.tsv $FMD > proj.tsv 2> proj.log"
timed project_nosnap $NQ $T sh -c "$RB3 extract --lift=$LIFT --ref-prefix=B73 --project-only --no-snap -t $T -I maize_v2_b12.isa -f queries.tsv $FMD > proj_nosnap.tsv 2> proj_nosnap.log"
timed stitch_model 160000 1 $PY $X/gw_build.py stitch --part model --windows windows.tsv --depths 0.1 1.0 --table proj.tsv --out-dir stitch
timed stitch_panel_ceiling 25000 1 $PY $X/gw_build.py stitch --part panel --windows windows.tsv --table proj.tsv --out-dir stitch
timed crosscheck_rb3_vs_faidx $(awk -F'\t' 'NR>1 && ($10=="ok"||$10=="ok_clamped")' proj.tsv | wc -l) $T $PY $X/gw_crosscheck.py proj.tsv crosscheck --isa maize_v2_b12.isa --threads $T
timed switch_artifact 4000 $T $PY $X/gw_switch_artifact.py --windows windows.tsv --out-dir switch_artifact --threads $T
mkdir -p projtest && cd projtest
for s in snap nosnap; do
  a=""; [ $s = nosnap ] && a="--no-snap"
  $RB3 extract --lift=$LIFT --ref-prefix=B73 --project-only $a -t $T -I ../maize_v2_b12.isa -f $V1/projtest/q.tsv $FMD > p_v3$s.tsv 2>/dev/null
  ln -sf $V1/projtest/b73_to_*.paf .
done
$PY $X/gw_projection_accuracy.py ../windows.tsv p_v3snap.tsv p_v3nosnap.tsv $V1/projtest/p_lw20k.tsv > end_accuracy.txt
cd "$W"
timed mm2_index_B97 1 $T $MM2 -x asm20 -t $T -d mm2_index_test.mmi /workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/B97.fa
ls -l mm2_index_test.mmi > mm2_index_test.size; rm -f mm2_index_test.mmi
for L in $LINES; do
  timed align_model_$L 0 $T $PY $X/gw_align.py --stitch stitch --out-dir align --set model --lines $L --threads $T
  timed align_panel_ceiling_$L 0 $T $PY $X/gw_align.py --stitch stitch --out-dir align --set panel --lines $L --threads $T
done
timed score_incl_trimmed 160000 1 $PY $X/gw_score.py --windows windows.tsv --stitch stitch --align align --ril-truth ril_truth.tsv --out results
timed report 80 1 $PY $X/gw_report.py results
