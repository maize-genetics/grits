#!/bin/bash
# Gene-window (gene centre +-5 kb) imputation-accuracy eval, end to end, as run on 2026-10-06.
# Steps: windows -> truth checks -> projection queries -> ropebwt3 --project-only -> plausibility filter
#        -> stitch (HaplotypeBuilder, gap policy N) -> extractor cross-check -> minimap2 to truth
#        -> score (truth assignment, panel ceiling) -> report.
# Threads: keep total <= 4 on the shared machine.
set -euo pipefail
GW=/local/workdir/zrm22/HackathonJun2026/grits_workdir
X=$GW/sh-distfix-wt/experiments/seq-extract
W=${W:-$GW/scratch/gene_window_eval}
RES=${RES:-$GW/results/gene_window_eval}
ENV=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default
PY="env LD_LIBRARY_PATH=$ENV/lib PYTHONDONTWRITEBYTECODE=1 $ENV/bin/python -W ignore"
RB3=$GW/scratch/seq_extract_ropebwt/ropebwt3
IDX=/workdir/zrm22/HackathonJun2026/ropebwt_refMap/rope_bwt_index_v2
MM2=$GW/scratch/tools/minimap2-2.28_x64-linux/minimap2
mkdir -p "$W" "$RES" "$W/mmi"
cd "$W"

$PY $X/gw_windows.py --out windows.tsv
$PY $X/gw_verify_truth.py truth_verify.tsv
$PY $X/gw_ril_truth.py windows.tsv ril_truth.tsv
[ -s maize_v2_b12.isa ] || $RB3 isa -o maize_v2_b12.isa $IDX/maizeFastaIndex_SampleContig_v2.fmd
for L in B73 B97 CML103 Il14H Oh43; do
  [ -s mmi/$L.asm20.mmi ] || $MM2 -x asm20 -t 3 -d mmi/$L.asm20.mmi /workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/$L.fa
done
for L in A188 CML459 EP1 Ia453 Tx303; do
  [ -s mmi/$L.asm20.mmi ] || $MM2 -x asm20 -t 3 -d mmi/$L.asm20.mmi $GW/data/maize_v2_heldout/asm/$L.fa
done

$PY $X/gw_build.py queries --windows windows.tsv --depths 0.1 1.0 --out queries.tsv
$RB3 extract --lift=$IDX/maizeFastaIndex_SampleContig_v2_s200.lift --ref-prefix=B73 --project-only \
  -I maize_v2_b12.isa -f queries.tsv $IDX/maizeFastaIndex_SampleContig_v2.fmd > proj.tsv
$PY $X/gw_filter_proj.py proj.tsv proj_filt.tsv --max-extra 50000
$PY $X/gw_build.py stitch --windows windows.tsv --depths 0.1 1.0 --table proj_filt.tsv --out-dir stitch
$PY $X/gw_crosscheck.py proj_filt.tsv crosscheck --isa maize_v2_b12.isa --threads 2
$PY $X/gw_switch_artifact.py --windows windows.tsv --out-dir switch_artifact
$PY $X/gw_align.py --stitch stitch --out-dir align --threads 3
$PY $X/gw_score.py --windows windows.tsv --stitch stitch --align align --ril-truth ril_truth.tsv --out results
$PY $X/gw_report.py results
