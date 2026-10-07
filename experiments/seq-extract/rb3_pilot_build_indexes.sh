#!/bin/bash
# Per-truth-line ropebwt3 index (FMD + ssa -s8 + len.gz), timed. Contig names are already chr1..10 (+ scaffolds).
set -euo pipefail
B=/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/seq_extract_ropebwt/ropebwt3_v3
cd /local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/gene_window_eval/rb3_pilot
for spec in Oh43:/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/Oh43.fa Tx303:/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_v2_heldout/asm/Tx303.fa; do
  L=${spec%%:*}; FA=${spec#*:}
  /usr/bin/time -v -o time_build_$L.txt $B build -t 3 -d -o $L.fmd $FA 2> build_$L.log
  /usr/bin/time -v -o time_ssa_$L.txt $B ssa -t 3 -s8 -o $L.fmd.ssa $L.fmd 2> ssa_$L.log
  cut -f1,2 $FA.fai | gzip > $L.fmd.len.gz
done
echo DONE
