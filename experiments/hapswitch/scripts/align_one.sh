#!/bin/bash
# align_one.sh SRC OUT : canonical (emE) alignment of one calibration row, skip if done
set -u
SRC=$1; OUT=$2; W=/local/workdir/zrm22/HackathonJun2026/grits_workdir
[ -s $OUT/windowed_k25native_wcount.npy ] && exit 0
mkdir -p $OUT
$W/scripts/align_promoted.sh $(readlink -f $SRC/reads.fastq) $OUT ${T:-8} $SRC/labels.bed > $OUT/align.out 2>&1 || exit 1
ln -sfn $(readlink -f $SRC/reads.fastq) $OUT/reads.fastq
rm -f $OUT/raw.tsv
echo "$(date +%m-%d_%H:%M) aligned $(basename $OUT)" >> $W/scratch/hapswitch/status.txt
