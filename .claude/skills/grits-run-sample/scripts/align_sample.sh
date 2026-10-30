#!/bin/bash
# align_sample.sh -- the promoted GRITS alignment: FASTQ -> founder-support matrix -> windowed model input.
#
#   align_sample.sh OUTDIR THREADS READS.fq[.gz] [MORE.fq[.gz] ...]
#
# All FASTQ files are read as one single-end stream (paired R1/R2 files are simply concatenated, which is
# how the promoted model was evaluated). Writes into OUTDIR:
#   raw.npy + raw.npy.{bins,gametes,layout}.tsv   per-bin founder support (refmap)
#   raw.ps4g, raw.log                             PS4G counts, refmap log
#   align_summary.txt                             reads in, and how many were placed (EXACT/PLACED) or dropped
#   windowed_k25native_wcount.npy                 model input, [windows, 512, 77] int8
#   align.done                                    marker; the script skips a finished OUTDIR
# The per-read table raw.tsv is deleted (it is large and not used downstream); set KEEP_TSV=1 to keep it.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "${GRITS_ENV:-$HERE/grits.env}"
REPO=$(cd "$HERE/../../../.." && pwd)
[ $# -ge 3 ] || { sed -n '2,13p' "$0"; exit 2; }
OUT=$1; T=$2; shift 2
mkdir -p "$OUT"
[ -f "$OUT/align.done" ] && { echo "already aligned: $OUT"; exit 0; }

# refmap reads one uncompressed file: concatenate the inputs into a temporary FASTQ unless there is just one
FQ="$OUT/reads.fastq"
if [ $# -eq 1 ] && [[ "$1" != *.gz ]]; then FQ=$1; else
  for f in "$@"; do case "$f" in *.gz) zcat "$f";; *) cat "$f";; esac; done > "$FQ"
fi

echo "[$(date +%H:%M:%S)] refmap ($T threads) -> $OUT"
"$GRITS_ROPEBWT3" refmap --ref-prefix=B73 --max-occ=-1 -l 19 --lift="$GRITS_LIFT" -t "$T" \
  --ps4g="$OUT/raw.ps4g" --npy="$OUT/raw.npy" --anchor-dist-npy --anchor-dist-thresh=2000 \
  --lift-local-win=20000 --lift-chain-nb=3 --lift-chain-interp \
  --placed-min-chr-agree=1 --exact-min-asm-agree=1 --exact-max-ref-hits=1 \
  --placed-max-spread=1000000 --min-core-len=100 \
  --err-noclip --err-max-events=1 \
  "$GRITS_INDEX" "$FQ" > "$OUT/raw.tsv" 2> "$OUT/raw.log"
awk -F'\t' '{n[$3]++; t++} END {printf "reads\t%d\n", t; for (s in n) printf "%s\t%d\t%.1f%%\n", s, n[s], 100*n[s]/t}' \
  "$OUT/raw.tsv" | sort > "$OUT/align_summary.txt"
[ "${KEEP_TSV:-0}" = 1 ] || rm -f "$OUT/raw.tsv"
[ "$FQ" = "$OUT/reads.fastq" ] && rm -f "$FQ"

echo "[$(date +%H:%M:%S)] windowing -> windowed_k25native_wcount.npy"
LD_LIBRARY_PATH=$GRITS_LD:${LD_LIBRARY_PATH:-} PYTHONPATH="$REPO/src:$REPO" \
  "$GRITS_PY" "$REPO/src/python/crf/ropebwt_npy_to_matrix.py" --npy="$OUT/raw.npy" \
  --bins="$OUT/raw.npy.bins.tsv" --gametes="$OUT/raw.npy.gametes.tsv" --num-parents 25 --window-size 512 \
  --anchor-dist-npy --emit-read-counts --out="$OUT/windowed_k25native_wcount.npy" > "$OUT/n2m.log" 2>&1
touch "$OUT/align.done"
echo "[$(date +%H:%M:%S)] done"
