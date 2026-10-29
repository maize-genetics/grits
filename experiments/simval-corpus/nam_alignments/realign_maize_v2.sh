#!/usr/bin/env bash
# Realign the 24 NAM founders in index_asms/maize_v2/ against the new,
# chr+organelle-trimmed B73 via PHGv2's align-assemblies (AnchorWave).
# See /home/zrm22/.claude/plans/nifty-napping-crystal.md for full context.
#
# Ref prep (gff2seq + minimap2-vs-ref) is done ONCE, then reused across all
# 24 alignments via --reference-sam/--reference-cds-fasta -- this guarantees
# every founder shares identical anchors and avoids redoing the ref side of
# minimap2 24 times.
#
# Resumable: skips any founder whose <align>/<sample>.maf already exists and
# is non-empty, so re-running after a crash/interrupt continues where it
# left off. AnchorWave's own proali failures are swallowed by PHG (caught
# and logged, execution falls through to dot-plot generation) -- so success
# is judged here by MAF existence/size, not by phg's exit code.
#
# Usage:
#   scripts/nam_alignments/realign_maize_v2.sh [PARALLEL] [THREADS_PER_RUN]
#
# Env overrides: PHG_BIN, JAVA_HOME, CONDA_ENV_PREFIX_ANCHORWAVE,
#                REF_FASTA, REF_GFF, PANEL_DIR, OUT_DIR

set -uo pipefail

PARALLEL="${1:-4}"
THREADS_PER_RUN="${2:-8}"

PHG_BIN="${PHG_BIN:-/local/workdir/zrm22/HackathonJun2026/DebugSim/phg/bin/phg}"
JAVA_HOME="${JAVA_HOME:-/programs/jdk-21.0.1}"
export PATH="$JAVA_HOME/bin:$PATH"

CONDA_ENV_PREFIX_ANCHORWAVE="${CONDA_ENV_PREFIX_ANCHORWAVE:-/home/zrm22/mambaforge/envs/phgv2-conda}"

PANEL_DIR="${PANEL_DIR:-/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2}"
REF_FASTA="${REF_FASTA:-$PANEL_DIR/B73.fa}"

OUT_DIR="${OUT_DIR:-/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_v2_rebuild}"
REF_DIR="$OUT_DIR/ref"
ALIGN_DIR="$OUT_DIR/align"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$REF_DIR" "$ALIGN_DIR" "$LOG_DIR"

REF_GFF="${REF_GFF:-$REF_DIR/B73_v2.gff3}"
REF_CDS_FASTA="$REF_DIR/ref.cds.fasta"
REF_SAM="$REF_DIR/B73.sam"

FOUNDERS=(B97 CML103 CML228 CML247 CML277 CML322 CML333 CML52 CML69 HP301
          Il14H Ki11 Ki3 Ky21 M162W M37W Mo18W Ms71 NC350 NC358 Oh43 Oh7B
          P39 Tzi8)

if [ ! -s "$REF_GFF" ]; then
  echo "[realign_maize_v2] FAILED: reference GFF not found at $REF_GFF (run scripts/nam_alignments/prepare_ref_gff.sh first)" >&2
  exit 1
fi

if [ ! -s "$REF_CDS_FASTA" ] || [ ! -s "$REF_SAM" ]; then
  echo "[realign_maize_v2] running ref prep (--just-ref-prep)"
  "$PHG_BIN" align-assemblies \
    --just-ref-prep \
    --gff "$REF_GFF" \
    --reference-file "$REF_FASTA" \
    --assembly-file "$PANEL_DIR/B97.fa" \
    -o "$REF_DIR" \
    --total-threads "$THREADS_PER_RUN" \
    --conda-env-prefix "$CONDA_ENV_PREFIX_ANCHORWAVE" \
    > "$LOG_DIR/ref_prep.log" 2>&1
  if [ ! -s "$REF_CDS_FASTA" ] || [ ! -s "$REF_SAM" ]; then
    echo "[realign_maize_v2] FAILED: ref prep did not produce $REF_CDS_FASTA / $REF_SAM -- see $LOG_DIR/ref_prep.log" >&2
    exit 1
  fi
  echo "[realign_maize_v2] ref prep done: $REF_CDS_FASTA, $REF_SAM"
else
  echo "[realign_maize_v2] ref prep already present, skipping"
fi

align_one() {
  local sample="$1"
  local asm="$PANEL_DIR/${sample}.fa"
  local maf="$ALIGN_DIR/${sample}.maf"

  if [ -s "$maf" ]; then
    echo "[realign_maize_v2] $sample already aligned ($maf exists), skipping"
    return 0
  fi
  if [ ! -s "$asm" ]; then
    echo "[realign_maize_v2] SKIP $sample: no assembly FASTA at $asm" >&2
    return 1
  fi

  echo "[realign_maize_v2] aligning $sample"
  local t0
  t0=$(date +%s)
  "$PHG_BIN" align-assemblies \
    --gff "$REF_GFF" \
    --reference-file "$REF_FASTA" \
    --reference-cds-fasta "$REF_CDS_FASTA" \
    --reference-sam "$REF_SAM" \
    --assembly-file "$asm" \
    -o "$ALIGN_DIR" \
    --total-threads "$THREADS_PER_RUN" \
    --in-parallel 1 \
    --ref-max-align-cov 1 \
    --query-max-align-cov 1 \
    --conda-env-prefix "$CONDA_ENV_PREFIX_ANCHORWAVE" \
    > "$LOG_DIR/align_${sample}.log" 2>&1
  local t1
  t1=$(date +%s)

  if [ ! -s "$maf" ]; then
    echo "[realign_maize_v2] FAILED: $sample -- expected MAF not found: $maf (see $LOG_DIR/align_${sample}.log and $ALIGN_DIR/proali_${sample}_outputAndError.log)" >&2
    return 1
  fi
  echo "[realign_maize_v2] $sample done in $((t1 - t0))s -> $maf ($(du -h "$maf" | cut -f1))"
}
export -f align_one
export PHG_BIN REF_GFF REF_FASTA REF_CDS_FASTA REF_SAM PANEL_DIR ALIGN_DIR LOG_DIR THREADS_PER_RUN CONDA_ENV_PREFIX_ANCHORWAVE

printf '%s\n' "${FOUNDERS[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'align_one "$@"' _ {}

echo "[realign_maize_v2] batch complete. Verify with:"
echo "  ls -la $ALIGN_DIR/*.maf | wc -l   # expect 24"
