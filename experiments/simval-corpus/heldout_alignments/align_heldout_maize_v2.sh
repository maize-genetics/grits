#!/usr/bin/env bash
# Realign the 5 held-out (out-of-index) assemblies in
# data/maize_v2_heldout/asm/ against the new, chr+organelle-trimmed B73 via
# PHGv2's align-assemblies (AnchorWave) -- same reference, same anchors, same
# parameters as scripts/nam_alignments/realign_maize_v2.sh used for the 24
# in-panel founders, so the held-out gVCFs are directly comparable.
#
# Unlike the founder script, this one does NOT run ref-prep itself -- it
# hard-requires the founder run's ref-prep artifacts
# (data/maize_v2_rebuild/ref/{ref.cds.fasta,B73.sam,B73_v2.gff3}) to already
# exist, and fails loudly if they don't. Reusing those exact anchors (rather
# than regenerating slightly-different ones) is the whole point -- every
# sample against the same GFF/CDS/SAM means only the query side varies.
#
# See /home/zrm22/.claude/plans/ok-now-that-we-warm-minsky.md.
#
# Resumable: skips any sample whose <align>/<sample>.maf already exists and
# is non-empty. align-assemblies swallows proali failures and still exits
# 0 -- so success is judged here by MAF existence/size, not phg's exit code.
# MAFs are written incrementally during proali, so delete any partial .maf
# before re-running after a kill/crash.
#
# Usage:
#   scripts/heldout_alignments/align_heldout_maize_v2.sh [PARALLEL] [THREADS_PER_RUN]
#
# Env overrides: PHG_BIN, JAVA_HOME, CONDA_ENV_PREFIX_ANCHORWAVE,
#                REF_FASTA, REF_DIR, ASM_DIR, OUT_DIR

set -uo pipefail

PARALLEL="${1:-5}"
THREADS_PER_RUN="${2:-10}"

PHG_BIN="${PHG_BIN:-/local/workdir/zrm22/HackathonJun2026/DebugSim/phg/bin/phg}"
JAVA_HOME="${JAVA_HOME:-/programs/jdk-21.0.1}"
export PATH="$JAVA_HOME/bin:$PATH"

CONDA_ENV_PREFIX_ANCHORWAVE="${CONDA_ENV_PREFIX_ANCHORWAVE:-/home/zrm22/mambaforge/envs/phgv2-conda}"

REF_FASTA="${REF_FASTA:-/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/B73.fa}"
REF_DIR="${REF_DIR:-/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_v2_rebuild/ref}"
REF_GFF="$REF_DIR/B73_v2.gff3"
REF_CDS_FASTA="$REF_DIR/ref.cds.fasta"
REF_SAM="$REF_DIR/B73.sam"

OUT_DIR="${OUT_DIR:-/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_v2_heldout}"
ASM_DIR="${ASM_DIR:-$OUT_DIR/asm}"
ALIGN_DIR="$OUT_DIR/align"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$ALIGN_DIR" "$LOG_DIR"

SAMPLES=(Tx303 A188 EP1 CML459 Ia453)

for f in "$REF_GFF" "$REF_CDS_FASTA" "$REF_SAM"; do
  if [ ! -s "$f" ]; then
    echo "[align_heldout_maize_v2] FAILED: required founder ref-prep artifact missing/empty: $f" >&2
    echo "  (this script deliberately does not regenerate ref-prep -- run scripts/nam_alignments/realign_maize_v2.sh first, or copy these 3 files into place)" >&2
    exit 1
  fi
done

align_one() {
  local sample="$1"
  local asm="$ASM_DIR/${sample}.fa"
  local maf="$ALIGN_DIR/${sample}.maf"

  if [ -s "$maf" ]; then
    echo "[align_heldout_maize_v2] $sample already aligned ($maf exists), skipping"
    return 0
  fi
  if [ ! -s "$asm" ]; then
    echo "[align_heldout_maize_v2] SKIP $sample: no assembly FASTA at $asm (run make_heldout_symlinks.sh first)" >&2
    return 1
  fi

  echo "[align_heldout_maize_v2] aligning $sample"
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
    echo "[align_heldout_maize_v2] FAILED: $sample -- expected MAF not found: $maf (see $LOG_DIR/align_${sample}.log and $ALIGN_DIR/proali_${sample}_outputAndError.log)" >&2
    return 1
  fi
  echo "[align_heldout_maize_v2] $sample done in $((t1 - t0))s -> $maf ($(du -h "$maf" | cut -f1))"
}
export -f align_one
export PHG_BIN REF_GFF REF_FASTA REF_CDS_FASTA REF_SAM ASM_DIR ALIGN_DIR LOG_DIR THREADS_PER_RUN CONDA_ENV_PREFIX_ANCHORWAVE

printf '%s\n' "${SAMPLES[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'align_one "$@"' _ {}

echo "[align_heldout_maize_v2] batch complete. Verify with:"
echo "  ls -la $ALIGN_DIR/*.maf | wc -l   # expect 5"
