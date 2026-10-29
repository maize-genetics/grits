#!/usr/bin/env bash
# MAF -> gVCF for the 24 realigned founders, via biokotlin-tools
# maf-to-gvcf-converter. Deliberately NO -f (fill-gaps) and NO
# --del-as-symbolic: project convention is that AnchorWave coverage gaps
# stay as gaps (never asserted to match reference), and symbolic <INS>/<DEL>
# encoding belongs inside merge-gvcfs, not upstream of it.
# See /home/zrm22/.claude/plans/nifty-napping-crystal.md.
#
# Resumable: skips any sample whose gvcf/<sample>.g.vcf.gz already exists
# and is non-empty.
#
# Usage:
#   scripts/nam_alignments/build_maize_v2_gvcfs.sh [PARALLEL]
#
# Env overrides: BIOKOTLIN_TOOLS_BIN, JAVA_HOME, REF_FASTA, ALIGN_DIR, GVCF_DIR

set -uo pipefail

PARALLEL="${1:-8}"

BIOKOTLIN_TOOLS_BIN="${BIOKOTLIN_TOOLS_BIN:-/local/workdir/zrm22/HackathonJun2026/biokotlin-tools/build/install/biokotlin-tools/bin/biokotlin-tools}"
JAVA_HOME="${JAVA_HOME:-/programs/jdk-21.0.1}"
CONDA_ENV_PREFIX_ANCHORWAVE="${CONDA_ENV_PREFIX_ANCHORWAVE:-/home/zrm22/mambaforge/envs/phgv2-conda}"
export PATH="$JAVA_HOME/bin:$CONDA_ENV_PREFIX_ANCHORWAVE/bin:$PATH"

REF_FASTA="${REF_FASTA:-/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/B73.fa}"

OUT_DIR="${OUT_DIR:-/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_v2_rebuild}"
ALIGN_DIR="${ALIGN_DIR:-$OUT_DIR/align}"
GVCF_DIR="${GVCF_DIR:-$OUT_DIR/gvcf}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
mkdir -p "$GVCF_DIR" "$LOG_DIR"

FOUNDERS=(B97 CML103 CML228 CML247 CML277 CML322 CML333 CML52 CML69 HP301
          Il14H Ki11 Ki3 Ky21 M162W M37W Mo18W Ms71 NC350 NC358 Oh43 Oh7B
          P39 Tzi8)

build_one() {
  local sample="$1"
  local maf="$ALIGN_DIR/${sample}.maf"
  local gvcf_plain="$GVCF_DIR/${sample}.g.vcf"
  local gvcf_out="${gvcf_plain}.gz"

  if [ -s "$gvcf_out" ]; then
    echo "[build_maize_v2_gvcfs] $sample already built, skipping"
    return 0
  fi
  if [ ! -s "$maf" ]; then
    echo "[build_maize_v2_gvcfs] SKIP $sample: no MAF at $maf" >&2
    return 1
  fi

  echo "[build_maize_v2_gvcfs] building $sample"
  local t0
  t0=$(date +%s)
  "$BIOKOTLIN_TOOLS_BIN" maf-to-gvcf-converter \
    --reference-file "$REF_FASTA" \
    --maf-file "$maf" \
    --output-file "$gvcf_plain" \
    --sample-name "$sample" \
    > "$LOG_DIR/gvcf_${sample}.log" 2>&1
    # deliberately NO -f, NO --del-as-symbolic -- see header comment
  local t1
  t1=$(date +%s)

  if [ ! -s "$gvcf_out" ]; then
    echo "[build_maize_v2_gvcfs] FAILED: $sample, expected $gvcf_out (see $LOG_DIR/gvcf_${sample}.log)" >&2
    return 1
  fi
  echo "[build_maize_v2_gvcfs] $sample done in $((t1 - t0))s -> $gvcf_out"
}
export -f build_one
export ALIGN_DIR GVCF_DIR LOG_DIR BIOKOTLIN_TOOLS_BIN REF_FASTA

printf '%s\n' "${FOUNDERS[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'build_one "$@"' _ {}

echo "[build_maize_v2_gvcfs] batch complete. Verify with:"
echo "  ls -la $GVCF_DIR/*.g.vcf.gz | wc -l   # expect 24 (25 once B73 added)"
