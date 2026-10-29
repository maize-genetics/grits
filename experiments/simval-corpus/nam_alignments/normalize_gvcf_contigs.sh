#!/usr/bin/env bash
# merge-gvcfs' internal ValidateVCFsUtils requires every input gVCF's
# ##contig header to declare the SAME set/order of contigs. AnchorWave-
# derived gVCFs only declare contigs that actually got an alignment record
# in that founder's own MAF -- and whether a founder's assembly happens to
# align to Pt/Mt is founder-specific (confirmed: Oh43 aligned Mt only, no
# Pt; other founders vary). The old reference had no organelles at all, so
# this divergence never showed up before.
#
# NOTE: `bcftools reheader -f <fai>` looked like the right tool but its
# internal ##contig reconstruction order is NOT deterministic across
# separate process invocations (confirmed empirically: two sequential
# manual test runs happened to match by coincidence, but an 8-way
# concurrent batch produced two different orderings). Instead we build ONE
# literal canonical contig block (fixed order, from the .fai) and splice it
# into each file's own header text (preserving that file's own #CHROM
# sample name), then apply via `bcftools reheader -h <header-file>` -- a
# pure text substitution, not reliant on any internal hash iteration.
# See /home/zrm22/.claude/plans/nifty-napping-crystal.md.
#
# Resumable: skips any sample whose normalized output already exists.
#
# Usage:
#   scripts/nam_alignments/normalize_gvcf_contigs.sh [PARALLEL]

set -uo pipefail

PARALLEL="${1:-8}"

CONDA_ENV_PREFIX_ANCHORWAVE="${CONDA_ENV_PREFIX_ANCHORWAVE:-/home/zrm22/mambaforge/envs/phgv2-conda}"
export PATH="$CONDA_ENV_PREFIX_ANCHORWAVE/bin:$PATH"

FAI="${FAI:-/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/B73.fa.fai}"

OUT_DIR="${OUT_DIR:-/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_v2_rebuild}"
GVCF_DIR="${GVCF_DIR:-$OUT_DIR/gvcf}"
NORM_DIR="${NORM_DIR:-$OUT_DIR/gvcf_normalized}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
mkdir -p "$NORM_DIR" "$LOG_DIR"

if [ ! -s "$FAI" ]; then
  echo "[normalize_gvcf_contigs] FAILED: no .fai at $FAI" >&2
  exit 1
fi

CANONICAL_CONTIGS="$OUT_DIR/ref/canonical_contigs.txt"
awk '{print "##contig=<ID="$1",length="$2">"}' "$FAI" > "$CANONICAL_CONTIGS"

normalize_one() {
  local gvcf="$1"
  local sample
  sample=$(basename "$gvcf" .g.vcf.gz)
  local out="$NORM_DIR/${sample}.g.vcf.gz"
  local header_file="$LOG_DIR/header_${sample}.txt"

  if [ -s "$out" ]; then
    echo "[normalize_gvcf_contigs] $sample already normalized, skipping"
    return 0
  fi

  echo "[normalize_gvcf_contigs] normalizing $sample"
  bcftools view -h "$gvcf" 2>/dev/null | grep -v "^##contig" | awk -v contigs="$CANONICAL_CONTIGS" '
    /^#CHROM/ { while ((getline line < contigs) > 0) print line; close(contigs) }
    { print }
  ' > "$header_file"

  bcftools reheader -h "$header_file" -o "$out" "$gvcf" > "$LOG_DIR/reheader_${sample}.log" 2>&1
  if [ ! -s "$out" ]; then
    echo "[normalize_gvcf_contigs] FAILED: $sample (see $LOG_DIR/reheader_${sample}.log)" >&2
    return 1
  fi
  bcftools index -c "$out"
  echo "[normalize_gvcf_contigs] $sample done -> $out"
}
export -f normalize_one
export NORM_DIR LOG_DIR CANONICAL_CONTIGS

find "$GVCF_DIR" -maxdepth 1 -name '*.g.vcf.gz' | sort | xargs -P "$PARALLEL" -I{} bash -c 'normalize_one "$@"' _ {}

echo "[normalize_gvcf_contigs] batch complete. Verify with:"
echo "  ls -la $NORM_DIR/*.g.vcf.gz | wc -l   # expect 25"
