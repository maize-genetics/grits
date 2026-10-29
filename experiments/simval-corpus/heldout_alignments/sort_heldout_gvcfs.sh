#!/usr/bin/env bash
# Physically re-sort each held-out gVCF's records into canonical
# chr1..chr10,Pt,Mt order, matching the founder panel's convention
# (scripts/nam_alignments/normalize_gvcf_contigs.sh + sort_gvcfs.sh).
#
# These 5 gVCFs are never merged with the founder panel, so this step is
# not load-bearing the way it was for merge-gvcfs -- but none of the 5
# held-out assemblies carry organelles, so each one's own ##contig set/order
# (AnchorWave/MAFToGVCF emission order, e.g. Mt,chr1,chr10,chr2,...) differs
# from the founders' and from each other. Doing this now removes that trap
# for any future tool (this project's own merge-gvcfs included) that reads
# contig order from the record/index data rather than the header text alone
# -- see scripts/nam_alignments/README.md's "Gotchas" section, which this
# mirrors: header rewrite alone is insufficient, bcftools sort physically
# reorders the body.
#
# Reuses the founders' canonical contig list
# (data/maize_v2_rebuild/ref/canonical_contigs.txt) as the single source of
# truth for what "canonical order" means, rather than rebuilding it.
#
# See /home/zrm22/.claude/plans/ok-now-that-we-warm-minsky.md.
#
# Resumable: skips any sample whose sorted output already exists.
#
# Usage:
#   scripts/heldout_alignments/sort_heldout_gvcfs.sh [PARALLEL]
#
# Env overrides: OUT_DIR, GVCF_DIR, SORT_DIR, CANONICAL_CONTIGS

set -uo pipefail

PARALLEL="${1:-5}"

CONDA_ENV_PREFIX_ANCHORWAVE="${CONDA_ENV_PREFIX_ANCHORWAVE:-/home/zrm22/mambaforge/envs/phgv2-conda}"
export PATH="$CONDA_ENV_PREFIX_ANCHORWAVE/bin:$PATH"

OUT_DIR="${OUT_DIR:-/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_v2_heldout}"
GVCF_DIR="${GVCF_DIR:-$OUT_DIR/gvcf}"
SORT_DIR="${SORT_DIR:-$OUT_DIR/gvcf_sorted}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
TMP_ROOT="${TMP_ROOT:-$OUT_DIR/tmp_sort}"
CANONICAL_CONTIGS="${CANONICAL_CONTIGS:-/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_v2_rebuild/ref/canonical_contigs.txt}"
mkdir -p "$SORT_DIR" "$LOG_DIR" "$TMP_ROOT"

if [ ! -s "$CANONICAL_CONTIGS" ]; then
  echo "[sort_heldout_gvcfs] FAILED: no canonical contig list at $CANONICAL_CONTIGS" >&2
  exit 1
fi

sort_one() {
  local gvcf="$1"
  local sample
  sample=$(basename "$gvcf" .g.vcf.gz)
  local out="$SORT_DIR/${sample}.g.vcf.gz"
  local tmpd="$TMP_ROOT/${sample}"
  local header_file="$LOG_DIR/header_heldout_${sample}.txt"

  if [ -s "$out" ]; then
    echo "[sort_heldout_gvcfs] $sample already sorted, skipping"
    return 0
  fi

  echo "[sort_heldout_gvcfs] normalizing header + sorting $sample"
  bcftools view -h "$gvcf" 2>/dev/null | grep -v "^##contig" | awk -v contigs="$CANONICAL_CONTIGS" '
    /^#CHROM/ { while ((getline line < contigs) > 0) print line; close(contigs) }
    { print }
  ' > "$header_file"

  local reheadered="$TMP_ROOT/${sample}.reheadered.g.vcf.gz"
  bcftools reheader -h "$header_file" -o "$reheadered" "$gvcf" > "$LOG_DIR/reheader_heldout_${sample}.log" 2>&1
  if [ ! -s "$reheadered" ]; then
    echo "[sort_heldout_gvcfs] FAILED: reheader $sample (see $LOG_DIR/reheader_heldout_${sample}.log)" >&2
    return 1
  fi

  mkdir -p "$tmpd"
  local t0
  t0=$(date +%s)
  bcftools sort -T "$tmpd/" -O z -o "$out" "$reheadered" > "$LOG_DIR/sort_heldout_${sample}.log" 2>&1
  local t1
  t1=$(date +%s)
  rm -rf "$tmpd" "$reheadered"

  if [ ! -s "$out" ]; then
    echo "[sort_heldout_gvcfs] FAILED: sort $sample (see $LOG_DIR/sort_heldout_${sample}.log)" >&2
    return 1
  fi
  bcftools index -c "$out"
  echo "[sort_heldout_gvcfs] $sample done in $((t1 - t0))s -> $out"
}
export -f sort_one
export GVCF_DIR SORT_DIR LOG_DIR TMP_ROOT CANONICAL_CONTIGS

find "$GVCF_DIR" -maxdepth 1 -name '*.g.vcf.gz' | sort | xargs -P "$PARALLEL" -I{} bash -c 'sort_one "$@"' _ {}

echo "[sort_heldout_gvcfs] batch complete. Verify with:"
echo "  ls -la $SORT_DIR/*.g.vcf.gz | wc -l   # expect 5"
