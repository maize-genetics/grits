#!/usr/bin/env bash
# merge-gvcfs' ValidateVCFsUtils derives each file's "contig order" from the
# actual body/index record order (first-appearance order in the bgzipped
# file), NOT from the ##contig header lines -- confirmed empirically:
# rewriting the header text alone (normalize_gvcf_contigs.sh) did not fix
# validation, because `bcftools query -f '%CHROM\n' | uniq` on a
# header-normalized file still showed the original AnchorWave/MAFToGVCF
# emission order (e.g. Mt, chr1, chr10, chr2, ...). The actual variant
# records must be physically re-sorted into a consistent chromosome order
# across all 25 files. Runs on the header-normalized set
# (gvcf_normalized/), since sorting from there also carries forward the
# canonical ##contig header block.
# See /home/zrm22/.claude/plans/nifty-napping-crystal.md.
#
# Resumable: skips any sample whose sorted output already exists.
#
# Usage:
#   scripts/nam_alignments/sort_gvcfs.sh [PARALLEL]

set -uo pipefail

PARALLEL="${1:-6}"

CONDA_ENV_PREFIX_ANCHORWAVE="${CONDA_ENV_PREFIX_ANCHORWAVE:-/home/zrm22/mambaforge/envs/phgv2-conda}"
export PATH="$CONDA_ENV_PREFIX_ANCHORWAVE/bin:$PATH"

OUT_DIR="${OUT_DIR:-/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_v2_rebuild}"
NORM_DIR="${NORM_DIR:-$OUT_DIR/gvcf_normalized}"
SORT_DIR="${SORT_DIR:-$OUT_DIR/gvcf_sorted}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
BCFTOOLS_TMP_ROOT="${BCFTOOLS_TMP_ROOT:-/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_v2_rebuild/tmp_sort}"
mkdir -p "$SORT_DIR" "$LOG_DIR" "$BCFTOOLS_TMP_ROOT"

sort_one() {
  local gvcf="$1"
  local sample
  sample=$(basename "$gvcf" .g.vcf.gz)
  local out="$SORT_DIR/${sample}.g.vcf.gz"
  local tmpd="$BCFTOOLS_TMP_ROOT/${sample}"

  if [ -s "$out" ]; then
    echo "[sort_gvcfs] $sample already sorted, skipping"
    return 0
  fi

  mkdir -p "$tmpd"
  echo "[sort_gvcfs] sorting $sample"
  local t0
  t0=$(date +%s)
  bcftools sort -T "$tmpd/" -O z -o "$out" "$gvcf" > "$LOG_DIR/sort_${sample}.log" 2>&1
  local t1
  t1=$(date +%s)
  rm -rf "$tmpd"

  if [ ! -s "$out" ]; then
    echo "[sort_gvcfs] FAILED: $sample (see $LOG_DIR/sort_${sample}.log)" >&2
    return 1
  fi
  bcftools index -c "$out"
  echo "[sort_gvcfs] $sample done in $((t1 - t0))s -> $out"
}
export -f sort_one
export NORM_DIR SORT_DIR LOG_DIR BCFTOOLS_TMP_ROOT

find "$NORM_DIR" -maxdepth 1 -name '*.g.vcf.gz' | sort | xargs -P "$PARALLEL" -I{} bash -c 'sort_one "$@"' _ {}

echo "[sort_gvcfs] batch complete. Verify with:"
echo "  ls -la $SORT_DIR/*.g.vcf.gz | wc -l   # expect 25"
