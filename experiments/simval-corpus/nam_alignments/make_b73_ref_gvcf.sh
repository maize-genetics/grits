#!/usr/bin/env bash
# Synthesize an all-refblock B73 gVCF: one <NON_REF> record per contig,
# spanning the WHOLE contig (POS=1..length). B73 IS the reference, so this
# is correct by construction and gives literal 100% coverage -- unlike an
# AnchorWave self-alignment, which leaves telomeres/gaps unaligned.
# See /home/zrm22/.claude/plans/nifty-napping-crystal.md, Step 3.
#
# The header (##fileformat/##FORMAT/##INFO block) is copied verbatim from a
# real founder gVCF produced by build_maize_v2_gvcfs.sh, so merge-gvcfs'
# validation pass sees an identical schema. Falls back to a hardcoded copy
# of that same block if no founder gVCF exists yet.
#
# Usage:
#   scripts/nam_alignments/make_b73_ref_gvcf.sh [B73_FASTA] [OUT_DIR] [HEADER_SOURCE_GVCF]

set -euo pipefail

B73_FASTA="${1:-/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/B73.fa}"
OUT_DIR="${2:-/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_v2_rebuild/gvcf}"
HEADER_SOURCE_GVCF="${3:-}"

JAVA_HOME="${JAVA_HOME:-/programs/jdk-21.0.1}"
CONDA_ENV_PREFIX_ANCHORWAVE="${CONDA_ENV_PREFIX_ANCHORWAVE:-/home/zrm22/mambaforge/envs/phgv2-conda}"
export PATH="$JAVA_HOME/bin:$CONDA_ENV_PREFIX_ANCHORWAVE/bin:$PATH"

FAI="${B73_FASTA}.fai"
if [ ! -s "$FAI" ]; then
  echo "[make_b73_ref_gvcf] FAILED: no .fai at $FAI" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
GVCF_PLAIN="$OUT_DIR/B73.g.vcf"
GVCF_OUT="${GVCF_PLAIN}.gz"

if [ -z "$HEADER_SOURCE_GVCF" ]; then
  HEADER_SOURCE_GVCF=$(find "$OUT_DIR" -maxdepth 1 -name '*.g.vcf.gz' ! -name 'B73.g.vcf.gz' -print -quit || true)
fi

HEADER_BLOCK_FILE=$(mktemp)
trap 'rm -f "$HEADER_BLOCK_FILE"' EXIT

if [ -n "$HEADER_SOURCE_GVCF" ] && [ -s "$HEADER_SOURCE_GVCF" ]; then
  echo "[make_b73_ref_gvcf] using header from $HEADER_SOURCE_GVCF"
  zcat "$HEADER_SOURCE_GVCF" | grep "^##" | grep -v "^##contig" > "$HEADER_BLOCK_FILE"
else
  echo "[make_b73_ref_gvcf] no founder gVCF found, using hardcoded header block"
  cat > "$HEADER_BLOCK_FILE" <<'EOF'
##fileformat=VCFv4.2
##FORMAT=<ID=AD,Number=3,Type=Integer,Description="Allelic depths for the ref and alt alleles in the order listed">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth (only filtered reads used for calling)">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PL,Number=3,Type=Integer,Description="Normalized, Phred-scaled likelihoods for genotypes as defined in the VCF specification">
##INFO=<ID=AF,Number=3,Type=Integer,Description="Allele Frequency">
##INFO=<ID=ASM_Chr,Number=1,Type=String,Description="Assembly chromosome">
##INFO=<ID=ASM_End,Number=1,Type=Integer,Description="Assembly end position">
##INFO=<ID=ASM_Start,Number=1,Type=Integer,Description="Assembly start position">
##INFO=<ID=ASM_Strand,Number=1,Type=String,Description="Assembly strand">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=END,Number=1,Type=Integer,Description="Stop position of the interval">
##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of Samples With Data">
EOF
fi

{
  cat "$HEADER_BLOCK_FILE"
  while read -r contig length _; do
    echo "##contig=<ID=${contig},length=${length}>"
  done < "$FAI"
  printf '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tB73\n'
  while read -r contig length _; do
    base=$(samtools faidx "$B73_FASTA" "${contig}:1-1" | tail -n +2 | tr -d '\n')
    printf '%s\t1\t.\t%s\t<NON_REF>\t.\t.\tASM_Chr=%s;ASM_End=%s;ASM_Start=1;ASM_Strand=+;END=%s\tGT:AD:DP:PL\t0:30,0:30:0,90,90\n' \
      "$contig" "$base" "$contig" "$length" "$length"
  done < "$FAI"
} > "$GVCF_PLAIN"

bgzip -f "$GVCF_PLAIN"
bcftools index -c "$GVCF_OUT"

N_RECORDS=$(zcat "$GVCF_OUT" | grep -vc "^#")
N_CONTIGS=$(wc -l < "$FAI")
if [ "$N_RECORDS" -ne "$N_CONTIGS" ]; then
  echo "[make_b73_ref_gvcf] FAILED: expected $N_CONTIGS records, got $N_RECORDS" >&2
  exit 1
fi

echo "[make_b73_ref_gvcf] OK: $N_RECORDS records -> $GVCF_OUT"
