#!/bin/bash
# bed_to_vcf.sh -- founder-pair BEDs -> genotype VCF (optional last step).
#
#   bed_to_vcf.sh OUTDIR SAMPLE
#
# Every SNP/indel record of the 25-founder panel VCF that lies inside a called BED interval gets the
# genotype (parent1's allele, parent2's allele); records outside called intervals have no call.
# Writes OUTDIR/SAMPLE_imputed.vcf.gz (+ .tbi), autosomes chr1-chr10 only. Takes ~45 min per sample
# (the panel VCF is ~14 GB).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "${GRITS_ENV:-$HERE/grits.env}"
[ $# -eq 2 ] || { sed -n '2,9p' "$0"; exit 2; }
OUT=$1; S=$2
export PATH=$GRITS_JAVA_HOME_BIN:$GRITS_HTSLIB_BIN:$PATH
ls "$OUT"/bed/"${S}"_*_imputed.bed > /dev/null || { echo "no BEDs for $S in $OUT/bed (run decode_sample.py)"; exit 1; }
"$GRITS_BED2VCF" bed-to-vcf --bed-dir="$OUT/bed" --reference-panel-vcf="$GRITS_PANEL_VCF" \
  --out-file="$OUT/${S}_imputed.all.vcf"
bcftools view -t chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10 -Oz -o "$OUT/${S}_imputed.vcf.gz" \
  "$OUT/${S}_imputed.all.vcf"
tabix -f -p vcf "$OUT/${S}_imputed.vcf.gz"
rm -f "$OUT/${S}_imputed.all.vcf"
echo "wrote $OUT/${S}_imputed.vcf.gz"
