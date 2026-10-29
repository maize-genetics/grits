#!/usr/bin/env bash
# Translate the RefSeq GCF_902167145.1 annotation (NCBI accession seqids)
# into the new maize_v2 B73's naming (chr1..chr10, Pt, Mt), and drop the 675
# NW_* scaffold records that name sequences absent from the trimmed
# reference. The mapping below was derived from the source FASTA's own
# headers and independently verified: every one of the 12 kept seqids'
# ##sequence-region length matches B73.fa.fai exactly (12/12).
#
# Note the organelle mapping is NOT alphabetical:
#   NC_001666.2 = chloroplast -> Pt
#   NC_007982.1 = mitochondrion -> Mt
#
# Usage:
#   scripts/nam_alignments/prepare_ref_gff.sh [IN_GFF] [B73_FASTA] [OUT_GFF]

set -euo pipefail

IN_GFF="${1:-/workdir/shared_files/grits_crf_evaluation/index_asms/GCF_902167145.1_Zm-B73-REFERENCE-NAM-5.0_genomic.gff}"
B73_FASTA="${2:-/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/B73.fa}"
OUT_GFF="${3:-/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_v2_rebuild/ref/B73_v2.gff3}"

FAI="${B73_FASTA}.fai"
if [ ! -s "$FAI" ]; then
  echo "[prepare_ref_gff] FAILED: no .fai at $FAI (run samtools faidx first)" >&2
  exit 1
fi

echo "[prepare_ref_gff] translating $IN_GFF -> $OUT_GFF"

awk '
  BEGIN {
    OFS = "\t"
    map["NC_050096.1"] = "chr1"
    map["NC_050097.1"] = "chr2"
    map["NC_050098.1"] = "chr3"
    map["NC_050099.1"] = "chr4"
    map["NC_050100.1"] = "chr5"
    map["NC_050101.1"] = "chr6"
    map["NC_050102.1"] = "chr7"
    map["NC_050103.1"] = "chr8"
    map["NC_050104.1"] = "chr9"
    map["NC_050105.1"] = "chr10"
    map["NC_001666.2"] = "Pt"
    map["NC_007982.1"] = "Mt"
  }
  /^##sequence-region/ {
    acc = $2
    if (acc in map) {
      print "##sequence-region", map[acc], $3, $4
    }
    next
  }
  /^#/ { print; next }
  {
    acc = $1
    if (acc in map) {
      $1 = map[acc]
      print
    }
  }
' "$IN_GFF" > "$OUT_GFF"

echo "[prepare_ref_gff] verifying output"

N_SEQIDS=$(awk '!/^#/{print $1}' "$OUT_GFF" | sort -u | wc -l)
if [ "$N_SEQIDS" -ne 12 ]; then
  echo "[prepare_ref_gff] FAILED: expected 12 distinct seqids, got $N_SEQIDS" >&2
  exit 1
fi

for name in chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 Pt Mt; do
  if ! awk -v n="$name" '!/^#/ && $1==n{found=1} END{exit !found}' "$OUT_GFF"; then
    echo "[prepare_ref_gff] FAILED: seqid $name missing from output GFF" >&2
    exit 1
  fi
  if ! grep -q "^$name	" "$FAI"; then
    echo "[prepare_ref_gff] FAILED: seqid $name not present in $FAI" >&2
    exit 1
  fi
done

for name in Pt Mt; do
  n_cds=$(awk -v n="$name" '!/^#/ && $1==n && $3=="CDS"' "$OUT_GFF" | wc -l)
  if [ "$n_cds" -eq 0 ]; then
    echo "[prepare_ref_gff] FAILED: organelle $name has 0 CDS records after translation" >&2
    exit 1
  fi
  echo "[prepare_ref_gff] $name: $n_cds CDS records"
done

echo "[prepare_ref_gff] OK. 12/12 seqids present, both organelles annotated."
echo "[prepare_ref_gff] output: $OUT_GFF"
