#!/usr/bin/env bash
# Symlink the 5 held-out (out-of-index) assemblies into one directory,
# renamed to <Sample>.fa -- align-assemblies derives both the output MAF
# name and (via --sample-name passed separately downstream) the gVCF sample
# column from File(assemblyFile).nameWithoutExtension
# (AlignAssemblies.kt:606), so this is what turns the old run's ugly
# basename-derived names (ep1.genome, Zm-Ia453-REFERENCE-FL-1.0) into clean
# sample names (EP1, Ia453) throughout the rest of the chain.
#
# Source paths are the same 5 rows as the existing
# scratch/heldout_assembly_eval/worklist_full5.tsv.
# See /home/zrm22/.claude/plans/ok-now-that-we-warm-minsky.md.
#
# Usage:
#   scripts/heldout_alignments/make_heldout_symlinks.sh
#
# Env overrides: OUT_DIR

set -euo pipefail

CONDA_ENV_PREFIX_ANCHORWAVE="${CONDA_ENV_PREFIX_ANCHORWAVE:-/home/zrm22/mambaforge/envs/phgv2-conda}"
export PATH="$CONDA_ENV_PREFIX_ANCHORWAVE/bin:$PATH"

OUT_DIR="${OUT_DIR:-/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_v2_heldout}"
ASM_DIR="$OUT_DIR/asm"
mkdir -p "$ASM_DIR"

NON_INDEX="/workdir/shared_files/grits_crf_evaluation/non_index_asms/maize"
IA453="/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/external_assemblies/Zm-Ia453-REFERENCE-FL-1.0.fa"

declare -A SOURCES=(
  [Tx303]="$NON_INDEX/Zm-Tx303-REFERENCE-NAM-1.0.fa"
  [A188]="$NON_INDEX/Zm-A188-REFERENCE-KSU-1.0.fa"
  [EP1]="$NON_INDEX/ep1.genome.fa"
  [CML459]="$NON_INDEX/CML459.chromosomes.v1.fa"
  [Ia453]="$IA453"
)

for sample in "${!SOURCES[@]}"; do
  src="${SOURCES[$sample]}"
  dst="$ASM_DIR/${sample}.fa"

  if [ ! -s "$src" ]; then
    echo "[make_heldout_symlinks] FAILED: source not found or empty for $sample: $src" >&2
    exit 1
  fi

  if [ -L "$dst" ] && [ "$(readlink -f "$dst")" = "$(readlink -f "$src")" ]; then
    echo "[make_heldout_symlinks] $sample already linked, skipping"
  else
    ln -sf "$src" "$dst"
    echo "[make_heldout_symlinks] $sample -> $dst (source: $src)"
  fi

  if [ ! -s "${dst}.fai" ]; then
    echo "[make_heldout_symlinks] building .fai for $sample"
    samtools faidx "$dst"
  fi
done

echo "[make_heldout_symlinks] done. Verify with:"
echo "  ls -la $ASM_DIR/*.fa | wc -l   # expect 5"
