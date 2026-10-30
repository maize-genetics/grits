---
name: grits-run-sample
description: Run the promoted GRITS founder-imputation pipeline on a user's own maize sequencing sample, from FASTQ reads to founder-haplotype calls (per-chromosome BED of founder pairs) and optionally a genotype VCF. Use this whenever someone wants to impute, genotype or "find the founders/parents" of their own maize line, hybrid, RIL or population sample with GRITS, asks how to align reads with ropebwt3 refmap for GRITS, how to decode or call a sample with the GRITS model, what the GRITS output files mean, or why a GRITS run failed or looks wrong, even if they don't say "skill" or name every step. Not for training or developing the model.
---

# Run GRITS on your own sample

GRITS describes a maize genome as a mosaic of 25 founder assemblies (B73 plus 24 NAM founders): along
each chromosome it calls which pair of founders the sample's two haplotypes look like. This skill runs
the **promoted** pipeline, exactly as it was evaluated, on a user's reads:

```
FASTQ ──align_sample.sh──▶ windowed model input ──decode_sample.py──▶ BED of founder pairs ──bed_to_vcf.sh──▶ VCF (optional)
          (ropebwt3 refmap, ~3 min/0.1x)           (GPU, ~1-2 min)                          (~45 min)
```

The three scripts in `scripts/` reproduce the promoted evaluation byte-for-byte (verified on held-out
and in-panel samples at 0.1x and 1x), so results are comparable with the published numbers.

## Before you start: what to ask and check

1. **Run `scripts/check_setup.sh`** (add `--vcf` if they want a VCF). It checks every resource listed in
   `scripts/grits.env` (index, lift, checkpoint, ropebwt3 build, Python env, GPU). First export three
   variables: `GRITS_WORKDIR` (the GRITS working directory holding the checkpoint and lift),
   `GRITS_REFMAP_ROOT` (the directory holding `rope_bwt_index_v2/`) and `GRITS_PYTHON_ENV` (a Python
   environment with torch). The other paths default to locations under these. If a line says FAIL, read
   `references/inputs.md`: it says what each resource is and how to build or obtain it. Point the scripts at other paths by exporting the
   variables or with `GRITS_ENV=/path/to/copy.env`.
2. **Find out about the sample**, because it decides how much to trust the result:
   - **Species and panel.** Maize only, and the model works with exactly this 25-founder panel. Other
     species or panels need a new index and retraining, which this skill does not cover.
   - **What the sample is**: an inbred, an F1 hybrid, a RIL or other cross. The model works this out
     itself (see "Routing" below); you only need it to sanity-check the result.
   - **How related it is to the panel.** A NAM founder or a cross of them ("in panel") is imputed almost
     perfectly. Material whose parents are not among the 25 ("out of panel") can only be described by its
     closest founders, with several percent error (see `references/interpreting.md`). Set expectations
     early.
   - **Depth.** The model was tuned at about 0.1x and 1x genome coverage (roughly 1.5M and 15M 150 bp
     reads). Lower or much higher depth works but is outside what was calibrated.
3. **Pick an output directory per sample** with enough space (~1 GB at 0.1x, ~5 GB at 1x while running).

## Step 1: align

```bash
scripts/align_sample.sh OUTDIR THREADS reads.fq.gz [more.fq.gz ...]
```

- Paired-end data: pass R1 and R2 together. They are read as one single-end stream on purpose; that is
  how the model was evaluated, and the paired mode did not improve accuracy.
- Uses 10-14 GB RAM. With 8 threads it takes about 3 min at 0.1x and 20 min at 1x.
- The read filters drop reads that are likely misplaced, so typically only 50-70% of reads are placed
  (`EXACT` + `PLACED` in `OUTDIR/align_summary.txt`; the rest are `FILTERED`, `MULTI` or `UNPLACED`).
- It's safe to re-run: a finished directory (`align.done`) is skipped.

## Step 2: decode

```bash
LD_LIBRARY_PATH=$GRITS_LD $GRITS_PY scripts/decode_sample.py --dir OUTDIR --sample NAME
```

(`source scripts/grits.env` first to get `$GRITS_PY`/`$GRITS_LD`.)

- Needs a GPU with about 30 GB free for a 1x sample (less at 0.1x). If memory is short, use `--batch-size 64`.
  On a shared machine with `scripts/gpu_slot.sh` in the GRITS workdir, prefix the command with
  `gpu_slot.sh -m 32` so it waits for a GPU with room. `--device cpu` works but is slow.
- The sample name must not contain `_chr`; the VCF step splits file names on it.
- Writes `OUTDIR/bed/NAME_<chrom>_imputed.bed`, one file per chromosome, `OUTDIR/decode_summary.txt`
  (routing line, then calls and heterozygous fraction per chromosome) and the full log `OUTDIR/decode.log`.

**Routing.** The decode first decides whether the sample is a hybrid and prints a line like
`route: disj=0.47 p90=0.35 hybrid=True in_panel=True` (also the first line of `decode_summary.txt`). Non-hybrids (inbreds and RILs) are restricted to
homozygous founder pairs. Always check this line against what the user knows about the sample; a wrong
route is the most damaging single failure (see `references/interpreting.md`).

## Step 3 (optional): VCF

```bash
scripts/bed_to_vcf.sh OUTDIR NAME     # -> OUTDIR/NAME_imputed.vcf.gz, chr1-10
```

Each panel variant inside a called interval gets the two founders' alleles. This takes about 45 min.
Most users who want "which founder where" are better served by the BED files directly; offer the VCF when
they need genotypes for downstream tools.

## Reading the output

Each BED row is `chrom start end parent1 parent2`: from `start` to `end` the sample's two haplotypes match
those founders. `parent1 == parent2` means homozygous there. Gaps between rows are regions with no call
(a chromosome's last partial window of reads, or stretches where no founder fits). When summarising for the
user, give per-chromosome founder pairs, the share of each founder genome-wide, and the heterozygous
fraction (in `decode_summary.txt`). Explain what the calls do and do not mean using
`references/interpreting.md`, especially for out-of-panel material.

## Several samples

Run the steps per sample. Alignment is CPU-bound, so run a few samples at once with THREADS chosen so the
total does not exceed the machine's free cores. Decodes are short, but don't start several on one GPU at
the same moment (each peaks at 20-30 GB). Keep one output directory per sample: the VCF step reads every
BED in `OUTDIR/bed`.

## Reference files

- `references/inputs.md`: each resource in `grits.env`, how to build the index and lift, and how to get the
  checkpoint and tools. Read it whenever `check_setup.sh` fails or the user is on another machine.
- `references/interpreting.md`: expected accuracy by sample type and depth, the routing line, and the
  common ways results go wrong. Read it before summarising results, or when results look odd.
