# Running GRITS on your own sample (manual guide)

This guide runs the **promoted** GRITS pipeline by hand, from FASTQ reads to founder-haplotype calls. It is the
same pipeline, with the same flags, that produced the published evaluation numbers. Claude Code users can ask
Claude to do this instead: the `grits-run-sample` skill in `.claude/skills/` drives the same three scripts.

GRITS describes a maize genome as a mosaic of 25 founder assemblies (B73 plus 24 NAM founders). Along each
chromosome it calls which *pair* of founders the sample's two haplotypes look most like.

```
reads.fq.gz ──align_sample.sh──▶ windowed model input ──decode_sample.py──▶ BED of founder pairs ──bed_to_vcf.sh──▶ VCF (optional)
             ropebwt3 refmap, CPU                       GPU, 1-2 min                                ~45 min
             ~3 min at 0.1x, ~20 min at 1x (8 threads)
```

All three scripts live in `.claude/skills/grits-run-sample/scripts/`. Below, `S` is that directory:

```bash
S=/path/to/grits/.claude/skills/grits-run-sample/scripts
```

## 1. What you need

| Resource | Variable | Notes |
|---|---|---|
| GRITS code | — | `git clone https://github.com/maize-genetics/grits`, on a branch containing this guide (the decode needs preset `placedfix_hetdel_hs_peel`). |
| ropebwt3 | `GRITS_ROPEBWT3` | Build [maize-genetics/ropebwt3-phg](https://github.com/maize-genetics/ropebwt3-phg) branch `develop` with `make`. Older builds lack `--err-noclip`. |
| 25-founder FM-index | `GRITS_INDEX` | `.fmd`, with `.fmd.ssa` and `.fmd.len.gz` beside it, ~8 GB. Copy the lab's v2 index. Rebuilding is possible but must match exactly (see `references/inputs.md` in the skill). |
| Lift | `GRITS_LIFT` | `ropebwt3 lift --ref-prefix=B73 -k 50 -s 200 -t 16 -o k50_s200.lift INDEX.fmd B73.fa`, ~4 min, 10 GB RAM. It must be k50/s200. |
| Model checkpoint | `GRITS_CKPT` | `v6cfullfe-stage2B/last.ckpt` (23 MB). Get it from the lab; it can't be rebuilt from the assemblies. |
| Python | `GRITS_PY`, `GRITS_LD` | torch, pytorch_lightning, numpy, pandas. Use the repo's default pixi env: `pixi install` (on Linux its torch build already supports CUDA GPUs). The `gpu` env is not needed and currently fails to build `mamba-ssm`. |
| GPU | — | ~30 GB free for a 1x sample; less at 0.1x. CPU works but is slow. |
| CPU / RAM | — | Alignment uses 10-14 GB RAM. |

The index, lift and checkpoint must all come from the same 25-founder panel. The panel, in index order, is:
B73, B97, CML103, CML228, CML247, CML277, CML322, CML333, CML52, CML69, HP301, Il14H, Ki11, Ki3, Ky21, M162W,
M37W, Mo18W, Ms71, NC350, NC358, Oh43, Oh7B, P39, Tzi8.

## 2. Point the scripts at your resources

Every path is read from `$S/grits.env`. Either export the variables, or copy that file, edit it and point
`GRITS_ENV` at the copy. The simplest setup is to set each path explicitly:

```bash
export GRITS_WORKDIR=/path/to/grits_workdir          # required by grits.env; other defaults live under it
export GRITS_REFMAP_ROOT=/path/to/index_parent_dir   # holds rope_bwt_index_v2/
export GRITS_ROPEBWT3=/path/to/ropebwt3-phg/ropebwt3
export GRITS_INDEX=/path/to/maizeFastaIndex_SampleContig_v2.fmd
export GRITS_LIFT=/path/to/k50_s200.lift
export GRITS_CKPT=/path/to/v6cfullfe-stage2B/last.ckpt
export GRITS_PYTHON_ENV=/path/to/grits/.pixi/envs/default  # or set GRITS_PY to a python binary

$S/check_setup.sh        # add --vcf to also check the VCF-step resources
```

`check_setup.sh` prints one line per resource and must end with `All required resources found.`

## 3. Align

```bash
$S/align_sample.sh OUTDIR THREADS reads.fq.gz [more.fq.gz ...]
```

- For paired-end data, pass R1 and R2 together. They are deliberately read as one single-end stream; that is
  how the model was evaluated.
- Typically 50-70% of reads are placed (`EXACT` + `PLACED` in `OUTDIR/align_summary.txt`). The rest are
  filtered as likely misplaced, multi-mapping or unplaceable.
- Output: `OUTDIR/windowed_k25native_wcount.npy` (the model input) and `OUTDIR/align.done`. A finished
  directory is skipped on re-run.

## 4. Decode

```bash
source $S/grits.env
LD_LIBRARY_PATH=$GRITS_LD $GRITS_PY $S/decode_sample.py --dir OUTDIR --sample NAME
```

- `NAME` must not contain `_chr`.
- If the GPU is short of memory, add `--batch-size 64`. `--device cpu` works but is slow.
- Output:
  - `OUTDIR/bed/NAME_<chrom>_imputed.bed`, one per chromosome, columns `chrom start end parent1 parent2`;
  - `OUTDIR/decode_summary.txt`: the routing line, calls and heterozygous fraction per chromosome, and a
    provenance line (checkpoint, preset, commit);
  - `OUTDIR/decode.log`.

**Check the routing line** (first line of `decode_summary.txt`), e.g.
`route: disj=0.47 p90=0.35 hybrid=True in_panel=True`. The sample is treated as a hybrid when `disj > 0.25`;
non-hybrids (inbreds, RILs) are decoded as homozygous only. If `hybrid` contradicts what you know about the
sample, treat the result with suspicion. This is the most damaging single failure.

## 5. VCF (optional)

```bash
$S/bed_to_vcf.sh OUTDIR NAME     # -> OUTDIR/NAME_imputed.vcf.gz, chr1-10, ~45 min
```

This needs the panel VCF (`GRITS_PANEL_VCF`), the `sample bed-to-vcf` CLI (`GRITS_BED2VCF`), and Java 21 plus
bcftools, bgzip and tabix on `PATH`. If they are installed elsewhere, set `GRITS_JAVA_HOME_BIN` and
`GRITS_HTSLIB_BIN` to their directories.

## 6. Several samples

Use one output directory per sample. Alignment is CPU-bound: run a few at once, keeping total threads within
the machine's free cores. Don't start several decodes on one GPU at the same moment; each peaks at 20-30 GB.

## 7. What accuracy to expect

Measured on simulated reads from held-out assemblies. Each cell is SNP+RefCall genotype error % / gene-window
error % per event.

| Sample type | 0.1x | 1x |
|---|---|---|
| In-panel inbred | 0.00 / 0.001 | 0.00 / 0.001 |
| In-panel hybrid (F1 of two founders) | 0.06 / 0.003 | 0.27 / 0.004 |
| In-panel RIL | 0.01 / 0.001 | 0.02 / 0.002 |
| Mixed hybrid (one parent in panel) | 5.4 / 0.23 | 4.3 / 0.20 |
| Mixed RIL | 2.1 / 0.21 | 1.8 / 0.16 |
| Out-of-panel inbred | 4.6 / 0.41 | 3.8 / 0.32 |
| Out-of-panel hybrid | 9.6 / 0.44 | 7.7 / 0.39 |
| Out-of-panel RIL | 4.6 / 0.42 | 3.8 / 0.34 |

For material whose parents are not among the 25 founders, a call means "this founder is the closest of the
25", not "this region is that founder". Founder switches inside an F1 hybrid are the closest-founder effect,
not crossovers. More detail, including sanity checks and a table of common failures, is in
`.claude/skills/grits-run-sample/references/interpreting.md`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `check_setup.sh`: ropebwt3 too old | build ropebwt3-phg `develop` |
| `expected the 25-founder panel` | wrong index; it must be the v2 25-founder index that matches the checkpoint |
| `unknown preset` | GRITS checkout too old |
| CUDA out of memory | `--batch-size 64`, or wait for a free GPU |
| `GLIBCXX ... not found` | run with `LD_LIBRARY_PATH=$GRITS_LD` |
| `window count does not match` | re-run `align_sample.sh` into a fresh directory |
| Permission denied on the index, lift or checkpoint | ask the lab to make them readable, or copy them |
