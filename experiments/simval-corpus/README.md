# simval-corpus

The core evaluation pipeline for the GRITS-CRF diploid founder-path imputer:
simulate reads for a known-truth individual, align+decode through `refmap`,
score against truth. This is the pipeline behind every "run dataset X at
coverage Y" result in the project, including the RIL2 founder-pair pipeline.

Everything here is a standalone driver script (invoked directly or via
`subprocess`, never imported as a package) — the same convention as the
sibling `experiments/refmap-founder-eval/` and `experiments/tripsacum-diploid-crf/`
directories. `simval_paths.py` is the central config module nearly
everything else imports.

## Environment

```bash
PY=/home/zrm22/mambaforge/envs/phg-ml/bin/python
export LD_LIBRARY_PATH=            # must be empty -- clears a host-global lib
                                    # path that breaks the conda torch/htslib stack
export PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src
```
Every command below assumes these three are set. Third-party binaries used:
`refmap` (a ropebwt3 fork) at
`/workdir/zrm22/HackathonJun2026/ropebwt_refMap/ropebwt3-phg/.claude/worktrees/refmap-ps4g-numpy/ropebwt3`,
`wgsim` at `/programs/samtools-1.20/bin/wgsim`, `bgzip` at
`/programs/htslib-1.20/bin/bgzip`.

**GPU**: inference uses CUDA if available, else CPU automatically (no flag).

## The dataset corpus (9 datasets, not 8)

The corpus manifest and build script live **outside this repo**, in
`/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/`
(shared, read-only-in-spirit resource). The dataset list is defined in
that location's own `scripts/config.py`, **not** in `simval_paths.py`
here. The corpus was originally designed as 8 datasets; `IDX-RIL2` was
added 2026-08-21 as a 9th (a true inbred RIL — `IDX-RIL`, one of the
original 8, is NOT actually a RIL: it has independent per-haplotype
breakpoints and ends up ~50% heterozygous; use `IDX-RIL2` for any RIL
claim).

| dataset_id | class | kind | individuals |
|---|---|---|---|
| `IDX-INBRED` | indexed | inbred | B73, Oh43, Il14H, B97, CML103 |
| `IDX-HYB` | indexed | hybrid | B73xOh43, B73xCML103, Oh43xIl14H, B97xCML103, Il14HxB97 |
| `IDX-RIL` | indexed | ril (not a real RIL — see above) | same 5 pairs |
| `IDX-RIL2` | indexed | ril2 (true inbred RIL) | same 5 pairs |
| `OUT-INBRED` | heldout | inbred | Tx303, A188, EP1, CML459, Ia453 |
| `OUT-HYB` | heldout | hybrid | Tx303xA188, Tx303xCML459, A188xEP1, EP1xIa453, CML459xIa453 |
| `OUT-RIL` | heldout | ril | same 5 held-out pairs |
| `MIX-HYB` | mixed | hybrid | B73xTx303, Oh43xA188, Il14HxEP1, B97xCML459, CML103xIa453 |
| `MIX-RIL` | mixed | ril | same 5 mixed pairs |

Coverages: `0.01, 0.1, 0.5, 1.0, 2.0` (only 2.0x is actually simulated;
lower rungs are nested prefix subsamples).

Already built — **do not regenerate** unless you have a real reason to:
- Reads: `.../simulated_validation/<DATASET_ID>/<individual>/<individual>.<cov>x.{R1,R2}.fastq.gz`
- Manifest (225 rows): `.../simulated_validation/manifest.tsv`
- Truth: indexed-founder gVCFs at
  `<repo-or-grits_workdir>/data/maize_v2_rebuild/gvcf_sorted/<Sample>.g.vcf.gz`;
  composed mosaics at `.../simulated_validation/truth/<DATASET_ID>/...`

If you must regenerate: `cd .../simulated_validation/scripts && $PY
build_read_datasets.py --dataset IDX-RIL2` (resumable, skip-if-exists).

## Checkpoint under evaluation

```
diploid-affinity-sim512-h3/d-epoch=04-val_pair_acc=0.6179.ckpt
```
Defined as `simval_paths.CKPT_DIPLOID` (also duplicated as
`heldout_assembly_eval.AFFINITY_CKPT`). Sibling checkpoints in the same
directory (`d-epoch=02-...`, `last.ckpt`) are **not** the evaluated one.
Model class `GRITSCRFDiploid`. Other constants in `simval_paths.py`:
`SOURCE_K=25`, `TARGET_K=24`, `FIXED_DROP_IDX=23` (see gotcha (a) below),
the v2 index/panel paths.

## Running one (dataset, coverage) row

Preferred — via the batch wrapper (handles manifest lookup, scratch/tmp
dirs, resumability, result-TSV append):

```bash
cd <this repo root>
$PY experiments/simval-corpus/scripts/simval_batch.py \
    --dataset IDX-HYB --coverage 0.1 --samples B73xOh43 \
    --parallel-align 1 --parallel-score 1 --threads-per-align 20
```
Add `--dry-run` first. Drop all filters for the full 225-row corpus.
Other flags: `--kind --limit --force --drop-idx --region --no-cleanup`.

Direct single-row invocation (get exact `--r1/--r2/--truth-h1/--truth-h2`
from `manifest.tsv` rather than typing them):

```bash
$PY experiments/simval-corpus/scripts/simval_eval_one.py \
  --stage align --sample B73xOh43 \
  --r1 <fastq R1> --r2 <fastq R2> \
  --truth-h1 <B73 truth gvcf> --truth-h2 <Oh43 truth gvcf> \
  --outdir <scratch dir> --out-json <scratch dir>/stage_out.json \
  --threads 20 --drop-idx 23 \
  --dataset-id IDX-HYB --coverage 0.1 --dataset-class indexed --kind hybrid
# then --stage score with the same args
```
`--kind` is required at align time (selects the `homo_scale` prior).
`--stage all` runs both.

## SNP+RefCall re-scoring

A site counts only if **both** truth and imputed classify to
`{HOMREF, SNP}` — any site where either side calls an indel is skipped.
Both scripts require step above's align+score to already have produced
the cached `*_imputed.autosomes.vcf.gz` — neither re-runs refmap/inference.

**Full-corpus** (`simval_snp_refcall_rescore.py`) — **hardcoded to the
0.1x rung only**, no flag to change this:
```bash
$PY experiments/simval-corpus/scripts/simval_snp_refcall_rescore.py \
  --parallel 12 --verify-against results/simval_results.tsv
```
`--verify-against` recomputes the all-sites metric alongside and asserts
it reproduces the given TSV exactly — this is the built-in correctness
gate.

**RIL2-specific, filter-sweep grid** (`run_ril2_snp_scoring.py`) — 5 pairs
x 5 coverages x 3 tags (`unfiltered-bin`, `hitfrac0.3-bin`,
`hitfrac0.5-bin`) = 75 rows:
```bash
$PY experiments/simval-corpus/scripts/run_ril2_snp_scoring.py --parallel 12
```
**Gotcha (e):** this script's `write_tsv()` **truncates and overwrites**
its output TSV with only the rows in that invocation — always run without
`--limit` to regenerate the full 75-row table, never trust a stale-looking
partial file.

## RIL2 — canonical driver vs. one-off sweep variants

**Canonical "run RIL2 end to end for a pair": `run_ril2_all_pairs.py`
(align) + `run_ril2_snp_scoring.py` (score, above).** No single script
does both.

```bash
# align, unfiltered baseline, all 5 coverages, one pair
$PY experiments/simval-corpus/scripts/run_ril2_all_pairs.py --pairs B73xOh43

# align, hit-fraction-filtered arm, all pairs, one coverage
$PY experiments/simval-corpus/scripts/run_ril2_all_pairs.py \
    --coverages 0.1 --max-hit-frac 0.5
```
Founder-path scoring without gVCF/comparator (replays the corpus's own
breakpoint RNG):
```bash
$PY experiments/simval-corpus/scripts/founder_path_error.py \
    --parent-a Oh43 --parent-b Il14H --tags unfiltered-bin hitfrac0.3-bin hitfrac0.5-bin
```

**Everything else RIL2-shaped is a one-off investigation variant, not a
routine driver — see `../ril2-error-regions/README.md`.** Two are
excluded entirely and stayed in `grits_workdir/scripts/`, not brought
into this repo: `filter_ps4g.py` and `run_ril2_windowfilter_test.py`,
both hard-dependent on `--max-hit-frac` living only on the (excluded, see
top-level `docs/HANDOFF.md`) `windowing-quality-filters` branch.

## Results locations

Only curated summary tables + findings docs are committed here — the
per-row raw JSON/log output (`simval_logs/`, `simval_events/`,
`simval_snp_refcall/`, batch summary JSONs, etc.) is regeneratable and
was deliberately kept local-only (still in `grits_workdir/results/` if
you need it without re-running):

- `results/simval_results.tsv` — the published all-sites site-level matrix (200 rows)
- `results/simval_events_results.tsv` — event-level rescore summary
- `results/simval_snp_refcall_results.{tsv,md}` — SNP+RefCall (0.1x rung)
- `results/simval_oracle_results.tsv`, `results/simval_panel_floor.tsv`

RIL2-specific results (error-region diagnostics, SNP+RefCall filter-sweep
tables) live in `../ril2-error-regions/results/`.

## Gotchas

**(a) `drop_idx=23` (`FIXED_DROP_IDX`).** Pinned from pilot row
`IDX-INBRED/B73/2.0x`, held fixed across all corpus rows so the coverage
axis stays comparable — **correct and intentional** for `simval_batch.py`
runs. It is **wrong** as a per-sample choice for most RIL2 work (P39 is
not always the lowest-hit founder for a given pair) — use
`adaptive_drop.adaptive_drop_idx(raw_npy, gamete_names)` instead when
scoring a specific pair, not the corpus as a whole.

**(b)** `run_ril2_all_pairs.py`'s `--max-hit-frac` arm asserts a path
into `grits-windowfilter-worktree` (the excluded `windowing-quality-filters`
branch) exists. If that worktree is ever removed, the filtered arm
hard-fails at import — the unfiltered arm is unaffected.

**(c)** `simval_snp_refcall_rescore.py` cannot be pointed at a non-0.1x
rung without editing its source.

**(d)** `results/simval_results.tsv` is the regression baseline both
re-scoring passes' `--verify-against` compares to — don't overwrite it
casually.

**(e)** See above — `run_ril2_snp_scoring.py` truncates its own TSV.
