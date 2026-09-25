# Heterozygous replacements in the encoder + CRF

> **Audience:** Claude Code and human collaborators. Living design document —
> update the §0 progress log with dated entries as stages land.

> **Workflow diagram:** [`results/het_replacement_workflow.png`](results/het_replacement_workflow.png)
> (stages colored reused / new / modified, capped by the §6 acceptance gate).
> Regenerate with `scripts/het_replacement_workflow_diagram.py` whenever the
> stage list in §2 changes, so the picture never drifts from this document.

## 0. Progress log

- **2026-09-25** — Plan written. No code changed yet. Branch `out-het-replacement`
  (cut from `warmstart-hparam-guard`, the tip of the indel-training line with
  the warm-start flag guard and shuffled split; `develop` has no training code).
- **2026-09-25** — §2.1 started. Calibration generator copied to
  `.../reads/maize/simulated_calibration/scripts/` (only `config.py` changed; it
  asserts no overlap with the evaluation corpus's lines). New held-out lines
  DK105, F7, K0326Y, PHB47, CML442 building truth gVCFs + anchorspro under
  `grits_workdir/data/maize_v2_calib/` (AnchorWave, per-sample work dirs). IDX
  datasets (CML247, Ki3, M162W, Tzi8, NC350) building now; OUT/MIX follow once
  the truth builds finish.
- **2026-09-25** — Priority check (the "does the model fail on simulated data
  too?" test): ternary-baseline on 40 held-out *simulated* hybrids calls
  heterozygous deletions right 76.0% of the time (vs 44.5% on real OUT hybrids at
  the same settings), and simulated data contains **no** homozygous deletions
  (0.0% of rows vs 24.5% of real hybrid sites) because overlay indels are drawn
  per founder independently. The data gap dominates, so §2.3 (lineage-shared
  replacements) now comes before §2.4; §2.4 stays but is re-tested after the
  simulator change.

## 1. Why

Held-out (OUT) samples decode far worse than in-panel (IDX) samples. A series of
diagnostics on 2026-09-23/24 (scored with `fast_snprc_score.py`, which reproduces
the full bed-to-vcf + comparator pipeline exactly) narrowed down where:

- **Truth "deletions" are mostly replacements.** Every AnchorWave-derived gVCF,
  in-panel founders included, marks 36-41% of B73 autosomal bp as literal
  deletions, balanced by an equal amount of inserted sequence (Ia453 chr1:
  121 Mb deleted vs 120 Mb inserted). B73 sequence is *replaced* (mostly TE
  turnover), not missing. Deletion spans are 40-51% of scored sites.
- **Homozygous OUT samples are close to the panel ceiling on event-level and
  SNP metrics.** For deletion events the comparison read-HMM reaches 68.5% vs a
  71.4% ceiling; for SNP+RefCall, 5.2% vs 3.4%. The per-site gap comes from a
  smaller number of long replaced blocks.
- **Hybrids are the main problem, and it is heterozygous replacements.** 34% of
  OUT-hybrid scored sites have exactly one haplotype deleted. The panel ceiling
  calls 97% of them right; the routed encoder + CRF calls 41% (and 24% as
  deleted on both haplotypes); the comparison HMM calls 60%.
- **The reads carry the signal; coverage does not.** On A188xEP1, read density
  is 1.00 / 0.67 / 0.68 for 0 / 1 / 2 deleted copies (1 vs 2 is not separable),
  but read *types* separate them: EXACT (B73-matching) reads are 52 / 35 / 8%,
  and PLACED reads name founders deleted at the locus 10 / 62 / 88% of the time.
- **The model cannot use it.** The pair emission is additive
  (`train_diploid_indel.py:320`, `emis_p = emis_f[pi] + emis_f[pj]`): each founder
  gets one score per site. At a heterozygous replacement the right pair is
  (founder *with* the sequence, founder *without* it), and a founder that is
  deleted at the site scores badly against the B73-matching reads there. An
  additive score cannot express "one of each matches this read mix".
- **Training never shows it realistically.** The training-vs-real audit found the
  simulator's deletion features unrealistic (distance channel off by orders of
  magnitude; no replacement reads placed as PLACED reads carrying the carriers'
  founder sets).
- More read depth does not help (0.1x -> 1x flat or slightly worse), and better
  PLACED-read placement (ropebwt3-phg branch `lift-adaptive-projection`) helps
  only 0.3-0.5 points; neither is the main lever.

## 2. Plan

Principles (standing rules, see memory):
- The **encoder + CRF** is the architecture being improved. HMMs are comparison
  and diagnostic baselines only (a colleague develops the HMM route separately).
- **All training uses simulated data only** (`simulate_alleles`). Real data (reads
  simulated from real assemblies, run through refmap) is used to calibrate
  simulator parameters and for evaluation.
- **Calibration and evaluation data are disjoint.** The existing simval corpus
  (`/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/`)
  stays the evaluation set; nothing is tuned on it.

### 2.1 Calibration corpus — NEW

A second corpus built with the same generator
(`simulated_validation/scripts/`: `config.py`, `build_read_datasets.py`,
`mosaic.py`, `truth.py`), with lines disjoint from the evaluation corpus:

- **5 new held-out (OUT) lines**, not in the 25-founder panel and not among the
  evaluation's held-out five (Tx303, A188, EP1, CML459, Ia453). Proposed, for
  diversity: DK105, F7, K0326Y, PHB47 (chromosome-scale) and CML442 (contig-level)
  from `/workdir/shared_files/grits_crf_evaluation/non_index_asms/maize/`.
- **5 new in-panel (IDX) founders**, excluding the evaluation's five (B73, Oh43,
  Il14H, B97, CML103). Proposed: CML247, Ki3, M162W, Tzi8, NC350.
- **Types:** INBRED, HYB, RIL2 for IDX / OUT / MIX (MIX pairs one new IDX with one
  new OUT line), 5 each, matching the evaluation corpus layout.
- **Coverage:** 0.1x (primary) and 1x.
- **Truth:** each new held-out assembly needs `phg align-assemblies` (AnchorWave)
  + `maf-to-gvcf-converter` via `scripts/build_truth_gvcf.sh`, plus its
  `*_B73.anchorspro` map for RIL mosaics. This is the main cost (hours per
  assembly); IDX founders already have both.
- Output under a new root (proposed `.../reads/maize/simulated_calibration/`),
  never mixed with the evaluation corpus.

### 2.2 Measure real statistics — NEW

From the calibration corpus only:
- replacement blocks: length distribution, spacing, DEL/INS balance, how often a
  block is shared by several panel founders (lineage sharing);
- per dosage class (0/1/2 deleted): EXACT vs PLACED read shares, how often PLACED
  reads name founders deleted at the locus, read density;
- PLACED-read placement shift distribution (collinear vs replacement sequence),
  using the read names' true origin and the gVCF `ASM_*` coordinate maps;
- per-founder feature distributions (ternary MATCH/DIV/DEL shares, distance and
  count channels) to replace the simulator's current mismatched values.

### 2.3 Simulator — MODIFIED (`src/python/crf/simulate_alleles.py`)

- Replacements as paired DEL + INS at the same locus, shared across the founders
  carrying the same lineage there (calibrated lengths and density from §2.2).
- Reads from replacement sequence emitted as PLACED-like rows: founder set = the
  founders carrying that replacement, position shifted by the calibrated
  distribution; B73-matching rows only from haplotypes whose founder has the
  sequence.
- Hybrids get heterozygous replacements naturally (the two haplotypes' founders
  differ); check the simulated 0/1/2 dosage shares against §2.2.
- Distance and count channels generated to match the real distributions
  (audit items #2 and #3).
- Off-by-default flags where the change alters existing behaviour, so older
  datasets stay reproducible.

### 2.4 Pair emission — MODIFIED (`GRITSCRFDiploidIndel`, `train_diploid_indel.py:320`)

Replace the additive pair score with a presence-aware mixture:
- The encoder emits **two scores per founder per site**: how well the founder
  explains collinear / B73-matching evidence, and how well it explains
  replacement evidence (the ternary DEL/DIV/MATCH input and read-type signal are
  already in the features).
- Pair (i, j) score combines them as a mixture over which haplotype produced the
  evidence (log-sum-exp), so (present, deleted) pairs can score well exactly when
  the read mix looks like one of each.
- Shared weights applied to every founder — **no per-founder-index parameters**
  (repo convention). CRF transitions and decoding unchanged; no `[B,T,P,P]`
  tensors.
- Warm-start from `ternary-baseline` where shapes allow (the flag guard must
  pass); new heads start near the additive behaviour.

### 2.5 Synthetic dosage check — NEW

Before any real-data run: on held-out *simulated* individuals, the model must
separate 0 / 1 / 2 deleted copies (confusion table like §1's). If it cannot on
simulated data, stop and fix §2.3/§2.4 first.

### 2.6 Train — REUSED (`train_diploid_indel.py`)

Full recipe flags (`--time-local-emis --warmup-steps 500 --homo-penalty 3.0
--spike-skip --founder-affinity`), warm-start flag guard, shuffled individual
split with per-class validation metrics. Simulated data only.

### 2.7 Evaluate and report — REUSED + NEW

- `fast_eval_ckpt.py` on the existing evaluation corpus: OUT (test), IDX, MIX at
  0.1x, plus 1x.
- Report: all-sites error, SNP+RefCall, error outside truth deletions,
  deletion-event accuracy, false-deletion events, and the **0/1/2 dosage
  confusion table** for hybrids.
- Comparison rows: ternary-baseline, routed ternary, diploid-affinity (all with
  the same inbred/hybrid call handling stated explicitly), and the read-HMM as a
  labelled comparison only.

## 3. Leakage guardrails

- No feature derived from a sample's own truth gVCF; the panel's per-founder
  deletion map is allowed (it is panel knowledge available at inference).
- Assert in code that training data and panel never include the evaluation's
  held-out lines, and that calibration lines are disjoint from evaluation lines.
- Thresholds / hyperparameters tuned only on calibration data. Earlier in-sample
  choices (router thresholds, `switch_scale=0.03`, mosaic breakpoints, HMM λ)
  are flagged as optimistic and re-derived on calibration data if reused.
- Oracle runs stay diagnostics and never enter result tables unlabeled.

## 4. Reused vs new

| stage | status | code |
|---|---|---|
| Calibration corpus | new data, reused generator | simval corpus `scripts/`, `build_truth_gvcf.sh` |
| Real statistics | new | new script under `scripts/` |
| Simulator | modified | `simulate_alleles.py` |
| Pair emission | modified | `train_diploid_indel.py` (`GRITSCRFDiploidIndel`) |
| Synthetic dosage check | new | new script / test |
| Training | reused | `train_diploid_indel.py` |
| Evaluation | reused | `fast_eval_ckpt.py`, `fast_snprc_score.py` |
| Report | new | dosage-confusion + event summary script |

## 5. Open questions

- Final choice of the 5 new OUT lines (contig-level assemblies such as CML442
  lengthen AnchorWave and add RIL-coverage undershoot).
- Whether 1x calibration data is needed or 0.1x suffices.
- Whether the adaptive lift projection should be rolled into refmap for both
  training-feature calibration and evaluation (it changes real features; keep
  it off until decided, so calibration and evaluation use the same refmap).

## 6. Acceptance criteria

- OUT-HYB heterozygous-deletion sites called correctly: up from 41% (routed
  encoder + CRF), and at least the comparison HMM's 60%.
- IDX stays below 1% error on every class.
- OUT improves on all-sites error, SNP+RefCall and deletion events without
  regressing MIX.
- Nothing tuned on the evaluation corpus.
