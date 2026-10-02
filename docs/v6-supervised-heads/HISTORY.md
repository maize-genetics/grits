# GRITS maize decoding, Aug–Oct 2026: what was tried and why

This is the record of the work that led to **v6 fit B**, the current best model, including every
direction that did not work and why. Read `README.md` in this folder first for the current state and
how to run things. Numbers are SNP+RefCall genotype error (%, lower is better) on the
simulated-validation corpus unless a line says otherwise; "0.1x"/"1x" are read depths.

## 0. Setup that everything below shares

- **Corpus.** `/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv`:
  reads simulated (wgsim) from real assemblies, 5 depths (0.01, 0.1, 0.5, 1, 2x). Classes:
  IDX (both parents in the 25-founder NAM panel), OUT (neither parent in the panel), MIX (one in, one
  out), each as INBRED / HYB (F1) / RIL2. 5 samples per class.
- **Truth.** Per-sample gVCFs from AnchorWave (`phg align-assemblies` → MAF → gVCF).
- **Metric.** SNP+RefCall error: genotype mismatch over sites where truth and call are both REF or
  SNP. The older "all-sites" error also counts deletion spans site by site; 37–41% of B73 bp are
  deletions in every truth gVCF (they are *replacements*: B73 sequence swapped for other sequence),
  so all-sites error is dominated by per-base scoring of replacement blocks. Always lead with
  SNP+RefCall.
- **Scorer.** `experiments/simval-corpus/scripts/fast_snprc_score.py` (exact, ~20 s per BED;
  validated 0.0 difference vs the full comparator) and `fast_eval_ckpt.py` (decode + score a
  checkpoint over the corpus).
- **Calibration corpus** (`.../simulated_calibration/`, IDX lines CML247/Ki3/M162W/Tzi8/NC350,
  OUT lines DK105/F7/K0326Y/PHB47/CML442, disjoint from evaluation) is the only real-read data any
  simulator parameter was fitted to. Training is on simulated data only.
- **Baselines.**
  - `diploid-affinity`: pre-indel model, ckpt `diploid-affinity-sim512-h3/d-epoch=04-val_pair_acc=0.6179.ckpt`.
  - `ternary-baseline`: first ternary/indel model, ckpt `diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt`.
    Every later run warm-starts from it.

## 1. In-panel indel line (Aug–mid Sep): works in-panel, fails out-of-panel

- **Ternary input + indels (v3-overlay, K=25).** Per-founder ternary features (match / diverged /
  deleted) from ropebwt3 refmap, simulated indels. Big in-panel win: IDX-HYB 7.2% → 0.2% vs
  diploid-affinity.
- **Fixed founder drop removed.** An old convention dropped one founder (P39) to make K=24, which
  makes that founder uncallable. Retrained at K=25: better (pair accuracy 99.35 vs 99.18%).
- **homo_scale router.** A fixed homozygous penalty (3.0) broke inbreds (1.5% pair accuracy). Fix:
  estimate inbreeding per sample (`homo_scale_from_affinity`, the "p90" classifier) and scale the
  penalty. Works in-panel; **fails on OUT samples** (all OUT p90 values sit 0.43–0.64, 5 of 15
  misclassified, and those 5 are the worst rows). The disjoint-row router (`route_real_sample`) is
  correct on 40/40 samples and is what every eval uses now.
- **row-collapse** (merge identical stacked rows): ~30× better IDX-HYB but 6–17 points *worse* than
  diploid-affinity on every OUT/MIX class. Not promoted. This started the out-of-panel investigation.

## 2. Out-of-panel investigation (mid–late Sep): mostly negative, found the real problems

All-sites genotype error here (pre-SNP+RefCall), 0.1x.

| Attempt | Idea | Outcome | Why it failed |
|---|---|---|---|
| region-fix | restore the nominal 6 crossovers/individual (a buffering bug had cut it to 1.16) | worse everywhere (IDX-HYB pair acc 96 → 82%) | real in-panel samples have **zero** recombination; more crossovers made training less realistic |
| held-out augment v1 | hide a founder, relabel its rows to a random lineage-mate | looked like a win; was a degenerate always-homozygous model (training flags dropped) | 2 bugs: missing recipe flags; validation split was single-class (file-order tail) |
| held-out augment, full recipe | same, flags fixed | beats ternary-baseline on 15/15 OUT rows but IDX-HYB 0.26 → 4.72% | per-site random relabel = label noise; that noise was its only switch signal |
| held-out mosaic / sticky relabel | remove the label noise; mosaic hidden founder | worse than v1 | removing noise removed switch pressure |
| low-rate | diploid-affinity's crossover magnitude | small win (~1–2 points) | still far behind diploid-affinity |
| decode-time switch_scale | cheaper switching at decode | small | — |

What the investigation established (still true):
- **OUT samples switch founders often:** a panel-ceiling Viterbi needs ~7 closest-founder changes per
  512-row window to reach ~2% error on OUT inbreds; models switched ~10× too little.
- **An untrained read HMM beat every neural model on OUT** (13.8 / 29.2 / 13.7 vs 16.5 / 39.6 / 16.4
  all-sites, INBRED/HYB/RIL2) → the neural models were losing read information.
- **Deletions are replacements** and dominate all-sites error; depth did not help (not read-limited).
- **Read placement bug.** PLACED reads were projected 10–100 kb off (median of all lift anchors within
  ±500 kb). Fixed in ropebwt3-phg with `-s 200` lift sampling + adaptive local projection
  (`--lift-local-win=20000`): placement +62% correct. **Adopted as the standard alignment.**
- **refmap read filters** (preset F: cut wrong-chromosome reads 7.7 → 1.65%; Ia453 3.88 → 3.53) were
  implemented but **not adopted** (small gain, untested at low depth).
- **Ternary −1 semantics.** −1 = founder unmatched and no lift anchor within 2 kb; only 72% precise / 54%
  recall vs truth deletions. Making −1 perfect at decode changed nothing; the models barely used it.

## 3. Het-replacement and likelihood CRF (late Sep)

Branch line `out-het-replacement`. Goal: teach heterozygous replacements, then decode OUT/hybrids.

- **Simulator replacement indels** with lineage-aware groups (`repl_groups=lineage`), parameters
  calibrated on the calibration corpus (`repl_params_v3.json`), plus a refmap-like observation table
  for −1/0 and distances (`--obs-table`).
- **Tie-aware loss.** Founders sharing an ancestral lineage are indistinguishable in the input; the
  loss accepts any lineage-equivalent pair (`--tie-aware-loss`). Kept in everything since.
- **Two-stage training / auxiliary losses** (colleague's idea): within noise. Diagnosis from it:
  every hybrid error kept one parent right; the **pair emission was additive**, so Viterbi picked the
  two individually best founders, not two that explain different reads.
- **Mixture pair emission** (`logaddexp` over the two haplotypes): fixes that; kept.
- **Likelihood emission** (`--emission likelihood`): the encoder no longer scores founders; a small
  table of read-state log-likelihoods is gated by the encoder. This follows the colleague's principle:
  the encoder gates and estimates recombination, the CRF decodes. Kept.
- **Distance-banded likelihood** and **multi-depth training data** (0.1/0.5/1/2x): small effects.
- Results at 0.1x (SNP+RefCall): likelihood models matched ternary-baseline in-panel but not OUT
  (OUT-HYB ~21 vs 18.7).

## 4. Supervised heads (sh3 → sh5, 29 Sep – 1 Oct)

Train the encoder directly on simulator truth: **gate** (row trustworthy), **crossovers per window**,
**het**, **affinity**. Plug the outputs into the likelihood CRF. CRF numbers set two ways: **fit A**
(computed from simulator truth) and **fit B** (tuned on simulated windows). See the separate
"Fit A vs fit B" doc.

| Step | Finding |
|---|---|
| sh3 | Hybrids much better (OUT-HYB 18.7 → 11.8). But a ~3% floor on in-panel inbreds/RILs. |
| per-row het prior | Cause of the floor: het head read ~0.999 on real inbreds; its prior, summed over 512 rows, outweighed the reads. `--het-prior off` or `segment` removes the floor. |
| fit B crossover scale | Raw scale went negative (−0.19) → switch probability clamped → crossovers off. Fixed: log-space scale; stay bonus no longer fitted. |
| het head artifact | Duplicating real rows sends the het head to exactly 0: it learned "identical neighbouring rows = homozygous", because the simulator drew one SNP window per site. |
| sh4: distance structure | Real distances share values across founders (anchor-gap geometry); simulator made them independent. Fixed (`--dist-structure`), but the het head was unchanged. |
| sh5: per-read SNPs | Each read gets its own SNPs (`--per-read-snps`). Het head improved but not calibrated. |
| sh5 ablations | Distance channel: no consistent gain → dropped. Affinity head: no gain over the read match rate → dropped. Regional / per-window affinity: no effect. |
| affinity loss | Plateaued at 0.31 vs a 0.175 floor: a window was scored against the genome-wide target. `--aff-target-scope window` fixes the target (not needed once the head was dropped). |

sh5 best at 0.1x: OUT-HYB 10.6, OUT-INB 7.7–9.4, vs diploid-affinity 19.6 / 5.4 (fair re-score).

## 5. v6 (30 Sep – 2 Oct): the current best

Training-data and label changes (all calibrated on the calibration corpus):
- per-read true-founder misses at 1.23% of reads (`--per-read-bad --bad-frac 0.031`);
- more founder sharing between haplotypes (`--read-snps 5 --derived-sfs 0.15`);
- **held-out augment v2**: 27 founders, 2 hidden per individual, 30% of individuals have their own
  founders hidden, tie-aware relabel (no label noise), faster ancestral switching (16/individual);
- gate label = "row matches its true founder" (`--gate-target support`); the old provenance label
  marked off-site replacement reads noisy although 100% of them match their true founder;
- emission table from all rows (`lik_table.py --rows all`): mismatch −4 to −6 instead of −17.

Model: ternary only (`--no-distance`), CRF founder prior from the read match rate
(`--aff-source reads`), het prior off. **v6** adds deletion-dosage and out-of-panel heads
(`--extra-heads`); **v6-base** is the same without them.

| 0.1x | IDX-INB | IDX-HYB | IDX-RIL2 | MIX-HYB | MIX-RIL2 | OUT-INB | OUT-HYB | OUT-RIL2 |
|---|---|---|---|---|---|---|---|---|
| diploid-affinity (fair, s200, inferred kind) | 0.00 | 3.70 | 0.02 | 13.85 | 2.50 | 5.42 | 19.58 | 5.48 |
| ternary-baseline | 0.00 | 0.11 | 0.06 | 12.29 | 3.63 | 6.07 | 18.74 | 5.97 |
| sh5 fit B | 0.05 | 0.21 | 0.40 | 6.01 | 4.24 | 9.38 | 10.62 | 9.33 |
| v6-base fit B | 0.00 | 0.04 | 0.18 | 4.73 | 1.11 | 4.63 | 4.45 | 4.38 |
| **v6 fit B** | 0.00 | 0.08 | 0.14 | 7.83 | 0.98 | 3.58 | 5.78 | 3.50 |

| 1x | IDX-INB | IDX-HYB | IDX-RIL2 | MIX-HYB | MIX-RIL2 | OUT-INB | OUT-HYB | OUT-RIL2 |
|---|---|---|---|---|---|---|---|---|
| diploid-affinity | 0.00 | 7.10 | 0.07 | 15.11 | 2.28 | 4.95 | 20.40 | 5.02 |
| **v6 fit B** | 0.03 | 0.43 | 0.29 | 4.54 | 1.13 | 3.13 | 4.79 | 2.91 |

What we learned after v6:
- **Fit B now beats fit A almost everywhere** (up to sh5, A won OUT inbreds/RILs); held-out
  individuals in the tuning data stopped B from over-learning "rarely switch".
- **The gains come from the data and labels**, not the architecture: v6-base already holds most of it.
- **Knockouts (decode-time, v6 fit B, 0.1x)**: the founder prior changes 16–22% of OUT/MIX calls,
  the gate 6–10%, deletion dosage 6–7% (no accuracy benefit), per-window crossovers and out-of-panel
  under 1% (no effect). A **gate-only CRF** (reads × gate + one switch rate per sample) scores mean
  2.36 vs 2.74 for v6 fit B as fitted at 0.1x and 2.29 vs 2.16 at 1x.
- **Founder prior strength depends on depth and sample type.** Zeroing it helps every MIX/OUT class at
  0.1x (mean 2.74 → 2.27) but costs hybrids at 1x (OUT-HYB 4.79 → 6.07). Refitting B with the prior
  frozen at 0 is worse (B makes switching ~0 instead).
- **Het rate as a CRF prior does not help**: once per segment at global / per-chromosome / regional
  scale = no change; per row at weight ≥ 0.05 hurts everything (the per-row head is still noisy).
- **Crossover head is weak**: on simulated data it under-predicts windows with a switch; on real data
  it predicts ~0.09 per window for every class. Simulated held-out individuals have only 0.03–0.04
  lineage-aware switches per window, so the training data still shows few switches.

## 6. Principles that held up

1. Real-data accuracy is the only verdict; simulated validation loss repeatedly misled (het head at
   0.0015 loss was wrong on real data; val split was single-class for weeks).
2. Most wins came from finding a simulator-vs-real mismatch and fixing it with a measurement on the
   calibration corpus, never on evaluation lines.
3. Per-row priors summed over a 512-row window swamp the reads unless the head is calibrated per row;
   apply priors once per segment or not at all.
4. Fit B learns whatever the simulator rewards; when it pushes a term to an extreme (affinity 4.0,
   crossover scale ~0) it is reporting a simulator gap.
5. When warm-starting, pass the full recipe flags; the guard now refuses a mismatch.

## 7. Open directions (ranked)

1. **Re-measure real OUT switching in lineage-aware terms** and make simulated held-out individuals
   switch at that rate (currently 0.03–0.04 per window at 0.1x).
2. **Sample-dependent founder prior** (stronger for hybrid-looking samples; v6's het head is
   calibrated per sample), or a depth-dependent weight.
3. **Il14H×EP1** (MIX-HYB) drives most of v6's MIX-HYB error at 0.1x (16.65%).
4. **Which v6 data change mattered** (held-out augment, support gate, all-rows table, per-read misses,
   sharing); only the table can be isolated without retraining.
5. **Gate-only decode + all heads in training** as the simpler production configuration.
6. In-panel RILs (0.14–0.29 vs diploid-affinity 0.02–0.07).

## 8. Where the older material lives

The pre-v6 experiment directories were not copied to this branch; they are on their own pushed
branches, e.g. `bring-in-simval-corpus` (simval corpus, ril2-error-regions, refmap-founder-eval,
reference-bias, tripsacum/cassava), `indel-region-buffer-fix-lowrate`, `indel-heldout-augment`,
`indel-training-loop`, `windowing-quality-filters`. See `CLEANUP_LOG.md` for the worktrees removed in
the Oct 2026 cleanup and where their documents went.
