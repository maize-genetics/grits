# Fixing the depth-confidence error increase

2026-09-18. Genome-wide genotype rescoring (`compare_gvcf_truth_diploid.py
--snp-refcall-metrics`, excludes all indel-classified sites independently)
found real HYB SNP-only genotype error rises sharply with real depth
(0.64%→2.65% SNP-class error, 0.1x→2.0x, Oh43xIl14H) even though
`ibd_adjusted_accuracy`'s window-level raw-read-count check reports flat
~99.6-99.9% founder accuracy at every depth. Root cause (fork
investigation, ruled out alternatives with real numbers — see
`ibd_adjusted_accuracy_correction` in project memory): NOT a PLACED-vs-
EXACT read-mix shift (identical 49.5% at both depths), NOT new "hard
sites" entering the scored pool (`compared_sites` grew 0.04% between
depths while mismatches grew 6.5x — the *same* sites are miscalled far
more often), NOT homozygous collapse. The real mechanism: the CRF's
Viterbi decode gets **more confident, not more correct**, in an
already-ambiguous whole-window majority call as more real reads
accumulate — `stay_bonus` reinforces within-window consistency, and more
real depth means more real rows/timesteps within a window for that
reinforcement to compound on the same ambiguous mapping-support signal
(repetitive sequence, paralogy, anchor/liftover ambiguity).

```mermaid
flowchart LR
    A["more real depth"] --> B["more real rows/timesteps<br/>per 512-site window"]
    B --> C["stay_bonus reinforces the<br/>SAME majority call across<br/>more timesteps"]
    C --> D["decode confidence rises"]
    D -->|"signal was genuinely<br/>ambiguous (mapping tie)"| E["confidently WRONG,<br/>more often, not less"]

    classDef bad fill:#f7e6e6,stroke:#b84a4a,color:#3a0b0b
    class E bad
```

## Candidate fixes

Ordered roughly cheapest/fastest-to-test → biggest investment. Inference-
time options need no retraining and can be tested directly against the
already-cached real data for this checkpoint (`diploid-indel-v3-k25-
overlay-affinity`).

| # | Fix | Where | Cost to test |
|---|---|---|---|
| 1 | Cap effective evidence per window — subsample real reads down to a fixed target depth before decoding whenever real depth exceeds it | inference | cheap |
| 2 | Reduce / make `stay_bonus` adaptive to evidence density | inference | cheap (it's a learned scalar `nn.Parameter`, overridable at eval time with no retrain) |
| 3 | Switch decode mode away from pure Viterbi (posterior-marginal decoding) | inference | needs checking whether `GRITSCRFDiploidIndel`'s kernels support a non-Viterbi decode already |
| 4 | Temperature-scale emission scores by local read density | inference | moderate (new calibration logic) |
| 5 | Depth-subsampling augmentation during training | training | large (new training run) |
| 6 | Recalibrate simulator coverage regime against measured real per-window read density | training | large (needs the train/test coverage-regime question resolved first, not yet tested) |

## Plan

Test #1 and #2 first, on a fast representative subset — one HYB pair and
one RIL2 pair (both Oh43xIl14H, since it already has the puzzling
depth-degradation numbers and 0.1x/2.0x genotype-level ground truth to
compare against), at 0.1x and 2.0x. Score with the same
`score_ibd_adjusted_accuracy`/`infer_real_founder_pairs` founder-decode
metrics used throughout this session (fast, no VCF pipeline needed for
the initial screen); if a mitigation looks promising at the founder-
decode level, validate the winner with a real genotype-level SNP+RefCall
rescore before treating it as confirmed (founder-decode accuracy and
genotype accuracy have already diverged once this session — see the
correction above — so a founder-decode-only win is not sufficient
evidence on its own).

## Results

### Fix #2 (stay_bonus sweep) — ruled out

Swept `stay_bonus` from 0.0 to 3.0 (trained default ≈2.01) at inference
time (it's a learned `nn.Parameter`, freely overridable, no retrain
needed — `experiments/depth-confidence-fix/scripts/stay_bonus_sweep.py`)
on HYB Oh43xIl14H and RIL2 Oh43xIl14H at 0.1x and 2.0x:

| sample/depth | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|---|
| HYB Oh43xIl14H / 0.1x | 99.03% | 99.03% | 99.03% | 99.03% | 99.03% | 99.03% | 99.04% |
| HYB Oh43xIl14H / 2.0x | 96.49% | 96.49% | 96.49% | 96.49% | 96.50% | 96.50% | 96.50% |
| RIL2 Oh43xIl14H / 0.1x | 99.62% | 99.62% | 99.62% | 99.62% | 99.62% | 99.62% | 99.62% |
| RIL2 Oh43xIl14H / 2.0x | 99.42% | 99.42% | 99.42% | 99.42% | 99.42% | 99.42% | 99.42% |

**Completely flat.** `stay_bonus` is not the lever — the decode outcome
is robust to it across this whole range, meaning the emission-score gap
between the best and second-best pair states is already decisive on its
own; transition-level smoothing isn't what's driving the depth-confidence
effect. Ruled out; the mechanism must live in the emission scores
themselves scaling with evidence density, not the CRF's transition term.
Points toward fix #1 (cap evidence per window) or #4 (temperature-scale
emissions) as more promising next tests.

### Fix #1 (cap effective evidence) — original design was wrong; corrected design finds the real cause

The original `evidence_cap_sweep.py` masked a random fraction of
currently-populated ternary sites back to PAD, assuming higher real depth
means more founders are directly evidenced per site. That assumption is
**false**: per-founder ternary fill rate is 1.0000 (25/25 founders
evidenced at every site) at *both* 0.1x and 2.0x, for both HYB and RIL2 —
`TERN_PAD` essentially never appears in real converted data at all,
because the ternary state is derived from anchor/liftover distance (a
property of each founder's genome relative to B73), which is available
once any read lands near a bin, independent of depth. The ternary-state
distribution (match/diverged/deletion fractions) and mean anchor distance
are also nearly identical between depths. So masking sites to simulate
"fewer reads" only destroys real information without corresponding to
anything that actually differs between 0.1x and 2.0x real data — the
96.5%→39.8% accuracy crash it produced was an artifact of deleting real
signal, not a demonstration of the depth-confidence mechanism.

**The real difference is genomic span, not evidence density.**
`ropebwt_npy_to_matrix.py` builds each `T=512` window from 512
*consecutive covered rows* (`raw.npy.bins.tsv`, sorted by contig then
bin), not a fixed genomic interval. At higher real depth there are more
covered bins genome-wide, so the same 512-row window spans a much smaller
genomic region:

| sample/depth | median gap between covered bins | mean window span (bin units) |
|---|---|---|
| HYB Oh43xIl14H / 0.1x | 3.0 | 3,809.7 |
| HYB Oh43xIl14H / 2.0x | 0.0 | 292.5 |
| RIL2 Oh43xIl14H / 0.1x | 3.0 | 3,902.0 |
| RIL2 Oh43xIl14H / 2.0x | 1.0 | 354.4 |

2.0x windows are ~13x more genomically compressed than 0.1x windows. A
geographically tiny window means the 512 CRF timesteps are largely
redundant re-observations of the *same* local mapping-ambiguity signal
(nearby bp positions share the same underlying reads/paralogy context),
which the CRF reads as confidence without it being correctness.

**Test** (`genomic_span_subsample.py`): re-derive the exact same
`[tern|labels|dist]` row transform `ropebwt_npy_to_matrix.py` computes,
straight from the cached pre-windowing `raw.npy` (verified byte-identical
to the cached windowed `.npy` at `stride=1`), then re-window the *same
real 2.0x reads* — no masking, no synthetic degradation, no depth
reduction — by taking every Nth covered row so a 512-row window spans
approximately 0.1x's real genomic footprint:

| sample | stride=1 (today) | stride=2 | stride=4 | stride=8 | stride=13 (≈0.1x span) | stride=20 |
|---|---|---|---|---|---|---|
| HYB Oh43xIl14H pair_acc | 96.50% | 97.52% | 98.49% | 99.20% | **99.23%** | 99.51% |
| HYB Oh43xIl14H ibd_adj | 99.74% | 99.77% | 99.88% | 99.99% | **100.00%** | 100.00% |
| RIL2 Oh43xIl14H pair_acc | 99.42% | 99.57% | 99.67% (peak) | 99.63% | 99.56% | 99.20% |
| RIL2 Oh43xIl14H ibd_adj | 99.92% | 99.91% | 99.82% | 99.76% | 99.62% | 99.28% |

**Confirmed.** For HYB — the sample where the depth-confidence problem
was actually observed (SNP+RefCall genotype error 0.64%→2.65%,
0.1x→2.0x) — spacing out 2.0x's own real reads to match 0.1x's genomic
span recovers pair_acc from 96.50% to 99.23% (ibd_adj to 100.00%),
*exceeding* the raw 0.1x baseline (99.03%), using the identical
checkpoint and identical real reads, only changing how rows are grouped
into windows. RIL2 — which the SNP-level rescore already showed is worst
at *low* depth, not high depth, the opposite pattern from HYB — shows
only a small, non-monotonic effect here, consistent with RIL2's error
being driven by something else (IBD/homology, not window compression).

**Implication**: `ropebwt_npy_to_matrix.py`'s row-count-based windowing
(`--window-size 512` consecutive covered rows) should become genomic-
span-aware — e.g. stride/step chosen adaptively so a window targets a
roughly constant bp footprint regardless of local read density, rather
than a fixed row count. This is an inference-time (and training-data-
generation-time) windowing change, not a model change — no retraining
required to validate further, though the training data's own windowing
convention would need the same fix for train/inference consistency
before promoting a production default.

**Note**: the stride fix above is diagnostic only (discards real reads)
and was not shipped. A parallel, still-unresolved attempt at a
non-wasteful whole-window aggregate confidence feature (`window_density`,
branch `indel-density-features`) gave a negative, confounded real-data
result — see that branch's own writeup. Its own probe gate rejected an
earlier per-row (gap/stack) design as flat/non-predictive of correctness,
which is what motivated the per-cell read-support-count idea below.

## Per-cell read-support count — branch `indel-readcount-feature`

2026-09-18, later the same day. Lab-meeting suggestion (colleague, via
user): keep the per-founder ternary state as-is (`{-1,0,1}`), but add an
explicit per-cell **read-support count** — how many reads actually voted
for that call, at that (site, founder). Different from `window_density`'s
whole-window aggregate: a genuinely finer-grained, per-cell confidence
signal the current ternary encoding throws away entirely (a match backed
by 1 read looks identical to one backed by 30).

**Validated directly against real cached data before any implementation**
(`scripts/readcount_probe.py`, using `raw.npy`'s already-cached counts
block and the deployed model's own predictions — no new pipeline code
needed to test the hypothesis). Error rate drops sharply and
monotonically with true-founder read support, at 2.0x (pooled HYB+RIL2):

| true-founder support | 0 reads | 1 | 2 | 3–5 | 5–10 | >10 |
|---|---|---|---|---|---|---|
| error rate | **4.40%** | 0.83% | 0.74% | 0.65% | 0.60% | **0.36%** |

Over 10x reduction worst-to-best — a real signal, and starkly different
from `window_density`'s per-row gap/stack probe (flat, 2.08–2.32% across
most of its range). A class-breakdown check also confirmed this hits SNP
and indel sites roughly equally (HYB indel-class error 0%→3.70% vs
SNP-class 0%→3.07%, 0.01x→2.0x) — a founder-decode problem, not a
variant-class artifact.

### Implementation

- `simulate_alleles.py --emit-read-counts`: `_indel_chunk` derives a
  per-(window,site,founder) count. Rows sharing one site are **not** all
  identical (on1/on2/insertion-kind rows draw independently — verified
  against `test_indel_chunk_tiny_exact`'s own fixture), so the count must
  be a genuine tally of how many rows at that site show MATCH per
  founder, not the site's total row count regardless of founder (an
  earlier design that would have over-counted). Deterministic from
  already-drawn `tern`/`cnt` — zero new RNG draws, golden hashes
  untouched.
- `ropebwt_npy_to_matrix.py --emit-read-counts`: the real per-(row,
  founder) count already exists in refmap's `--anchor-dist-npy` export
  (`arr[:, :K]`) — this script just discarded it until now. Threaded
  through as a 4th block. No refmap/alignment rerun needed — verified
  against the already-cached real HYB 2.0x `raw.npy` this session has
  used throughout.
- `IndelFounderPathEncoder.count_proj`: zero-initialized `Linear(1,
  d_model)`, a separate additive term on the cell embedding (not folded
  into `_cell_input`'s 5-dim representation, which would blow up
  `fast_cells`' 516-state table to 516×127). Constructed **last** in
  `__init__` — the RNG-order-safety lesson from `window_density`,
  applied proactively this time; `TestOverfitSmoke` passed clean with no
  rework needed.
- Count lives **in the main array** (2K+2 → 3K+2), not a sidecar —
  `_check_width`/Dataset/`infer_real_founder_pairs` just detect width.
  Old 2K+2 data and checkpoints keep working unchanged.
- `TestBitIdenticalWarmStart`: checkpoint loaded `strict=False`
  reproduces its own pre-`count_proj` real-data predictions exactly —
  13/13 real-data tests passed first try.

### Isolated-variable retrain — avoiding last round's mistake

`window_density`'s retrain confounded two changes at once (mixed-depth
data + a synthetically-harder founder-count composition,
`--min-founders 2 --max-founders 25`), so its HYB regression couldn't be
attributed to either cleanly. This round: training data generated with
the **same single-coverage regime as the original v3-K25 recipe**
(`coverage_model=linear`, `coverage=2.0`) and a **clean, real-sample-like
composition** — two sub-populations, `--inbreeding 1.0` (100 individuals,
verified `het_frac=0.000`, matching real INBRED) and `--inbreeding 0.0`
(100 individuals, verified `het_frac=0.969`, matching real 2-founder
HYB), concatenated and individual-shuffled, **not** the grouped/
min-max-founders mechanism that caused last round's regression. `--sites
8192` per individual (smaller than the original's 60000, for a fast
validation-scale run), sliced to `T=512` via `scripts/slice_contigs.py`
(needed zero changes — it slices along the row axis, agnostic to column
width, verified by direct content comparison against the source array).
200 individuals × 16 sub-windows = 3,200 total training windows.
`--emit-read-counts` on both sub-populations. Real eval data
(`ropebwt_npy_to_matrix.py --emit-read-counts`) regenerated for HYB/RIL2
Oh43xIl14H at all 5 depths — window counts matched the previously-known
values exactly, confirming correctness.

Warm-started `diploid-indel-v3-k25-overlay-affinity` via the new
`--warm-start-ckpt` flag, 5 epochs. Training healthy (val_pair_acc
0.021→0.25, much higher than `window_density`'s 0.21 on a harder task).

### Real-data result — regression vs. full-scale baseline, but a genuine ablation resolves why

| depth | v3-K25 baseline (full-scale) | readcount-isolated (small-scale, with count) |
|---|---|---|
| 0.01x | 100.00% | 100.00% |
| 0.1x | 99.03% | 98.35% |
| 0.5x | 98.08% | 96.92% |
| 1.0x | 97.13% | 95.34% |
| 2.0x | 96.50% | 94.40% |

HYB regressed at every depth from 0.1x on — this time genuinely with
`count` non-null on both sides (unlike `window_density`'s eval, which
never exercised its own mechanism). Rather than stop here, ran one more
cheap ablation (retraining took ~15s): the exact same isolated-variable
data and individuals, warm-started identically, with the count block
simply stripped (`2K+2`, not `3K+2` — `count_proj` never fires during
this training, so its weights stay exactly zero throughout — a true
"same everything except count" control,
`diploid-indel-v3-scale-control-nocount`).

| depth | v3-K25 baseline (full-scale) | scale-control (small-scale, no count) | readcount-isolated (small-scale, with count) |
|---|---|---|---|
| 0.01x | 100.00% | 100.00% | 100.00% |
| 0.1x | 99.03% | 97.83% | 98.35% (**+0.52pp** vs control) |
| 0.5x | 98.08% | 96.16% | 96.92% (**+0.76pp**) |
| 1.0x | 97.13% | 95.16% | 95.34% (**+0.18pp**) |
| 2.0x | 96.50% | 94.48% | 94.40% (−0.08pp, flat) |

**This resolves the confound.** The bulk of the gap vs. the full-scale
baseline is explained by training **scale** (200 individuals vs. the
original's ~1000) — the scale-only control regresses almost as much as
the count version did, at every depth. Once that's controlled for, the
count feature shows a **small, consistent, real improvement at 3 of 4
non-ceiling depths** (0.1x–1.0x, +0.18 to +0.76pp), going flat only at
2.0x, the depth where the original problem is worst. RIL2 is flat across
all three variants (±0.1pp, noise) at every depth, consistent with its
error being IBD/homology-driven, not depth-confidence-driven — no
mechanism tested this session has moved RIL2 either direction
meaningfully.

**Verdict: promising but inconclusive.** The count feature has a real,
positive, if modest, effect once cleanly isolated — but this validation-
scale test can't yet tell whether it would close the depth-confidence
gap, because training scale is the dominant limiting factor right now,
not the feature itself. Not yet promoted; not disproven either. Branch
`indel-density-features` (and `window_density` within it) remains a
separate, still-unresolved hypothesis on its own branch.

### Full-scale retrain — matches original recipe's exact scale, but a severe, founder-specific regression appears

Repeated the isolated-variable recipe above at the original v3-K25's
exact scale: 500+500 individuals (`--inbreeding 1.0`/`0.0`, same clean
split), `--sites 60000` (not 8192), `--indel-region-mult 4`,
`--emit-read-counts` — sliced output was **`(117000, 512, 77)`, 1000
individuals × G=117**, an exact match to the original `maize_v3_k25
_sliced.npy`'s own `N=117,000 individuals=1000` (only the width differs,
2K+2→3K+2). `scripts/slice_contigs.py` needed no changes at all — it
slices along the row axis, agnostic to column width (verified by direct
content comparison against the source array before trusting it).
Warm-started identically, 5 epochs, `val_pair_acc` climbed to 0.4533 —
much higher than the validation-scale run's 0.2527, as expected with far
more data. Real eval data (`--emit-read-counts`) generated for **all 15**
IDX-INBRED/IDX-HYB/IDX-RIL2 samples at all 5 depths (not just
Oh43xIl14H), per request, regardless of how the result looked.

Mean `pair_acc` / `ibd_adjusted_accuracy` by kind and depth, baseline vs.
full-scale readcount checkpoint (`diploid-indel-v3-readcount-fullscale`,
epoch 3, val_pair_acc=0.4533):

| kind | metric | 0.01x | 0.1x | 0.5x | 1.0x | 2.0x |
|---|---|---|---|---|---|---|
| INBRED | baseline pair_acc | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| INBRED | new pair_acc | 99.91 | 99.49 | 99.14 | 98.65 | **97.74** |
| INBRED | baseline/new ibd_adj | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| HYB | baseline pair_acc | 100.00 | 99.23 | 97.73 | 96.95 | 96.16 |
| HYB | **new pair_acc** | **50.88** | **38.79** | **31.68** | **29.72** | **28.11** |
| HYB | baseline ibd_adj | 100.00 | 99.87 | 99.69 | 99.66 | 99.63 |
| HYB | new ibd_adj | 86.73 | 79.92 | 83.26 | 84.90 | 85.96 |
| RIL2 | baseline pair_acc | 96.67 | 98.83 | 98.59 | 98.49 | 98.30 |
| RIL2 | new pair_acc | 96.53 | 92.61 | 89.82 | 88.98 | 88.16 |
| RIL2 | baseline/new ibd_adj | ~99.6-99.8 | ~99.5-99.6 | ~99.7 | ~99.8 | ~99.8 |

**This is not a mild, uniform regression like the validation-scale
round — HYB pair_acc collapsed to near-chance.** Per-sample detail
revealed a clean, severe, founder-specific pattern, not noise:

| sample | baseline pair_acc (2.0x) | new pair_acc (2.0x) |
|---|---|---|
| INBRED B73 | 100.00% | **88.70%** (only non-perfect INBRED; all 4 others stayed exactly 100.00% at every depth) |
| HYB B73xOh43 | 95.29% | **6.39%** |
| HYB B73xCML103 | ~95% | **5.50%** |
| HYB Oh43xIl14H | 96.50% | 42.68% |
| HYB B97xCML103 | ~96% | 45.23% |
| HYB Il14HxB97 | ~96% | 40.73% |
| RIL2 B73xCML103 | ~99% | **69.57%** |
| RIL2 B73xOh43 | ~99% | **73.13%** |
| RIL2 B97xCML103 | 99.42% | 99.43% |
| RIL2 Il14HxB97 | ~99% | 99.35% |
| RIL2 Oh43xIl14H | 99.42% | 99.32% |

**Every sample involving B73 (founder index 0 in this panel) is
severely degraded; every sample not involving B73 shows the milder,
roughly-expected pattern** (matching the validation-scale round's
magnitude for non-B73 HYB pairs, and near-baseline for non-B73 RIL2).
Severity scales with how actively the decode must *distinguish* B73 from
alternatives: INBRED B73 (must just stay homozygous) degrades mildly;
RIL2 B73-pairs (mixed local homo/het) degrade moderately; HYB B73-pairs
(must actively pick B73 vs. one specific alternative every window)
collapse to near-chance. `ibd_adjusted_accuracy` for HYB (80-87%) is
well above raw `pair_acc` (28-51%) but still far below baseline's
~99.6-99.9%, so this isn't just "confidently wrong on ties" — the raw
founder-pair prediction itself is often genuinely broken specifically
where B73 is a candidate.

**Not (obviously) a code bug.** Nothing in `--emit-read-counts` or the
model changes special-cases founder index 0 anywhere — the count
aggregation, `count_proj`, and `_cell_input` all operate identically
across all K founders. B73 is the real panel's reference genome, and the
simulator's `--indel-ref-founder` defaults to `-1` (no founder is given
special reference-like treatment during generation) — a genuine
simulated/real domain mismatch for whichever index happens to be B73 in
the real panel. But **this exact mismatch was equally present in the
original v3-K25 training** (same default), and its checkpoint handles
B73 fine (95-100% across all B73-involving samples) — so the mismatch
alone doesn't explain why this specific retrain broke it. Leading
hypothesis: a genuine warm-start/retraining instability specific to this
run's data realization, not (yet) shown to be inherent to the count
feature or to scale generally — this project has documented history of
exactly this class of pathology (`baseline_training_log_2026-09-15.md`'s
"the homo-penalty detour": `--homo-penalty 3` caused loss to go flat/
oscillate under certain data compositions in earlier training attempts).
**Not diagnosed further this round** — the natural next check (not yet
run) is the same full-scale scale, warm-start, and composition with the
count block stripped (mirroring the validation-scale `scale-control`
ablation that cleanly resolved the earlier confound) to see whether this
B73-specific collapse is present without the count feature too, which
would point away from count entirely and toward something in the retrain
recipe itself.

### Verdict, updated

The validation-scale round's cautiously positive read (count helps
0.1x-1.0x once isolated from a composition confound) **does not
generalize cleanly to full scale** — at minimum, this specific full-scale
run introduced a severe, B73-specific regression that swamps any
depth-confidence signal for the affected samples. Genuinely unresolved:
whether this is a count-feature problem, a general full-scale-retrain
instability (independent of count), or a data-realization fluke specific
to this run's `--seed 401`/`402`. **Not promoted. Do not use
`diploid-indel-v3-readcount-fullscale` for anything.**

### Next steps (not yet done)

1. **Full-scale no-count control** — same scale/composition/warm-start,
   count block stripped (`maize_v3_readcount_sliced_nocount.npy`'s
   full-scale sibling doesn't exist yet) — directly tests whether the
   B73-specific collapse is count-caused or a general full-scale-retrain
   issue. The single highest-value next experiment.
2. If the control also collapses on B73: this is unrelated to the count
   feature, points at something about this retrain recipe at this scale
   (or this specific data realization) more generally — investigate
   independent of the read-count-feature hypothesis.
3. If the control does NOT collapse on B73: isolates the problem to the
   count feature specifically at full scale — worth checking whether
   `count_proj`'s learned weights are unusually extreme/unstable, or
   whether the simulator's count-derivation simplification (site-uniform
   count magnitude, PLAN.md's documented simplification) interacts badly
   with founder index 0's simulated-vs-real asymmetry specifically.
4. **Genotype-level confirmation** — the plan's Verification item 5
   (genome-wide SNP+RefCall rescore) still not run for any of these
   checkpoints; remains required before any promotion decision regardless
   of how the above resolves.
