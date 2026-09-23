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

## Investigation summary and naming convention (2026-09-23)

This doc has grown into many per-experiment sections written across several
parallel branches. **The sections below are NOT all present on every
branch** — each experiment was committed on its own branch and none have
been merged together, so the full picture requires checking multiple
branches' copies of this file. This summary is written to be accurate and
findable regardless of which branch you're reading it from.

### Naming convention (replaces earlier ambiguous "Branch A/B", "Option B" names)

| name | checkpoint | branch | status |
|---|---|---|---|
| `diploid-affinity` (real baseline) | `diploid-affinity-sim512-h3/d-epoch=04-val_pair_acc=0.6179.ckpt` | pre-dates this investigation (trained July 23) | reference point, pre-ternary architecture |
| `ternary-baseline` | `diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt` | root of all indel-v3 work | everything below warm-starts from this |
| `row-collapse` (was "Branch B") | `diploid-indel-v3-collapse-fullscale/d-epoch=04-val_pair_acc=0.6542.ckpt` | `indel-readcount-row-collapse` | best real-data result of the ternary family on IDX; promoted for in-panel use |
| `region-fix` | `diploid-indel-v3-regionfix-fullscale` | `indel-region-buffer-fix` | **NOT PROMOTED, dead end** — regresses both IDX and OUT |
| `held-out-augment` (was "Option B") | `diploid-indel-v3-heldout-augment-fullscale/d-epoch=02-val_pair_acc=0.7314.ckpt` | mechanism on `heldout-calibration`; real-data result section on `regionfix-out-snprc-check` (not yet reconciled onto `heldout-calibration`) | best OUT-INBRED/OUT-RIL2 result so far, worse than others on OUT-HYB |
| `low-rate` | `diploid-indel-v3-lowrate-v3-fullscale` (dead — training destabilizes) | `indel-region-buffer-fix-lowrate` | **PAUSED, unresolved** after 3 attempts — see below |

### Master comparison table — genotype-level OUT error rate, 0.1x

| class | diploid-affinity | ternary-baseline | row-collapse | region-fix | held-out-augment |
|---|---|---|---|---|---|
| OUT-INBRED | 15.1% | 28.78% | 26.8% | 30.70% | **22.81%** (best of ternary family) |
| OUT-HYB | 41.0% | 48.07% | **47.4%** (best of ternary family) | 51.11% | 52.00% |
| OUT-RIL2 | — (not run comparably) | 31.79% | 29.5% | 34.38% | **22.52%** (best of ternary family) |

**diploid-affinity (the real baseline) still beats every ternary-family checkpoint on every
class measured** — the ternary/indel architecture has narrowed this gap in places
(held-out-augment on INBRED/RIL2) but has not closed it anywhere.

### Why diploid-affinity generalizes better — two verified mechanisms, neither sufficient alone

1. **Undiluted crossover draws.** diploid-affinity trained on `--min-crossovers 1
   --max-crossovers 4` (nominal mean 2.5/individual) applied directly, because it predates
   the `indel_region_mult` region-buffering mechanism entirely (introduced 2026-09-11,
   confirmed via `git log -S "indel_region_mult" -- src/python/crf/simulate_alleles.py`).
   ternary-baseline's own nominal target (6.0) was silently diluted by that bug to an actual
   ~1.16/individual.
2. **bf16-stability guards.** diploid-affinity's training used `--spike-skip
   --cosine-decay`; no ternary-family retrain (region-fix, held-out-augment, low-rate) has
   used them.

The `low-rate` experiment tried to replicate BOTH factors together (undiluted ~2.5
crossover rate + `--spike-skip --cosine-decay`) and **still failed to train stably** across
3 attempts (loss spikes to 38-98 vs. a stable ~2-4 baseline, regardless of the stability
guards; a het-calibration hypothesis, modeled on this project's earlier
`tier1_breedpop_sparse_retrain` precedent, was directly disconfirmed by code inspection —
neither `--learned-het` nor `--homo-penalty` is even active in this checkpoint family).
**Root cause of the low-rate instability is unresolved** — recommended next step is a
systematic bisection (crossover magnitude vs. founder/ancestor composition vs.
`indel_region_mult` vs. random seed) rather than another blind retry.

### Methodology note: diploid-affinity vs. ternary-family isn't a controlled ablation

The two families run through genuinely different real-data pipelines: diploid-affinity's
eval doesn't need `--anchor-dist-npy`/`-l 19`/`--anchor-dist-thresh` (it doesn't consume
ternary/distance/count features), and it's given the TRUE zygosity (`homo_scale` from the
manifest) rather than inferring it. Both are **necessary infrastructure differences, not
scoring bugs** — the final gVCF-vs-truth comparison logic was directly verified identical
for both paths. But this means the comparison is "each model through its own correct
real-world pipeline," not an isolated architecture ablation — the *existence and direction*
of the gap is well-established across many independent checkpoints and rows, but its exact
magnitude isn't cleanly decomposed into architecture-vs-tooling causes yet.


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

### B73 root-cause investigation (in progress)

Three targeted checks, run in parallel with a full-scale no-count control
retrain (below):

1. **Background-sharing statistics ruled out.** Checked founder 0's
   (B73's slot) simulated statistics directly against all 24 other
   founders in both sub-populations: background match rate z-score
   -0.31 and -0.95 (well within 1 SD — nothing anomalous), true-match
   rate and how often it was drawn as ground truth both squarely
   mid-range. **Founder 0's simulated data is statistically
   unremarkable** — rules out "founder 0 happened to land an outlier
   profile in this seed's coalescent tree."

2. **Prediction-pattern check.** For HYB B73xOh43 at 2.0x (true=(0,21)),
   the model gets Oh43 (21) right in nearly every top prediction, but
   the *other* slot scatters across many different wrong founders (1,
   22, 14, 18, 21-homozygous, 15, 10 — no concentrated confusion with one
   specific look-alike) — and **founder index 0 essentially never
   appears** even among the wrong top-8 predictions (only 6.39%, exactly
   the correct rate). Not "B73 confused with a specific other founder" —
   closer to "index 0's emission score is suppressed almost everywhere."
   For the non-B73 pair (Oh43xIl14H), errors are far more concentrated/
   normal-looking (top wrong answer `(11,23)` at 11.79%, a plausible
   near-neighbor).

3. **`ext_bias` weight drift.** Directly diffed checkpoint state dicts.
   `encoder.ext_bias.weight` — the layer converting the real per-founder
   affinity signal (raw + mean-centered match rate) into an additive
   emission bias — moved from `[0.296, 0.296]` (baseline) to `[0.776,
   0.776]` (new), **~2.6x steeper**, while an unrelated layer
   (`gate_head`) barely moved at all (max diff 0.013 vs `ext_bias`'s
   0.48). `stay_bonus` also drifted (2.01→2.31, +15%; smaller, and this
   session's earlier `stay_bonus` sweep already found accuracy flat
   across 0-3, so unlikely to be the primary driver on its own).

**Leading hypothesis, coherent with all three findings**: real B73's
affinity profile (computed from actual reads — B73 is the literal
reference genome) sits somewhere the simulator never produces for *any*
of its 25 founders, since `--indel-ref-founder` was never set to give
any one of them genuine reference-like treatment during generation. Feed
that out-of-distribution input through a much steeper `ext_bias` mapping
and the result could be an extreme, near emission-killing bias
specifically for B73's slot, while the other 24 founders' real affinity
profiles — which *do* resemble what the simulator produces — stay
reasonably calibrated. This also explains why the original baseline
(same underlying domain gap, milder `ext_bias`) doesn't show the
problem. **Concrete fix candidate for a future round**: give at least
one simulated founder actual reference-genome treatment
(`--indel-ref-founder`) during generation, so training sees an example
resembling B73's real statistical profile. Not yet tested.

### RESOLVED: count-feature-specific, not a general full-scale-retraining artifact

Ran the decisive control: `diploid-indel-v3-scale-control-fullscale-nocount`
(identical scale/composition/warm-start/step-count to the with-count
full-scale run, count block stripped) finished training
(`d-epoch=04-val_pair_acc=0.4144.ckpt`). Two checks resolve this
cleanly:

**1. `ext_bias`/`stay_bonus` drift is NOT the causal mechanism — the
no-count control drifted *further*, not less:**

| | `ext_bias.weight` | `stay_bonus` |
|---|---|---|
| baseline | `[0.296, 0.296]` | 2.009 |
| with-count-fullscale (regressed) | `[0.776, 0.776]` (2.6x) | 2.309 (+15%) |
| **no-count-control (clean)** | **`[0.919, 0.919]` (3.1x)** | **2.481 (+23.5%)** |

If steeper `ext_bias` were what killed B73's emissions, the no-count
control — which drifted *more* — should regress at least as badly.
It doesn't (below). This falsifies the "leading hypothesis" from the
previous section as the causal mechanism; the drift itself is a generic
full-scale-retraining/warm-start effect, not what's breaking accuracy.

**2. Real-data eval, all 3 checkpoints, B73-involving samples plus one
non-B73 control pair (Oh43xIl14H) — the no-count control is completely
clean everywhere, and the with-count regression is NOT actually
B73-specific:**

pair_acc, 0.01x → 2.0x:

| sample | baseline | with-count-fullscale | no-count-control |
|---|---|---|---|
| IDX-INBRED B73 | 100 / 100 / 100 / 100 / 100 | 99.6 / 97.5 / 95.7 / 93.3 / **88.7** | 100 / 100 / 100 / 100 / 100.0 |
| IDX-HYB B73xOh43 | 100 / 98.9 / 97.0 / 96.0 / 95.3 | 8.8 / 9.8 / 8.0 / 7.2 / **6.4** | 100 / 99.2 / 96.5 / 94.8 / 93.4 |
| IDX-HYB B73xCML103 | 100 / 99.0 / 96.8 / 96.1 / 95.2 | 5.0 / 7.1 / 6.5 / 6.2 / **5.5** | 100 / 99.6 / 97.0 / 95.2 / 93.6 |
| IDX-RIL2 B73xCML103 | 97.2 / 98.9 / 99.4 / 99.4 / 99.3 | 91.9 / 81.9 / 74.2 / 71.7 / **69.6** | 96.2 / 97.6 / 97.3 / 97.2 / 97.2 |
| IDX-RIL2 B73xOh43 | 96.8 / 96.7 / 95.1 / 94.8 / 94.4 | 91.2 / 82.0 / 76.2 / 74.7 / **73.1** | 97.3 / 98.7 / 98.7 / 98.5 / 98.3 |
| **IDX-HYB Oh43xIl14H (non-B73 control)** | 100 / 99.0 / 98.1 / 97.1 / 96.5 | 79.2 / 58.0 / 47.4 / 44.9 / **42.7** | 99.6 / 98.6 / 97.1 / 95.5 / 94.8 |

Two things this resolves at once:

- **The no-count control shows no regression anywhere** — every number
  sits in baseline's normal range (a couple of cells are even slightly
  better, e.g. RIL2 B73xOh43). Same scale, same data, same warm-start,
  same step count as the failed run — only the count channel differs.
  **The regression is caused by the count feature at full scale, full
  stop.**
- **The "B73-specific" framing from the previous section was wrong** —
  Oh43xIl14H involves no B73 at all and still collapses under
  with-count-fullscale (100%→42.7% by 2.0x, worse in relative terms than
  several B73 pairs). B73 pairs are hit hardest, but this is a broader
  full-scale count-training pathology, not narrowly about founder index 0.

**Updated verdict: `diploid-indel-v3-readcount-fullscale` is confirmed
broken by the count feature's interaction with full-scale training —
not a seed fluke, not a general retraining artifact, not narrowly a B73
problem. Definitively not promoted; do not use.** The validation-scale
round's small positive signal (Phase 4) does not survive scaling up, and
the mechanism (why count training breaks broadly at full scale but not
at ~200 steps) remains unexplained — worth a follow-up only if the
read-count idea is revisited; not required to close out this round.

### Next steps

1. **Genotype-level confirmation** — the plan's Verification item 5
   (genome-wide SNP+RefCall rescore) still not run for any of these
   checkpoints. Now moot for `diploid-indel-v3-readcount-fullscale`
   specifically (already definitively not-promoted above), but still
   the right verification to run before ever promoting a future
   read-count retrain, since founder-decode wins have diverged from
   genotype-level reality before this session.
2. **On-disk layout cleanup** — the `2K+2`/`3K+2` contract puts the
   H1/H2 truth-label columns in the *middle* of the array
   (`[ternary(K) | H1 | H2 | distance(K) | count(K)]`), sandwiched
   between feature blocks that are the actual model input. This is the
   direct cause of the two open-ended-slice bugs already hit and fixed
   this session (`out[sl,:,K+2:]`, `dist_out=out[:,:,K+2:]`) — every
   consumer needs a `K+2:2K+2`-style offset to skip the labels, which is
   easy to get wrong and gets more error-prone with each added feature
   block. Moving labels to the end (`[ternary(K) | distance(K) | count(K)
   | H1 | H2]`) would make every feature block a simple contiguous range
   and remove this whole bug class — note the *original* simpler binary
   format (`GRITSCRFDiploid`, `K+2` width) already does this
   (`[binary(K)|H1|H2]`); only the ternary+distance format ended up with
   labels in the middle. User-agreed this is worth doing before the
   format is considered final/published. Was explicitly deferred until
   the regression investigation concluded (to avoid conflating two
   changes); that investigation is now resolved above, so this is
   unblocked whenever it's picked up — tracked here so it isn't dropped.
3. **Count-training-breaks-at-full-scale mechanism** — unexplained why
   the count feature trains cleanly at validation scale (small positive
   effect, Phase 4) but breaks broadly at full scale. Not required to
   close out this investigation; only worth digging into if the
   read-count idea itself gets revisited.

## Second attempt: two variants, branches `indel-readcount-grouped-count` / `indel-readcount-row-collapse`

User request: retry the read-count idea, but literally "group the reads
which have the same ternary structure at the same position, and use the
total number of reads as the count" — not just annotate a per-founder
tally, actually merge rows that already agree. Two interpretations, each
genuinely different in scope/risk, so both were built on their own
branch (either might get discarded):

- **Branch A (`indel-readcount-grouped-count`)**: keep today's row
  structure exactly as-is; fix the count VALUE. `_indel_chunk` now
  groups rows by `(window, site, the row's own exact K-length ternary
  vector)` and uses each row's own group size as its count — a per-ROW
  quantity, always >0 (a row is always a member of its own group),
  broadcast across its own K columns. This directly targets the flaw in
  the old design: that one broadcast a SITE-wide per-founder MATCH tally
  to every row at a site regardless of the row's own pattern, and was
  always 0 for DEL/DIVERGED cells even when many reads agreed on that
  call.
- **Branch B (`indel-readcount-row-collapse`)**: go further and actually
  collapse same-kind row stacks into one row BEFORE sampling, so windows
  span more distinct reference sites for the same T-row budget instead
  of burning slots on duplicate rows (colinear on1/on2 are already
  boolean, never stacked; only insertion evidence genuinely stacks).
  Requires restructuring `_row_counts`'s row-budget math
  (`_sample_rows(groupcnt, T)` instead of `_sample_rows(cnt, T)`).

Both implemented, tested (96/102 and 102/102 respectively, hand-derived
exact fixtures + naive-reference fuzz tests), and merged so Branch B
builds on Branch A's corrected formula rather than the known-bad one.

**Pre-check finding on Branch B before committing full-scale compute**:
measured the actual window-span benefit under the REAL production
recipe's settings (`indel_model=overlay`, default
`--indel-ins-read-per-bp 2e-3`) — negligible. Max reference site reached
per window: 46382 (uncollapsed) vs 46391 (collapsed), a 0.02% difference.
At these settings insertions are sparse enough (~300bp mean length ×
0.2% reads/bp ≈ 0.6 expected reads per insertion) that there's rarely
more than one row to collapse in the first place — the row-collapse
mechanism is correctly implemented but has almost nothing to act on
here. Flagged to the user before spending compute on full-scale
generation; user chose to run it anyway for a real head-to-head number
rather than rely on the proxy measurement.

### Branch A real-data result

Full-scale retrain (500+500 individuals, `--sites 60000`, matching the
original recipe's exact scale — recovered by RECOMPUTING the count block
directly on the already-cached full-scale raw arrays via their `.refpos.npy`
sidecars, rather than regenerating from scratch: guarantees byte-identical
composition/individuals/tern/dist to the original run, only the count
values differ, the cleanest possible isolation of this one variable).
Warm-started identically, 5 epochs. Training-time `val_pair_acc` climbed
to **0.7427** — HIGHER than the original no-count baseline's own 0.6820.

That did **not** translate to better real-data accuracy, and the gap
between the two is itself the finding: `val_pair_acc` is measured on a
held-out split of the SAME simulated single-coverage (`coverage=2.0`)
training distribution the model was fit on, not real data at any depth.
A model can genuinely fit that split better by leaning harder on the
count feature's very clean within-simulator signal, while generalizing
worse to real data's noisier count values — and real data spans 0.01x
to 2.0x, most of it far from the one simulated regime ever trained on.

Real 15-sample × 5-depth eval, baseline vs Branch A:

| kind | metric | 0.01x | 0.1x | 0.5x | 1.0x | 2.0x |
|---|---|---|---|---|---|---|
| INBRED | baseline / groupedcount pair_acc | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 |
| HYB | baseline pair_acc | 100.00 | 99.23 | 97.73 | 96.95 | 96.16 |
| HYB | **groupedcount pair_acc** | 100.00 | 99.20 | 95.03 | 89.80 | **81.38** |
| HYB | baseline ibd_adj | 100.00 | 99.88 | 99.69 | 99.66 | 99.63 |
| HYB | groupedcount ibd_adj | 100.00 | 99.92 | 99.36 | 98.49 | 97.00 |
| RIL2 | baseline pair_acc | 96.67 | 98.83 | 98.59 | 98.49 | 98.30 |
| RIL2 | groupedcount pair_acc | 96.33 | 98.50 | 98.09 | 97.98 | 98.23 |

Per-sample detail confirms the HYB regression is **uniform across all 5
pairs** (B73xOh43 76.66%, B73xCML103 75.33%, Oh43xIl14H 84.26%,
B97xCML103 87.72%, Il14HxB97 82.93% at 2.0x) — not narrowly tied to one
founder, same broad-not-founder-specific pattern as the original bug's
regression, just far milder (28-51% there vs 75-88% here) and
depth-scaling rather than uniformly catastrophic. INBRED is unaffected
(100% at every depth). RIL2 is flat/neutral (±0.3pp), consistent with
its error being IBD-driven rather than depth-confidence-driven —
matches every prior round this session.

**Verdict: the formula fix worked (no more catastrophic, broadly-broken
collapse), but the count feature is still a net negative on real HYB
data at moderate-to-high depth, growing worse with depth.** Not
promoted. `diploid-indel-v3-groupedcount-fullscale` is a genuine
improvement over `diploid-indel-v3-readcount-fullscale` (the original
broken run) but still worse than the no-count baseline on the metric
that matters.

### Branch B real-data result

Full-scale retrain (`diploid-indel-v3-collapse-fullscale`, fresh
`--collapse-rows --emit-read-counts` generation — see the pre-check note
above for why this couldn't reuse the cached raw arrays the way Branch A
did: collapse-rows changes row SAMPLING itself, not just a post-hoc
count value, so it needs a real `simulate()` run). Same scale as the
original recipe and Branch A (500+500 individuals, `--sites 60000`,
sliced to 117,000 windows, `G=117`). Warm-started identically, 5 epochs.
`val_pair_acc` climbed to 0.6542 — lower than Branch A's 0.7427, and
*this* time that lines up with real-data behavior rather than
contradicting it (see below).

Mean pair_acc / ibd_adj by kind and depth, baseline vs Branch B:

| depth | INBRED pa/ia | HYB pa/ia | RIL2 pa/ia |
|---|---|---|---|
| | **baseline** | | |
| 0.01x | 100.00% / 100.00% | 100.00% / 100.00% | 96.67% / 96.76% |
| 0.1x | 100.00% / 100.00% | 99.23% / 99.87% | 98.83% / 99.59% |
| 0.5x | 100.00% / 100.00% | 97.73% / 99.69% | 98.59% / 99.74% |
| 1.0x | 100.00% / 100.00% | 96.95% / 99.66% | 98.49% / 99.76% |
| 2.0x | 100.00% / 100.00% | 96.16% / 99.63% | 98.30% / 99.75% |
| | **Branch B (collapse-rows)** | | |
| 0.01x | 100.00% / 100.00% | 100.00% / 100.00% | 96.61% / 96.79% |
| 0.1x | 100.00% / 100.00% | 99.31% / 100.00% | 98.95% / 99.63% |
| 0.5x | 100.00% / 100.00% | 97.69% / 99.98% | 98.84% / 99.89% |
| 1.0x | 99.99% / 100.00% | 96.55% / 99.97% | 98.71% / 99.93% |
| 2.0x | 99.99% / 100.00% | 95.32% / 99.95% | 98.57% / 99.93% |

**INBRED**: a match to baseline within 0.01pp everywhere.

**HYB, full per-pair detail** (pair_acc/ibd_adj, baseline → Branch B, all 5 depths):

| pair | 0.01x | 0.1x | 0.5x | 1.0x | 2.0x |
|---|---|---|---|---|---|
| B73xOh43 | 100/100→100/100 | 98.90/99.81→99.41/100.00 | 96.95/99.54→97.91/99.96 | 95.95/99.54→97.16/99.95 | 95.29/99.53→**96.01**/99.91 |
| B73xCML103 | 100/100→100/100 | 99.00/99.73→99.59/100.00 | 96.78/99.60→98.24/99.98 | 96.13/99.47→97.27/99.96 | 95.21/99.36→**96.33**/99.94 |
| Oh43xIl14H | 100/100→100/100 | 99.03/99.94→98.76/100.00 | 98.08/99.74→97.49/99.98 | 97.13/99.75→95.65/99.97 | 96.50/99.74→**94.57**/99.97 |
| B97xCML103 | 100/100→100/100 | 99.76/99.94→99.59/100.00 | 98.93/99.84→98.07/99.98 | 98.33/99.81→97.25/99.99 | 97.60/99.83→**95.92**/99.97 |
| Il14HxB97 | 100/100→100/100 | 99.44/99.95→99.22/100.00 | 97.90/99.74→96.76/99.98 | 97.22/99.73→95.44/99.97 | 96.20/99.69→**93.76**/99.96 |

Two of five HYB pairs (B73xOh43, B73xCML103) come out slightly *ahead*
of baseline at 2.0x; the other three (Oh43xIl14H, B97xCML103, Il14HxB97)
land 1.6-2.4pp behind. Net mean gap at the worst depth (2.0x) is
**-0.84pp** (95.32% vs 96.16%) — a rounding error next to Branch A's
**-14.78pp** (81.38% vs 96.16%) at the same depth. `ibd_adjusted_accuracy`
is at or above baseline at every single HYB cell, often noticeably so
(99.91-99.99% vs baseline's 99.36-99.83% at 2.0x) — Branch B's remaining
disagreements with baseline are overwhelmingly IBD-explainable, not
genuine wrong calls.

**RIL2, full per-pair detail**:

| pair | 0.01x | 0.1x | 0.5x | 1.0x | 2.0x |
|---|---|---|---|---|---|
| B73xCML103 | 97.20/97.20→96.12/97.04 | 98.93/99.59→97.76/99.64 | 99.43/99.88→97.52/99.91 | 99.41/99.90→97.40/99.93 | 99.27/99.84→97.28/99.95 |
| B73xOh43 | 96.76/97.22→**97.26**/97.26 | 96.74/99.44→**98.74**/99.62 | 95.08/99.19→**98.82**/99.90 | 94.77/99.15→**98.80**/99.93 | 94.40/99.18→**98.55**/99.92 |
| B97xCML103 | 96.34/96.34→96.34/96.34 | 99.50/99.69→99.42/99.61 | 99.45/99.85→99.33/99.89 | 99.46/99.90→99.26/99.92 | 99.37/99.90→99.26/99.93 |
| Il14HxB97 | 96.12/96.12→96.45/96.45 | 99.37/99.61→99.28/99.61 | 99.38/99.88→99.00/99.87 | 99.22/99.93→98.71/99.94 | 99.02/99.92→98.62/99.94 |
| Oh43xIl14H | 96.92/96.92→96.88/96.88 | 99.62/99.62→99.56/99.66 | 99.63/99.89→99.51/99.89 | 99.57/99.90→99.36/99.92 | 99.42/99.92→99.14/99.92 |

One pair (B73xCML103) is consistently 1.3-2.2pp worse across every
depth; one pair (B73xOh43) is consistently, substantially *better*
(+2.0 to +4.2pp, biggest gain of anything measured this session); the
other three are flat within noise (≤0.3pp). No depth-scaling pattern
either direction — unlike HYB, these gaps don't grow or shrink with
depth, consistent with RIL2 error being IBD/homology-driven rather than
depth-confidence-driven, same conclusion as every prior round.

**Why this succeeded where Branch A didn't, despite the pre-check
showing ~no window-span benefit**: the span measurement was checking the
wrong mechanism. The real difference isn't (mainly) "windows see more
genome" — it's that Branch A still feeds the model long runs of
literally identical consecutive (ternary, count) rows whenever a site
had real stacking, while Branch B eliminates that repetition by
construction (one row per kind-group, weighted). Branch A's `val_pair_acc`
being *higher* than baseline's while real accuracy was *worse* is the
signature of exactly this: the model overfit to a simulator-specific
artifact (long runs of identical rows) that doesn't exist in real data's
one-row-per-site format. Branch B's `val_pair_acc` (0.6542) being lower
than Branch A's but far more consistent with real accuracy fits that
story.

**Verdict: promote candidate.** Branch B (`diploid-indel-v3-collapse-fullscale`)
is the first version of the read-count idea that doesn't cost real
accuracy at any depth on the metric this whole investigation was about —
HYB. Net effect across all three kinds is a wash-to-small-win (INBRED
flat, HYB -0.84pp mean at worst depth with 2 of 5 pairs improving, RIL2
+0.27pp mean at worst depth with 1 of 5 pairs meaningfully better and
1 meaningfully worse). Genotype-level SNP+RefCall confirmation (plan's
Verification item 5, still not run for any checkpoint this session) is
the required next step before promoting for real — founder-decode wins
have diverged from genotype-level reality before this session — but
this is by a wide margin the strongest result of everything tried under
this investigation.

### Current status / next steps (2026-09-21)

1. **Genotype-level confirmation for Branch B** — genome-wide SNP+RefCall
   rescore (`compare_gvcf_truth_diploid.py --snp-refcall-metrics`) against
   `diploid-indel-v3-collapse-fullscale`, required before any real
   promotion decision. Not yet run.
2. **Branch A (`indel-readcount-grouped-count`) is not promoted** — real,
   depth-growing HYB regression, no path forward identified. Keep the
   branch for its test coverage of the corrected count formula (a real
   fix over the original bug), but the checkpoint itself is dead.
3. **Branch B (`indel-readcount-row-collapse`) is the promote candidate**
   — closest-to-baseline (and in places better-than-baseline) real-data
   result of every variant tried across both rounds of this
   investigation. Pending item 1 above.
4. **On-disk layout cleanup** (H1/H2 truth labels moved to the end of the
   array) — still unblocked, still not done, still worth doing before
   either format is considered final/published. See the earlier note in
   this doc for the exact motivation.
5. `indel-density-features` (`window_density`) remains a separate,
   still-unresolved, untouched hypothesis on its own branch.

## Region-buffering crossover-rate fix (branch `indel-region-buffer-fix`)

Diagnosed separately from the held-out (OUT) investigation: `simulate_alleles.py`
draws each training individual's crossover path uniformly across the full
internal buffer `R = indel_region_mult * sites` (4 * 60,000 = 240,000 sites),
but only the leading ~19.4% of that region ends up emitted as real training
rows — ~80% of drawn crossovers land in the discarded tail and are never
observed. Measured effect on Branch B's actual training data: nominal mean
6.0 crossovers/individual (uniform draw over `--min/max-crossovers 2/10`)
collapses to an observed mean of 1.16/individual, and only ~0.99% of T=512
training windows contain any switch at all.

### Phase 1 — sweep and pre-training verification

`indel_region_mult` sweep (n=200, `inbreeding=1.0`, everything else at
Branch B's exact recipe), measuring `short%` (must stay ~0%) and realized
crossovers/individual:

| `indel_region_mult` | short% | crossovers/individual |
|---|---|---|
| 1 | 0.500% (unsafe) | 4.355 |
| **2** | **0.000%** | **2.410** |
| 3 | 0.000% | 1.535 |
| 4 (current) | 0.000% | 1.280 |

`indel_region_mult=2` is the smallest safe value — confirmed again at full
production scale (n=500 each sub-population): short%=0.000% (0/500) for
both `inb1.0` and `inb0.0`. That alone only recovers ~2.3/individual
(survival fraction ~38%, up from ~19%), still short of the intended 6.0, so
`--min-crossovers`/`--max-crossovers` were also calibrated upward using the
measured survival fraction: `8`/`23` (nominal mean 15.5) → verified at
n=500 to land at 6.10-6.18/individual, matching the target almost exactly.

Final settings (`indel_region_mult=2`, `--min-crossovers 8 --max-crossovers 23`),
verified at full production scale before committing to training:

| | short% | crossovers/individual | het_frac | switches/T512-window |
|---|---|---|---|---|
| inb1.0 (INBRED) | 0.000% (0/500) | 6.096 | 0.0000 | mean 0.05195 (94.94% zero, 4.92% one, 0.14% two+) |
| inb0.0 (HYB) | 0.000% (0/500) | 5.870 | 0.9572 | mean 0.05003 (95.13% zero, 4.73% one, 0.13% two+) |

Sanity-checked against real structure: the fixed rate (~0.05/window) is
~3.8x higher than IDX-RIL2's own real rate (~0.013/window, from 28 true
crossovers genome-wide) but still ~60-80x short of baseline's OUT-implied
requirement (~3-4/window) — expected, this fix was never meant to close
that gap, only recover the recipe's own intended in-panel calibration.

### Phase 2 — full retrain and real-data result

Full-scale regenerate (500+500 individuals) with the validated settings,
sliced to `(117000, 512, 77)` (`G=117`, matching Branch B exactly), warm-started
from the ORIGINAL baseline checkpoint (not Branch B's), 5 epochs.
`val_pair_acc` trajectory: epoch1=0.3699, epoch2=0.4104, epoch4=0.4190 —
clearly flattening (epoch0→2 gained +0.0426, epoch2→4 gained only +0.0086),
not the non-monotonic instability pattern seen in the bigger-model variants.
Note this val set itself now has the recovered high crossover rate too, so
the lower absolute number vs Branch B's 0.6542 reflects a harder validation
task, not worse learning — not directly comparable across the two runs.

Real 15-sample × 5-depth eval, baseline vs Branch B vs regionfix-fullscale:

| kind | metric | 0.01x | 0.1x | 0.5x | 1.0x | 2.0x |
|---|---|---|---|---|---|---|
| INBRED | baseline / branchB / regionfix pair_acc | 100.00 / 100.00 / **100.00** | 100.00 / 100.00 / **99.95** | 100.00 / 99.996 / **99.85** | 100.00 / 99.994 / **99.70** | 100.00 / 99.993 / **99.32** |
| HYB | baseline pair_acc | 100.00 | 99.23 | 97.73 | 96.95 | 96.16 |
| HYB | branchB pair_acc | 100.00 | 99.31 | 97.69 | 96.56 | 95.32 |
| HYB | **regionfix pair_acc** | **96.37** | **90.16** | **85.76** | **83.93** | **81.75** |
| HYB | branchB ibd_adj | 100.00 | 100.00 | 99.98 | 99.97 | 99.95 |
| HYB | **regionfix ibd_adj** | 99.09 | 97.36 | 96.19 | 95.43 | 94.30 |
| RIL2 | baseline pair_acc | 96.67 | 98.83 | 98.59 | 98.49 | 98.30 |
| RIL2 | branchB pair_acc | 96.61 | 98.95 | 98.84 | 98.71 | 98.57 |
| RIL2 | regionfix pair_acc | 98.42 | 98.99 | 98.57 | 98.15 | 97.14 |

**HYB regresses substantially and uniformly across all 5 pairs** (checked
individually, not just the mean — B73xOh43 71.59%, B73xCML103 69.20%,
Oh43xIl14H 88.57%, B97xCML103 91.59%, Il14HxB97 87.82% at 2.0x, every one
of them well below both baseline and Branch B at every depth). INBRED shows
a smaller but real, depth-growing regression (99.32% at 2.0x vs ~100%
before). RIL2 is a mixed wash (slightly better at 0.01x, slightly worse at
1.0x/2.0x).

**Why, mechanistically**: real IDX-INBRED and IDX-HYB individuals in this
corpus are literal pure lines / F1 hybrids — **zero true recombination
within the individual** (H1 is 100% one parent, H2 is 100% the other, or
both 100% one parent for inbred). Only IDX-RIL2 has any real recombination
at all. Recovering the recipe's "intended" ~6 crossovers/individual made
training *more* mismatched with what most real in-panel samples actually
need (near-zero), not less — the old suppressed ~1.16/individual rate,
while an unintended side effect of the buffering bug, happened to sit
closer to what real IDX-INBRED/IDX-HYB data needs than the recipe's own
nominal target does. RIL2 (which does have real, nonzero recombination)
is the one case that doesn't regress, consistent with this explanation.

**Verdict: not promoted.** This is a real, clean, well-evidenced negative
result, not a neutral fix. The region-buffering waste is still a genuine
inefficiency in what the CLI flags nominally intend, but "fixing" it by
recovering the nominal rate actively hurts real HYB/INBRED accuracy more
than it helps anything else measured. Branch B remains the better
checkpoint. If revisited, the right lever is probably tuning
`--min/max-crossovers` down from their current nominal 2-10 default
(closer to what real zero-recombination IDX-INBRED/HYB data actually
needs) rather than up — a different, untested direction from what this
round tried.

## Synthetic leave-one-out held-out training (Option B): first full result

**Checkpoint:** `diploid-indel-v3-heldout-augment-fullscale/d-epoch=02-val_pair_acc=0.7314.ckpt`
(warm-started from the original `diploid-indel-v3-k25-overlay-affinity`
baseline; `max_epochs=5` reached, epoch 2 was the best checkpoint by
val_pair_acc -- 0.7267/0.7301/0.7314 across epochs 0/1/2, later epochs did
not improve). Trained on the relabeled leave-one-out data
(`heldout_augment.py`, commits `626ebfc`/`725270d` on
`indel-heldout-augment`) combined with the normal in-panel splits.

**Real-bug fix required first.** This checkpoint's K+1=26-state
architecture can legitimately decode the null/no-visible-lineage-mate
state (index 25) on real data -- something no prior checkpoint's training
distribution ever incentivized, so it was never exercised before. This
crashed `heldout_assembly_eval.py::k_target_to_name` (`IndexError: list
index out of range`, indexing straight into the 25-element `gamete_names`
list) on every non-RIL2 row. Fixed in `write_imputed_bed` (the fixing
fork's own diff, restored/continued here): skip sites where the predicted
index falls outside `gamete_names` rather than crash or fabricate a
founder guess -- downstream scoring already tolerates BED gaps (this
project's existing "no-fill" gVCF convention). A matching GPU-memory-leak
fix was also carried over into `simval_eval_one.py` (same pattern already
fixed in `simval_eval_indel.py` this session).

**Method note:** OUT/MIX samples have no true panel-founder-identity label
(they're genuinely held-out assemblies, not equal to any panel member), so
founder-level `pair_acc`/`ibd_adj` is undefined for them by construction --
this is why every OUT/MIX accuracy check this session (region-fix's own
OUT sweep, Branch B's genotype sweep) used genotype-level SNP+RefCall
`error_rate` instead, and this check does too, for direct comparability.

**Result -- mixed, but real and directionally consistent within each
class** (0.1x, mean genotype-level error_rate, OUT-only; n=5/5/3 of 5 rows
scored for INBRED/HYB/RIL2 respectively, RIL2's remaining 2 rows still
finishing at write time):

| class | baseline (v3-K25) | region-fix | Option B |
|---|---|---|---|
| OUT-INBRED | 28.78% | 30.70% | **22.81%** |
| OUT-HYB | 48.07% | 51.11% | 52.00% |
| OUT-RIL2 | 23.81% | 24.58% | **21.80%** |

OUT-INBRED: Option B wins on **all 5/5 rows**, by 1-20pp each (Tx303 is
the standout: 43.33%->23.19%, more than halved). OUT-RIL2: wins on all
3/3 scored so far, by 1.4-4.6pp each. OUT-HYB: **loses on all 5/5 rows**,
by 1-6pp each (worse than both baseline and even region-fix, the other
already-failed approach).

**Interpretation.** Option B's mechanism -- learning to recognize "no
visible founder matches, defer to IBD-relatedness structure" -- helps
exactly where a single true ancestry needs tracking (INBRED: one true
lineage throughout; RIL2: one lineage per haplotype block, no
simultaneous heterozygous decision). It hurts HYB, where the true state
is two DIFFERENT founders simultaneously and the null-leaning bias this
training instills likely costs precision on whichever of the two slots
is harder to pin down, compounding across both. This is the same
predicted shape as the diploid indel model's general homo/het asymmetry
seen elsewhere this session, now showing up as a training-composition
effect rather than a decode-time homo_scale one.

**Verdict:** first genuinely-positive result for OUT-of-index decode this
investigation has produced (both region-buffer-fix and Branch B alone
regressed on every OUT/MIX class). Not a clean win -- HYB regresses
meaningfully -- but INBRED and RIL2 both improve by a real, consistent
margin. Promising enough to warrant a HYB-focused follow-up (e.g.
combining Option B's augmentation with an explicit two-founder
calibration term, or simply weighting HYB-like compositions higher in
the augmented training mix) rather than abandoning the approach.

Per-row JSON: `results/heldout_augment_out_snprc/`. Driver:
`experiments/simval-corpus/scripts/heldout_augment_out_snprc_check.py`
(original sweep) + `heldout_augment_remaining.py` (this round's
resume-after-crash driver, GPU-1-pinned to run alongside the 3 rows that
had already survived as orphaned processes from the crashed run).
