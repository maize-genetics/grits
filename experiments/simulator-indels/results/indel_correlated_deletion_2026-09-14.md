# Correlated deletion across IBD-adjacent founders: calibration sweep

2026-09-14. Follow-up to `calibration_sweep_2026-09-14.md`'s identified
root cause and PLAN.md §5 item 1's "next step": today's `_indel_tracts`
draws structural variation **per LINEAGE, independently across
lineages** — two different (but possibly closely related) lineages have
zero correlation in their deletion patterns. That was confirmed to make
a heterozygous (outbred, cassava-like) individual's "either homolog
covered" rate come out too high (80.72% at the earlier doc's specific
config; 82.34% at this doc's own baseline config below) against a
~70–75% real-data target, and no single `--indel-density` value could
close that gap without overshooting the separate indel-affected-fraction
target.

## What was built

- **`_gem_partition(rng, n, M, theta, C)`** (new function,
  `simulate_alleles.py`): a STATIC (one draw per window, no genomic/
  along-T path) Ewens/GEM(theta) partition of `M` items into `C`
  groups. Reuses the exact stick-breaking construction `_gem_lineages`
  already uses for its per-segment lineage draw, but skips that
  function's segment/path-building machinery entirely. Hand-verified
  against a fixed-input trace (n=1, M=3, C=2, theta=1.0: beta=[0.3,0.7]
  → cdf=[0.37975, 1.0]; u=[0.1,0.4,0.9] → group=[0,1,1]) before being
  pinned as a test. Separately confirmed smaller theta genuinely
  concentrates mass into fewer effective groups (mean distinct groups
  used out of C=24, M=24, n=2000 windows): theta=0.3→1.99, 1.0→3.76,
  2.0→5.65, 5.0→8.98 — monotone, not just a plausible-sounding claim.
- **`simulate()`'s indel branch**: when `--indel-shared-frac > 0`, each
  window also draws a supercluster partition of the `M` SNP-sharing
  lineages via `_gem_partition(..., theta=indel_cluster_theta, C=M)`,
  then calls `_indel_tracts` TWICE — once at density
  `indel_density * (1 - indel_shared_frac)` over the lineages
  ("private"), once at density `indel_density * indel_shared_frac`
  over the SAME `[n,M,R]` slot shape but semantically indexed by
  supercluster id ("shared"). The shared component is gathered onto
  lineage-space via the existing `_gather_by_lineage` helper (the same
  mechanism that already makes two IBD founders share byte-identical
  indel content) and OR'd/summed with the private component; insertions
  inside the combined deletion mask are re-zeroed, mirroring
  `_indel_tracts`'s own internal cleanup. `indel_shared_frac=0.0` (the
  value that reproduces today's pre-existing behavior, though no longer
  the CLI default — see Verdict below) skips this entirely and
  reproduces the exact pre-change rng draw sequence — verified
  byte-identical (`data`, `.ibd`, `.refpos`, and the `true_cov` QC
  counts) against `simulate_alleles.py` as of git rev `cc26018` (this
  branch's parent tip), pinned as a golden-hash regression test.
- New CLI flags `--indel-shared-frac` and `--indel-cluster-theta`,
  threaded additively through `simulate()`'s signature.
- 8 new unit tests (`tests/python/crf/test_simulate_alleles.py`):
  hand-computed exact / shape-dtype-range / determinism / theta-
  concentration-monotonicity tests for `_gem_partition`; a golden-hash
  regression test pinning `indel_shared_frac=0.0`'s byte-identical
  output; a same-vs-different-supercluster deletion co-occurrence fuzz
  test (30 seeds) confirming lineages sharing a supercluster really do
  show higher mean deletion overlap than lineages in different
  superclusters; a sanity check that a positive shared-frac actually
  changes output. All 57 tests pass (49 pre-existing + 8 new).

## Calibration sweep

Same methodology as `calibration_sweep_2026-09-14.md`: direct calls to
`simulate()` (not the CLI), `--founders 24 --sites 8192 --windows 60
--min-crossovers 2 --max-crossovers 6 --sharing-model coalescent
--sharing-theta 4.0`, `--indel-density 2.3e-3` (today's calibrated
default) held fixed throughout — this sweep is ONLY about
`--indel-shared-frac`/`--indel-cluster-theta`, not density. Averaged
over 6 seeds (1–6) per setting, reporting mean ± std to make seed noise
visible (this scale showed ~0.3–4.3pp std across settings — noisier
than the 2026-09-14 doc's ~1.7–2.2pp, because the shared/private split
adds its own extra Poisson-draw variance on top of the base process).

### HET (outbred, cassava-like: `--inbreeding 0.0`) — the actual target

Real targets: indel-affected ~32.7%, either-covered ~72.1%.

| shared_frac | cluster_theta | either-covered | indel-affected |
|---|---|---|---|
| 0.00 (baseline) | — | 82.34% ± 0.29 | 35.93% ± 0.88 |
| 0.30 | 1.0 | 81.29% ± 0.88 | 33.36% ± 1.84 |
| 0.50 | 1.0 | 81.01% ± 2.82 | 33.25% ± 2.93 |
| 0.70 | 1.0 | 77.93% ± 4.27 | 34.56% ± 3.67 |
| 0.50 | 0.3 | 78.10% ± 2.67 | 33.64% ± 2.89 |
| 0.50 | 2.0 | 78.99% ± 3.09 | 35.85% ± 2.61 |
| 0.60 | 0.3 | 74.83% ± 2.81 | 34.35% ± 2.28 |
| 0.80 | 0.3 | 72.70% ± 2.49 | 33.96% ± 2.44 |
| 1.00 | 0.3 | 68.45% ± 5.13 | 35.18% ± 5.10 |

theta=0.3 (the most concentrated value swept) gave a clearly stronger
either-covered reduction than theta∈{1.0, 2.0} at the shared_frac=0.5
checkpoint, so the higher-shared_frac follow-up (stage 4) fixed
theta=0.3 and swept shared_frac up to 1.0. **shared_frac=0.80 lands
almost exactly on both targets simultaneously** — confirmed with a
second, independent batch of 6 fresh seeds (7–12) to rule out a
lucky first draw:

| batch | seeds | either-covered | indel-affected |
|---|---|---|---|
| original | 1–6 | 72.70% ± 2.49 | 33.96% ± 2.44 |
| confirmation | 7–12 | 72.85% ± 4.04 | 33.20% ± 4.35 |
| **pooled (n=12)** | 1–12 | **72.78%** | **33.58%** |

vs. targets either-covered 72.1% (0.68pp off) / indel-affected 32.7%
(0.88pp off) — both comfortably inside 1 standard error of the pooled
estimate. `shared_frac=1.00` (all density routed through the shared
component, no private draw at all) overshoots PAST the target in the
other direction (68.45%, below 72.1%) with much higher seed variance
(±5.13pp vs ±2.49pp at 0.80) — diminishing/reversing returns past 0.8,
not a monotone "more is always better" knob.

### INBRED (`--inbreeding 1.0`) — sanity check, not the mechanism's target

| shared_frac | cluster_theta | either-covered | indel-affected |
|---|---|---|---|
| 0.00 (baseline) | — | 64.42% ± 2.27 | 35.93% ± 0.88 |
| 0.50 | 1.0 | 67.15% ± 2.76 | 33.25% ± 2.93 |
| 0.80 (chosen default) | 0.3 | 66.11% ± 2.65 | 33.96% ± 2.44 |

By construction (h1≡h2 whenever `--inbreeding=1.0`), a single
individual's either-covered depends only on whether ITS ONE active
lineage is deleted — the shared/private split's population-level
marginal deletion rate per lineage is, in expectation, the union of two
independent Poisson processes at the same total rate, so it should not
systematically move either metric for this scenario. The observed
+2.7pp either-covered / -2.7pp affected shift at shared_frac=0.5 is the
same size as this sweep's seed noise (std 2.3–2.9pp) and moves in the
direction consistent with a slightly lower realized deletion rate at
that specific seed set, not a real effect of the mechanism — not
evidence the mechanism breaks inbred calibration.

## Verdict: the outbred/cassava-like gap closes; the inbred gap does not

For the scenario this mechanism actually targets — outbred (het,
cassava-like) individuals, where two structurally-independent homologs
were the whole problem — **the gap closes almost completely**:
either-covered moves from 82.34% (baseline, a 10.24pp miss vs. the
72.1% target) to a pooled 12-seed 72.78% (a 0.68pp miss) at
`shared_frac=0.80, cluster_theta=0.3` — **~93% of the gap closed**,
without materially disturbing indel-affected (35.93% → 33.58%, actually
slightly *closer* to its own 32.7% target, not further from it). This
was confirmed real and not a lucky single seed batch: two independent
6-seed batches (1–6 and 7–12) landed within 0.15pp of each other
(72.70% / 72.85%). The mechanism has a real optimum, not a "more is
strictly better" shape — pushing `shared_frac` to 1.0 (no private
component at all) overshoots past the target (68.45%) with much wider
seed-to-seed variance (±5.13pp vs. ±2.49pp at 0.80), consistent with
the private component doing real work damping variance, not just being
redundant with the shared one.

**Default changed**: `--indel-shared-frac` 0.0 → **0.8**,
`--indel-cluster-theta` 1.0 → **0.3** — the setting that landed closest
to both the outbred either-covered and indel-affected targets
simultaneously in this sweep, confirmed with a second independent seed
batch rather than accepted on a single lucky draw.

**What this result does NOT do**: it does not touch the inbred
either-covered gap (62–67% vs. 71.6% target, **under**, not over,
target — the opposite sign from the outbred problem). This is by
design, not an oversight this round failed to catch: at
`--inbreeding=1.0`, h1≡h2 always, so there is only ever one active
lineage per individual — no second, independent homolog for a
correlated-deletion mechanism to act on. The small movement observed at
the inbred sanity checkpoints (+1.7 to +2.7pp either-covered as
shared_frac rises) is within this sweep's own seed noise (std
2.3–2.8pp) and should not be read as the mechanism doing real work
there; it is a separate, still-open problem needing its own fix (most
likely revisiting `--indel-density` specifically for the inbred case,
or a different mechanism — not explored this round, which was scoped to
the correlated-deletion follow-up PLAN.md §5 item 1 called out
specifically).

## What's still open

- **The inbred either-covered gap** (62–67% vs. 71.6% target) — real,
  separate, unaddressed by this change (see above). This is arguably
  now the more actionable of the two either-covered gaps, since the
  outbred one is (for the first time) within noise of its target.
- **Coverage dispersion** (0.13–0.22 vs. ≫1 target,
  `calibration_sweep_2026-09-14.md`) — unrelated to this mechanism,
  unaffected by it, remains open.
- This sweep held `--indel-density` fixed at its already-calibrated
  2.3e-3 throughout. It's plausible a joint (density, shared_frac,
  cluster_theta) sweep could do even better, but was not attempted —
  the univariate sweep here already reaches within ~1pp of both
  outbred targets simultaneously, and further tuning risked overfitting
  to this specific seed set rather than reflecting a real improvement
  (per PLAN.md's standing instruction not to silently tune to hit
  numbers).
- Not re-validated at other `--sharing-theta` / `--founders` / `--sites`
  configurations — this sweep used the same fixed config as
  `calibration_sweep_2026-09-14.md` throughout (K=24, sites=8192,
  windows=60, sharing-theta=4.0) for direct comparability; whether
  shared_frac=0.8/cluster_theta=0.3 generalizes to materially different
  configs (e.g. much larger K, or a different sharing_theta) is
  untested.
