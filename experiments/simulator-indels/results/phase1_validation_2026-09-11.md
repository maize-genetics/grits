# Phase 1 indel model — implementation + real-data validation

2026-09-11. Implements `PLAN.md` §5 Phase 1 ("Simulator core") behind
`--simulate-indels` in `src/python/crf/simulate_alleles.py`. This
document reports the actual measured numbers from running the new
simulator, checked against `PLAN.md` §2.7's acceptance criteria and
`indel_biology_notes.md`'s real-data targets — honestly, including the
targets that are **not** yet hit.

## What changed

- New `2K+2` ternary(`K`)+distance-code(`K`)+label(2) output layout,
  gated by `--simulate-indels` (default off = today's exact `K+2`
  output, byte-identical — pinned by a golden-hash regression test).
- Output shape stays `[windows, T, 2K+2]`, same `T` as today. Internally,
  generation runs over a larger reference region
  (`R = --indel-region-mult * T`, default `4T`) and each window's `T`
  output rows are a **plain prefix** of the real sampled reads in
  reference order — matching how real `ropebwt_npy_to_matrix.py`
  windowing already works (position is for ordering only). No padded
  row budget.
- New `.refpos.npy` sidecar (`[windows, T]` int32) — the reference site
  each output row projects to, needed since row index no longer implies
  reference position 1:1.
- 16 new `--indel-*` CLI flags controlling density, the two-component
  length mixture, coverage, insertion stacking, region buffer size, and
  soft recombination suppression near indel tracts. Full list and
  calibration sources in `parse_args()`'s help text.
- 49 new unit tests, `tests/python/crf/test_simulate_alleles.py` (all
  passing): golden-hash regression, shape/dtype, structural invariants,
  hand-computed exact tiny-input values, and fuzz-vs-independent-
  reference-oracle tests for every probabilistic component.

## Validation runs

Two runs at `--founders 24 --sites 8192 --windows 60
--min-crossovers 2 --max-crossovers 6 --sharing-model coalescent
--sharing-theta 4.0 --seed 1`, all other `--indel-*` flags at their
calibrated defaults:

- **inbred** (`--inbreeding 1.0`, the CLI default — H1≡H2 always)
- **het** (`--inbreeding 0.0` — H1/H2 draw independently)

| stat | inbred | het | target |
|---|---|---|---|
| indel-affected ref bp | 30.27% | 35.70% | maize 39.3% / cassava 32.7% |
| either-founder covered | 97.74% | 97.54% | **70–75%** (today's non-indel sim: ~96%) |
| hemizygous ref sites (among covered sites) | 0.00%¹ | 54.77% | qualitative: "many" |
| nullizygous ref sites (among covered sites) | 0.00%² | 0.00%² | qualitative: "many" |
| genome-wide site coverage (of the R-site region) | 12.28% | 17.07% | — (new stat) |
| coverage dispersion (var/mean rows-per-site) | 0.1325 | 0.2221 | want **≫1** |
| windows needing padding | 3.33% | 1.67% | should be ~0% |
| insertion stacking, max rows/site | 130 | 65 | — |
| insertion stacking, % of covered sites stacked | 100.00% | 45.22% | — |
| ternary: deletion / diverged / match | 30.27/38.24/31.49% | 32.39/37.46/30.15% | — |
| distance code: mean / p99 / saturated frac | 31.38 / 124.0 / 0.00% | 33.61 / 123.0 / 0.00% | — |

¹ **Not a bug** — at `--inbreeding=1.0` (default), H1 and H2 are
literally the same founder at every site, so "hemizygous" (exactly one
homolog's founder deleted) is impossible by construction. The `het` run
shows this isn't a code defect: with independent homologs, hemizygous
jumps to 54.77%, in the right ballpark for `2·p·(1−p)` at
`p≈0.30–0.36` deletion probability (~0.42–0.46 under independence;
54.77% is somewhat higher, consistent with founders not being fully
independent — see limitations below).

² **A real measurement limitation, not a code bug, found and confirmed
by direct debugging this round**: a genuinely nullizygous site (both
true homologs' founders absent) emits **zero rows by construction** —
there is no homolog left to sample a colinear read from — so it can
never appear in `refpos`/the emitted-row dedup this stat is computed
over. It will read ≈0% regardless of the true rate. The best available
proxy is the new "genome-wide site coverage" line: only 12–17% of the
`R`-site generation region produced any output row at all, so a
meaningful share of the uncovered 83–88% is plausibly nullizygous
positions, not just "beyond the first-T-rows prefix-take." This needed
either a `simulate()` return-signature change (to surface true R-site
deletion counts) or a documented gap — given the acceptance-criteria
section already says "report the gap plainly rather than silently
tuning," the gap is documented here rather than adding more surface
area under time pressure.

## Against §2.7's acceptance criteria

- **Indel-affected fraction near ~40%**: partially met. 30–36%
  depending on inbreeding scenario, in the right regime but on the low
  side of both real targets, closer to cassava's 32.7% than maize's
  39.3%. Not re-tuned to force a closer hit — reported as measured.
- **Either-founder-covered near 70–75%**: **not met.** Both runs land
  at ~97.5–97.7%, essentially unchanged from today's non-indel
  simulator's ~96%. This is §2.7's stated "sharpest test" and it fails
  it. Root cause, best current understanding: with `--indel-coverage
  2.0` (2 reads/homolog/site in expectation) and only ~30–36% of sites
  indel-divergent for a **single** given founder, the *other* homolog
  (or, at hemizygous sites, the surviving homolog alone) still very
  often carries a real matching read — reaching a ~97% either-covered
  rate needs either much higher joint deletion overlap between the two
  homologs' active founders, or materially lower `--indel-coverage`, or
  correlated (not independent) deletion structure between IBD-adjacent
  founders beyond what the lineage-level generation currently produces.
  This is exactly the "open empirical question" §2.2/§2.7 flagged in
  advance — not hit on the first real run, and not silently re-tuned to
  force a hit.
- **Coverage dispersion ≫1 ("blotchy")**: **not met, in the wrong
  direction.** 0.13–0.22 — noticeably **under**-dispersed relative to
  Poisson (≈1), not over. At current defaults, insertion stacking
  (max 65–130 rows at a single site) is real but confined to a small
  minority of covered sites (45–100% is misleadingly high here because
  it's "% of sites with count>1," not weighted by how much excess mass
  those sites carry — most stacked sites still cluster near count≈2).
  The bulk of sites sit very close to the coverage-driven mean (~2),
  which suppresses variance well below Poisson. `--indel-ins-read-per-bp`
  (default `2e-3`) is the most direct lever to raise this if a future
  round targets it explicitly.
- **Windows needing padding ~0%**: close but not exact — 1.67–3.33%,
  the rare-edge-case fallback path triggering more often than "should
  essentially never" at these settings. A real finding per the plan's
  own instruction not to quietly widen `--indel-region-mult` to hide it
  — reported as-is; raising `--indel-region-mult` or `--indel-coverage`
  is the documented lever if a future round wants to drive this to
  zero.

## Indel size/event distribution vs. real biology

Drawn directly from `_indel_lengths` at its calibrated defaults
(`large_frac=0.027, small_alpha=1.7, small_max=50, large_logmean=8.6,
large_logsd=1.6, max_len=65536`), 500,000 events, independent of the
spatial placement logic:

| | simulator | CML103 chr1 (real, `indel_biology_notes.md`) |
|---|---|---|
| events ≤1bp | 47.39% of events / 0.13% of bp | 41.52% of events / 0.07% of bp |
| events ≥4096bp | 1.52% of events / 93.01% of bp | ~2.70% of events / ~94.54% of bp |
| median event length | 2 bp | (not directly reported; small-component-dominated) |
| ins:del ratio | 0.5 (by construction — drawn independently of length) | ~0.95–1.07 across size classes |

The event/bp-weight **inversion** — tiny events dominate the count,
huge events dominate the affected bp — reproduces well (93.0% vs. 94.5%
bp for the ≥4kb tail; both agree the large component is where nearly
all indel-affected bp comes from, not the event histogram). ins:del
symmetry is exact by construction, close to the real ~1:1 finding.

**One implementation artifact found while producing this table**: the
small component's `rng.zipf(1.7, ...)` clipped to `[1, small_max=50]`
produces a visible **pile-up spike exactly at 50**, since the zipf
tail is heavy and many draws exceed 50 before clipping. This inflates
the 32–63bp histogram bin (6.07% of events) above both its 16–31bp
(3.87%) and 8–15bp (6.59%) neighbors — non-monotonic, unlike the real
data's smooth decline. The large component shows the same effect at
its `max_len=65536` ceiling (0.17% of events land exactly at the cap).
Not fixed this round — flagged as a real, minor calibration artifact
for a future pass (e.g. a smoother taper near the clip boundary,
rather than a hard clip) rather than something worth widening scope to
fix under this round's plan.

## Summary

Phase 1 (`PLAN.md` §5 item 1) is **implemented and tested, but does not
yet meet its own acceptance criteria** — most notably either-founder-
covered (~97% vs. 70–75% target) and coverage dispersion (0.13–0.22 vs.
≫1 target). The architecture, tests, and CLI surface are solid (49/49
tests pass, golden-hash regression proves the flag-off path is
untouched, size-distribution shape matches real biology well); what's
missing is calibration depth on the two sharpest §2.7 tests. Per the
plan's own instruction, this is reported as a calibration follow-up,
not silently tuned to hit the numbers. See `PLAN.md` §0 for the dated
log entry and §5 for updated phase status.
