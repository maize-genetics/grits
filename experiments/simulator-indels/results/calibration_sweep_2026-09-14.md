# Calibration sweep, and a real bug found in the "either-founder-covered" QC stat

2026-09-14. Follow-up to `phase1_validation_2026-09-11.md`, prompted by
a request to calibrate toward that report's two failing acceptance
criteria (either-founder-covered, coverage dispersion) before using
the indel-mode simulator for training. **Headline finding: the
either-founder-covered number in the 2026-09-11 report was measuring
the wrong thing.** The real gap is much smaller than reported, and what
remains points at a specific, well-understood modeling gap rather than
a vague "needs more tuning."

## What was wrong with the old stat

`_print_indel_summary`'s "either-founder covered" was computed by
deduplicating over **emitted output rows**, then checking whether any
row at that site matched h1's or h2's founder. This is conditioned on
a row existing at all. But a row can only be sampled from a homolog
that is physically present (see `_row_counts`) — a site where **both**
homologs' founders are deleted (nullizygous) can never produce a row
by construction, so it can never enter that dedup. The stat was
structurally blind to exactly the phenomenon it was supposed to
measure, and read a near-tautological ~97% (≈ "when a read exists, did
genotyping-error noise corrupt which founder it matched") regardless
of the true rate.

**Proof this was the bug, not a coincidence**: a sweep across
`--indel-density` (2.65e-3 → 1e-2, more than tripling the realized
indel-affected fraction) left the old row-conditioned stat completely
flat at ~97%. A model whose real either-covered rate depended on
density even slightly should have moved; it didn't, because the stat
literally cannot see the effect.

## The fix (shipped)

`_indel_chunk` now also computes `(n_either, n_hemi, n_null)` directly
from `pres1`/`pres2` — the per-(window, R-site) presence booleans it
already builds internally — over the **full R-site generation region**,
not the T-row output. `simulate()` accumulates these across chunks and
returns them as a new `true_cov_out` value (return arity 8→9; the one
other caller, `simulate_wholegenome.py`, updated to match, same pattern
as the original implementation round). `_print_indel_summary` now
prints both numbers, clearly labeled: the new genome-wide TRUE stat
(the one PLAN.md §2.7's target actually refers to) and the old
row-conditioned one (kept, relabeled as a secondary diagnostic, not the
acceptance check). One new hand-verified exact test
(`test_indel_chunk_tiny_exact`, extended) checks `(n_either, n_hemi,
n_null)` against a by-hand-traced 2-founder fixture. All 49 tests still
pass (49/49, was 48 non-extended + this one extended).

## Corrected numbers (same K=24/sites=8192/windows=60/seed=1 config as the 2026-09-11 report)

| metric | inbred (maize-like) | het (cassava-like) | target |
|---|---|---|---|
| indel-affected ref bp | 30.27% | 35.70% | maize 39.3% / cassava 32.7% |
| **either-founder TRUE covered (genome-wide)** | **62.17%** | **80.72%** | **70–75%** |
| hemizygous / nullizygous (genome-wide, TRUE) | 0.00%¹ / 37.83% | 37.31% / 19.28% | qualitative |
| either-founder covered (rows only — old stat, now relabeled) | 97.74% | 97.54% | not the target metric |
| coverage dispersion (unchanged, still correctly measured) | 0.1325 | 0.2221 | want ≫1 |

¹ hemi is exactly 0 at `--inbreeding=1.0` (h1≡h2) by construction, as
established in the 2026-09-11 report.

**The real gap is ~9pp in each direction, not ~25pp**: inbred is
*under* target (62.17% vs. 71.6%), het is *over* target (80.72% vs.
72.1%) — opposite signs, which matters for what fixes it (below).

## Why a single `--indel-density` knob can't close both gaps

Checked directly: lowering density from the calibrated default
(2.65e-3) to 1.6e-3 brings inbred's either-covered to **72.15%** — a
near-exact hit on the maize target. But at that density the realized
indel-affected fraction drops to ~28%, undershooting *both* real
targets (39.3% maize / 32.7% cassava) by 5–11pp — and it moves het's
already-too-high either-covered even further from target, since less
deletion means *more* joint coverage, the wrong direction for het.

This is the concrete, mechanistic version of the correlated-deletion
gap flagged (but not confirmed) in the first report: in the current
model, indel tracts are drawn **per lineage independently** — h1's and
h2's founders lose coverage at unrelated sites even when they're
IBD-adjacent. Real biology apparently produces some middle ground: a
high population-wide "some founder is missing here" rate (driving
indel-affected fraction up) *combined with* enough shared/correlated
structural variation between related founders that a het individual's
two homologs jointly lose coverage more often than independent draws
would predict (driving het's either-covered down without needing to
raise density further, which would only inflate indel-affected past
target). No density value reproduces that shape — it needs a
correlation mechanism, not a magnitude adjustment.

## Revised assessment

Not "not met" in the way the first report implied. The indel model is
within ~9 percentage points of both real-data acceptance ceilings, in
directions that are individually explicable and point at one specific,
scoped follow-up (correlated/clustered deletion across IBD-adjacent
founders in `_indel_tracts`) rather than a broad recalibration. Coverage
dispersion (0.13–0.22 vs. ≫1 target) is unaffected by this correction
and remains a real, separate gap — under-dispersed coverage, not
resolved by anything in this round.

**Default `--indel-density` was left unchanged in the first pass above,
then revised later the same day — see the addendum below.**

## Addendum (same day): decoupling maize from cassava

The comparison above mixed two different things in one number: an
*organism* (maize vs. cassava, different real indel-affected targets)
and a *mating structure* (inbred vs. het, paired arbitrarily with
maize/cassava respectively). Prompted by a review question — should
maize be calibrated on its own first, since it's this project's actual
training target, before treating cassava as a secondary generalization
check — the two were separated cleanly.

**Maize alone (inbred, NAM-founder-like), both real maize targets
simultaneously** (indel-affected ~39.3%, either-covered ~71.6%):
a `--indel-density` sweep, averaged over 6–8 seeds at
K=24/sites=8192 to control for real seed-to-seed noise (std ≈
1.7–2.2pp at this scale — itself worth knowing when reading any single
validation run), found the joint-error-minimizing density at
**~2.2–2.3e-3**: indel-affected ≈35.2%, either-covered ≈64.8% (errors
≈4–7pp on each target, both directions, roughly balanced) — a real,
modest improvement over the original 2.65e-3 default's ≈39%/≈63%
(≈0pp / ≈−8.6pp — nearly exact indel-affected but a much worse
either-covered miss). **Default `--indel-density` changed from
2.65e-3 to 2.3e-3** on this basis (`simulate()` and `--indel-density`'s
CLI default both updated; help text revised to cite this joint
calibration).

**Cassava alone (het, outcrossing-like), as the generalization
check** (indel-affected ~32.7%, either-covered ~72.1%): the *same*
knob cannot get close on both. The either-covered target is reachable
— density ≈3.5e-3 gives either-covered ≈70.9% (within ~1.2pp of
72.1%) — but at that density indel-affected is ≈47.1%, **14.4
percentage points above** cassava's own 32.7% target, a much larger
and cleaner miss than maize's. This is the clearest evidence yet for
the root cause identified earlier: heterozygous individuals draw h1
and h2's structural-variant state *independently*, so reaching
real-world joint coverage loss requires far more aggregate deletion
than the population's true indel-affected rate would suggest — and no
single density value resolves that tension, for either organism, but
it is far sharper for the outcrossing (het) case than the inbred case.

**Conclusion**: decoupling maize and cassava was the right move
methodologically (removes a real confound, and correctly reflects that
maize — not cassava — is this project's actual training target), but
it does not, by itself, close the gap. It sharpens the case for the
one concrete follow-up already identified: add correlated/clustered
deletion structure across IBD-adjacent founders to `_indel_tracts`.
Until that lands, no single `--indel-density` value — organism-specific
or otherwise — will jointly satisfy both real acceptance criteria for
an outcrossing organism; an inbred-line organism like maize gets much
closer (as shown above) because h1≡h2 removes the independence problem
entirely.

Also fixed in this pass, for consistency with the either-covered fix
above: "indel-affected ref bp" had the same (smaller) row-conditioning
bias — computed by dedup over sites *this window's own h1/h2 happened
to cover*, rather than the true population-wide rate. `_indel_chunk`
now also returns `(n_deleted, n_founder_sites)` from the already-
materialized `dist_kt` array (all K founders x all R sites, no extra
computation cost), and `_print_indel_summary` prints both the corrected
TRUE genome-wide rate and the old row-conditioned one (relabeled, kept
as a secondary diagnostic). `simulate()`'s `true_cov_out` return grew
from a 4-tuple to a 6-tuple accordingly. One new hand-verified exact
assertion added to `test_indel_chunk_tiny_exact`; 49/49 tests still
pass.

## Training readiness, reassessed (updated per the addendum)

Previously: hold off, because the model looked nearly indistinguishable
from the pre-indel simulator on the metrics that mattered. That
conclusion doesn't hold anymore — the model **is** measurably
different and much closer to real-data coverage-loss behavior than it
appeared, especially for maize (this project's actual training target):
indel-affected ≈35% vs. 39.3%, either-covered ≈65% vs. 71.6%, both
within ~5–7pp after the default-density fix. Two real gaps remain
before training on maize-shaped data specifically: coverage dispersion
is still flat (not blotchy), and maize's either-covered is still ~7pp
under target. Cassava-shaped (het) data has a materially bigger,
better-understood gap (the independence-of-homologs issue above) and
would need the correlated-deletion follow-up before it's a good
training target regardless of density tuning. Whether the maize-side
gaps are worth closing before training, versus training now on maize
data and treating them as a documented limitation, is a real tradeoff
— not attempted here; flagged for discussion rather than decided
unilaterally.
