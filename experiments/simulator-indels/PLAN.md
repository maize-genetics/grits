# Realistic indel modeling for the training-data simulator

> **Audience:** Claude Code, and any human collaborator working on this
> project. This is a living design document, not a one-off summary —
> update it (with a dated Progress log entry, §0, same convention as
> `docs/PLAN.md`) as design decisions change or implementation phases
> land, so a fresh Claude Code session or a collaborator with no prior
> context can pick this up correctly. **Last reconciled with the code:
> 2026-09-09** — no simulator/model code has changed yet; see §0.

> **Scope of this document:** reworking `src/python/crf/simulate_alleles.py`
> to (a) generate realistic insertion/deletion (indel) patterns relative
> to reference, and (b) emit the two-matrix training format that the new
> `refmap --anchor-dist-npy` feature produces, in a separate repo
> (`ropebwt3-phg`, branch `lift-ridx-ternary-dist-map`, pushed, not yet
> merged there either). See `docs/HANDOFF.md`'s top entry for how this
> connects to the rest of the project's history.

---

## 0. Progress log

### 2026-09-09 — Design phase complete, no code changed

Full design worked out (this document), grounded in real maize founder
gVCF data plus general plant-genome indel-mutation literature (see
`results/indel_biology_notes.md`). Branch `simulator-indel-modeling`
created off `bring-in-simval-corpus`. **No changes made yet to
`simulate_alleles.py`, `train_diploid.py`, or `train_crf.py`** —
implementation is intentionally deferred to separately-approved future
rounds (§5). This entry exists so the next session (human or Claude)
knows exactly where this stands: design is settled enough to start §5
Phase 1, but Phase 1 has not been started.

### 2026-09-09 — Training-side design reframed as a new model line

The training-loop side of §5 Phase 2 (below) was originally designed as
an in-place widening of `FounderPathEncoder`/`GRITSCRFDiploid`, gated
behind a checkpoint-compatibility hyperparameter. That framing was
explicitly rejected: this is being built as **a new model with its own
training script(s)**, not an upgrade to the existing one, reusing
existing data structures/classes only where they're genuinely
feature-width-independent. Full design now lives in
[`TRAINING_PLAN.md`](TRAINING_PLAN.md); §5.2 below is a pointer to it,
not the design itself. Still no code changed.

---

## 1. Why this work exists

Two confirmed root causes from this project's RIL2 founder-path
investigation motivate the new `refmap --anchor-dist-npy` feature:
binarization of read-support data loses the signal that would resolve
identity-by-descent (IBD) zone founder confusion (e.g. the confirmed
chr5:86–134Mb B73×CML103 IBD zone), and pangenome repetitiveness is
invisible to the model in its current input representation. Beyond
that, this design pass surfaced a second, equally important goal: the
same ternary state / distance features are the intended mechanism for
giving the model a *direct, observable* signal to disambiguate three
locus states it currently conflates —

- **heterozygous** — both homologs present at a locus, carrying
  genuinely different founders (ordinary biallelic variation);
- **hemizygous** — one homolog carries a real deletion relative to
  reference at that locus, so only one copy of the segment truly exists
  (a structural absence, not an allelic difference);
- **recombination** — a genuine founder-path switch on one homolog, no
  deletion involved at all.

An earlier, unimplemented idea for this same disambiguation problem
(`docs/PLAN.md`'s "E10", a per-read position float) is **explicitly
superseded and deferred** — see §6. The ternary state gives the model
this signal directly (an explicit per-founder `deletion` label) rather
than requiring the model to infer it indirectly from read-position
clustering, which is what E10 proposed before this feature existed.

## 2. Design

### 2.1 Coordinate system: B73/reference is the backbone, not a new one

The simulator today has no base-pair coordinate concept at all — sites
are pure integer indices `0..T-1` (`simulate_alleles.py`, confirmed by
source read, 765 lines, no bp anywhere). Rather than inventing an
independent coordinate space, this design keeps the reference (B73)
positions as the backbone directly — window sites are reference
positions at whatever bin resolution the rest of the pipeline already
trains on — and gives each non-reference founder a **running offset
relative to that reference track**: zero through colinear sequence,
diverging through an indel run, sized by the indel's own length. This
precisely mirrors how real `.lift`-file anchors work in the actual
`ropebwt3-phg` implementation: a founder's own coordinate advances
slower than the reference's during a deletion (relative to reference),
faster during an insertion (relative to reference) — the same "2D indel
offset" mechanism underlying `rb3_lift_nearest_ref` in `lift.c`.

Indels are discrete events placed onto a founder's mosaic segments,
reusing the existing tract-placement primitives already in
`simulate_alleles.py`:
- `_segment_index` (`:219–242`) — already used for E11 heterozygosity
  tracts; drops in directly for indel-tract placement.
- `_good_mask` (`:270–289`) — a 2-state Markov tract generator (
  stationary probability + mean run length); its own docstring already
  invokes "SV blocks" as the motivating case.
- `_rate_map` (`:64–72`) — for making indel density spatially
  heterogeneous rather than uniform, if desired.

The ternary state and distance-to-anchor features are then *derived*
from walking each founder's offset track, not drawn independently of
it — matching exactly how the real feature works.

### 2.2 Indels are a property of the founder's genome, not the individual

Any individual/segment that inherits a given founder's haplotype at a
locus inherits that founder's indel structure there — this is how real
inheritance works, and it is what makes the three-state disambiguation
in §1 fall out naturally rather than needing special-case logic:

- H1's founder has a deletion, H2's doesn't → **hemizygous**.
- Both H1's and H2's founders have a deletion (including the fully
  homozygous/inbred case where H1 and H2 carry the same founder) →
  **nullizygous** (matches this project's established terminology for
  the fully-homozygous case, distinct from hemizygous).
- Otherwise → ordinary heterozygous variation.

This reuses the existing founder-subset/individual-assignment machinery
(`_build_paths_subset`, the E11/Tier2 breeding-population mechanism)
with no new grouping code required — indels attach to the founder-id
axis that already exists.

### 2.3 Output: exactly two matrices, verified against the real implementation

Verified directly against the actual committed C code in `ropebwt3-phg`
(branch `lift-ridx-ternary-dist-map`, `ps4g.c`'s `rb3_ps4g_npy_finalize`,
commit `26d04e0`) rather than assumed from memory. In the production
refmap tool, the widened `.npy` keeps a *third*, legacy-unchanged
binary/count block for backward compatibility with refmap's existing
consumers — that constraint does not apply to a brand-new simulator
output with no legacy consumers. **The simulator's matrix 1 is the
ternary state directly, replacing the current binary values rather than
sitting alongside them:**

- **Matrix 1** (occupies the *same column position* as today's binary
  features, `row[:, :K]`): ternary read-sharing state per founder,
  values `{-1, 0, 1}` = deletion / diverged / match relative to
  reference — matching `rb3_lift_ternary_state`'s exact semantics. The
  reference (B73) column is always `match`; it is never run through
  deletion logic, since it has no anchors relative to itself.
- **Matrix 2** (new trailing block, appended *after* the existing H1/H2
  label columns): distance to the nearest point where this founder's
  offset last matched the reference track — i.e., distance out of the
  current indel run, near-zero in colinear sequence — per founder,
  non-negative, with a sentinel value for "no nearby anchor." Matches
  `rb3_lift_nearest_ref`'s semantics.

Truth labels (H1/H2 founder-path, today at columns `K`/`K+1`) **stay at
their current fixed offsets, unchanged**. No VCF-shaped output is
planned: the model trains directly on these `.npy` arrays and never
reads a VCF, and truth already lives in the label columns.

**Target on-disk layout:** `[ternary(K) | H1(1) | H2(1) | distance(K)]`,
total width `2K+2` (up from today's `K+2`). Labels don't move — this
narrows the training-side blast radius to "matrix 1's value range
changed" plus "one new trailing block," not a full column renumbering.

### 2.4 Indel size model: two-component mixture

Real maize founder gVCF data is genuinely bimodal in indel length (see
`results/indel_biology_notes.md` for the full histogram), and the two
modes map cleanly onto two distinct, well-documented plant-wide
mutational mechanisms. The simulator implements this as an explicit
two-component mixture, not a single distribution, so it stays
mechanistically defensible across plant species by adjusting each
component's own parameters, not the model structure:

- **Small component** (replication slippage at homopolymer/microsatellite
  repeat tracts): geometric/power-law-decaying length, 1bp dominant.
- **Large component** (LTR-retrotransposon insertion/removal, including
  the solo-LTR partial-deletion mechanism): a heavier-tailed
  distribution centered in the kb range.

Mixture weight and each component's own parameters are config knobs.
Defaults are maize-calibrated from the real numbers in
`results/indel_biology_notes.md` (excluding A188, a confirmed
assembly-quality outlier — see that document), but the two-component
*structure* is the general-plant-biology claim, portable to other
species by re-fitting parameters, not rewriting the generator.

## 3. Terminology (use precisely and consistently)

- **Insertion / deletion relative to reference (B73)** — never bare
  "insertion"/"deletion" without the reference frame; a pairwise
  alignment alone can't establish ancestral state.
- **Heterozygous / hemizygous / nullizygous** — see §1 and §2.2; these
  are three distinct states, not interchangeable.
- **Identity by descent (IBD)** — segments inherited from a common
  ancestor without recombination.
- **Presence/absence variation (PAV)** — the standard term for the
  structural-variant class this feature targets.
- Avoid the word "ambiguous" (standing project preference) — prefer
  specific phrasing like "read support that cannot discriminate between
  founders" or "the model must infer this rather than observe it
  directly."

## 4. Reusable vs. genuinely new (source-verified)

**Reusable as-is:**
- `_segment_index`, `_good_mask`, `_rate_map` (§2.1) — tract placement.
- Founder-subset/individual-assignment machinery (`_build_paths_subset`,
  E11/Tier2 breeding-population mechanism) — indel presence attaches to
  the founder-id axis these already manage.
- The model-side zero-init, checkpoint-compatible extension pattern
  already established for `ext_bias`/`het_head`
  (`src/python/crf/train_crf.py:177–194`) — the template for adding the
  new distance-embedding path without invalidating existing checkpoints.

**Genuinely new:**
- Any presence/absence or ternary state at all — today's features are
  binary `int8` end to end.
- A per-founder deletion semantic distinct from "no match."
- The offset-tracking mechanism connecting indel events to the
  ternary/distance derivation (§2.1).
- The two-component mixture length model with real-data-calibrated
  defaults (§2.4).
- `train_crf.py`'s `binary_cells` fast path (`:218–222`) hardcodes a
  2-row `{0,1}` lookup table — must be widened or disabled for ternary
  data.
- `train_diploid.py`'s `HET_INBRED`/`HET_OUTBRED` calibration constants
  and `_het_scale`'s Jaccard computation / `_founder_affinity`'s mean
  (`:252–308`) all assume bounded `0/1` input — need recalibration or
  restriction to a binarized view of the ternary matrix.
- An encoding scheme for the distance block that fits the existing
  `int8` array dtype — needs log-scale or binned compression, similar
  to the existing rate-track's `np.clip(np.rint(...), 1, 127)`
  precedent at `simulate_alleles.py:492–493`.

## 5. Future work (not started — each phase needs its own review/approval)

1. **Simulator core.** Add the indel-tract generator (§2.1, §2.4) and
   the offset-tracking mechanism to `simulate_alleles.py`; emit the new
   `2K+2` layout (§2.3) behind a new CLI flag (e.g. `--simulate-indels`)
   so existing behavior is preserved when the flag is off.
2. **Training-side format updates.** Full design now in
   [`TRAINING_PLAN.md`](TRAINING_PLAN.md) — reframed (2026-09-09, see
   §0) as a new model with its own training script(s)
   (`train_diploid_indel.py`), not an in-place widening of
   `FounderPathEncoder`/`GRITSCRFDiploid`; existing CRF kernels,
   `EMACallback`, and Lightning wiring are reused, while the
   cell-embedding transform, `_het_scale`/`_founder_affinity`
   equivalents, and Dataset classes are rebuilt for the ternary value
   range. No checkpoint-compatibility constraint with
   `diploid-affinity-sim512-h3` — see `TRAINING_PLAN.md` §1.
3. **Retrain and evaluate** against this project's established
   SNP+RefCall baselines (genome-wide 0.0711% error; B73×CML103
   chr5:86–134Mb IBD-zone 1.181% SNP-class error — see
   `experiments/simval-corpus/`).
4. **Decide whether to merge** `ropebwt3-phg`'s
   `lift-ridx-ternary-dist-map` branch, once (3) shows the feature is
   worth keeping on real (not just synthetic) data.

## 6. Explicitly deferred, not in scope

- **`docs/PLAN.md`'s E10** (per-read position float, for the same
  heterozygous/recombination disambiguation goal) — superseded by the
  ternary+distance mechanism (§1). If E10 is ever revisited, resolve its
  conflict with this feature over `FounderPathEncoder.cell`'s
  per-cell embedding slot *before* starting E10, not before starting
  this work.
- **`src/python/cross/pick_crossovers.py`'s bp-space model** — not
  needed given the reference-anchored-offset design in §2.1; that
  module remains dormant/disconnected from training as before.
- **Any VCF-shaped simulator output** — see §2.3.
- **`docs/PLAN.md`'s E3 `meta.sv_blocks`** truth-span idea — worth
  revisiting as an eval sidecar once the core feature lands, following
  the existing `.ibd.npy`/`.ind.npy`/`.cls.npy` sidecar pattern
  (`simulate_alleles.py:652–681`), but not required for the core
  mechanism.

## 7. Open questions (not yet decided — update as resolved)

- Exact homopolymer/repeat-context bias implementation for the small
  indel component (§2.4) — flagged as a refinement, not yet designed.
- Exact distance-block encoding scheme to fit `int8` (§4) — log-scale
  vs. binned, and the choice of clip/sentinel values.
- Whether the solo-LTR partial-deletion mechanism (§2.4,
  `results/indel_biology_notes.md`) gets its own explicit sub-model or
  folds into the general large-indel component's length distribution.
