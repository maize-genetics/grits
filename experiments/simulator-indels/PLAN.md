# Realistic indel modeling for the training-data simulator

> **Audience:** Claude Code, and any human collaborator working on this
> project. This is a living design document, not a one-off summary —
> update it (with a dated Progress log entry, §0, same convention as
> `docs/PLAN.md`) as design decisions change or implementation phases
> land, so a fresh Claude Code session or a collaborator with no prior
> context can pick this up correctly. **Last reconciled with the code:
> 2026-09-11** — no simulator/model code has changed yet; see §0.

> **Scope of this document:** reworking `src/python/crf/simulate_alleles.py`
> to (a) generate realistic insertion/deletion (indel) patterns relative
> to reference, and (b) emit the two-matrix training format that the new
> `refmap --anchor-dist-npy` feature produces, in a separate repo
> (`ropebwt3-phg`, branch `lift-ridx-ternary-dist-map`, pushed, not yet
> merged there either). See `docs/HANDOFF.md`'s top entry for how this
> connects to the rest of the project's history.

> **Workflow diagrams:** [`results/simulator_workflow_before.png`](results/simulator_workflow_before.png)
> (today — 5 stages, binary `K+2` output, grounded directly in
> `simulate()`'s current body) vs.
> [`results/simulator_workflow.png`](results/simulator_workflow.png)
> (planned — the §2/§4 pipeline, color-coded reused / new / modified,
> capped by the §2.7 acceptance gate). Same box grid/size in both so
> they compare directly. Regenerate via
> `scripts/simulator_workflow_diagram.py` /
> `scripts/simulator_workflow_before_diagram.py` after any change to
> §2/§4's stage list or to `simulate()` itself, so neither picture
> drifts from the prose or the code.

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

### 2026-09-10 — Collaborator review incorporated

A collaborator reviewed this design and returned seven points, folded in
below: (1) report the large-indel size distribution explicitly —
bp-weighted, not just event-weighted; (2) the read-sampling model must
produce **stacking** (many rows at one reference position for large
insertions) since ~40% of the maize genome is indel-divergent between
two inbreds; (3) add Cassava as a second validation system (deferred —
cassava has assemblies, not gVCFs, and needs data the collaborator will
supply); (4) answer how recombination is modeled across indels — not
previously addressed anywhere in this document, now §2.6; (5) indels
should follow the same Ewens/GEM(θ) coalescent as SNPs, possibly a
shorter one — now §2.2; (6) the acceptance bar is that heterozygous
individuals show **hemizygous-looking read sharing at many positions**
and homozygous lines show **blotchy coverage** — now §2.7, with real
targets from `docs/notes/cassava_data_diagnostic.md`.

The 40% figure is now measured directly (not just taken on faith):
39.0–42.6% of reference bp across B97/CML103/Tzi8 chr1 carries no
homologous founder sequence (deletion+gap, i.e. excluding insertion bp,
which doesn't consume reference span — see the definitional note in
`results/indel_biology_notes.md`). Full multi-chromosome, multi-founder
figures in `results/indel_size_distribution.png`, reproducible via
`scripts/indel_size_report.py`.

One design point from the 2026-09-09 entry was revisited and **held**:
I initially flagged a conflict between §2.3's strict ternary matrix 1
and the stacking requirement (a pile-up would encode identically to a
single read under a value-only representation). Per the collaborator,
stacking is a **row-multiplicity** phenomenon, not a cell value — the
existing PS4G convention already expands read counts into consecutive
duplicate rows (verified in `ropebwt_npy_to_matrix.py`'s docstring and
in real matrices, where zero rows are ever all-zero). §2.3's `2K+2`
ternary layout is correct as committed; see the note added there.

### 2026-09-11 — Standing practice: a workflow diagram alongside this plan

Going forward, a plan document in this project should ship with a
diagram showing its pipeline stages and which are reused / new /
modified — not just prose. First one built: `results/simulator_workflow.png`
(source: `scripts/simulator_workflow_diagram.py`), covering §2's 8
pipeline stages plus the §2.7 acceptance gate. Keep the diagram and the
prose in sync — regenerate the PNG (hand-authored SVG, rasterized via
`rsvg-convert`) whenever §2 or §4's stage list changes, rather than
letting the picture go stale.

### 2026-09-11 — Cassava calibration data landed and measured

Real cassava gVCFs arrived (`grits_workdir/cassava/gvcfs/`, 106
haplotype-resolved files); measured with the same
`scripts/indel_size_report.py` used for maize, no changes needed beyond
two generality fixes caught while reusing it (the figure title was
hardcoded to "maize" — now an `--organism` flag; the chromosome-sort in
panel D assumed a literal `chr` prefix and would have crashed on
cassava's `Chromosome01`-style names — now sorts by trailing digits
regardless of prefix). §2.7's cassava paragraph and
`results/indel_biology_notes.md` updated with the findings — see there
for numbers. The event/bp-weighted size inversion (§2.4's core claim)
reproduces closely in cassava; the indel-affected-fraction target is
organism-specific (32.7% cassava vs. 39.3% maize), not a universal 40%.

### 2026-09-11 — "Before" diagram added for direct comparison

Built `results/simulator_workflow_before.png`
(`scripts/simulator_workflow_before_diagram.py`), grounded directly in
`simulate()`'s current body (`simulate_alleles.py:354–497`, not
inferred from this document's own prose): 5 stages — founder paths,
genotyping-error mask, the inline H1-or-H2 "one read" pick (no separate
read-sampling function exists today), SNP match features, write
`K+2`/binary matrix. Uses the identical box grid/size as
`simulator_workflow.png` so the two are directly comparable side by
side; a red "not present today" panel occupies the same canvas space
the planned diagram's stages 6–8 fill.

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

**Indel presence is drawn from the same Ewens/GEM(θ) coalescent process
as SNP sharing, not an independent distribution.** The simulator already
draws SNP mini-haplotype sharing this way: `_gem_lineages`
(`:245–267`) assigns each founder's mosaic segments a lineage via
GEM(θ) stick-breaking (`--sharing-theta`), and `_coalescent_feats`
(`:292–351`) indexes per-lineage SNP alleles (`lin_alleles`) through
that same `lineage [n,K,T]` array — two founders on the same lineage at
a site are identical-by-descent and share the same mini-haplotype. Draw
per-lineage indel presence/absence exactly in parallel, indexed by the
*same* `lineage` array: two IBD founders then share indel content and
mini-haplotype identically, which is the biologically correct coupling
(a real segment inherited without recombination carries both together)
and gives consistent co-support structure rather than two independently
noisy signals.

Default: indels reuse `--sharing-theta` directly (no separate draw). Add
`--indel-theta` (default: unset, inherits `--sharing-theta`) to let
indels follow a shorter coalescent than SNPs if calibration later shows
that fits better — real indel tract boundaries plausibly turn over
faster than point-mutation lineages. Treat as a tuning knob to explore
once real simulator output exists, not a default to guess ahead of data.

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

**Why this stays a value-only matrix rather than needing a count block:**
a collaborator review (2026-09-10, §0) raised a real concern here — a
strict ternary/binary *value* can't distinguish a 40-read pile-up from a
single read, which matters once read stacking (§2.5) is modeled. The
resolution is that stacking is represented as **row multiplicity, not a
cell value**: matching the existing PS4G convention, where multiple
reads at the same region become additional consecutive rows carrying
the same founder profile, not a larger count in one row (confirmed
against `ropebwt_npy_to_matrix.py`'s "non-collapsed, row-not-position"
windowing convention, and against real training matrices, where no row
is ever all-zero). So the `2K+2` ternary layout is unaffected by
stacking — see §2.5 for how rows get produced.

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

### 2.5 Read sampling and coverage: the mechanism that produces stacking

The simulator currently has **no coverage model at all**: per its own
docstring (`:21`), "each of the T positions is ONE read" — exactly 1×
by construction — and `_coalescent_feats` returns one binary match value
per site. Nothing can stack, and nothing can be uncovered. This is a
real gap, not a refinement — reproducing the acceptance criteria in
§2.7 requires an actual read-sampling layer:

1. Sample reads in **founder (assembly) coordinates**, proportional to
   founder sequence content — a founder carrying a large insertion
   contributes reads across the full length of the inserted sequence,
   same as real sequencing coverage would.
2. Project sampled reads onto the reference coordinate system (§2.1).
   Inserted sequence has no reference span, so every read landing inside
   an insertion projects to the **single reference anchor point**
   flanking it → many consecutive rows at one reference position. This
   *is* the stacking, emitted as row duplication per §2.3's note, not a
   count value.
3. Reads sampled from within a deletion interval don't exist — that
   founder contributes **zero rows** at those reference positions.
4. Net effect: rows per unit reference bp becomes highly non-uniform
   along the genome — dense at large insertions, absent through
   deletions — which is the mechanism behind the coverage "blotchiness"
   in §2.7.

### 2.6 Recombination across indels

Not previously addressed anywhere in this design; a collaborator asked
directly (2026-09-10, §0). Answer:

- **Crossovers are drawn on the reference backbone** (§2.1), the frame
  in which homology — and therefore meiotic pairing — is defined,
  reusing `_segment_index`/`_build_paths`/`_rate_map` unchanged.
- **Indels are atomic with respect to crossover**: no breakpoint falls
  inside an indel tract. Mechanistic basis — crossover requires
  homologous synapsis, and a hemizygous insertion has no pairing partner
  on the other homolog at that interval. Implementation: after drawing
  breakpoints, snap any landing inside an indel interval out to the
  nearest flanking colinear position.
- **Large indels locally suppress crossover** beyond strict atomicity —
  no new mechanism needed: `_segment_index` already accepts a per-site
  `rate`, so scale it down inside and near indel-divergent intervals,
  reusing the existing hidden `--recomb-span` rate-map machinery (E2).
- **Truth labels stay defined everywhere**, including inside deletions —
  a reference position has a well-defined founder assignment even where
  the individual carries no sequence there. Those become *labeled but
  uncovered* positions, which is exactly the blotchiness in §2.7 and is
  currently excluded from SNP+RefCall scoring project-wide.
- **The coupling that matters**: indel content is a lineage-level
  property (§2.2), and a crossover switches which lineage is active — so
  crossing over changes which indel structure an individual carries
  downstream. This is the mechanism by which hemizygosity varies along a
  chromosome instead of being a fixed per-individual property.

### 2.7 Calibration targets and acceptance criteria

The collaborator's stated priority, more important than any single
parameter choice: heterozygous individuals must show **hemizygous-
looking read sharing at many positions**, and homozygous lines must show
**high blotchiness in coverage**. Both already have measured real-data
targets, recorded in `docs/notes/cassava_data_diagnostic.md`
(2026-06-24), which found a then-unexplained ceiling that the indel
model is now a strong candidate explanation for:

| metric | real cassava (het) | real maize (inbred) | current simulator |
|---|---|---|---|
| true H1 founder has ≥1 supporting read | 42.4% | 71.6% | ~96% |
| **either true founder (H1 or H2) has a read** | **72.1%** | **71.6%** | **~96%** |

At ~40% of reference bp indel-divergent (see `results/indel_biology_notes.md`),
a large share of positions genuinely have no homologous founder
sequence to sample a read from — so reproducing the real **~72% ceiling**
instead of the simulator's current ~96% is the sharpest available test
that the indel model (§2.5, §2.6) is doing real work, not just adding
noise. Acceptance criteria for Phase 1 (§5):

- Simulated either-true-founder-covered rate should land near 70–75%
  for outbred (heterozygous) individuals, not ~96%.
- Add a **coverage-dispersion statistic** for blotchiness: index of
  dispersion (variance/mean) of rows-per-reference-bin, plus the
  zero-coverage run-length distribution. Poisson (uniform) sampling
  gives an index ≈1; real blotchy coverage should be markedly ≫1. No
  real-data target number yet — flagged in §7.
- Realized genome-wide indel-affected fraction should land near the
  measured **~40%** target (§2.4, `results/indel_size_distribution.png`),
  reported as an explicit simulator QC line, not just implied by config.

**Cassava as a second validation system** — biological calibration
**done** (2026-09-11); simulator-output validation still pending. Real
haplotype-resolved cassava gVCFs landed
(`grits_workdir/cassava/gvcfs/`, 106 files, same PHGv2 `ASM_Start`/
`ASM_End` format as maize — the earlier concern about needing the
`.lift`-anchor route instead was moot once real gVCFs arrived).
Measured 12 haplotypes × 18 chromosomes: the event/bp-weighted size
inversion reproduces almost exactly (43.9% of events at 1bp vs. 0.17%
of bp; ~1.3% of events ≥4kb vs. 88.1% of bp), directly supporting
§2.4's two-component-mixture claim as general-plant biology, not a
maize-only fit. Indel-affected fraction is measurably lower than
maize's (32.7% pooled mean vs. 39.3%) — reported as found, cause not
yet determined. Full write-up and figure:
`results/indel_biology_notes.md`, `results/cassava_indel_size_distribution.png`.
**What's still deferred**: the simulator-*output* side of cassava
validation (reproducing the ~72%-either-founder-covered acceptance
check from §2.7's table) — that needs §2.5's read-sampling/coverage
layer actually built, not just biological calibration input.

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
- `_gem_lineages` (`:245–267`) and the `lineage [n,K,T]` array it
  produces — indel presence draws through the same lineage assignment
  as SNP sharing (§2.2), no new partition machinery needed.
- `--recomb-span`'s hidden per-site rate map (E2) — reused to suppress
  crossover near large indels (§2.6), no new rate mechanism needed.
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
- A genuine read-sampling/coverage layer (§2.5) — the simulator has none
  today (exactly 1 read/site by construction); this is what produces
  stacking and blotchiness, not a refinement of existing code.
- Breakpoint-snapping to keep crossovers out of indel tracts (§2.6).
- The coverage-dispersion QC statistic (§2.7) — new, no existing
  analogue in the simulator's diagnostics.
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

1. **Simulator core.** Add the indel-tract generator (§2.1, §2.2, §2.4),
   the offset-tracking mechanism, the read-sampling/coverage layer
   (§2.5), and breakpoint-snapping (§2.6) to `simulate_alleles.py`; emit
   the new `2K+2` layout (§2.3) behind a new CLI flag (e.g.
   `--simulate-indels`) so existing behavior is preserved when the flag
   is off. Done when it meets the acceptance criteria in §2.7, not just
   when it runs.
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
- `--indel-theta` (§2.2): whether indels genuinely need a shorter
  coalescent than SNPs, and if so by how much — untunable until real
  simulator output exists to check co-support structure against.
- Exact read-sampling distribution for §2.5 (uniform over founder
  assembly coordinates is the starting assumption; real sequencing
  coverage has its own biases — e.g. GC content — not modeled here).
- No real-data target yet for the coverage-dispersion statistic (§2.7)
  — only the qualitative "≫1, not ≈1" expectation is established.
- Breakpoint-snapping distance in §2.6 (how far "near" an indel
  suppression extends) — not yet parameterized.
