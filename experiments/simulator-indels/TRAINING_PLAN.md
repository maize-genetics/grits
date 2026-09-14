# Training-loop plan for the ternary+distance ("indel-aware") model

> **Supersedes an earlier draft.** An in-place-widening approach was
> designed and considered first — gate `FounderPathEncoder.cell`'s
> width behind a new hyperparameter so `strict=True` checkpoint loading
> of `diploid-affinity-sim512-h3` still works, mirroring the `ext_bias`
> pattern. That approach was explicitly rejected in favor of the
> framing below: *"I think we should make a new round of training
> scripts as the input is quite different. I view this as a different
> model. Share any data structures or classes as needed."* If you're
> looking for the abandoned in-place design, it no longer exists in
> `PLAN.md` §5 either — this document is now the source of truth for
> the training side. Checkpoint-compatibility concerns from that draft
> do **not** carry over here (see §1) — that's not an oversight.

## 0. Progress log

- **2026-09-09**: Training-side design reframed as a new model line
  (this document created); no code changed.

### 2026-09-14 — §2/§3 implemented against a synthetic fixture (real simulator output still pending)

All of §2 (shared code) and §3 (new code) built on branch `indel-training-loop`
(worktree off `simulator-indel-modeling`, since the simulator side was
mid-edit in the shared checkout at the time — no changes made to
`simulate_alleles.py` or its in-progress `TERN_*`/`DIST_*`/`LABEL_PAD`
constants; this script defines its own literal copies with a note to
reconcile once both land on the same branch). Also delivered: two
architecture diagrams (`results/model_architecture_before.png` /
`model_architecture_after.png`, source `scripts/model_architecture_*_diagram.py`),
following the same hand-authored-SVG convention as `simulator_workflow.png`.

**§2 (shared)**: `src/python/crf/crf_kernels.py` — `build_pair_tables`,
`_dcrf_nll`, `_dcrf_viterbi`, `_dcrf_marginal`, `_dcrf_viterbi_factored`
extracted verbatim from `train_diploid.py`, which now re-imports them under
the same names so every external `from python.crf.train_diploid import
_dcrf_viterbi`-style call site (≈15 across `experiments/*/scripts/` and
`src/python/crf/*.py`) keeps working unchanged. Verified: loaded
`diploid-affinity-sim512-h3` post-extraction and scored its full 10,000-row
val split — `val_pair_acc = 0.6181` vs. the checkpoint's own recorded 0.6179
(within noise), confirming the extraction is a pure move with zero effect on
the deployed model.

**§3 (new), open question resolved**: encoder-sharing went with **(a) sibling
class** — `IndelFounderPathEncoder` added to `train_crf.py` alongside
`FounderPathEncoder`, which is untouched. Cell embedding: one-hot(4) ternary
state + one scalar log-distance → `Linear(5, d_model)`, plus an exact
516-row (4 ternary × 129 distance codes) lookup table (`fast_cells=True`),
mirroring `binary_cells`. `recomb_head` widened by one input:
`del_frac = (ternary == -1).mean(-1)` alongside the existing
`depth = log1p((ternary == 1).sum(-1))` — PLAN.md §2.6's local-crossover-
suppression claim made a directly observable recomb-head input.

New training script `src/python/crf/train_diploid_indel.py`:
`IndelDiploidDataset`/`IndelDiploidIndividualDataset`/`IndelDiploidAffinityDataset`
reading the `2K+2` layout, `GRITSCRFDiploidIndel`, `parse_args()`/`main()`
following `train_diploid.py`'s flag names plus `--fast-cells`,
`--het-inbred`/`--het-outbred` (defaults 0.23/0.50, unchanged from the binary
model — explicitly flagged as **not yet recalibrated**, printed at Dataset
construction so the real distribution falls out of the first real-data run's
log). `_founder_affinity` is imported and reused bit-identically (no
calibration constants); `_het_scale`'s body is reused with the two endpoint
constants parameterized rather than hardcoded (`_het_scale_indel`). Both
operate on `M = (ternary == 1)` — proven bit-identical to today's binary
"supported" semantic (`ternary==1` IS today's binary presence, verified
against `rb3_lift_ternary_state` in the ropebwt3-phg C source this session).

Two corrections to this document's earlier text, found reading the code
directly rather than assuming:
- **`founder_mask` does NOT exclude the pad founder** (`train_diploid.py:
  416-417`: `founder_mask = ones(B, K)`, all-ones). The null founder is a
  live decode state, so the pad value matters numerically. Resolved: the
  null-founder pad is `(TERN_DIV, DIST_PAD)`, not `(0, 0)` — `DIST_PAD` a
  genuine "no anchor" sentinel, distinct from a real distance of 0
  ("sitting exactly on an anchor," the opposite fact). Tested directly
  (`test_dist_pad_differs_from_distance_zero`).
- **`log1p` is unsafe on the distance channel too**, not just ternary — this
  document's §3 claimed distance "is already non-negative," but the real
  feature's `rb3_lift_nearest_ref` returns `-1` for "no anchor" (verified in
  `lift.c`), so `DIST_PAD=-1` breaks `log1p` exactly like `TERN_DEL=-1`
  breaks it on the ternary side. The distance scalar in the cell embed is a
  linear `dist/127` mapping (with `-1.0` for the pad), never `log1p`.

Verification run (`tests/python/crf/test_train_diploid_indel.py`, 8/8
passing, run in isolation per this repo's established convention): CRF
kernel extraction bit-exact against an independent reference reimplementation;
`fast_cells` bit-identical to the per-cell MLP path; Dataset round-trip
(`LABEL_PAD` → `K` not founder 0, `DIST_PAD` → `-1.0` not `0.0`); an
overfit-one-batch smoke test on a synthetic fixture (known founder paths +
one planted deletion tract per window) — loss falls to <50% in 150 steps,
hap_acc rises well above chance. Pair-state accuracy itself lags hap_acc in
this test (expected: no het prior is active, and `GRITSCRFDiploid`'s own
`homo_penalty` comment documents single-read data biasing pair-state decode
toward homozygous pairs without one) — hap_acc is the correct diagnostic for
"do gradients reach every new component," not pair_acc alone.

**What's still open**: real simulator output. The fixture above is
synthetic, built only to exercise the code path — it does not resemble the
real indel/coverage statistics PLAN.md §2.7 targets. Training cannot start
for real until the simulator emits the actual `2K+2` format (§5 below,
Phase 1). Also open: `--het-inbred`/`--het-outbred` recalibration (still the
binary model's 0.23/0.50 placeholders) and the §4 evaluation-tooling
`--model-class` selector, both explicitly deferred to once a real checkpoint
exists to test against.

## 1. Why this is a new model, not an upgrade

`experiments/simulator-indels/PLAN.md` scopes the simulator emitting a
new two-matrix format: `[ternary(K) | H1(1) | H2(1) | distance(K)]`,
width `2K+2`, where the first block is a **signed ternary state**
(`{-1, 0, 1}` = deletion / diverged / match relative to reference) —
not today's binary read-sharing count — and the second is a **new
distance-to-nearest-anchor block** with no analogue in the current
format at all.

Because the representation is materially different (2 channels per
cell instead of 1, with different semantics), this is treated as **a
new model with its own checkpoint lineage**, not an in-place widening
of `FounderPathEncoder`/`GRITSCRFDiploid`. This removes the dominant
source of complexity in the earlier draft — there is no need to gate
anything behind a hyperparameter so that `load_from_checkpoint`'s
default `strict=True` still accepts the old checkpoint, because the new
code never needs to load it.

**`diploid-affinity-sim512-h3`** (val_pair_acc 0.6179, trained
2026-07-23 — confirmed this session to be the exact checkpoint driving
every RIL2 dataset evaluation throughout this project, wired as
`simval_paths.py`'s `CKPT_DIPLOID`) stays deployed and untouched. It
remains the **comparison target** the new model gets measured against,
using the same established SNP+RefCall baselines (genome-wide 0.0711%
error; B73×CML103 chr5:86–134Mb IBD-zone 1.181% SNP-class error — see
`experiments/simval-corpus/`) — an evaluation-time role, not something
the new training code needs to be able to load.

This document is grounded in a source-verified audit of the current
training code (`src/python/crf/train_crf.py`, `train_diploid.py`,
`train_haploid.py`) done in the same session as this write-up — every
file:line citation below was directly read, not inferred.

## 2. What's genuinely shared (reuse via import, not duplication)

Confirmed feature-width-independent by direct read of the code. These
should move to, or be imported from, a common location — not be
copy-pasted into a new file:

- **The diploid CRF math** — `_dcrf_nll`, `_dcrf_viterbi`,
  `_dcrf_marginal`, `_dcrf_viterbi_factored`, `build_pair_tables`
  (module-level functions, `train_diploid.py:50–203`, two of the four
  kernels `@torch.jit.script`-compiled). These take no raw `X` at all
  — the pair-state transition/emission math operates purely on
  per-timestep emission/transition scores, completely independent of
  what the per-cell features look like or how many channels they have.
  **Recommend extracting these into a new shared module**,
  `src/python/crf/crf_kernels.py`, imported by both the existing
  `train_diploid.py` and the new training script. A pure extraction —
  same code, new location, both callers updated to import from it —
  rather than having the new model's file import from "the old model's
  file," which would create an odd, backwards-looking dependency for
  something meant to stand on its own.
- **`EMACallback`** (`src/python/crf/callbacks.py:11–54`) and the
  Lightning `Trainer`/`ModelCheckpoint`/`EarlyStopping` wiring pattern
  (`train_diploid.py:653–670`) — reusable as-is in the new script's own
  `main()`, no changes needed.
- **The encoder's outer structure** — `fpool` (`nn.MultiheadAttention`
  pooling), `pos_encoder` (`nn.TransformerEncoder`), `gate_head`,
  `recomb_head`, `_entropy`, `_posenc` in `FounderPathEncoder`
  (`train_crf.py:159–277`) — none of these depend on cell width; only
  `self.cell`/`_embed_cells` do. Two ways to share this, worth deciding
  once implementation starts rather than locking in now:
  - **(a) New sibling encoder class** (e.g. `IndelFounderPathEncoder`)
    in `train_crf.py`, alongside the existing `FounderPathEncoder`,
    with its own `_embed_cells` but otherwise structurally identical —
    simpler, lower-risk, duplicates ~80 lines of unchanged pooling/
    transformer/recomb-head logic.
  - **(b) Refactor `FounderPathEncoder` to accept a pluggable cell
    embedder**, with the existing binary embedder and a new
    ternary+distance embedder both implementing the same small
    interface, sharing everything else. No duplication, but requires
    touching the existing encoder and verifying — numerically, not
    just by inspection — that the existing binary path's behavior is
    bit-for-bit unchanged after the refactor.
  - **Recommendation: start with (a).** Lower risk, doesn't touch code
    the deployed checkpoint depends on at all. (b) is a reasonable
    follow-up cleanup once the new model's shape has stabilized and
    proven itself — not a prerequisite for getting it working.
  - **This is an open engineering-judgment call**, not a settled
    decision — flagged explicitly so a future implementer (or
    collaborator) revisits it rather than assuming (a) is final.
- **The split-selection pattern** used for train/val/test partitioning
  (deterministic head-slice by dataset) — same logic shape, applied to
  a new function that reads the new `.npy` layout instead of the old
  one.

## 3. What's genuinely new

- **A new cell-embedding transform.** The ternary channel cannot reuse
  `log1p` the way the binary path does — `log1p(-1) = -inf`, so a
  `deletion` state would poison the embedding outright. Needs its own
  transform (candidates: embed the raw `{-1, 0, 1}` value directly with
  no log compression, or apply a small fixed/learned offset before any
  log-like compression). The distance channel is already non-negative
  and can safely reuse `log1p`. **This is a real design decision to
  make explicitly during implementation**, not a detail to leave
  implicit or copy from the old path by default.
- **A new `depth`-equivalent feature** feeding `recomb_head`, analogous
  to today's `depth = log1p(X.sum(-1))` (`train_crf.py:257`) — must be
  derived from the ternary channel using a definition that stays
  meaningful with signed values present (e.g. a count of non-deletion/
  matching founders, not a raw sum that can go negative and make
  `log1p` produce NaN). No backward-compatibility concern here — there
  is no legacy binary path in the same class to stay consistent with —
  this is purely a from-scratch design choice, made once, correctly.
- **A new null-founder pad value** for the reference/B73 slot in the
  2-vector case (`train_diploid.py:416` pads a zero column for the old
  1-vector case) — decide explicitly what `[ternary, distance]` the pad
  founder gets. Candidate: it may not matter numerically as long as
  `founder_mask` excludes the pad founder before it reaches
  `depth`/`recomb_head` — **confirm this holds by reading the masking
  path, don't assume it** before picking a pad value.
- **New Dataset class(es)** reading the `2K+2` layout from
  `experiments/simulator-indels/PLAN.md` §2.3 (ternary block, H1, H2,
  distance block, in that column order; truth labels live in the
  `.npy` itself, no VCF needed) — structured similarly to
  `PreWindowedDiploidDataset` (`train_diploid.py:210–228`) but written
  fresh, not a modified copy of it.
- **New affinity/heterozygosity-conditioning logic.** The existing
  `_het_scale` (`train_diploid.py:255–263`, Jaccard-overlap-based,
  constants `HET_INBRED=0.23`/`HET_OUTBRED=0.50` at `:252`) and
  `_founder_affinity` (`:301–308`, mean "match rate") both assume
  bounded `0/1` input — neither is valid over `{-1, 0, 1}`. Since this
  is new code, redesign both cleanly instead of retrofitting: the
  natural analogue is to define "supported" in terms of the **match
  state only** (ternary `== 1`), the closest correct correspondent to
  the old binary "supported" semantic. The exact calibration constants
  (the `HET_INBRED`/`HET_OUTBRED` equivalents) will need **fresh
  empirical values**, not reused ones — the underlying statistic's
  distribution changes once diverged/deletion states exist alongside
  match. **Flag this as needing its own empirical calibration pass**
  once real simulator output exists (Phase 1, below) — not something to
  hand-pick blind ahead of having real data to calibrate against.
- **A new main training script and CLI.** Recommend
  `src/python/crf/train_diploid_indel.py`, with its own
  `GRITSCRFDiploidIndel` LightningModule and its own `parse_args()`.
  Should closely follow `train_diploid.py`'s existing flag naming and
  structure where the underlying concept carries over unchanged (e.g.
  `--founder-affinity`, `--time-local-emis`, `--homo-penalty`, the
  Lightning/checkpoint flags at `train_diploid.py:552–613`) so it's
  immediately recognizable to anyone familiar with the current script
  — but as a fully independent file. No new flags on the *old* script,
  no branching inside it.

**Explicitly out of scope for this document:** a haploid counterpart.
The project's active training/eval path is diploid; a
`train_haploid_indel.py`-equivalent should wait until/unless it's
actually needed, rather than being built speculatively alongside the
diploid version. (`train_haploid.py` does duplicate a null-founder pad
and a copy of `_founder_affinity` internally — noted here only so a
future haploid effort knows both spots exist, not as something this
phase touches.)

## 4. Evaluation-side implication (flagged, not designed here)

Once a new-model checkpoint exists, `experiments/simval-corpus/`'s
evaluation scripts (`simval_eval_one.py` and the other pipelines that
call `GRITSCRFDiploid.load_from_checkpoint`, e.g.
`experiments/refmap-founder-eval/scripts/nam_diploid.py`,
`experiments/tripsacum-diploid-crf/scripts/tripsacum_diploid.py`,
`experiments/simval-corpus/scripts/heldout_assembly_eval.py`,
`src/python/crf/infer_wholegenome_real.py` — the 5 hardcoded
`CKPT_DIPLOID`-style paths already known from this session's audit)
need a way to know **which model class to instantiate** before loading
a checkpoint — today every call site assumes `GRITSCRFDiploid`.
Recommend an explicit selector (e.g. a `--model-class` flag, or a small
try-`GRITSCRFDiploidIndel`-then-fall-back-to-`GRITSCRFDiploid`
dispatch) rather than silent duck-typing on tensor shapes. This is a
real, necessary follow-on change, but belongs with the evaluation
tooling once a real new-model checkpoint exists to test against — not
designed in detail in this training-loop document.

## 5. Future work (phased, each needs its own review/approval)

1. **Simulator core** (scoped separately in `../PLAN.md` §5.1) —
   prerequisite for everything below; this document's Dataset classes
   and calibration passes need real simulator output to work against.
2. **Training-loop implementation** — everything in §2–4 above:
   `crf_kernels.py` extraction, the new encoder class, new Dataset
   class(es), new cell-embedding/depth/affinity logic, and
   `train_diploid_indel.py` itself.
3. **First training run(s)** on simulator output from Phase 1, sized
   similarly to the existing `sim512`-style runs.
4. **Retrain and evaluate** against the established SNP+RefCall
   baselines (genome-wide 0.0711% error; B73×CML103 chr5:86–134Mb
   IBD-zone 1.181% SNP-class error) with `diploid-affinity-sim512-h3`
   as the explicit comparison point — not a checkpoint to load, a
   number to beat.
5. **Evaluation-tooling update** (§4) — once a checkpoint worth
   evaluating broadly exists.
6. **Decide whether to merge** `ropebwt3-phg`'s
   `lift-ridx-ternary-dist-map` branch, once (4) shows the feature is
   worth keeping on real (not just synthetic) data — same decision
   point already named in `../PLAN.md` §5.4.

## 6. Open questions

**Resolved (2026-09-14, see §0):**
- Encoder-sharing strategy — went with **(a) sibling class**
  (`IndelFounderPathEncoder`). Not revisited: (b)'s pluggable-embedder
  refactor remains a reasonable future cleanup once this model's shape has
  stabilized, per the original recommendation.
- Ternary cell-embedding transform — one-hot(4) ternary state + scalar
  `dist/127` (with `-1.0` for `DIST_PAD`) → `Linear(5, d_model)`, plus an
  exact 516-state lookup fast path.
- Null-founder pad value — `(TERN_DIV, DIST_PAD)`. `founder_mask` does
  **not** exclude it (verified false, not confirmed true as this document
  originally hoped) — see §0's 2026-09-14 entry for why the pad value
  matters numerically as a result.

**Still unresolved, need real simulator output:**
- `HET_INBRED`/`HET_OUTBRED`-equivalent calibration constants
  (`--het-inbred`/`--het-outbred` in `train_diploid_indel.py`, currently the
  binary model's 0.23/0.50 placeholders) — code prints the measured
  het-proxy distribution at Dataset construction so the real values fall out
  of the first real-data run's log, but they haven't been measured yet.
