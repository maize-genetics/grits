# simulator-indels

Reworking `src/python/crf/simulate_alleles.py` to emit realistic
insertion/deletion (indel) patterns relative to reference, and to output
the two-matrix training format (per-founder ternary read-sharing state +
distance-to-anchor) that the new `refmap --anchor-dist-npy` feature
produces in the separate `ropebwt3-phg` repo (branch
`lift-ridx-ternary-dist-map`, pushed, not yet merged).

**Start here:** [`PLAN.md`](PLAN.md) — the living design document. Read
it before making any change in this area; update it (with a dated
Progress log entry, same convention as `docs/PLAN.md`) as work
progresses, so any collaborator's Claude Code session can pick up
exactly where the last one left off.

**At a glance, before vs. after:**
[`results/simulator_workflow_before.png`](results/simulator_workflow_before.png)
(today's 5-stage pipeline, binary `K+2` output) vs.
[`results/simulator_workflow.png`](results/simulator_workflow.png)
(the planned 8-stage pipeline, color-coded reused / new / modified,
capped by the acceptance gate) — same box grid in both, meant to be
viewed side by side. Regenerate via `scripts/simulator_workflow_before_diagram.py`
/ `scripts/simulator_workflow_diagram.py` whenever `simulate()` or the
stage list changes.

**Biological grounding:** [`results/indel_biology_notes.md`](results/indel_biology_notes.md)
— general plant indel-mutation mechanisms plus real calibration numbers
from **two organisms' own gVCFs**, maize
([`results/indel_size_distribution.png`](results/indel_size_distribution.png))
and cassava
([`results/cassava_indel_size_distribution.png`](results/cassava_indel_size_distribution.png)),
both reproducible via `scripts/indel_size_report.py` (`--organism` flag
only affects the figure title).

**What "done" means for Phase 1:** `PLAN.md` §2.7 — acceptance criteria
grounded in real sim-vs-real gaps already measured
(`docs/notes/cassava_data_diagnostic.md`), not just "the code runs."

**Training side (Phase 2):** [`TRAINING_PLAN.md`](TRAINING_PLAN.md) — the
model/training-loop design, implemented on branch `indel-training-loop`
(`src/python/crf/crf_kernels.py`, `IndelFounderPathEncoder` in `train_crf.py`,
`train_diploid_indel.py`). Model architecture, before vs. after:
[`results/model_architecture_before.png`](results/model_architecture_before.png)
vs.
[`results/model_architecture_after.png`](results/model_architecture_after.png)
— same visual convention as the simulator-pipeline diagrams above.
Verified against a synthetic fixture only (`tests/python/crf/
test_train_diploid_indel.py`); real training still needs Phase 1's simulator
output.

## Status

Phase 1 (simulator core): in progress, no `2K+2` write path yet — see
`PLAN.md` §0 (Progress log) and §5 (Future work).

Phase 2 (training loop): implemented and verified against a synthetic
fixture — see `TRAINING_PLAN.md` §0's 2026-09-14 entry. Blocked on Phase 1
for real training data.
