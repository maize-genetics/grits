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

**Biological grounding:** [`results/indel_biology_notes.md`](results/indel_biology_notes.md)
— general plant indel-mutation mechanisms plus real maize calibration
numbers measured from this project's own founder gVCFs.

## Status

Design phase — no code changes yet. See `PLAN.md`'s "This round's
deliverables" and "Future work" sections for exactly what's done and
what's next.
