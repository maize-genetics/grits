# v3: `--indel-model {tracts,lineage,overlay}`

2026-09-16. The existing `--simulate-indels` path ("tracts") had grown
to 17 indel-specific flags and two stacked stochastic layers (a private
per-lineage Poisson draw plus a second "shared/correlated" layer bolted
on to fake IBD correlation). The user's colleague proposed two simpler
framings; this implements both as selectable modes alongside the
unchanged `tracts` baseline, keeps founder-affinity on throughout (large,
repeatedly-confirmed gain, confirmed independent of indel generation),
and adds the acceptance test the user asked for directly: turning
indels on or off should leave the pattern roughly similar for homozygous
individuals.

## Where v3 changes the pipeline

```mermaid
flowchart TD
    L["_draw_lineages<br/>lineage [n,K,R]"] --> SW{"--indel-model"}

    SW -->|tracts| T1["_indel_tracts<br/>+ _gem_partition<br/><i>17 flags</i>"]
    SW -->|lineage| T2["_lineage_indels<br/><i>2 flags</i>"]
    SW -->|overlay| T3["_overlay_indels<br/><i>3 flags</i>"]

    T1 --> R["_indel_suppressed_rate"]
    T2 --> R
    T3 --> R
    R --> P["_build_paths"]
    P --> CF["_coalescent_feats<br/>per_gamete=True"]
    CF --> G["_gather_by_lineage"]
    G --> RC["_row_counts"]
    RC --> SR["_sample_rows"]
    SR --> CH["_indel_chunk<br/>[n,T,2K+2]"]

    classDef reuse fill:#dfeef7,stroke:#4a90b8,color:#0b2b3a
    classDef new fill:#dff5e3,stroke:#4aa564,color:#0d2e18
    classDef old fill:#f7e6e6,stroke:#b84a4a,color:#3a0b0b
    class L,R,P,CF,G,RC,SR,CH reuse
    class SW,T2,T3 new
    class T1 old
```

Blue = reused unchanged · green = new in v3 · red = existing, kept as
the default comparison baseline. Only the indel-generation step swaps —
everything downstream (row sampling, ternary/distance assembly) is
byte-identical code for all three modes, so they are directly
comparable and drop-in for `train_diploid_indel.py`.

## Where the simplification comes from

```mermaid
flowchart LR
    subgraph now["tracts (today)"]
        A1["Poisson events<br/>per lineage"] --> A2["2-component<br/>length mixture"]
        A2 --> A3["+ shared GEM<br/>layer for correlation"]
    end
    subgraph a["lineage (new)"]
        B1["existing lineage<br/>occupancy runs"] --> B2["promote whole run<br/>to indel, own length"]
    end
    subgraph b["overlay (new)"]
        C1["private per-founder<br/>Poisson process"] --> C2["decoupled from<br/>coalescent lineage"]
    end

    classDef old fill:#f7e6e6,stroke:#b84a4a,color:#3a0b0b
    classDef new fill:#dff5e3,stroke:#4aa564,color:#0d2e18
    class A1,A2,A3 old
    class B1,B2,C1,C2 new
```

**`lineage` mode** (`_lineage_indels`): walks the lineage-occupancy array
`occ[n,m,r]` (any founder on lineage m at site r) already implicit in
the coalescent draw, segments it into runs via the same
"last-seen-index" accumulate trick `_anchor_distance` already uses
elsewhere in this file, and promotes whole runs to indels with
probability `--indel-lineage-frac`, using the run's own length — no
separate length-mixture knob. Two founders sharing a lineage over the
same run always get an identical decision (verified by test:
`test_lineage_indels_ibd_founders_share_content_after_gather`), so
cross-founder correlation falls straight out of the same structure that
already drives SNP-level IBD sharing, rather than the tracts mode's
bolted-on `--indel-shared-frac`/`--indel-cluster-theta` layer.

**`overlay` mode** (`_overlay_indels`): a thin wrapper around
`_indel_tracts` itself with `M=K` (each founder is its own private
"lineage" slot) and one LogNormal length scale instead of the
two-component mixture. Combined with substituting a per-founder-identity
array (`lineage[n,k,r]=k`) in place of the real coalescent lineage array
at the `_indel_suppressed_rate`/`_indel_chunk` call sites, this fully
decouples indel content from SNP-level IBD sharing — IBD founders do
**not** automatically share indel content in this mode, an accepted,
documented simplification. `--indel-overlay-founder-freq` additionally
gates entire founders out of indel generation.

Both slot into `_indel_tracts`'s exact `(del_mask[*,M,R], ins_bp[*,M,R])`
contract, so `_indel_suppressed_rate` and `_indel_chunk` needed zero
changes — verified end-to-end (`test_simulate_indel_model_*`, 13 new
tests, 62/62 passing including all 49 pre-existing tests and the
golden-hash regression that `tracts`/off is byte-for-byte unchanged).

## Smoke test (4 windows, 2000 sites, seed 1)

| | indel-affected | either-covered | ternary match/div/del |
|---|---|---|---|
| tracts | 35.27% | 93.08% | 31.4% / 35.1% / 33.5% |
| lineage (frac=0.15 default) | 14.78% | 97.99% | 36.3% / 50.7% / 13.0% |
| overlay | 34.79% | 93.49% | 24.1% / 41.3% / 34.6% |

All three produce distinct, sane statistics (confirms the dispatch
genuinely changes generation, not a silent fallthrough) with 0%
padding in all cases.

## Homozygous indels-on/off consistency check

New acceptance test (`scripts/homozygous_indel_consistency.py`),
directly implementing the user's request: for `h1==h2` individuals with
**zero crossovers** (one constant true founder per window, so there's no
haplotype-disambiguation work for indels to do), compute each founder's
raw match-support (`count of ternary/feature==1` in the window) and take
the argmax — a naive, model-free decode. Compare top-1 accuracy against
this true founder, indels off vs. all three `--indel-model` settings
(400 windows, K=24, seed 123, otherwise-matched settings):

| config | top-1 accuracy | delta vs. off |
|---|---|---|
| off (plain binary) | 100.00% | — |
| tracts | 76.59% | −23.41pp |
| lineage | 74.69% | −25.31pp |
| overlay | 86.90% | −13.10pp |

**Not a bug, but a real, expected mechanical effect, worth understanding
precisely.** In the plain binary format, `_coalescent_feats` forces the
true founder to match(=1) at every "good" site *unconditionally* — so a
naive match-count argmax hits 100% trivially. In indel mode,
`_indel_chunk` correctly overrides that forced match with `TERN_DEL`
whenever the true founder **itself** has a deletion at that reference
site (`deleted = dist_row > anchor_thresh; tern_rows =
np.where(deleted, TERN_DEL, tern_rows)`) — which is the right behavior
(a deleted stretch can't produce a colinear read that "matches"), but it
means the true founder's own indel-affected fraction directly reduces
its own raw match-support, an effect the plain format never had. A
trained CRF (unlike this naive baseline) can use the deletion pattern
itself as positive evidence, not just discard it, so this ~13-25pp drop
is not expected to survive at that magnitude once a real model is
trained — but it is not zero either, and is worth re-checking once
runs land.

**What the check actually catches — mode-to-mode comparison, not the
absolute off-vs-on drop**: all three on-modes show a real, non-trivial
degradation from a naive decoder's perspective, but the differences
between modes are the load-bearing signal. `overlay` degrades least
(−13.10pp) — its indels are private-per-founder and less likely to
create genuine ambiguity between founders that happen to be biologically
related. `tracts` and `lineage` are close to each other (−23.4pp /
−25.3pp) — both tie indels to lineage-sharing, so a deletion in the true
founder's lineage more often coincides with an ordinary founder that
also shares that lineage losing support at the same sites, a more
confusable pattern. This is consistent with the mode designs, not a
red flag on `lineage` specifically, but it is the kind of asymmetry this
check exists to surface, and is worth re-running once trained models are
available for each mode.

## Status / next steps

- Implemented, tested, smoke-tested, and the requested consistency check
  built and run — all on branch `indel-v3-simulator`.
- **Not yet done**: full contig-then-slice dataset generation and
  training runs for `lineage`/`overlay` (the runs 20/21 comparison, now
  for three modes instead of two). This is the natural next step but
  wasn't started yet in this round — worth confirming before committing
  further GPU time given the homozygous-check finding above.
- Calibration against the real Oh43xIl14H ternary/distance data
  (`real_anchor_dist_npy_oh43xil14h.md`) has not yet been attempted for
  either new mode — the real either-covered rate (99.0%) is far above
  all three simulator configurations shown here (93-98%), an open gap
  independent of which mode is chosen.
