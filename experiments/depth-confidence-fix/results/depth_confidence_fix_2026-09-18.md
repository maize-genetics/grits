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

### Window-density feature (`window_density`) + mixed-depth retrain — negative result so far, on branch `indel-density-features`

The stride/subsample fix above is diagnostic-only (it discards real
reads). The non-wasteful alternative: give the model an explicit
per-window evidence-density signal so it can *discount* redundant
clustered evidence itself while still seeing every real read, rather
than treating a genomically-compressed high-depth window as 512
independent observations.

**Design, revised after a probe gate.** A per-ROW version (gap-to-
previous-row, read-stack-depth) was designed first and tested directly
against the deployed model's real predictions
(`scripts/density_probe.py`) before any implementation: per-row values
do **not** separate correct from incorrect predictions within a fixed
depth (HYB 2.0x: mean_gap[correct]=0.568 vs mean_gap[wrong]=0.647;
stack-bucketed error rate flat 2.08–2.32% across stack=1–10). The large
depth→error effect is real but lives at the whole-window level, not the
per-row level. Redesigned to two per-window scalars —
`compute_window_density(site_idx, valid)` in `simulate_alleles.py`:
`window_span` (last−first reference-site index among valid rows) and
`window_uniq_sites` (distinct site count; `T/uniq_sites` is mean stack
depth) — delivered as a small `[N,2]` sidecar (`--emit-density`, both
producers, byte-identical main-array contract otherwise) rather than
widening the `[N,T,2K+2]` array. Consumed via a zero-initialized
`density_proj` in `IndelFounderPathEncoder`, broadcast-added once per
window right where positional encoding is added — reaches emissions,
gate, and transition potential. Zero-init (constructed *last* in
`__init__`, after every other layer, so its weight-init RNG draws don't
shift any other layer's fixed-seed initialization — verified against
`TestOverfitSmoke`) means a freshly-widened model is numerically
identical to the deployed checkpoint, confirmed by a new regression
test (`TestBitIdenticalWarmStart`) pinning exact pre-existing pair_acc/
ibd_adj numbers on real HYB/RIL2 data.

**Mixed-depth training data.** The default `--indel-coverage-model
linear` gives `P=1.0` (guaranteed read at every present site) at the
module's own defaults — simulated data never varies genomic compression
at all. Generated a small validation-scale mixed-depth set instead:
`--indel-coverage-model reads` (real contiguous-fragment sampling) at
0.1x/0.5x/1.0x/2.0x (region-mult tuned per depth for ~0% short-window
padding: 40/10/6/6), 200 windows/depth, `--windows-per-individual 20`,
concatenated and individual-block-shuffled (so train/val/test splits
mix depths, not confounded by file order) into one 800-window set.
Decoded `window_density` confirms a clean depth gradient survived
concatenation: mean span 8,619→666 bin-units, 0.1x→2.0x (~13x, matching
the real-data compression ratio that motivated this whole investigation).

**Retrain**: new `--warm-start-ckpt` flag (`train_diploid_indel.py`) —
`--resume`'s `ckpt_path=` mechanism requires an exact architecture match
(Lightning's strict-mode state-dict load), which fails once
`density_proj` exists; `--warm-start-ckpt` loads only the weights
(`strict=False`) and starts a genuinely fresh optimizer/epoch, not a
resumed one. Warm-started from `diploid-indel-v3-k25-overlay-affinity`,
5 epochs, same hyperparameters recovered from that checkpoint
(`time_local_emis`, `homo_penalty=3.0`, `spike_skip`, `founder_affinity`,
`batch_size=64`, `bf16-mixed`). Training healthy — no crash, val_pair_acc
climbed monotonically each epoch (0.021→0.033→0.090→0.133→0.208) on this
new, harder synthetic task (`--min-founders 2 --max-founders 25`, up to
25-way founder mixtures — not directly comparable to v3-K25's own 0.68).

**Real-data result — negative, and doesn't yet test the actual
hypothesis.** `ropebwt_npy_to_matrix.py --emit-density` was never run
against the cached real HYB/RIL2 data, so `window_density=None`
throughout this eval on *both* checkpoints — `density_proj` never fired.
What this measures is purely 5 epochs of warm-start fine-tuning on the
small mixed-depth set:

| depth | HYB v3-K25 (baseline) | HYB density-mixed-depth (new) | RIL2 v3-K25 | RIL2 density-mixed-depth |
|---|---|---|---|---|
| 0.01x | 100.00% | 98.65% | 96.92% | 97.77% |
| 0.1x | 99.03% | 96.05% | 99.62% | 99.74% |
| 0.5x | 98.08% | 93.93% | 99.63% | 99.73% |
| 1.0x | 97.13% | 92.51% | 99.57% | 99.64% |
| 2.0x | 96.50% | 91.73% | 99.42% | 99.58% |

HYB — the case this whole investigation is about — got **worse at every
depth**, and proportionally worse depth-degradation, not better (baseline
drops ~3.5pp 0.01x→2.0x; new checkpoint drops ~7pp). RIL2 improved
slightly at every depth, a small win against a much clearer HYB loss.

Most likely cause: **catastrophic forgetting**, not a failure of the
density-feature idea — the validation set's `--min-founders 2
--max-founders 25` individual composition looks nothing like a real
2-founder F1 hybrid, so 5 epochs fine-tuning on it plausibly pulled the
model's other weights away from what it already did well on that
specific case, while incidentally helping RIL2-like harder cases.
**The density-feature hypothesis itself remains untested on real data**
— `density_proj` never received a non-zero input in this eval. Next
steps before any further verdict: (1) run `--emit-density` against the
cached real HYB/RIL2 npy so `window_density` is actually non-null at
real inference time, (2) fix the training-composition mismatch (narrower
founder-count range matching real hybrids more closely, and/or more
scale/epochs) so warm-starting doesn't erode existing hybrid-decoding
skill before the density signal gets a fair test.

Branch `indel-density-features` (cut from `indel-v3-simulator`, disposable
per this session's explicit agreement) — not merged, not promoted.
