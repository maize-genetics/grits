# Session handoff — crf-relatedness (2026-09-09, indel-modeling design)

Status snapshot for resuming work. Full per-experiment detail is in
[`docs/RESULTS.md`](RESULTS.md); experiment spec in [`docs/PLAN.md`](PLAN.md).
Older active threads (2026-08-28 branch consolidation, 2026-07-13 sim→real
transfer, 2026-07-10 checkpoint/eval hardening + sim data regen, 2026-06-28
whole-genome, 2026-06-26 het, 2026-06-25 IBD-ceiling, 2026-06-24 E1-maize) are
below as history.

## ACTIVE (2026-09-09): realistic indel modeling for the training simulator — DESIGN DONE, no code changed yet

**Context.** The 2026-08-28 entry below committed to a follow-through
chain after that session's `ropebwt3-phg` work (branch
`lift-ridx-ternary-dist-map`, in the separate `ropebwt3-phg` repo,
pushed, not yet merged there): rework `src/python/crf/simulate_alleles.py`
to emit synthetic training data matching the new `refmap --anchor-dist-npy`
feature (a per-founder ternary read-sharing state — match/diverged/
deletion — plus a distance-to-nearest-anchor value), then widen the
model's input, then retrain. This entry scopes and designs the simulator
side of that chain. This is now explicitly a **shared, ongoing project
between the user and collaborators working on the same machine** — see
`experiments/simulator-indels/` for the durable, git-committed design
document collaborators should read and edit directly, rather than
relying on any session-local planning file.

**Full design:** [`experiments/simulator-indels/PLAN.md`](../experiments/simulator-indels/PLAN.md)
— read this first before touching `simulate_alleles.py`. Biological
grounding (general plant indel-mutation literature + real maize
calibration numbers from this project's own founder gVCFs):
[`experiments/simulator-indels/results/indel_biology_notes.md`](../experiments/simulator-indels/results/indel_biology_notes.md).

**Summary of what was decided this round** (all detail, sources, and
exact code citations are in the linked `PLAN.md` — this is a summary,
not a duplicate):
- **Goal, clarified beyond the original ask:** the ternary+distance
  features aren't just about indel realism for its own sake — they're
  the intended mechanism for letting the model directly observe (rather
  than infer) whether a locus is genuinely **heterozygous** (both
  homologs present, different founders), **hemizygous** (one homolog has
  a real deletion relative to reference, so only one copy exists), or
  showing **recombination** (a real founder-path switch, no deletion).
  An earlier, unimplemented idea for this same problem (`docs/PLAN.md`'s
  "E10", a per-read position float) is explicitly **superseded and
  deferred** — it wanted read-position clustering to indirectly infer
  what the ternary state now observes directly. If E10 is ever revisited,
  its conflict with this feature over `FounderPathEncoder.cell`'s
  per-cell embedding slot must be resolved first.
- **Coordinate system:** no new/independent bp-space model. B73/reference
  is the coordinate backbone (matching the real pipeline exactly); each
  founder gets a running offset relative to it, diverging through indel
  runs — the same mechanism as the real `.lift`-anchor slope/distance
  logic in `ropebwt3-phg`.
- **Indels are a property of the founder's own genome**, not
  redrawn per individual — any individual inheriting a founder's
  haplotype at a locus inherits its indel structure there.
- **Output is exactly 2 matrices**, verified against the actual
  committed `ropebwt3-phg` C code (not assumed from memory): the
  simulator's matrix 1 **is** the ternary state, replacing today's
  binary features in place (same column position) — unlike the real
  refmap C code, which keeps a legacy-unchanged 3rd block purely for its
  own backward compatibility, a constraint that doesn't apply to a new
  simulator format. Matrix 2 (distance) is a new trailing block after
  the unchanged label columns. No VCF-shaped output — the model never
  reads one.
- **Indel size model:** a two-component mixture (small,
  replication-slippage-driven; large, LTR-retrotransposon-driven),
  mechanistically motivated and calibrated from real maize founder gVCF
  data (excluding a confirmed assembly-quality outlier, A188) — not a
  single distribution, and not maize-specific in structure.

**Explicitly not done in this round:** no changes to
`simulate_alleles.py`, `train_diploid.py`, or `train_crf.py`.
Implementation is deferred to separately-approved future phases listed
in `experiments/simulator-indels/PLAN.md` §5 (simulator core → training-
side format/encoder updates → retrain and evaluate against the
established SNP+RefCall baselines → decide on merging
`lift-ridx-ternary-dist-map`). Whoever picks this up next should expect
to start at Phase 1 of that list, and should update
`experiments/simulator-indels/PLAN.md`'s own Progress log (§0) directly
— that document, not this one, is now the source of truth for this
effort's day-to-day state.

## ACTIVE (2026-08-28): branch consolidation + grits_workdir scripts/results brought in — DONE, review pending

**Context.** Several weeks of RIL2 founder-path-decode-error investigation
happened in a separate, non-version-controlled working directory
(`grits_workdir`, `/local/workdir/zrm22/HackathonJun2026/grits_workdir`)
while this repo accumulated 5 uncoordinated feature branches. Before a
vacation break, both were brought under control in one pass.

**1. Branch sprawl resolved.** 5 branches
(`tripsacum-tests`, `tier2-constant-pair-individuals`,
`wholechrom-decode-real-data`, `windowing-quality-filters`,
`perf-allele-multiset-score`) turned out to be 3 real units with zero
file-level conflicts (`tripsacum-tests` was the shared base for two of
them). Unified onto `integration/aug-2026-consolidation` (off the real
`develop` tip), excluding `windowing-quality-filters` (mixed
`--max-hit-frac` results — see its own `RESULTS.md` on that branch,
still present, not deleted) — which also sidesteps a real
`src/python/crf/eval.py` vs. `eval/` module-shadowing bug that branch
introduced. The 4 merged branches are deleted, local + `origin`. One
real merge conflict, in `ps4g_io/ps4g.py`: `develop` had already
evolved `build_index_lookup` past a stale copy carried by
`tripsacum-tests`, plus an uncommitted WIP patch to that stale copy sitting
in the main checkout — resolved by taking `develop`'s already-general
`distinguish_gametes`-parameterized version wholesale. That uncommitted
WIP (BED→VCF coordinate/AC-AN/gamete-label correctness fixes, needed for
AF-binned accuracy tables that have been silently empty/wrong genome-wide)
is preserved as its own commit, explicitly labeled incomplete (2 known
test-expectation updates + a missing `VCFInfoHeaderLine` still needed).
`feat/ps4g-viewer-column-mode` (GUI/Rust/Tauri) confirmed genuinely
disjoint, left untouched. **Full detail, exact commands, and the
verification trail: `/home/zrm22/.claude/plans/wondrous-discovering-octopus.md`
(session-local; if inaccessible, this summary plus `git log
integration/aug-2026-consolidation` covers it).**

**2. `grits_workdir`'s scripts + results brought into the repo**, on
branch `bring-in-simval-corpus` (off `integration/aug-2026-consolidation`,
pushed, **not yet merged — awaiting review**), as five new
`experiments/` directories matching the existing
`experiments/refmap-founder-eval/`, `experiments/tripsacum-diploid-crf/`
convention:
- **[`experiments/simval-corpus/`](../experiments/simval-corpus/README.md)**
  — the core pipeline: build/locate the 9-dataset simulated-validation
  corpus, run one (dataset, coverage) row, SNP+RefCall re-scoring, the
  canonical RIL2 driver. **Start here for "how do I run RIL2 or any of
  the other datasets" — this README has copy-pasteable commands and every
  known gotcha** (a stale `drop_idx=23` default, an unmerged-branch
  dependency, a TSV-truncation footgun in the RIL2 scorer).
- **[`experiments/ril2-error-regions/`](../experiments/ril2-error-regions/README.md)**
  — the founder-path decode-error investigation itself (anchor density,
  IBD confirmation, zero-error control, position-randomization probes).
- **[`experiments/crf-model-internals/`](../experiments/crf-model-internals/README.md)**
  — encoder/CRF internals probing.
- **[`experiments/reference-bias/`](../experiments/reference-bias/README.md)**
  — the `refmap` vs. `chain`+PAV reference-bias study.
- **`experiments/cassava-diploid-crf/`** (new) + a 1-line correctness fix
  to the existing `experiments/tripsacum-diploid-crf/scripts/tripsacum_recombination.py`
  (`combo_name` was being called with the wrong signature — would have
  raised `TypeError` if run).
Scripts already identical in the repo (the early NAM founder-recovery era
— `kmer_sweep.py`, `nam_baseline.py`, `nam_diploid.py`, `nam_trim_sweep.py`,
`trim_reads.py` — and 2 of 3 Tripsacum scripts) were skipped rather than
duplicated; cross-directory imports (`nam_baseline`, and the ~25 scripts
across the new dirs that import `simval-corpus`'s core modules) were
fixed with explicit `sys.path` entries, verified by importing all 84
moved Python files in isolated subprocesses. Only scripts, `README.md`s,
and findings docs + small curated summary tables are committed — the raw
per-row JSON/log/`.npz`/`.png` output these scripts produce (~10MB+
across the 5 new dirs, most of it hundreds of files deep) was
deliberately kept out of git and stays local-only, in
`grits_workdir/results/`, regeneratable via the scripts. `filter_ps4g.py` and
`run_ril2_windowfilter_test.py` were deliberately left in `grits_workdir`
(hard-depend on the excluded `windowing-quality-filters` branch).

**3. Next steps — rework the simulator, update the model, retrain.**
This session also landed a new `refmap` feature on a **separate repo**
(`ropebwt3-phg`, branch `lift-ridx-ternary-dist-map`, pushed, not yet
merged): `--anchor-dist-npy` widens PS4G/npy output with a per-founder
ternary read-sharing state (match/diverged/deletion) and a
distance-to-nearest-`.lift`-anchor value, targeting the two confirmed
root causes above (binarization losing the signal that would resolve
IBD-zone confusion; repetitiveness being invisible to the model). The
follow-through, in order, **not done yet**:
1. Rework `src/python/crf/simulate_alleles.py` to emit matching synthetic
   ternary + distance features (it currently emits strictly binary ones)
   — needed before any retrain can use the new `refmap` output for real.
2. Widen `FounderPathEncoder.cell` in `src/python/crf/train_crf.py` from
   `nn.Linear(1, d_model)` to `nn.Linear(2, d_model)` (a per-cell 2-vector,
   not a wider founder axis) — checkpoint-compatible via zero-init on the
   new column, mirroring the existing `ext_bias`/founder-affinity pattern.
3. Retrain, re-evaluate via `experiments/simval-corpus/`'s SNP+RefCall
   methodology against the existing baselines (genome-wide 0.0711%,
   B73xCML103 chr5:86-134Mb IBD-zone 1.181% SNP-class error).
4. Decide whether to merge `lift-ridx-ternary-dist-map` once (3) shows
   the feature is worth keeping.
Whoever picks this up next may only reach step 1 before other commitments
— if so, leave a note here on exactly where it stopped.

## ACTIVE (2026-07-13): sim→real transfer eval + real-data consistency check — DONE

**Question (user-set).** Do the sim-trained checkpoints from 2026-07-10 (`haploid-sim`,
`diploid-sim512-h3`) carry any real signal on **real** data, and is
`fullMaizeDataset_all_diploid.npy` still consistent with the documented real-data findings
(the 72% either-match / true-founder-deficit build problem)?

**Data consistency — CONFIRMED unchanged.** Re-derived the either-match diagnostic (same
logic as `crf/sfs_sharing.py`'s `analyze()`; `matplotlib` isn't installed in this pixi env so
replicated the print-table computation standalone — no repo files changed) on the current
copies of both real files:

| file | het | H1 | H2 | either |
|---|---|---|---|---|
| Real maize | 0.00 | 72% | 72% | **72%** |
| Real cassava | 1.00 | 42% | 41% | **72%** |

Exact match to the documented table in `docs/notes/cassava_data_diagnostic.md`. The real-data
build (true-founder deficit, sim ~96% either-match vs real 72%) is unchanged since it was
diagnosed — still the real-data **BUILD problem** flagged to zrm22, not fixed by anything in
between, and not a folding-artifact/measurement issue (that caveat was specific to the SFS
*shape*, not this either-match number — see the 2026-06-25 section below).

**Sim→real transfer results** (scored with the new `crf/eval.py`; both checkpoints trained
2026-07-10 purely on simulated data, never on real data; rows appended to
`grits_workdir/results/eval.tsv`):
- **Haploid** (`haploid-sim` ckpt → real maize, `--limit-n 250000 --split test` — the
  identical held-out rows the historical E1-maize benchmark used): Viterbi acc **0.560** vs
  the real-*trained* benchmark 0.682 (CRF) / 0.614 (HMM); emission-only 0.279. The learned
  transition structure still adds real value cross-domain (+0.28 over emission-only) but
  breakpoints are heavily over-called (precision 0.052 @±0, pred 822,647 vs true 141,229) —
  sim-trained transitions read real data's missingness (72% either-match) as near-constant
  switching.
- **Diploid** (`diploid-sim512-h3` ckpt → real **cassava**, not maize — maize is inbred
  H1==H2 and a poor match for a model trained on *outbred* sim data; the real, outbred,
  same-layout comparison target is `NonCollapsedDatasets/singleShuffledCassavaDataset/
  fullMaizeDataset_all_diploid.npy` — mislabeled filename, content is real cassava,
  `(278053, 512, 26)`, het=1.00, discovered this session): pair_acc **0.0357**, hap_acc
  **0.2088** — near-identical to the documented diploid-HMM-on-cassava baseline (~0.049/0.215,
  `docs/notes/cassava_data_diagnostic.md`). On this file the 72%-either-match ceiling
  dominates: there's little daylight between the sim-trained CRF and a hand-tuned HMM here —
  data quality is the binding constraint, not the model.

**Conclusion.** Sim→real transfer partially works — haploid clearly beats its own
emission-only floor and shows genuine structure-learning transfer, but is capped hard by the
real-data build deficit, especially on cassava where the ceiling swallows any model
difference. This confirms the ropebwt3 reference-coordinate rebuild (next-week item) is the
actual lever to pull, not further CRF modeling — matches what was already suspected from the
E4/E-IBD threads.

**Commands (this session, GPU 0, single short inference pass each — not sustained training):**
```bash
# real maize (haploid) — historically-comparable split
CUDA_VISIBLE_DEVICES=0 ~/.pixi/bin/pixi run -- python src/python/crf/eval.py \
    --ckpt <WORKDIR>/checkpoints/haploid-sim/last.ckpt \
    --data NonCollapsedDatasets/singleShuffledMaizeDataset/fullMaizeDataset_all_diploid.npy \
    --num-parents 24 --limit-n 250000 --split test --workdir <WORKDIR>

# real cassava (diploid) — note the misleading filename, content is cassava not maize
CUDA_VISIBLE_DEVICES=0 ~/.pixi/bin/pixi run -- python src/python/crf/eval.py \
    --ckpt <WORKDIR>/checkpoints/diploid-sim512-h3/last.ckpt \
    --data NonCollapsedDatasets/singleShuffledCassavaDataset/fullMaizeDataset_all_diploid.npy \
    --num-parents 24 --split test --workdir <WORKDIR>
```

**Gotcha for next time:** this box gives us only **1 CPU core**, so CPU-only `eval.py` on
25k+ windows is impractically slow (hit a 5-minute timeout with no completion). Used a single
short GPU inference pass instead (confirmed both GPUs idle at the time) — that's a very
different footprint from a sustained training job, but still check GPU load first on this
shared machine before running even a quick eval.

**NEXT STEPS:** unchanged from 2026-07-10 — this branch still has no real-*trained* model
(only sim-trained checkpoints evaluated on real data so far); the branch/PR decision is still
open; and the ropebwt3 rebuild (next week) now has a concrete before/after baseline to check
against once it lands (does either-match rise meaningfully above 72%?).

## ACTIVE (2026-07-10): checkpoint/eval hardening + regenerated training data — DONE, pushed

**Branch note:** `crf-relatedness` is now **in PR** — do not commit new features to it. All
work this session is on a new branch **`crf-checkpoint-eval`** (branched off
`crf-relatedness`, pushed to `origin`). Same rule applies going forward until that PR lands.

**Done this session (committed `2620d1c`, pushed to `origin/crf-checkpoint-eval`):**
- **Checkpointing was already mostly working** (both `train_haploid.py`/`train_diploid.py`
  already had best-val `ModelCheckpoint`) but had a real bug: the monitored-metric filename
  token (`{val/loss}`, `{val/pair_acc}`) contains a `/`, which Lightning treats as a path
  separator — checkpoints were landing in a stray `val/` subdirectory instead of directly
  under `checkpoints/<run-name>/` (the old handoff's `e1-*-val/loss=*.ckpt` glob was a
  workaround for exactly this). **Fixed**: log a slash-free alias (`val_loss`, `val_pair_acc`,
  `val_total_loss` incl. legacy `train_crf.py`) and monitor/interpolate that instead. Also
  added `save_last=True` (stable resume anchor), `--resume` → `Trainer.fit(ckpt_path=...)`,
  and gave `train_haploid.py` the same `--ema`/`--ema-decay` option `train_diploid.py` already
  had (extracted `EMACallback` into new `crf/callbacks.py`, shared by both).
- **New unified `crf/eval.py`** — one evaluator for both `GRITSCRFHaploid` and
  `GRITSCRFDiploid` checkpoints, auto-detecting the type from the checkpoint's `state_dict`
  (diploid registers `pair_table`; haploid does not) instead of needing two separate scripts
  (`eval_test.py`/`eval_test_diploid.py`, left in place but superseded going forward). Founder-
  path accuracy only (no label-free masked-site metric this pass). Also fixed the old
  hardcoded-`cuda` device bug so eval runs CPU-only too.
- **Verified** via a CPU smoke test (tiny sims, 2-epoch train, `--resume`, `--ema`) before
  touching real scale — see commit `2620d1c` message for the full verification list.

**Data regen (the old `/workdir/esb33` — all sim data + the default workdir — is now
permission-denied to us):** regenerated the historical sims at a **new workdir**,
`/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/training/` (use this as the working
substitute for `/workdir/esb33` until access is restored):
```bash
# haploid (pure defaults; matches old sim_alleles.npy / E1-probe exactly)
python src/python/crf/simulate_alleles.py --workdir <WORKDIR>
# diploid outbred (matches old sim_diploid_512.npy / E4-probe: F=0, coalescent, cx 1-4/gamete)
python src/python/crf/simulate_alleles.py --workdir <WORKDIR> --out sim_diploid_512.npy \
    --sites 512 --min-crossovers 1 --max-crossovers 4 \
    --inbreeding 0 --gamete-balance 0.5 --sharing-model coalescent
```

**Trained + scored both (GPU 0 only, strictly sequential — GPU 1 left free the whole time for
another user on this shared machine):**
```bash
CUDA_VISIBLE_DEVICES=0 ~/.pixi/bin/pixi run -- python src/python/crf/train_haploid.py \
    --data <WORKDIR>/data/training/sim_alleles.npy --workdir <WORKDIR> \
    --time-local-emis --lr 1e-4 --warmup-steps 500 --precision bf16-mixed \
    --max-epochs 5 --run-name haploid-sim

CUDA_VISIBLE_DEVICES=0 ~/.pixi/bin/pixi run -- python src/python/crf/train_diploid.py \
    --data <WORKDIR>/data/training/sim_diploid_512.npy --workdir <WORKDIR> \
    --time-local-emis --lr 1e-4 --warmup-steps 500 --precision bf16-mixed \
    --max-epochs 5 --homo-penalty 3 --run-name diploid-sim512-h3

# eval (either checkpoint, either model type — eval.py auto-detects)
CUDA_VISIBLE_DEVICES=0 ~/.pixi/bin/pixi run -- python src/python/crf/eval.py \
    --ckpt <WORKDIR>/checkpoints/<run-name>/last.ckpt \
    --data <WORKDIR>/data/training/<sim>.npy --split test --workdir <WORKDIR>
```
- **Haploid** (5 epochs, d256/L6 5.07M): held-out **test Viterbi acc 0.9945** (emission-only
  0.2375; breakpoint F1 ±2 = 0.962) — matches the historical PLAN.md §0 architecture-probe
  result (val acc → 0.99).
- **Diploid** (5 epochs, homo-penalty 3): held-out **test pair_acc 0.6186, hap_acc 0.7467**,
  predicted-homozygous 0.0005 — near-exact match to the historical E4-probe result (pair_acc
  0.614, hap_acc 0.743) despite being a fresh regeneration with a new seed realization. This is
  a strong end-to-end validation that the regenerated sim, hardened training scripts, and new
  `eval.py` all reproduce prior results faithfully.
- Checkpoints: `<WORKDIR>/checkpoints/{haploid-sim,diploid-sim512-h3}/`. Results table:
  `<WORKDIR>/results/eval.tsv`.

**Gotcha for Monday — `pixi install --environment gpu` is BROKEN** (pre-existing, unrelated to
this session's changes): fails building `mamba-ssm` from source — conda's pinned
`cuda-toolkit 12.8` mismatches the PyPI `torch` wheel that resolves (`2.13.0+cu130`), so
`nvcc`/`torch.utils.cpp_extension` reject the mismatch. `mamba-ssm` is only needed by the
unused `bimamba/` reference encoder — **none of the CRF scripts import it**. Workaround (used
successfully all session): skip `--environment gpu` entirely, use the plain **`default`** env,
which already has a CUDA-capable `torch` (confirmed `torch.cuda.is_available()==True`, sees
both H200s) — `CUDA_VISIBLE_DEVICES=N ~/.pixi/bin/pixi run -- python <script>` is all GPU
training/eval needs.

**NEXT STEPS (Monday):**
1. Decide whether `crf-checkpoint-eval` merges into the `crf-relatedness` PR or opens its own
   against `crf-relatedness`.
2. **The original headline goal, still open**: train/eval on **real** maize data —
   `NonCollapsedDatasets/singleShuffledMaizeDataset/fullMaizeDataset_all_diploid.npy`
   (`(1346838, 512, 26)` int8, labeled, pre-windowed — feeds `make_splits`/
   `make_diploid_splits` directly, `--limit-n` for a subset as the old E1-maize run did). The
   sibling `fullMaizeDataset.npy` (no `_all_diploid`) is broken — no label column, don't use.
3. **Next week**: rebuild the real maize dataset via the new ropebwt3 reference-coordinate
   lookup option (sibling `../ropebwt_refMap/` dir) and re-score with `eval.py` — same matrix
   layout, no code changes needed, `eval.py` is the measurement tool for that comparison.
4. Broader `crf-relatedness` roadmap (E5 relatedness conditioning, E6 SNP r², etc. — see
   `docs/PLAN.md`) is unaffected by today's work and remains open.


## ACTIVE (2026-06-28): whole-genome scale + the imputation metric — DONE, pushed

**Done this session (committed + pushed to `origin/crf-relatedness`, head `731ec86`):**
- **E12 whole-genome** edge-free decode: encoder on overlapped 1024 windows, emissions
  stitched by center-crop, **one Viterbi over the full 100k-site chromosome**. Matches
  E11 (dense ALL 0.795 pair). `infer_wholegenome.py` (`--mode whole-chrom`).
- **Masked-site SNP accuracy** (`--mask-frac 0.01`, the imputation metric): hold out 1%
  of **whole sites**, decode, score **label-free either-match** at the held-out sites
  (correct if *either* decoded founder reproduces the hidden read — no founder truth
  needed → runs on real data). Dense SNP **0.93–0.96 ≫ founder-ID** (allele-redundancy
  cushion: outbred-het 0.55→0.88); sparse SNP ~0.967 at the bad-frac ceiling. RESULTS
  E12-msnp.
- **Affinity-pruned smaller CRF** (`--prune-affinity`/`--compare`): P 325→6 (k2-F2) /
  →45 (k8-S1); **~10× CPU decode** (launch-bound on GPU). **Safe on dense, UNSAFE on
  sparse** (drops sparsely-carried founders, k8-het 0.99→0.31). Keep opt-in, default off.
  RESULTS E12-prune; memory [[project_affinity_prune_sparse_unsafe]].
- **REPORT.md extended E1–E8 → E1–E12**; two new figures (`fig9_e12_wholegenome`,
  `fig10_e12_prune`) in `figures.py`.

**NEXT STEPS (user-reviewed 2026-06-28).** Real-genotype testing is the headline goal;
priority among the rest is by how much each *de-risks* that step in simulation first.

1. **Dosage r² benchmark + posterior dosage — USER LIKES THIS, do first.** Convert our
   accuracy (founder-path + either-match) into the metric the field uses:
   **MAF-stratified dosage r²** vs a fixed SNP panel. Pieces exist — simulator's
   `emit_snp_panel` (`G` = founder×SNP alleles, `_coalescent_feats(..., emit_panel=True)`)
   and the `--mask-frac` harness. Combine with the **forward-backward marginal**
   (`_dcrf_marginal`, already present) for a **posterior expected dosage** + per-site
   **confidence track**, and a rare-vs-common r² curve. Gives a Beagle/STITCH-comparable
   number *before* touching real data.

2. **Clustered (blocky) bad data — the real concern, NOT uniform bad-frac.** The real
   maize/cassava matrices had a **build bug**, so the old "72% either-match ceiling /
   28% missing-true-founder" number is suspect — don't treat it as fundamental. The
   failure mode to actually simulate is **clustered** dropout: runs/blocks of bad sites,
   not iid. The simulator **already supports this** — `_good_mask(rng, n, T, bad_frac,
   block)` is a 2-state Markov chain with **mean bad-run length = `block`** (the
   `--error-block` / `error_block` knob). So: sweep `error_block` (and bad_frac) up to
   realistic clustered regimes, measure where founder-path and r² fall off, and if it's
   steep give the model an explicit **"none-of-panel" escape** (the null founder index K
   exists but isn't trained as an out-of-panel emission). Cheaper to find the cliff in
   sim than on real data.

3. **Posterior confidence + auto pruning-gate — good idea (user).** The #1 marginal
   posterior also lets you (a) **flag/mask low-confidence calls** instead of forcing a
   hard Viterbi everywhere, and (b) **auto-gate pruning**: enable the reduced CRF only
   when held-out masked-SNP doesn't drop (exactly the sparse failure). Turns the
   pruning caveat into a feature.

4. **Batched GPU Viterbi — DEPRIORITIZED, open question.** The decode is launch-bound on
   the 100k-step T-loop *per chromosome*. **User's point: this may not matter** once you
   analyze **1M sites across all chromosomes together**, or run **~1000 samples at once** —
   the batch dimension already amortizes each kernel launch over B× work. So first just
   **measure** whole-genome throughput at a realistic batch (B samples × chromosomes);
   if the GPU saturates, #4 is moot and skip it. Only worth fused/batched launches if
   it stays launch-bound under real batching.

**Implementation note for #1/#2:** `simulate_wholegenome.py:86` already calls
`simulate(...)` and captures the panel return (`_panel`, currently discarded) and passes
`bad_frac=args.bad_frac` — but it does **not** pass `error_block` (defaults to 1.0 = iid)
or `emit_snp_panel=True`. So both next steps need a small thread-through there: expose
`--error-block` and `--emit-snp-panel`, persist `_panel` alongside the `.npy` (panel needs
`coalescent` mode, already on). The decode/marginal side (`_dcrf_marginal`) and the
masked-site harness are already in `infer_wholegenome.py`.

**Files (this session):** `infer_wholegenome.py` (`--mask-frac`, `--prune-affinity`,
`--compare`, `masked_snp`), `simulate_wholegenome.py`, `inspect_affinity.py`,
`figures.py` (fig9/fig10), `docs/REPORT.md`, `docs/RESULTS.md` (E12-*).

## ACTIVE (2026-06-26): diploid heterozygosity — diagnosed and fixed

**The question (user-set).** Why couldn't the encoder inform the local CRF *transition*
probabilities to force heterozygosity? Plant genotyping is ≤1 read/locus, so het must be
read from the **sustained rapid alternation** of two haplotypes in the single-read stream.

**Diagnosis (RESULTS E7-diag) — definitive.** It can't, and not for lack of training:
pull the `homo_penalty` crutch (`--homo-penalty 0`, stable recipe) and the model collapses
to homozygous — `predHet=0.000` even where `trueHet=0.795`; `c` flat het-vs-homo (1.01).
**Mechanism:** stable-het `{A,B}` and stable-homozygous `{A,A}` have ~equal emission over
an alternating-read window AND both have zero transitions, so no transition cost can break
the tie. Het preference must be **emission-side**. (`diag_c_het.py` instruments this.)

**Fix (RESULTS E7-fix) — works, best result yet.** `--learned-het`: a gated `het_head`
on the encoder emits a **per-locus** het logit from the window context; `softplus(het)`
becomes a per-site homozygous penalty replacing the fixed/per-individual `homo_penalty`.
**Outbred pair_acc 0.171 → 0.516** (beats hand-tuned hp3 and per-individual adaptive,
both 0.445); best all-band 0.689. `predHet` now tracks `trueHet` across F buckets — the
encoder learned to detect het regions from the alternation pattern. Ckpt `e7-learnedhet`
(best-val 0.6996, ep4).

**Open / next:** (1) slight inbred over-fire (`predHet 0.21` where true 0; inbred
0.826→0.753) — tune (sharper prior / het regularization / longer train); (2) diploid tail
still oscillates → best-val checkpointing; (3) 1M haploid full last-epoch monotonicity
(EMA/SWA or partition clamp); (4) decoupled-sim retrain for in-distribution few-founder
outbred. Code: `train_crf.py` (`het_head`/`emit_het`), `train_diploid.py`
(`--learned-het`), `diag_c_het.py`, `show_window.py`. Latest commit `187569b`.

## ACTIVE (2026-06-25): IBD-ceiling experiment — "is 60–80% the no-outside-info limit?"

**The question (user-set).** In low-recombination ([0,2]-bp / 512) windows the
haploid CRF tops out ~60–80% founder acc. Hypothesis: the residual errors are
**IBD confusion** — the true founder is identical-by-descent with ≥1 other
founder across the whole block, so their reads are bit-identical and *no*
read-only decoder can separate them. If so, 60–80% is at/near the ceiling
reachable **without outside information** (relatedness / genome-wide founder
presence — i.e. the whole reason for the `crf-relatedness` branch). The sim has
ground-truth IBD, so the ceiling is computable exactly.

**Prerequisite done — corrected the SFS record.** The real maize/cassava
allele-sharing SFS has a hard cutoff at K/2=12. **That cutoff is a FOLDING
ARTIFACT in the build, not the correct shape** (match count folded to the minor
side). So the real SFS is NOT a valid target. BOTH sides were wrong: our old sim
is bell-shaped/over-shared (peak ~K/A), real is folded. Correct target = the
*unfolded* neutral SFS (n_i ∝ θ/i, singleton-dominated, real tail to K), derived
from theory. Updated `docs/notes/cassava_data_diagnostic.md` and the memory note.
(The separate **true-founder deficit**, sim 96% either-match vs real 72%, remains
a real-data BUILD problem flagged to zrm22 — unchanged, not what this thread is.)

**Built this session (UNCOMMITTED — see `git status`):**
1. `crf/simulate_alleles.py` — new Ewens/GEM(θ) sharing model via
   `--sharing-theta`. Replaces the fixed-`A` ancestor island model: per-window
   stick-breaking lineage proportions (GEM), founders are mosaics whose segments
   draw iid lineages → Ewens partition, class sizes ∝ θ/i (unfolded,
   singleton-dominated, tail→K). Larger θ = more singletons / less IBD. Also
   writes a companion **`<out>.ibd.npy`** = per-site IBD lineage labels [N,T,K]
   (the ceiling ground truth). New helpers `_segment_index`, `_gem_lineages`;
   `_coalescent_feats` now returns `(feats, lineage)`; `simulate` returns
   `(out, ibd)`. **Bug fixed mid-session:** original truncation forced the last
   stick to 1.0 → one giant spurious class → sharing *rose* with θ. Now the
   truncated stick weights are **normalized** (no giant class); θ-sweep is
   monotonic: θ=2→0.36 mean share / 7% singleton; θ=8→0.17 / 19%; θ=16→0.12 /
   30%; max sharing always 24 (tail to K, no fold). ✓
2. `crf/analyze_ibd_ceiling.py` — NEW. For each constant-founder segment forms
   the indistinguishable set S (feature-based: founders whose K-wide feature
   column is bit-identical to truth over the whole segment = what a read-only
   decoder sees; IBD-based: same lineage over the segment = the mechanism).
   Reports per breakpoint band: CRF acc, **ceiling = mean 1/|S_feat|**, gap,
   mean |S|, %sites with |S|>1, and **%errors that are IBD-confusable**
   (pred ∈ S). Reuses `make_splits` + the model's `decode()`; slices `.ibd.npy`
   with the identical head-slice/test split.

**STATUS 2026-06-25 (resumed): THREAD COMPLETE — hypothesis CONFIRMED, committed
(`affcf92`), logged to RESULTS.md "E-IBD".** The θ=6 training died at epoch 2/8
when the prior session ended, but the epoch-2 checkpoint
(`e-ibd-th6/e1-epoch=02-val/loss=7.115.ckpt`) is already AT the ceiling, so the
conclusion is robust. Analyzer validated end-to-end. Results:
- **CRF on the ceiling:** gap (acc−ceiling) ≤ 2.6 pts in every breakpoint band
  (all = 0.8222 vs 0.8408), and **86–100% of errors are IBD-confusable**. The
  60–80% in high-recomb bands IS the read-only limit (6+ bp ceiling 0.745, ~3
  founders IBD-indistinguishable).
- **θ-sweep (`ceiling_sweep.py`, data-only, no model):** ceiling moves
  monotonically with relatedness — θ2 0.70 → θ4 0.80 → θ6 0.84 → θ8 0.87 →
  θ16 0.92 (all-band). Sweep sims at `sim_ewens_th{2,4,8,16}_sweep.npy`.
- **Conclusion:** the only lever above the ceiling is outside info → quantitative
  justification for the relatedness encoder (E5/E7). NEXT: build E5 (relatedness
  matrix conditioning) and show it beats the ceiling on these same sims; optional:
  finish θ=6 training to epoch 7 for a tighter headline (won't change conclusion).

**(historical) WHERE WE PAUSED — a training run is LAUNCHED and should be running:**
- Background job generated `/workdir/esb33/data/training/sim_ewens_th6.npy`
  (+`.ibd.npy`): 100k windows, 512 sites, K=24, **coalescent θ=6**, inbred (F=1,
  haploid), crossovers 0–8 (so the [0,2]-bp band is ~⅓ of windows), read-snps 16,
  ancestor-crossovers 4, seed 0. Then trains the full **d256/L6 (5M)** haploid
  CRF: `--time-local-emis --lr 1e-4 --warmup-steps 500 --precision bf16-mixed
  --batch-size 64 --max-epochs 8 --run-name e-ibd-th6` on GPU 0.
- **The analyzer was NOT yet validated end-to-end** (the earlier smoke run was
  killed before producing a checkpoint). First thing on resume: confirm a
  checkpoint exists, then run the analyzer (command below) and sanity-check the
  table (ceiling ≥ CRF acc? does meanS>1 concentrate in [0,2]-bp? are most errors
  IBD-confusable?).

**RESUME COMMANDS:**
```bash
# 1. find the trained checkpoint
ls "/workdir/esb33/checkpoints/e-ibd-th6/"     # e1-epoch=NN-val/loss=*.ckpt (val/ is a subdir)

# 2. run the IBD-ceiling analysis on the test split
CKPT=$(ls /workdir/esb33/checkpoints/e-ibd-th6/e1-*-val/loss=*.ckpt | sort | head -1)
LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  .pixi/envs/gpu/bin/python src/python/crf/analyze_ibd_ceiling.py \
    --ckpt "$CKPT" --data /workdir/esb33/data/training/sim_ewens_th6.npy \
    --split test --max-windows 20000
# writes <workdir>/results/ibd_ceiling.txt
```

**INTERPRETATION when results land:** if in the [0,2]-bp band CRF acc ≈ ceiling,
meanS>1, and most errors are IBD-confusable → hypothesis CONFIRMED: 60–80% is the
no-outside-info limit, motivating the relatedness/global-conditioning encoder
(E5/E7). If there's a real gap (errors NOT IBD-confusable) → the model is leaving
recoverable signal on the table, so improve the model first. Then: sweep θ
(less/more relatedness shifts the ceiling), log to RESULTS.md (new section
"E-IBD"), and update `crf/sfs_sharing.py` to overlay the unfolded-Ewens sim.

**Open knobs / notes:**
- θ=6 was a first guess (median sharing 4, 15% singletons). If the [0,2]-bp band
  is already ~100% (ceiling trivial) lower θ (more IBD); if CRF ≪ ceiling
  everywhere, the task may be too hard — raise θ or read-snps.
- The analyzer currently supports `--split test` only for the IBD slice.
- All four task-list items: #1 record ✓, #2 Ewens model ✓, #3 analyzer built
  (validation pending), #4 run+log (in progress — training launched).

---

# Session handoff — crf-relatedness (2026-06-24)

Status snapshot for resuming work. Full per-experiment detail is in
[`docs/RESULTS.md`](RESULTS.md); experiment spec in [`docs/PLAN.md`](PLAN.md).

**Branch:** `crf-relatedness` — clean, pushed to `origin` at the commit below.
All data/checkpoints/logs under `/workdir/esb33/`. Run via
`pixi run --environment gpu python …` (2× H200 visible; set
`CUDA_VISIBLE_DEVICES` to parallelize across GPU 0/1).

## Where things stand

### 1. Haploid vs HMM on REAL maize — DONE (E1-maize)
Data: `/workdir/zrm22/HackathonJun2026/NonCollapsedDatasets/singleShuffledMaizeDataset/fullMaizeDataset_all_diploid.npy`
(N=1.35M, T=512, 26 cols = 24 founder read-counts + H1 + H2; inbred H1==H2).
Trained on a 200k subset (`--limit-n 250000` → 200k/25k/25k), 5 epochs, single seed.

Held-out **test founder accuracy** (`eval_test.py`, `eval_hmm.py`):
| arm | params | test acc | bp F1(±2) |
|---|---|---|---|
| CRF full (per-site c) | 5.07M | **0.682** | 0.351 |
| CRF small d128/L4 | 0.88M | 0.669 | 0.337 |
| CRF per-window c | 5.07M | 0.665 | 0.288 |
| CRF large d384/L12 | 22M | 0.662 | — |
| **HMM Li–Stephens (best)** | ~0 | **0.614** | 0.252 |
| CRF no-transition | 5.07M | 0.284 | — |

Key findings: CRF beats HMM by +6.7 (even the 0.88M model by +5.5). The win is
**entirely in low-recombination windows** (stratified: [0,2] bp +12.9; [6+] bp
HMM edges by 1.9 on likely-noisy dense windows). The HMM over-segments
(hallucinates ~4× too many breakpoints). `c` does NOT collapse on real data
(within-window sd 0.49), so per-site c > per-window c.

### 2. Diploid pair-state CRF — DONE, first result (E4-probe)
`train_diploid.py`: joint decode over P=325 founder pairs on the shared encoder.
New diploid sim mode: each site = one read from a random gamete
(`--gamete-balance`), so two paths interleave and must be phased.

Best model (`--homo-penalty 3`, ckpt
`/workdir/esb33/checkpoints/diploid-h3/d-epoch=04-val/loss=88.089.ckpt`):
**test pair_acc 0.614, hap_acc 0.743** on `sim_diploid_512.npy` (100k windows,
F=0, ≤4 crossovers/gamete, coalescent).

**Two bugs fixed (don't reintroduce):**
- No hard two-switch ban — independent chromosomes ⇒ transition cost −c·nsw,
  nsw∈{0,1,2}, a 2-switch costs exp(−c)².
- **Homozygous collapse:** single-read pair emission emis_f[i]+emis_f[j] is
  maximized by (a,a) of the observed founder ⇒ decode goes 100% homozygous
  without a het prior. `--homo-penalty` (subtract from homozygous pair-states,
  like `diploid_hmm`) fixes it: 0.04→0.61 pair_acc. Use penalty 3.

## Key files (all on this branch)
- `src/python/crf/train_haploid.py` — haploid CRF + encoder glue; `make_splits`
  (deterministic head-slice; `--limit-n` caps N), `--window-c`, `--no-transition`,
  `--run-name` (per-arm ckpt/log dirs).
- `src/python/crf/train_diploid.py` — diploid pair-state CRF (`--homo-penalty`).
- `src/python/crf/simulate_alleles.py` — sim; diploid via `--inbreeding 0`
  `--gamete-balance`; coalescent relatedness; `--recomb-span` hidden-rate (E2).
- `src/python/crf/eval_test.py` — CRF test eval (Viterbi + emission-only + bp P/R).
- `src/python/crf/eval_hmm.py` — haploid Li–Stephens HMM baseline scorer
  (per-window, sweeps p_stay/weight).
- `src/python/crf/eval_diploid_hmm.py` — diploid Li–Stephens baseline (factored
  per-chromosome transition; sweeps p_stay×weight×homo_penalty).
- `src/python/crf/eval_test_diploid.py` — held-out evaluator for
  `GRITSCRFDiploid` checkpoints (pair_acc / sorted hap_acc / homo-pred fraction).
- `src/python/crf/eval_stratified.py` — compare arms by true-breakpoint band.
- `src/python/crf/metrics.py` — breakpoint precision/recall (±tol).
- `src/python/crf/train_crf.py` — shared `FounderPathEncoder` (don't rewrite);
  `--window-c` lives here.

### 3. Diploid HMM baseline + d128/L4 capacity — DONE (E4-probe-baseline, 2026-06-24)
New scorers `eval_diploid_hmm.py` (Li–Stephens baseline, factored per-chromosome
transition, sweeps p_stay×weight×homo_penalty) and `eval_test_diploid.py`
(held-out `GRITSCRFDiploid` evaluator). Same test split of `sim_diploid_512.npy`.

| arm | params | test pair_acc | test hap_acc |
|---|---:|---:|---:|
| CRF full d256/L6 (hp=3) | 5.07M | 0.614 | 0.743 |
| **CRF d128/L4 (hp=3)** | 0.88M | **0.618** | **0.746** |
| **Diploid HMM (best)** | ~0 | 0.552 | 0.700 |

CRF beats the diploid HMM by **+6.5 pair / +4.6 hap**; the 0.88M model ties the
5.07M one (capacity is not the bottleneck — same as haploid). Best HMM:
p_stay=0.99, w=0.5, hp=0.5. CRF predicted-homozygous fraction 0.0005 (well-
calibrated het prior). See RESULTS.md "E4-probe-baseline".

## Next steps (prioritized — user-approved direction is diploid)
1. **Window 1024** — user wants to test whether larger windows help diploid
   phasing: regenerate sim with `--sites 1024` and retrain.
2. **Founder × read matrix (new direction, 2026-06-24).** User wants the real
   matrix to be **founder × read**: the sequence axis runs over reads, not
   collapsed positions. Multiple reads at the same position = multiple
   observations (NOT summed). Contrast current
   `cross/ps4g_to_matrix.py:collapse_matrix_inference` which sums same-position
   rows via `np.add.at`. Coverage then = observation count, position becomes a
   covariate. **This replaces per-coverage stratification** (dropped — coverage
   is captured by observation count under this layout).
3. Deferred: ≥3-seed / converged / full-data headline (PLAN §8); relatedness
   encoder (E5); promote to real diploid maize.

### Future ideas (user-flagged 2026-06-24, also in Claude memory)
- **Masked-site accuracy (SNP-like metric):** mask a fraction of test sites and
  check whether the decoded path recovers the correct founder there —
  complements haplotype-path accuracy. Needs the sim to mark masked/bad sites
  (extra label column) and the eval to score only on those positions.
- **Sub-panel / bi-parental sim:** generate *test* sets where each sample
  descends from only a subset of founders (simplest: bi-parental, K=2). Train on
  the full panel; keep the feature matrix K-wide (other founders just have zero
  coverage) so the trained model evaluates unchanged. Add `--founders-subset` to
  `simulate_alleles.py`.
- **Proportional within-window positional encoding:** replace/augment the
  encoder's ordinal sinusoidal PE (keyed on timestep index) with position as the
  *fraction of the way through the window* (~[0,1]), NOT absolute genomic
  coordinates. User preference. Robust to variable window spans and gives two
  reads at the same position the same encoding (needed for the founder×read
  layout). Feed via the unused `dbp` hook in `FounderPathEncoder`
  (`crf/train_crf.py`); needs a per-observation position column.

## Gotchas
- The OTHER maize file `…/fullMaizeDataset.npy` (no `_all_diploid`) is BROKEN —
  all 25 cols are read counts, no label column. Use the `_all_diploid` file.
- `pixi` binary at `~/.pixi/bin/pixi`; imports use `from python.<mod>` (PYTHONPATH
  set by pixi to `src/`).
- bf16 required (fp16 overflows the CRF partition logsumexp); lr 1e-4 + warmup 500.
- `pixi run --environment gpu` HANGS on this cluster's NFS (blocks in `cl_syn` on
  env resolution, never reaches Python). Workaround: call the env Python directly:
  `PYTHONPATH=src CUDA_VISIBLE_DEVICES=N .pixi/envs/gpu/bin/python <script>`. For
  scripts that import scipy (e.g. `eval_test_diploid.py`) also set
  `LD_LIBRARY_PATH=.pixi/envs/gpu/lib` (bare Python misses the newer libstdc++).
