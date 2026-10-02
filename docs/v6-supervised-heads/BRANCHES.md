# Branch index: what each branch investigated and why we moved on

One row per branch, oldest first. All branches are on origin (`maize-genetics/grits` or
`maize-genetics/ropebwt3-phg`). Use this file to find the right branch; read `HISTORY.md` for the
reasoning and the result tables.

- **Status** says what happened to the branch's idea.
- **In v6** means the v6 line contains the branch's commits, because its tip is an ancestor of
  `supervised-heads-distfix`.
- Branches marked *not in v6* forked off and were never merged back. Their code exists only on that
  branch.

## GRITS: current

| Branch | What it is | Status |
|---|---|---|
| `v6-supervised-heads` | Clean v6 code + eval tools + these docs, cut from develop | **current best; start here** |
| `supervised-heads-distfix` | Working branch where sh4, sh5, v6, the knockouts and the all-depth comparison were developed (raw history behind v6) | active; superseded by `v6-supervised-heads` for reading |

## GRITS: July – August, pre-indel model (diploid-affinity era)

| Branch | Investigated | Why we moved on |
|---|---|---|
| `crf-relatedness` | Founder-relatedness and affinity conditioning for the CRF encoder; produced diploid-affinity | Merged to main (PR #186). Its diploid-affinity checkpoint is still a comparison baseline. |
| `crf-checkpoint-eval` | First simulated-to-real transfer eval: refmap founder recovery, Oh43 dropout, k-mer and trimming sweeps, synthetic-diploid real-read eval | Done; in v6. These diagnostics fed the move to the simval corpus. |
| `windowing-quality-filters` *(not in v6)* | `--max-hit-frac` fan-out filter + binarize-by-default on the PS4G matrix | Helped RIL2 (best at frac 0.2) but hurt hybrids (pair accuracy 85.3% → 81.6%). Not carried into the ternary/indel input. |
| `integration/aug-2026-consolidation`, `bring-in-simval-corpus` | Merged the August branches (whole-chromosome decode, constant-pair simulator, fast allele scorer) and imported `grits_workdir` scripts as `experiments/` | Done; in v6. The base for everything after. |

Branches mentioned in old notes that were folded into the consolidation and no longer exist
separately: `wholechrom-decode-real-data` (only cleaned up boundary artifacts),
`tier2-constant-pair-individuals` (did not fix the hybrid/inbred regression),
`perf-allele-multiset-score` (scorer speed-up, kept).

## GRITS: September, indel-aware ternary model

| Branch | Investigated | Why we moved on |
|---|---|---|
| `simulator-indel-modeling` | Design doc + Phase 1 realistic indel simulator, maize/cassava calibration | Done; in v6. Foundation of the ternary model. |
| `indel-training-loop` | Indel-aware training loop + `GRITSCRFDiploidIndel` | Done; in v6. |
| `indel-correlated-deletion` *(not in v6)* | Correlated deletions across IBD-adjacent founders in the simulator | Never merged into the v6 line. Later replacement-indel work (`repl_groups=lineage`) covered the same need. Currently checked out in the main checkout. |
| `indel-baseline-run` | First fully completed indel-aware baseline training (runs 20–21), batch and memory fixes, `--accumulate-grad-batches` | Done; in v6. |
| `indel-v3-simulator` | v3 simulator; automatic homo_scale from affinity (fixed inbreds); `infer_real_founder_pairs` as the single inference path; IBD-adjusted metric; root cause of depth-confidence error | In v6. The ternary-baseline checkpoint comes from this era. The IBD-adjusted metric was later shown to measure mapping ties, not genotype accuracy. |
| `indel-density-features` *(not in v6)* | Per-window evidence-density feature to fix depth confidence | Negative on real data. |
| `indel-readcount-feature` | Per-cell read-support count feature | Helped in ablation, but the full-scale retrain had a severe B73-specific regression. |
| `indel-readcount-grouped-count` | Fixed count formula (group by site and ternary pattern) | Fixed the collapse but still net-negative on hybrids. |
| `indel-readcount-row-collapse` *(not in v6)* | `--collapse-rows`: merge identical row stacks ("row-collapse") | Best in-panel result of that series (IDX-HYB about 30× better), but 6–17 points worse than diploid-affinity on every OUT/MIX class. This started the out-of-panel investigation. |
| `indel-collapse-bigger-model` *(not in v6)* | d_model 384, 8 layers | Unstable, with a B73-specific regression. |
| `indel-collapse-moredata` *(not in v6)* | 2× training individuals | Did not beat row-collapse (low-depth trade-off). |
| `indel-collapse-bigmodel-moredata` *(not in v6)* | Both together | Worse than capacity alone. |
| `indel-region-buffer-fix` | Restore the nominal 6 crossovers per individual (a buffering bug had cut it to 1.16) | More crossovers made everything worse (IDX-HYB pair accuracy 96 → 82%). Real in-panel samples have zero recombination. |
| `indel-region-buffer-fix-lowrate` | diploid-affinity's undiluted crossover rate + spike-skip/cosine-decay | Small win once flags were fixed, still far behind diploid-affinity. Branch holds the consolidated investigation summary and naming convention. |
| `indel-heldout-augment` *(not in v6)* | Synthetic leave-one-out: hide a founder, relabel its rows (held-out augment v1) + column permutation | First version was a degenerate always-homozygous model (dropped flags plus single-class validation split). The full-recipe retrain beat ternary-baseline on 15/15 OUT rows, but IDX-HYB regressed 0.26 → 4.72%. Replaced by held-out augment v2 in v6. |
| `heldout-calibration` *(not in v6)* | `--no-permute` escape hatch for held-out augment | Small tool commit; idea continued in v6. |
| `warmstart-hparam-guard` | Refuse warm-start on hyperparameter mismatch; shuffle individuals before the split | Kept; in v6. Fixed the single-class validation split. |
| `regionfix-out-snprc-check`, `archive/regionfix-out-snprc-check` *(not in v6)* | OUT SNP+RefCall check of region-fix and held-out augment; first fast scorer class breakdown; diagnostic homo_scale override | Superseded by the v6 fast scorer. The archive branch holds 8 commits that were ahead of origin, including the unique f053aef. |
| `heldout-mosaic` *(not in v6)* | Sticky relabel + fine-scale mosaic hidden founder (removes relabel noise) | Negative. Removing the noise removed the only switch pressure. |

## GRITS: late September, likelihood CRF and supervised heads

| Branch | Investigated | Why we moved on |
|---|---|---|
| `two-stage-training` *(not in v6)* | Colleague's two-stage idea: auxiliary switch/het losses, then frozen trunk; tie-aware loss; lineage replacement groups | Within noise of v3. Its diagnosis (the additive pair emission was the hybrid bug) led to the mixture emission. Tie-aware loss and lineage groups were carried forward. |
| `out-het-replacement` *(not in v6)* | Het-replacement simulator, calibration corpus, likelihood emission, distance-banded likelihood, depth-scaled training data, first supervised-heads diagrams | Matched ternary-baseline in-panel but not OUT. The likelihood CRF it introduced is the v6 decoder. Its scripts are on `v6-supervised-heads` under `experiments/het-replacement`. |
| `supervised-heads` *(not in v6)* | sh3: encoder heads trained on simulator truth (gate, window crossovers, het, affinity) + fit A/B | Hybrids much better, but a ~3% in-panel inbred floor from the per-row het prior. Continued on `supervised-heads-distfix` (sh4 → v6). |

## GRITS: other people's branches (not part of this work)

`encoder-decoder`, `mutation`, `auto-encoder`, `crf-tests`, `worktree-hmm-ps4g-port`,
`site-builder`, `testing_suite_bella`, `fix/ps4g-read-count-denominator`,
`feat/ps4g-viewer-column-mode`, `fix/bed-to-vcf-gamete-index`.

## ropebwt3-phg

| Branch | Investigated | Status |
|---|---|---|
| `v6-supervised-heads` (aad36c1) | `-s 200` lift sampling + `--lift-local-win 20000` adaptive local projection; fixes PLACED reads landing 10–100 kb off | **Adopted**; every v6 eval uses it. Same commit as local `lift-adaptive-projection`. |
| `lift-ridx-ternary-dist-map` | `--anchor-dist-npy`: per-founder ternary + lift-anchor distance block (the ternary input) | Adopted. Origin is 1 commit behind (817f4f6, now also on `v6-supervised-heads`). |
| `lift-syntenic-anchors` | Syntenic-anchor filter for the distance block | Not adopted; distances were later dropped from the model. |
| `refmap-read-filters` | Preset-F read filters (wrong-chromosome reads 7.7 → 1.65%) | Not adopted; small gain, untested at low depth. |
| `pav-ps4g-insertion-rows`, `chain-assembly-anchors`, `pav-breakpoint-anchoring` | Chain + PAV alignment route | Not adopted. 400–1300× worse than refmap in-panel, only 2–4× worse out-of-panel. |
| `refmap-stats`, `refmap-multi-speedup` | July `--stats` / `--locate-cap` instrumentation for the MULTI short-circuit | Diagnostic only. |
| `refmap-carrier-cap-report-occ`, `worktree-refmap-ps4g-numpy` | PS4G/npy export and `--report-occ` | Merged upstream; local worktree binary still used by older scripts. |
