# v6 supervised heads: current state and handoff

Start here. The history of everything tried (and why) is in `HISTORY.md`; what each branch
investigated and why we moved on is in `BRANCHES.md`; the removed worktrees are
listed in `CLEANUP_LOG.md`.

## Current best: v6 fit B

SNP+RefCall genotype error, simulated-validation corpus (5 samples per class):

| | IDX-INB | IDX-HYB | IDX-RIL2 | MIX-HYB | MIX-RIL2 | OUT-INB | OUT-HYB | OUT-RIL2 |
|---|---|---|---|---|---|---|---|---|
| v6 fit B, 0.1x | 0.00 | 0.08 | 0.14 | 7.83 | 0.98 | 3.58 | 5.78 | 3.50 |
| v6 fit B, 1x | 0.03 | 0.43 | 0.29 | 4.54 | 1.13 | 3.13 | 4.79 | 2.91 |
| diploid-affinity, 0.1x | 0.00 | 3.70 | 0.02 | 13.85 | 2.50 | 5.42 | 19.58 | 5.48 |

All-depth comparison (0.01–2x) vs ternary-baseline and diploid-affinity: see
`experiments/supervised-heads/scripts/bigpicture_pipeline.sh` (results tags `v6_B_<depth>`,
`tern_<depth>`, `diploid_affinity_s200_<depth>`).

## The model in one paragraph

Encoder (`GRITSCRFDiploidIndel`, `src/python/crf/train_diploid_indel.py`) reads the refmap ternary
rows of a 512-row window plus the sample's genome-wide founder match rate. Stage 1 trains heads on
simulator truth: gate, crossovers per window, het (trained, not used at decode), deletion dosage,
out-of-panel. Stage 2 (**fit B**) freezes the encoder and tunes a handful of CRF numbers on simulated
windows: read-state log-likelihoods, crossover scale, founder-prior weight, dosage and out-of-panel
weights. Decoding: Viterbi over founder pairs. Diagrams:
`experiments/supervised-heads/results/v6_fitB.png` (3-step), `v6_model.png`, `v6_training_data.png`.
What fit A and fit B are: the "Fit A vs fit B" Claude doc, summarised in `HISTORY.md` §4.

## Reproduce

Environment: the pixi env of the main checkout (`.pixi/envs/default`), `LD_LIBRARY_PATH=<env>/lib`,
`PYTHONPATH=<this checkout>/src`. GPU jobs: at most 3 eval processes per GPU (each up to ~30 GB).

1. **Align reads** with ropebwt3-phg branch `v6-supervised-heads` (commit aad36c1: `-s 200` lift
   sampling + `--lift-local-win 20000` adaptive projection; build with `make`):
   `experiments/het-replacement/scripts/align_calibration.py --corpus evaluation --coverages 0.1 ...`
   (preset `s200local`; binary path inside the script).
2. **Build training data**: `experiments/supervised-heads/scripts/v6_build.sh` (4 depths, ~1 h CPU).
3. **Train + fit + score**: `experiments/supervised-heads/scripts/v6_train.sh` (stage 1 ~1.5 h per run
   on one GPU, fit B ~15 min, scoring ~8 min per depth with one process per scope).
4. **Score any checkpoint**: `experiments/simval-corpus/scripts/fast_eval_ckpt.py --tag T --ckpt C
   --scope idx|out|mix --route --depth 0.1 --input-glob scratch/lift_s200_local20k` (run one process
   per scope). Diagnostics: `--knockout gate1|gatemean|xoconst|ood0` (comma-combinable),
   `--aff-region`, `--het-scale`. diploid-affinity: `fast_eval_diploid_affinity.py`.

Checkpoints: `/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/`
(`v6-stage1/`, `v6-stage2B/last.ckpt` = v6 fit B; `v6base-*`; derived decode-only copies
`v6-stage2B/last_*.ckpt` for the sweeps). Training data:
`.../indel_baseline/data/training/maize_v6ho_multidepth_fullscale_sliced.*`. Results JSONs:
`/workdir/zrm22/HackathonJun2026/grits_workdir/results/<tag>_{idx,out,mix}_fast/`.

Cluster-specific absolute paths remain for data (panel VCF, scorer cache
`scratch/fast_score_cache`, manifests, checkpoints); code paths resolve to this checkout.

## Conventions

- Train on simulated data only; fit simulator parameters on the calibration corpus only.
- Lead every table with SNP+RefCall; include diploid-affinity and ternary-baseline.
- Treat differences under ~0.5 points per class as noise (5 samples per class).
- Warm-start from ternary-baseline with the full recipe flags (the guard enforces it).

## Next steps

See `HISTORY.md` §7. Highest value: measure real OUT switching in lineage-aware terms and make the
simulated held-out individuals match it; a sample-dependent founder-prior weight.
