# GRITS — Claude Code Conventions

## Repository overview
GRITS (**G**enetic **R**ecombination **I**mputation **T**ool **S**et) imputes
founder haplotype paths in diploid plant genomes. The ML stack lives entirely
under `src/python/`.

## Running GRITS on a user's sample
If someone wants to impute, genotype or find the founders of their own maize sample, use the
**`grits-run-sample` skill** (`.claude/skills/grits-run-sample/SKILL.md`). Do not assemble the pipeline from
the training code or the experiment scripts.
- It runs the promoted pipeline: ropebwt3-phg `develop` refmap with the promoted flags, then
  `infer_real_founder_pairs(preset="placedfix_hetdel_hs_peel")` with checkpoint `v6cfullfe-stage2B`.
- Start with `scripts/check_setup.sh`. Resource paths come from `scripts/grits.env` via `GRITS_*`
  environment variables; never hard-code them.
- The human-readable version of the same steps is `docs/run-sample.md`; keep the two in sync.
- The `--workdir` convention below applies to training scripts, not to this skill: its outputs go in the
  user's chosen OUTDIR.

## Active branch for encoder work
All new encoder development goes on **`crf-relatedness`**, branched from
`crf-tests`. Do not commit encoder changes directly to `main`.

## Environment
```bash
# CPU (default)
pixi install
pixi run -- python <script>

# GPU (Linux only, requires CUDA 12+)
pixi install --environment gpu
pixi run --environment gpu -- python <script>
```
`PYTHONPATH` is set automatically by pixi to `${PIXI_PROJECT_ROOT}/src`,
so imports use `from python.<module>...`, never `from src.python...`.

## Workdir convention
**All data inputs, model checkpoints, logs, and result files must be written
to an explicit `--workdir` directory — never to hardcoded paths.**

- Training scripts must accept `--workdir <path>` (argparse).
- Default workdir is `/workdir/esb33/`.
- Sub-structure inside workdir:
  ```
  workdir/
    data/training/    # .npy matrices (train / val / test splits)
    data/held-out/    # reserved test set, never touched during training
    checkpoints/      # model .pth / Lightning checkpoints
    logs/             # TensorBoard / wandb logs
    results/          # per-experiment metric tables appended to RESULTS.md
  ```
- Never commit workdir contents; it is gitignored.

## Running tests
```bash
pytest --cov=src --cov-report=term-missing --cov-report=html
```
Tests live in `tests/python/` mirroring the `src/python/` structure.
Use `unittest.TestCase` for model tests, `pytest` as the runner.

## Key modules
| Module | Purpose |
|--------|---------|
| `src/python/crf/train_crf.py` | CRF training entry point (active) |
| `src/python/modernBERT/` | Transformer encoder (reference) |
| `src/python/bimamba/` | BiMamba SSM encoder (reference) |
| `src/python/ps4g_io/` | PS4G file parsing + PyTorch datasets |
| `src/python/hmm/` | Viterbi / HMM baseline |
| `src/python/vcf_eval/accuracy.py` | SNP accuracy metrics |

## CRF encoder architecture (crf-relatedness goal)
The `FounderPathEncoder` in `src/python/crf/train_crf.py` has an `ext_dim`
hook for external embeddings. The `crf-relatedness` branch will add:
1. A **pairwise relatedness matrix** (founders × founders) as a conditioning signal.
2. A **per-site recombination rate track** as an additional `dbp`-style covariate.

See `docs/PLAN.md` for the full experiment series.

## Coding conventions
- No comments unless the *why* is non-obvious.
- No per-founder-index parameters (breaks variable-panel transfer).
- Do not materialize `[B, T, P, P]` diploid transition tensors.
- Loss target is the **founder path**, never SNP dosages directly.
- Config-driven experiments: one YAML per experiment under `configs/`.
- Append to `RESULTS.md` after each experiment; never overwrite prior rows.
