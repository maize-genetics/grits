# Indel-aware baseline training: full experiment log

2026-09-15. A chronological record of every training run attempted while
trying to get a first working baseline for `train_diploid_indel.py`
(`GRITSCRFDiploidIndel`, the new ternary+distance model) on
maize-calibrated `--simulate-indels` data. Written because the process
involved several real dead ends worth not re-discovering — a wrong
pixi environment, an OOM, a genuine training-instability bug, two
mis-diagnoses along the way (one caught before wasting much compute,
one caught only after runs had already been launched), and a data-
composition mismatch against the comparison baseline. Every run below
used `src/python/crf/train_diploid_indel.py` against
`maize_indel_baseline.npy` (K=24, 3000 windows, 8192 sites/window,
`--simulate-indels`, seed 42) unless noted otherwise.

## Summary table

| # | Run / log | Config vs. previous | Outcome | Why killed / stopped |
|---|---|---|---|---|
| 1 | (log overwritten) | `--batch-size` unset (64), `.pixi/envs/gpu` interpreter directly | Crashed instantly | `ModuleNotFoundError: No module named 'torch'` — wrong pixi environment |
| 2 | `train.log` | Same flags, correct `.pixi/envs/default` via `pixi run --` | Crashed at step 0 | CUDA OOM — ~129GB allocated, batch=64 × T=8192 far exceeds what this CRF kernel was sized for |
| 3 | `train2.log` | `--batch-size 4` | Completed epoch 0, chance-level accuracy | `val_pair_acc=0.00567`, `val_hap_acc=0.00588` after epoch 0 — below random chance; loss spiking. Killed manually once diagnosed |
| 4 | `train3.log` | `--spike-skip --ema` added | Crashed instantly | `train_diploid_indel.py: No such file or directory` — a concurrent subagent's `git checkout -b` silently switched the shared checkout's branch out from under this launch. Fixed by creating an isolated git worktree |
| 5 | `train4.log` ("v2") | Same config, relaunched from the isolated worktree | **Completed all 5 epochs — real signal, then collapse** | See "Run 5" below — kept as the baseline checkpoint, not killed |
| 6 | `train5.log` ("v3") | `--homo-penalty 3` added | Loss flat at ~20-25k through 85% of epoch 0, never descending | Killed — diagnosed as the penalty fighting the data (see "The homo-penalty detour") |
| 7 | `train6.log` ("v4-lowlr") | + `--lr 3e-5` | Loss oscillating 1.95e4–2.47e4 through 35% of epoch 0, no descent | Killed alongside #6, same diagnosis |
| 8 | `train7.log` ("v5-fp32") | + `--precision 32-true` (replacing bf16) | Loss ~2.5e4, flat, 8% into epoch 0 | Killed alongside #6, same diagnosis (too early to be conclusive alone, but consistent) |
| 9 | `train8.log` ("v6-tightclip") | + `--grad-clip 0.3` (down from 1.0) | Loss ~2.5e4, flat, 8% into epoch 0 | Killed alongside #6, same diagnosis |
| 10 | `train9.log` ("v7-lowlr-nopenalty") | Homo-penalty dropped; `--lr 3e-5` | 8% into epoch 0 | Killed — superseded by the `--inbreeding` discovery (see below) before it could produce a real result |
| 11 | `train10.log` ("v8-fp32-nopenalty") | `--precision 32-true`, no penalty | 6% into epoch 0 | Killed, same reason as #10 |
| 12 | `train11.log` ("v9-repeat") | Exact repeat of run 5's config | 6% into epoch 0 | Killed, same reason as #10 |
| 13 | `train12.log` ("v10-weakpenalty") | `--homo-penalty 0.3` (weak) | 6% into epoch 0 | Killed, same reason as #10 |

## Run 1–2: environment and memory, before any real training happened

Run 1 used `.pixi/envs/gpu`'s interpreter directly — that environment
turned out not to have PyTorch installed at all (`gpu` is apparently
for a different part of the stack); `.pixi/envs/default` has CUDA-
enabled torch (2.12.1+cu130) and is the correct one. Fixed by using
`pixi run --` (or the equivalent direct interpreter path) instead.

Run 2, same flags otherwise (`--batch-size` left at the default of 64),
OOM'd immediately: `CUDA out of memory. Tried to allocate 26.00 MiB...
this process has 129.64 GiB memory in use`. Root cause: this CRF
kernel's forward-backward pass retains per-timestep activations across
the full sequence for backprop; at T=8192 (this project's indel-mode
window length) that scales far beyond what batch=64 can fit — the
existing deployed model only ever trained at T=512. Fixed by dropping
to `--batch-size 4`, which reproduces roughly the same total memory
per step (empirically, memory scales close to linearly with batch
size here).

## Run 3: chance-level accuracy, loss spikes

With batch-size fixed, training actually ran, but validation after
epoch 0 was `val_pair_acc=0.00567`, `val_hap_acc=0.00588` — below what
random guessing among 24 founders should give. Training loss showed
real spikes during the epoch (e.g. a jump to several thousand,
dropping back down the next step) — not just batch-size-4 noise.
Killed manually (`SIGTERM`) partway into epoch 1 once this was
diagnosed, rather than let a clearly-broken run continue.

## Run 4: branch-switch collision (process bug, not a training bug)

Relaunched with `--spike-skip --ema` (this project's own documented
levers for exactly this kind of instability) — and it crashed
instantly with the training script itself missing. Cause: a parallel
subagent working on a different feature (`indel-correlated-deletion`)
had run `git checkout -b ... origin/simulator-indel-modeling` **in the
same shared checkout** this training command was launched from,
silently switching its branch to one that predates the training-loop
code merge. Fixed by creating an isolated `git worktree` (a
`git worktree add` invocation directly, since the interactive
`EnterWorktree` tool isn't usable from this session's primary
directory) so the two efforts stop colliding on shared filesystem
state. All later runs launch from that worktree.

## Run 5 ("v2"): the real signal, and the real collapse

Same config (`--spike-skip --ema`, `--batch-size 4`), relaunched from
the isolated worktree. This is the run that actually produced a
baseline checkpoint. Corrected, exact per-epoch validation (an earlier
verbal summary of this run mis-attributed the collapse to epoch 2 —
tqdm's progress-bar redraws during the *next* epoch's early training
steps still display the *previous* epoch's validation numbers, which
reads misleadingly on a quick glance; the numbers below are the
de-duplicated true per-epoch values):

| Epoch | val_pair_acc | val_hap_acc | cumulative `train/skipped` |
|---|---|---|---|
| 0 | 0.666 | 0.666 | 78 |
| 1 | 0.000 | 0.000 | 259 |
| 2 | 0.000 | 0.000161 | 350 |
| 3 | 0.00367 | 0.00374 | 508 |
| 4 | 0.289 | 0.324 | 997 |

The model genuinely learned something real at epoch 0 (66.6% pair/hap
accuracy — a legitimate, non-trivial baseline signal), then collapsed
going into epoch 1 and stayed collapsed through epoch 3, with a
partial (not full) recovery by epoch 4. `train/skipped` (steps
`--spike-skip` caught and discarded) climbed the entire time — spike
protection was clearly firing continuously, not just for one isolated
event, but didn't prevent the underlying degradation. Lightning's
`ModelCheckpoint` correctly identified and saved the best state:

```
checkpoints/diploid-indel-baseline-maize-v2/d-epoch=00-val_pair_acc=0.6663.ckpt
```

This checkpoint is the actual usable artifact from this run — not the
run's final state, which is badly degraded.

## The homo-penalty detour (runs 6–9, then corrected)

Root-cause reasoning at the time: this codebase has a documented,
previously-diagnosed failure mode for exactly this "single-read
diploid" emission design (`docs/RESULTS.md`) — with one read per site,
`emis_f[i]+emis_f[j]` is maximized by guessing the pair is homozygous
at whichever founder the single visible read matches, regardless of
whether that's correct. The documented fix is `--homo-penalty`
(subtracts a fixed term from homozygous-pair emission scores), and the
established working recipe (`diploid-affinity-sim512-h3`) uses
`--homo-penalty 3`. Runs 6–9 tested that value (alone, +lower LR, +fp32,
+tighter grad-clip) — see the summary table.

**This was a real misdiagnosis, caught by inspecting the loss curves
directly rather than trusting the theory.** All four runs showed loss
completely flat or aimlessly oscillating around 20-25k with no
downward trend at all — a starkly different (and worse) pattern than
run 5's healthy early descent without any homo-penalty. The reason:
the training data for these baseline runs used `--inbreeding 1.0` (the
simulator's default) — **every individual is truly, 100% homozygous**.
A penalty that pushes predictions away from homozygous fights the
correct answer on every single training example in an all-inbred
dataset; it doesn't correct a bias, because there is no bias to
correct here. The documented failure mode this penalty was built for
was measured on data that was **97% heterozygous** — the opposite
composition. All four runs were killed once this was recognized, before
wasting further compute chasing a fix that couldn't work for this
data.

## Runs 10–13: superseded before completion

A corrected set (homo-penalty dropped, testing lower-LR / fp32 / a
plain repeat of run 5's config / a much weaker penalty in isolation)
was launched next — still against the same `--inbreeding 1.0` data.
Within minutes of launch, a second, more fundamental issue was found:
the actual comparison target, `diploid-affinity-sim512-h3` (and the
plain `diploid-sim512-h3` it's benchmarked against), was trained on
data generated with **`--inbreeding 0`** — i.e. essentially all
individuals heterozygous, the opposite mating structure from what
these baseline runs were using. This also retroactively explains why
the homo-penalty detour looked so wrong: that mechanism was calibrated
against heterozygous-dominant data, and the training data itself
(not just the penalty choice) was the actual mismatch. All four runs
were killed (processes launched outside harness tracking, killed
directly by PID) before completing a single epoch, rather than
continue testing hyperparameters against training data with the wrong
mating-structure composition.

## Current state (in progress at time of writing)

Regenerating the training set with `--inbreeding 0` (matching the
established comparison recipe) — same K=24/sites=8192/windows=3000/
`--simulate-indels`/seed=42 otherwise. Once that lands, the plan is to
relaunch with the *original*, unmodified established recipe
(`--homo-penalty 3 --spike-skip --ema --lr 1e-4 --warmup-steps 500
--precision bf16-mixed`) — homo-penalty=3 should now be appropriate
again, since the data will no longer be uniformly homozygous. Whether
this also resolves the epoch-1 collapse seen in run 5 is still an open
question; if it recurs even on correctly-composed data, that points at
something else (the confirmed loss/gradient scaling with the longer
T=8192 sequence length remains a live, not-yet-ruled-out candidate —
see the prioritized follow-up list already given in-session for
sub-window training, fp32, and lower-LR as the next things to test
cleanly, one variable at a time, once this data lands).

## Lessons for next time (process, not modeling)

- **A shared, non-worktree checkout is not safe for concurrent agents.**
  Run 4's crash cost nothing but a launch retry, but a `git checkout -b`
  from one effort silently redirecting another effort's file reads is a
  real, repeatable hazard whenever two processes share a working
  directory. Isolate with `git worktree add` proactively, not reactively.
- **Read the loss curve before trusting the theory.** The homo-penalty
  detour was mechanistically well-reasoned and backed by this project's
  own prior documentation — and still wrong for this specific data. The
  four flat/stuck loss curves were the actual signal that caught it,
  not further reasoning about the mechanism.
- **Match the comparison baseline's data composition, not just its
  model format.** Adding indels to the training data isn't the only
  axis that changed here — `--inbreeding` defaulting to 1.0 silently
  changed the mating-structure composition too, in a way that broke an
  established, working hyperparameter (`--homo-penalty`) that had
  nothing to do with indels at all.
