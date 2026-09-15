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
| 14 | `train14.log` ("v11-het") | `--inbreeding 0` data, established recipe restored (homo-penalty=3 back on) | Validation frozen bit-for-bit across 5 checkpoints (500 steps); loss oscillating, zero spike-skip interventions | Killed — data composition fix was necessary but not sufficient, see "Run 14" below |
| 15 | `train15.log` ("v12-batch16") | `--batch-size 16` (up from 4), same T=8192 het data | Fits (75.5GB); loss band narrower, `val/hap_acc` climbing slowly (0.0042→0.0111) | Left running for comparison, not killed |
| 16 | `train16.log` ("v13-t4096-batch64") | T halved to 4096, `--batch-size 64` | OOM on first real step (131.9GB) | Too ambitious even with T halved |
| 17 | `train17.log` ("v13-t4096-batch32") | T=4096, `--batch-size 32` | **Cleanest result so far** — smooth 10-checkpoint climb in `val/hap_acc`, tight/descending loss band | Still running, being watched through later epochs — see "Runs 15–17" below |

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

## Run 14 ("v11-het"): correct data composition alone is not sufficient

Data regenerated with `--inbreeding 0` (`maize_indel_baseline_het.npy`;
QC: either-founder-covered 73.56%, right in the 70–75% target band).
Relaunched with the untouched established recipe (`--homo-penalty 3
--spike-skip --ema --lr 1e-4 --warmup-steps 500 --precision bf16-mixed
--batch-size 4`). Instrumented with `--val-check-interval 100` (and
later `100`→ finer, see below) instead of the default once-per-epoch,
specifically to get faster feedback than a full ~17-minute epoch.

**Result: validation was frozen bit-for-bit identical across 5
checkpoints spanning 500 steps** (`val/pair_acc=0.000148`,
`val/hap_acc=0.0239`, unchanged every single check). At 500 steps —
half the EMA's ~1000-step effective window (`--ema-decay 0.999`) — EMA
lag alone can't explain a completely flat reading. Raw training loss
confirmed a real problem: oscillating persistently between healthy-
looking low values (43–350) and large spikes (1.4k–9.7k) from the very
first steps, never settling — and **`train/skipped` never appeared even
once** in 156+ steps, despite the obvious swings. Root cause:
`--spike-skip` flags a step as a spike only if it exceeds
`loss_spike_mult × a slow-moving EMA of loss` (`train_diploid_indel.py`
`training_step`) — built for a rare outlier against a stable baseline.
Here the swings *are* the steady-state behavior, so the EMA-of-loss
itself gets dragged up by them and the detector goes blind. Fixing the
`--inbreeding` mismatch was necessary but not sufficient — there is a
second, independent instability. Killed once this was established with
enough checkpoints to rule out EMA lag as the explanation.

## The real mechanism: O(T²) self-attention, not O(T) or the 2x format width

Traced directly in `train_crf.py`'s `IndelFounderPathEncoder.forward`:
`self.fpool` (a `MultiheadAttention` pooling step) attends only across
the K=24 founders per site — cheap, doesn't scale with T. But
`self.pos_encoder = nn.TransformerEncoder(...)` (called on `h` shaped
`[B,T,d_model]`, no mask) is genuine self-attention across the full
T-length sequence — **O(T²) memory**, not O(T). With this script's
defaults (`d_model=256, n_heads=8, n_layers=6`), one layer's attention-
weight tensor alone at T=8192/batch=64 is
`64 × 8 × 8192 × 8192 × 2 bytes (bf16) ≈ 68.7 GB` — before backward-pass
storage or the other 5 layers, already explaining most of run 2's OOM.
Contrast the two scaling factors directly: the ternary+distance format
width (K+2 → 2K+2) is **2x**; the sequence-length jump (T=512 → T=8192,
the deployed baseline's window length vs. this project's indel-mode
window length) is **256x (16²) for this specific component**. The
format-width increase — the one change that's actually about
indels — is a rounding error next to the architecture running at 16x
the sequence length it was ever tuned at. (Caveat: modern PyTorch can
dispatch to a memory-efficient/flash-attention kernel that avoids
materializing the full dense matrix; the real number could be smaller
than this back-of-envelope estimate if that's engaged here — not
profiled directly. The *order-of-magnitude gap* between the 2x and
256x factors holds regardless.)

This reframes run 14's oscillation mechanistically: at T=8192, one
training window is a huge, highly variable unit (either-covered is
only ~74%, so windows differ a lot in how much real signal they carry),
and `--batch-size 4` (forced down from 64 by the OOM above) means one
unlucky window is a quarter of a step's gradient signal with nothing to
average it out. `--grad-clip` doesn't fix this — it bounds gradient
*magnitude*, not the fact that the gradient *direction* is a
different, high-variance sample every step at this batch size.

## Runs 15–17: batch size and window length, tested directly

| # | Run / log | Config | Result |
|---|---|---|---|
| 15 | `train15.log` ("v12-batch16") | T=8192 (het data), `--batch-size 16` (up from 4) | Fits (75.5GB, no OOM). Loss band narrower than batch=4 (spikes now ~1.2k–5.2k, not into the tens of thousands), `val/hap_acc` climbing slowly but really (0.0042→0.0111 over 4 checkpoints) — batch size clearly helps, doesn't fully fix it |
| 16 | `train16.log` ("v13-t4096-batch64") | T=4096 (halved), `--batch-size 64` | **OOM on the first real optimizer step** — 131.9GB, over the ceiling; too ambitious even with T halved |
| 17 | `train17.log` ("v13-t4096-batch32") | T=4096, `--batch-size 32` | Fits comfortably (77.7GB). **Cleanest result of the whole investigation** — see below |

Run 17's `val/hap_acc` across 10 consecutive checkpoints (spanning
~150 steps, most of 2 epochs at 75 steps/epoch): `0.0662, 0.0663,
0.0666, 0.0673, 0.0676, 0.0682, 0.0686, 0.0691, 0.0698, 0.0696` —
smooth, monotonic, real progress, no oscillation in the metric itself.
Already 6x higher than run 15's comparable-point reading. Loss band is
also tighter and trending down (settles into ~1.3k–3.4k after an
initial ~5.3k, vs. run 15's wider and less-improving swings).
`val/pair_acc` (both haplotypes correct jointly, the harder target)
remains 0.000 throughout — expected this early, not itself a red flag.

**Halving T and spending the freed memory on batch size compound
rather than substitute for each other**: T=4096/batch=32 clearly beats
T=8192/batch=16 despite using a *smaller* real batch, consistent with
the O(T²) mechanism above — shrinking T pays down memory quadratically,
which is what actually made room for a batch large enough to average
out per-window variance. Growing batch alone at the original T hit
that ceiling much sooner.

Run 17 continued cleanly through epoch 2 (20% in: `val/hap_acc` had
settled to a small plateau around 0.0696–0.0698, loss tightened further
into a ~400–1300 band, no spikes) — confirms the oscillation problem is
genuinely fixed, not just delayed past where run 5 collapsed.

## A third, separate gap: training-set size never matched the comparison recipe

**6-7% `val/hap_acc` is low in absolute terms, and the stability fix
above does not explain that away.** Random guessing across K=24
founders (+1 null state) gives ~4% — 7% is only ~1.7x above chance,
nowhere near the established `diploid-affinity-sim512-h3` comparison
model's `pair_acc=57.6%` by the end of *epoch 0 alone* (final
`hap_acc` 73-76%). Investigating this surfaced a third real gap, on
the same level as the `--inbreeding` mismatch and the O(T²) memory
issue, and — same pattern as both of those — not caught until directly
asked about rather than checked upfront: **`maize_indel_baseline*.npy`
was generated with `--windows 3000` the entire time**, picked somewhat
arbitrarily when first generating data for this baseline effort,
never checked against what the actual comparison recipe used. The
established recipe trains on **100,000 windows** (1000 individuals ×
100 windows each) — over 30x more. Even accounting for the new T=4096
windows being 8x longer than the old T=512 ones (more sites seen per
window), total site-exposure is still roughly `2400×4096 ≈ 9.8M`
(this run's train split) vs. `80,000×512 ≈ 41M` (the comparison
recipe's) — a real ~4x gap in raw training signal on top of whatever
optimization gap may remain.

**Regenerating at 100,000 windows** (`maize_indel_baseline_het_t4096_full.npy`,
same T=4096/`--inbreeding 0`/K=24/density=2.3e-3/seed=42 otherwise) to
match the established recipe's scale — in progress at time of writing,
expected ~20GB, likely a long generation given the 33x window-count
increase. Once it lands, relaunch with the same T=4096/batch=32/
homo-penalty=3/spike-skip/ema recipe that proved stable in run 17, and
see whether accuracy actually climbs toward a comparable range with
enough data, or whether a real optimization gap remains even once data
volume is no longer a confound.

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
- **A "fix" that only removes one confound can still leave a second,
  independent problem standing.** Correcting `--inbreeding` was
  necessary and clearly right, but run 14 showed it wasn't sufficient —
  the frozen-validation/persistent-oscillation pattern was a genuinely
  different, unrelated issue (self-attention memory/variance at
  T=8192), not a residual symptom of the same one. Fixing a diagnosed
  problem is not the same as confirming there's only one problem.
- **Trace the actual memory-consuming code before estimating scaling
  factors.** An earlier, vaguer explanation for the original OOM
  ("CRF forward-backward retains activations across T") wasn't wrong
  that memory scales with T, but it missed the real, dominant term —
  `nn.TransformerEncoder`'s O(T²) self-attention, two full orders of
  magnitude bigger than the 2x width factor from the actual format
  change. Grepping for the attention modules and reading their call
  shapes took a few minutes and produced a testable, falsifiable
  number; reasoning from the general shape of "CRF + long sequence"
  didn't.
- **When two independent levers both plausibly help, test whether they
  compound rather than picking one.** Batch=16 alone (same T) helped
  partially; T=4096 alone would free memory but do nothing for
  averaging noise on its own. Combining both (smaller T *freeing room
  for* a much bigger batch) produced a clearly better result than
  either extrapolated alone — because they address the same underlying
  O(T²) cost from two directions (shrink it, and spend the savings on
  averaging).
- **Check every dimension of the comparison baseline upfront, not one
  at a time as each gets questioned.** Three real mismatches against
  `diploid-affinity-sim512-h3` surfaced in this investigation —
  `--inbreeding` composition, the T=8192 sequence length vs. its
  T=512, and now training-set size (3,000 windows generated here vs.
  its 100,000) — and none were caught by checking the recipe before
  generating data; each was found only after a symptom (an unstable
  run, then a suspiciously low accuracy number) prompted a direct
  question. A single upfront diff against the comparison recipe's full
  command line would have caught all three before spending any GPU
  time, rather than one at a time after the fact.
