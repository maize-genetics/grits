# Worktree cleanup, 2 Oct 2026

Before any removal, every commit that existed only locally was pushed to origin, so no code or
committed result was lost. Uncommitted files worth keeping were copied first (see "Saved").

`HJ` = `/local/workdir/zrm22/HackathonJun2026`, `GW` = `HJ/grits_workdir`,
`AG` = `HJ/test_crf_relatedness/.claude/worktrees`, `RW` = `HJ/ropebwt_refMap/ropebwt3-phg/.claude/worktrees`.

## Branches pushed (all new on origin; no existing remote branch was overwritten)

| Repo | Branch | Why |
|---|---|---|
| grits | `v6-supervised-heads` | the v6 code, eval tools and these docs |
| grits | `supervised-heads-distfix` | active working branch (74 unpushed commits; v6 was developed here) |
| grits | `heldout-mosaic` | sticky relabel / mosaic hidden founder (negative result, HISTORY §2) |
| grits | `out-het-replacement` | het-replacement line incl. 4 supervised-heads diagram commits |
| grits | `supervised-heads` | sh3/sh4 line: head-quality JSONs, track plots, BEDs |
| grits | `two-stage-training` | two-stage/aux-loss study + anatomy results (HISTORY §3) |
| grits | `warmstart-hparam-guard` | warm-start recipe guard + shuffled split |
| grits | `archive/regionfix-out-snprc-check` | local `regionfix-out-snprc-check` was 8 ahead of its origin branch; pushed under a new name instead of overwriting (unique commit f053aef: diagnostic homo_scale override) |
| ropebwt3-phg | `v6-supervised-heads` | aad36c1: `-s 200` + `--lift-local-win` alignment used by v6 |
| ropebwt3-phg | `lift-syntenic-anchors` | syntenic-anchor filter for `--anchor-dist-npy` (not adopted) |
| ropebwt3-phg | `refmap-read-filters` | preset-F read filters (not adopted, see HISTORY §2) |
| ropebwt3-phg | `refmap-stats`, `refmap-multi-speedup` | July `--stats` / `--locate-cap` instrumentation |

## Saved before removal

| From | To | What |
|---|---|---|
| `GW/het-replacement-wt` (untracked) | `experiments/het-replacement/results/` on this branch | `repl_params_v2_nocross.json`, 5 `sim_calib_check_*.json` simulator-calibration checks |
| `GW/supervised-heads-wt` (untracked) | `experiments/supervised-heads/results/head_windows/`, `scripts/head_windows_{dump,plot}.py` | per-window encoder-head visualisations (sh4) |
| `AG/agent-a45bea03197fcd883` (untracked) | `docs/v6-supervised-heads/archive/regionfix-agent-scripts/` | 5 ad-hoc region-fix calibration scripts |
| same + `AG/agent-a62d2f3fb2e32395b` (ignored logs) | `GW/archive_logs/` (not in git) | region-fix and held-out-calibration training logs |

## Worktrees removed (all 20 succeeded)

| Worktree | Branch / commit | Why safe |
|---|---|---|
| `HJ/grits-indeltrain-worktree` | indel-training-loop | pushed + contained in v6 history; only change was a `pixi.lock` diff identical to the main checkout. Freed its own 6 GB `.pixi` |
| `HJ/grits-perfscore-worktree` | detached 63fac18 | on origin; its one untracked file is byte-identical to the tracked copy |
| `HJ/grits-tier2-worktree` | detached a5a0587 | clean, on origin |
| `HJ/grits-wholechrom-worktree` | detached 5898fd2 | clean, on origin |
| `HJ/grits-windowfilter-worktree` | windowing-quality-filters | clean, on origin (has its own PLAN/RESULTS/HANDOFF) |
| `GW/lowrate-wt` | indel-region-buffer-fix-lowrate | clean, on origin |
| `GW/warmstart-guard-wt` | warmstart-hparam-guard | pushed above |
| `GW/heldout-mosaic-wt` | heldout-mosaic | pushed above |
| `GW/het-replacement-wt` | out-het-replacement | pushed above; untracked JSONs saved |
| `GW/regionfix-out-snprc-wt` | regionfix-out-snprc-check | pushed as archive/… |
| `GW/supervised-heads-wt` | supervised-heads | pushed above; untracked files saved |
| `GW/two-stage-wt` | two-stage-training | pushed above |
| `AG/agent-a45bea03197fcd883` | indel-region-buffer-fix | on origin; scripts and logs saved |
| `AG/agent-a62d2f3fb2e32395b` | heldout-calibration | on origin; log saved |
| `AG/agent-a793b2b6a2dd9fdc6` | indel-collapse-bigger-model | on origin (negative result) |
| `AG/agent-ae4f7ccbe2b7dc4bc` | indel-collapse-moredata | on origin (negative result) |
| `RW/worktree-refmap-ps4g-numpy` | worktree-refmap-ps4g-numpy | tip contained in origin branches; no binary built, nothing references it |
| `RW/lift-syntenic-anchors`, `RW/refmap-read-filters`, `RW/refmap-multi-speedup` | (see pushes) | pushed above; nothing references their binaries. `refmap-multi-speedup/tmp_bench/` (1.1 GB of July benchmark output) discarded |
| stale `/tmp/.../refmap-before-baseline` | detached | directory already gone; `git worktree prune` |

## Second pass (user-approved): orphaned watchers killed, 4 more worktrees removed

14 orphaned log-watcher shells (`tail -f` / `until` loops, 9-17 days old, left by earlier Claude
sessions) were killed: pids 192504 3867628 3892874 1117801 1328953 1450005 1538196 4006336 4035428
3207118 3581363 497836 3311956 3610456 (8 needed SIGKILL). Then removed:

| Worktree | Branch | Why safe |
|---|---|---|
| `GW/indel-baseline-run-wt` | indel-baseline-run | on origin; its one untracked `.md` is identical to the tracked copy in supervised-heads-distfix |
| `GW/indel-v3-simulator-wt` | indel-heldout-augment | clean, on origin |
| `AG/agent-aa0329c51068b915c` | worktree-agent-aa0329c51068b915c (local branch also deleted) | tip contained in origin/indel-readcount-row-collapse |
| `AG/agent-af19f5a0eaeb11150` | indel-collapse-bigmodel-moredata | clean, on origin (negative result) |

## Kept, and why

- `HJ/test_crf_relatedness` (main checkout): its `.pixi` environment runs every job.
- `GW/sh-distfix-wt`: active; the all-depth comparison pipeline runs from it.
- `GW/v6-wt`: this branch.
- ropebwt3 `RW/lift-adaptive-projection` (running alignments use its binary),
  `RW/lift-ridx-ternary-dist-map`, `RW/pav-ps4g-insertion-rows`, `RW/refmap-ps4g-numpy`
  (scripts point at their built binaries).
