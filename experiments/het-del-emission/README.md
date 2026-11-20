# Presence-aware (het-deletion) pair emission — 2026-10-14

**Problem.** The supervised-heads CRF scores a het pair (i, j) per row as a 50/50 mixture.
At a locus where founder i is deleted (its B73 sequence is replaced), almost every read comes from j
(i reads DEL). The true pair then pays log 2 on every such row against (j, j) or a look-alike
(X, j), so het-DEL sites get called with 0 deleted haplotypes. More rows per locus (higher depth)
means more log-2 charges, which is part of why IDX-HYB degrades with depth.

**Change** (`GRITSCRFDiploidIndel._presence_mix`, decode-time `model.del_rho` / `del_win`,
or `--del-rho-fit` for a stage-2 fitted value). Each haplotype contributes reads in proportion
to its read yield w = rho**d, where d = 1 if the founder reads DEL on the row:
mix = log(w_i e^l_i + w_j e^l_j) - log(w_i + w_j).
- rho = 1 gives the old mixture exactly.
- Homozygous and both-deleted pairs never change.
- So under homoroute, inbred and RIL2 decodes are bit-identical.

`del_win > 0` also infers d on MATCH rows from neighbouring rows. This hurts, because MATCH on
a deleted founder is real replacement-read evidence. **Use del_win = 0.**

**Selection.**
- Calibration corpus, hybrids only (`results/calib_sweep.txt`): rho 0.5 / 0.3 / 0.1 with w0, and
  0.3 with w16. Lower rho was better at 1x everywhere and at 0.1x for IDX and MIX.
- Chose **rho = 0.1, w0**.
- Validation corpus: `results/validation.txt`.

**Sim stage-2 fit** (`fit_rho.sh`, w16, ckpt `v6cfullfe-presence-stage2B`).
- It fitted rho = 0.67, but it also moved aff_w from 1.06 to 0.45, so the fit is confounded.
- Not evaluated. The simulator's replacement reads apparently over-yield compared with real data.
