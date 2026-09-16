# First real ternary+distance data: IDX-HYB Oh43xIl14H, 0.1x

2026-09-16. `ropebwt3-phg refmap --anchor-dist-npy` (branch
`lift-ridx-ternary-dist-map`, commit `26d04e0`) run against the real
0.1x hybrid read set, producing the first real (non-synthetic) data in
the `2K+2` ternary+distance training format.

## Run

```
ropebwt3 refmap --ref-prefix=B73 --max-occ=-1 -l 19 \
  --lift=maizeFastaIndex_SampleContig_v2.lift -t 20 \
  --label-bed=labels.bed --ps4g=raw.ps4g --npy=raw.npy \
  --anchor-dist-npy --anchor-dist-thresh=2000 \
  maizeFastaIndex_SampleContig_v2.fmd reads.fastq
```
Reads: `IDX-HYB/Oh43xIl14H/Oh43xIl14H.0.1x.{R1,R2}.fastq.gz` concatenated
(1,433,904 sequences). 52.6s real / 942.6s CPU (18-way parallel), 9.9GB
peak RSS. Output: `raw.npy` (1,117,328 x 77, `count|ternary|distance|gA,gB`
for K=25 founders), converted via a new `--anchor-dist-npy` mode added to
`ropebwt_npy_to_matrix.py` into `oh43xil14h_0.1x_ternary.npy`
(2176 windows x 512 x 52, `[ternary(K)|H1|H2|distance(K)]`, K=25,
log-coded distance matching `simulate_alleles.py`'s `_encode_dist`).
Labels are a placeholder (`"B73"` for every bin — real truth for a
hybrid lives in separate gVCFs, not embedded in refmap's own labels),
so label-coverage/het stats from this file are not meaningful; ignore
them.

## Findings

**All-panel (25 founders) ternary**: match 39.6% / diverged 19.8% /
deletion 40.6%.

**True-founders only** (Oh43 idx 21, Il14H idx 11) — the numbers that
actually matter:

| | match | diverged | deletion |
|---|---|---|---|
| Il14H | 67.0% | 10.0% | 22.9% |
| Oh43  | 67.4% | 12.0% | 20.6% |

- **Either true founder covered: 99.0%.** Both true founders match
  (uninformative site): 35.5%.
- Mean/median distance-to-anchor (decoded back to bp) for the true
  founders: Il14H 12,166 / 3,755bp, Oh43 10,917 / 2,655bp.

## Two things this surfaces, not yet resolved

**1. The either-covered rate (99.0%) is far higher than the simulator's
calibration target (~71.6% for maize)**, not lower as might be expected
from noisier real data. Two live hypotheses, not yet distinguished:
real coverage/homology genuinely leaves fewer true gaps than the
simulator assumes, or `--anchor-dist-thresh=2000` is generous enough
that most real gaps read as "diverged" rather than "deletion" and still
count toward "covered" (ternary `match` isn't gated by the threshold at
all, so this specific number is threshold-independent — the gap is real,
not a thresholding artifact, but still needs reconciling against how
the ~71.6% target was originally measured).

**2. The all-panel 40.6% deletion rate is likely confounded by anchor
granularity, not purely biological.** `ropebwt3 lift`'s default anchor
stride is `-s 2000`, and `--anchor-dist-thresh` was set to the matching
default of `2000`bp — i.e. the diverged/deletion cutoff sits right at
the anchor spacing itself, so a founder with genuinely uniform (no true
gap) but simply less-dense local anchoring can tip into "deletion" by
construction. The true-founder-only deletion rates (~21-23%) are far
lower than the all-panel number (40.6%, dragged down by less-related
founders), which is at least consistent with a real biological signal
underneath the confound, but the confound itself hasn't been isolated
(e.g. by re-running at a different `--anchor-dist-thresh` and checking
whether the deletion rate tracks it linearly, which would indict the
threshold, vs. staying flat, which would support a real signal).

## Next

These become the calibration targets for the v3 simulator modes
(lineage-derived and reference-stable-overlay) in place of the current
simulator's already-tuned-to-an-assumption constants — report the real
gap plainly rather than tuning either new mode to hit it artificially.
