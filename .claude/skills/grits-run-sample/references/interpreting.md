# Interpreting GRITS results

## What accuracy to expect

These numbers come from simulated reads of real assemblies; the samples were never seen in training.
Each cell is genotype error % at SNPs and reference calls / gene-window error % per event.
- **In panel**: both parents are among the 25 founders.
- **Mixed**: one parent is among them.
- **Out of panel**: neither parent is.

| Sample type | 0.1x | 1x |
|---|---|---|
| In-panel inbred | 0.00 / 0.001 | 0.00 / 0.001 |
| In-panel hybrid (F1 of two founders) | 0.06 / 0.003 | 0.27 / 0.004 |
| In-panel RIL | 0.01 / 0.001 | 0.02 / 0.002 |
| Mixed hybrid | 5.4 / 0.23 | 4.3 / 0.20 |
| Mixed RIL | 2.1 / 0.21 | 1.8 / 0.16 |
| Out-of-panel inbred | 4.6 / 0.41 | 3.8 / 0.32 |
| Out-of-panel hybrid | 9.6 / 0.44 | 7.7 / 0.39 |
| Out-of-panel RIL | 4.6 / 0.42 | 3.8 / 0.34 |

Nearly the whole genome receives a call (99.6-100%).

**Out-of-panel samples:** GRITS can only say which of the 25 founders each region looks *most like*.
A large part of the remaining error is because no founder is a close match, not the model's choice; more
assemblies in the panel would lower it. Say this plainly when reporting on non-NAM germplasm. "Region X
looks like Oh43" means "Oh43 is the closest of these 25", not "it is Oh43".

**Out-of-panel hybrids** are the weakest case. The errors concentrate near founder switches, where one of
the two haplotypes is called wrongly.

## The routing line

The decode prints, for example:

```
route: disj=0.47 p90=0.35 hybrid=True in_panel=True
```

- **`hybrid`**: the sample is called heterozygous when `disj > 0.25`. `disj` measures how often nearby
  reads support disjoint sets of founders. In tests, inbreds and RILs stayed below about 0.23 and hybrids
  were well above 0.3.
- **Non-hybrids are decoded as homozygous only.** A RIL's residual heterozygosity, or an F2's, is
  forced to one founder there. There is no separate RIL or F2 mode.
- **`in_panel`** is a guess from how well the reads fit the panel. It only changes a few tuning settings.
- **If `hybrid` contradicts what the user knows**, say so and treat the result with suspicion:
  - an F1 called non-hybrid will show no heterozygosity at all;
  - an inbred called hybrid will show spurious heterozygous blocks.
  - A `disj` close to 0.25 makes such errors more likely. It happens most at depths well above 1x, where
    `disj` creeps up for inbreds.

## What the calls do not tell you

- **Parent of origin.** `parent1`/`parent2` are an unordered pair, so a BED row can't say which founder
  came from which parent of a hybrid.
- **Crossovers, in an F1 hybrid.** An F1 carries two unbroken parental chromosomes. Every founder switch
  along an F1 chromosome is the model moving to whichever founder best matches that parent locally, not
  recombination. For RILs and other segregating material, long blocks do reflect real recombination, but
  short blocks between them are mostly the closest-founder effect.
- **Which model produced a run.** `decode_summary.txt` ends with a provenance line: checkpoint, preset and
  GRITS commit. Quote it when sharing results.

## Sanity checks to run on every result

- **Heterozygous fraction** (`decode_summary.txt`): about 0 for inbreds and RILs, and above 0.9 for F1
  hybrids. Values in between suit segregating material, but in this model only a hybrid-routed sample
  can have them.
- **Founder shares:**
  - An in-panel line should be almost entirely one founder.
  - An F1 of two founders should be almost entirely that pair.
  - A RIL from two founders should show long blocks of each.
  - Many founders in short blocks is normal for out-of-panel material; it is the "closest founder"
    effect above.
- **Reads placed** (`EXACT` + `PLACED` in `align_summary.txt`): typically 50-70%. Much lower suggests the wrong species or
  organism, heavy contamination, or reads much shorter than 150 bp.
- **Coverage gaps:**
  - A chromosome with no BED means fewer than 512 informative positions were placed on it. Usually the
    depth is far too low.
  - Each chromosome's final partial window is never called, so expect small uncalled tails at the
    chromosome ends.

## Depth

The model has two tuned settings, chosen automatically from read density: the 0.1x one up to about
1,500 placed rows per Mb, and the 1x one above that. Coverage between about 0.3x and 0.5x sits near that
boundary, and coverage far above 1x was not calibrated.
- More depth is not always better for out-of-panel samples: in tests their error was lowest around 0.5x.
- 2x or more mostly costs time.

## Common failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `check_setup.sh` FAIL on ropebwt3 options | old ropebwt3 build | build ropebwt3-phg `grits-pipeline` |
| `expected the 25-founder panel` | wrong index (e.g. a 24-founder one) | use the v2 25-founder index; it must match the checkpoint |
| `unknown preset` | GRITS checkout too old | use a branch with `placedfix_hetdel_hs_peel` |
| CUDA out of memory | GPU shared or too small | `--batch-size 64`, or wait for a free GPU |
| `GLIBCXX ... not found` | env's libstdc++ not on the library path | run with `LD_LIBRARY_PATH=$GRITS_LD` |
| `window count does not match` | windowing and bins from different runs | re-run `align_sample.sh` into a fresh directory |
| VCF has no calls on a chromosome | no BED for it (see coverage gaps) | check `decode_summary.txt` |
