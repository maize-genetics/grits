# Indel biology notes — grounding for the simulator's indel model

Compiled 2026-09-09 to ground `simulate_alleles.py`'s planned indel
modeling in real biology, not an arbitrary parametric choice. Combines
general plant-genome literature (so the model stays defensible across
species, not maize-specific) with real calibration numbers measured
directly from this project's own maize founder gVCF data. **Updated
2026-09-10** after collaborator review: bp-weighted size distribution,
a multi-founder/multi-chromosome measurement, and a figure
(`indel_size_distribution.png`), plus a real measurement bug caught and
corrected along the way (see below). **Updated 2026-09-11**: cassava
calibration data added (second organism, confirms the general-plant
framing — see below), reusing the same script unchanged.

## Two distinct mutational mechanisms, two distinct size regimes

Real indel-length data (below) is **bimodal**, and the two modes map
cleanly onto two well-documented, mechanistically distinct processes.
This is the basis for the simulator's two-component mixture design
(see `../PLAN.md`) rather than a single parametric distribution.

### Small indels (1bp–~50bp): replication slippage

DNA polymerase slippage during replication of short repetitive tracts
(homopolymers, microsatellites) is the established primary mutational
source for small indels across plant genomes generally — this is not a
maize-specific phenomenon. Mutability increases sharply, and
nonlinearly, with repeat-tract length: in *Arabidopsis thaliana*, a
16-nucleotide homopolymer tract was found to be roughly 80-fold more
mutable than a 7-nucleotide tract of the same base. AT-rich dinucleotide
repeats show elevated mutation rates relative to other repeat classes.
This motivates giving the small component a homopolymer/repeat-context
placement bias in addition to a length distribution (flagged as an open
refinement in `../PLAN.md`, not yet designed in detail).

Sources:
- [Microsatellite Instability in Arabidopsis Increases with Plant Development](https://pmc.ncbi.nlm.nih.gov/articles/PMC2971617/)
- [Two Distinct Modes of Microsatellite Mutation Processes: Evidence From the Complete Genomic Sequences of Nine Species](https://genome.cshlp.org/content/13/10/2242.full)

### Large indels (kb-scale): LTR-retrotransposon activity

LTR retrotransposons are the single largest contributor to plant
genome-size variation across the whole plant kingdom — genome size is
approximately a linear function of LTR-retrotransposon content across
species, ranging from ~3% of small plant genomes to ~85% of large ones.
Insertions occur via replicative transposition (copy-and-paste,
directly increasing genome size at the insertion site). A specific,
well-documented removal mechanism — unequal homologous recombination
between a single element's two flanking LTR repeats — produces a
"solo-LTR," effectively a large deletion relative to the full-length
insertion. This insertion/removal cycle is the direct biological
analogue of the large-indel component, and mechanistically, of the
presence/absence variation (PAV) pattern the whole ternary-state feature
is meant to help the model recognize. It is a general plant phenomenon;
maize's own genome is a well-studied instance of it (nested
LTR-retrotransposon insertions are documented as the dominant source of
maize's large indel accumulation specifically).

Sources:
- [Transposable Elements and Genome Size Variations in Plants](https://pubmed.ncbi.nlm.nih.gov/25317107/)
- [Co-evolution of plant LTR-retrotransposons and their host genomes](https://pmc.ncbi.nlm.nih.gov/articles/PMC4875514/)
- [Nested Insertions and Accumulation of Indels Are Negatively Correlated with Abundance of Mutator-Like Transposable Elements in Maize and Rice](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3903597/)
- [Structural Variation and Its Roles in Plant Genomes](https://doi.org/10.3390/plants15162498) — general cross-species finding that SV/indel abundance decreases monotonically with length within the >50bp structural-variant class (consistent with, though coarser than, the bimodal structure found here).

## Real maize calibration data

Measured directly from `data/founder_truth_gvcfs_nofill/*.g.vcf.gz`
(24 NAM founders, explicit REF/ALT sequence — **not** the merged panel
VCF, see the methodological trap below), cross-validated against this
project's own pre-existing, independently-computed
`results/simval_events_results.tsv` (39 individuals across 6 dataset
classes).

**Indel fraction of variant events: ~11.6–12.7%**, essentially constant
across founders:

| founder | indel_frac | ins:del ratio |
|---|---|---|
| CML103 (genome-wide) | 12.44% | 0.986 |
| B97 (chr1) | 12.52% | 0.991 |
| Oh43 (chr1) | 12.66% | 0.996 |
| Tzi8 (chr1) | 12.56% | 0.988 |
| Tx303, held-out (chr1) | 11.71% | 0.992 |

**Insertions and deletions are essentially symmetric in event count**
(~0.99:1), confirmed independently in both this fresh gVCF query and the
pre-existing events table.

**This ~12% figure is a different quantity from the ~40% "genome
fraction" figure below — don't conflate them.** This one counts indel
*events* as a share of *all* variant events (SNP + indel); it says
nothing about how much reference sequence those events cover, since a
single indel event can span from 1bp to >100kb. The ~40% figure (next
section) counts reference *base pairs* with no homologous founder
sequence, as a share of the whole reference genome. Both are real,
correctly measured, and simultaneously true — indels are a small share
of variant *count* and a large share of affected *bp* — precisely
because of the size-class inversion documented below.

### Reference-genome-homology fraction: the ~40% figure, and a real measurement bug caught along the way

**Definition.** For each founder, the fraction of reference bp where
that founder's assembly has **no homologous sequence at all** —
i.e. the bp are inside a deletion (relative to reference) or a true
assembly gap. This is the number that matters for hemizygosity: it is
literally "how much of the reference genome will this founder
contribute zero reads to."

**Insertions are deliberately excluded from this fraction.** An
insertion doesn't remove any reference bp — it adds founder-assembly
sequence anchored at a single reference point (this is exactly what
drives read-stacking in `../PLAN.md` §2.5, not reference-bp loss). The
first pass at this measurement (during the collaborator-review planning
session) summed *both* insertion and deletion deltas into the
"affected" numerator and got CML103 chr1 = **81.6%** — roughly double
the correct figure. The bug: insertion delta is bp that exist only in
the founder's own assembly coordinates and were never part of the
reference span to begin with, so including them in a
*reference*-genome fraction double-counts non-existent reference bp.
Restricting the numerator to `gap_bp + deletion_bp` only corrects this
to **41.2%** for the same file — caught by the regression check built
into `../scripts/indel_size_report.py`'s own verification pass before
any design conclusion was drawn from the wrong number. Recorded here so
the mistake isn't repeated by a future pass at this measurement.

**Multi-founder, multi-chromosome measurement** — 8 founders × 10
chromosomes (80 founder/chromosome pairs), deletion+gap fraction,
generated by `../scripts/indel_size_report.py`, full per-pair values in
`indel_size_report_summary.tsv`, plotted in `indel_size_distribution.png`
(panel D):

| founder | mean across chr1–10 | min | max |
|---|---|---|---|
| B97 | 36.7% | 30.9% | 41.7% |
| CML103 | 39.6% | 35.5% | 43.0% |
| Ki3 | 40.9% | 37.4% | 44.3% |
| Ky21 | 36.9% | 30.3% | 43.0% |
| NC350 | 40.6% | 37.7% | 44.7% |
| Oh43 | 37.0% | 31.4% | 42.4% |
| P39 | 42.7% | 38.9% | 46.5% |
| Tzi8 | 39.9% | 36.2% | 42.7% |
| **all 8, pooled** | **39.3%** | **30.3%** | **46.5%** |

A188 was not present in this gVCF directory, so the "exclude the
assembly-quality outlier" step (below) did not need to be applied here
— worth re-checking if A188 gVCFs are added to this corpus later.

The **≥4kb events account for 94.8%** of indel bp across the full
80-pair sweep — consistent with the single-chromosome 94.5% figure
above, confirming the event/bp inversion is a stable population-level
property, not a CML103-chr1 artifact.

The collaborator's **~40% figure is confirmed**: 39.3% pooled mean,
every founder's per-chromosome range falling within roughly 30–47%.
This also supersedes the earlier (coincidentally close, but differently
and less correctly derived) single-founder block-based estimate.

### Methodological trap: the panel VCF's encoding is misleading

`data/maize_v2_rebuild/panel/panel_25founders_v2.vcf` (the merged
25-founder panel) encodes every variant with **purely symbolic**
`<INS>`/`<DEL>` alleles — confirmed directly (`max_REF_len == 1` across
40 million scanned records; zero explicit-sequence indels anywhere in
the file). Critically, **deletions are span-encoded** (every reference
position falling inside a deletion carries a `<DEL>` allele for the
carrier haplotype) while **insertions are anchor-encoded** (only the
single insertion point carries `<INS>`). A naive count of `<DEL>` vs
`<INS>` alleles in this file gives an ~11:1 ratio — this is a pure
artifact of the span-vs-anchor encoding asymmetry, not a real 11:1
del:ins bias. **Any indel calibration must use the explicit-sequence
founder gVCFs, never the panel VCF's symbolic encoding.**

### Length distribution is bimodal, not monotone — and event-weighted vs. bp-weighted are nearly inverted

Fine-grained histogram, CML103 chromosome 1 (432,887 indel events), now
reported **both ways** — % of events in each size class, and % of total
indel bp those events account for. This distinction is not cosmetic: a
collaborator review (2026-09-10) asked for the large-size distribution
specifically, and the two weightings tell almost opposite stories.

| size class | % of events | % of indel bp | ins:del ratio |
|---|---|---|---|
| 1bp | 41.52% | 0.07% | 0.968 |
| 2–3bp | 25.54% | 0.11% | 0.999 |
| 4–7bp | 15.30% | 0.14% | 1.019 |
| 8–15bp | 7.36% | 0.13% | 1.022 |
| 16–31bp | 3.14% | 0.11% | 0.976 |
| 32–63bp | 1.05% | 0.08% | 0.953 |
| 64–127bp | 0.53% | 0.09% | 1.029 |
| 128–255bp | 0.74% | 0.23% | 1.066 |
| 256–511bp | 0.64% | 0.40% | 1.029 |
| 512–1023bp | 0.58% | 0.72% | 0.973 |
| 1024–2047bp | 0.46% | 1.16% | 1.025 |
| 2048–4095bp | 0.43% | 2.24% | 0.998 |
| 4096–8191bp | 0.73% | 7.87% | 0.967 |
| 8192–16383bp | 1.11% | **21.49%** | 0.980 |
| 16384–32767bp | 0.50% | **19.84%** | 0.978 |
| 32768–65535bp | 0.23% | **17.93%** | 0.982 |
| 65536–131071bp | 0.10% | **15.69%** | 0.950 |
| 131072–262143bp | 0.03% | 9.06% | 1.000 |
| 262144–524287bp | 0.00% | 2.05% | 2.750 |
| ≥1048576bp | 0.00% | 0.61% | — (n=1) |

**≤1bp events are 41.5% of all indel events but 0.07% of indel bp.
Events ≥4096bp are ~2.7% of events but ~94.5% of indel bp.** A
simulator calibrated to the event histogram (dominated by 1–3bp
slippage events) would produce almost no indel-affected genome at all —
the large component is not a minor addition to a small-indel-dominated
picture, it **is** where nearly all the affected reference bp comes
from — the calibration target below must be fit against the bp-weighted
table, not the event histogram. ins:del ratio stays close to 1.0 in
every size class (0.95–1.07, one small-n exception at the largest
bucket) — the symmetry already noted above holds size-by-size, not just
in aggregate.

Reproducible via `../scripts/indel_size_report.py`; multi-chromosome,
multi-founder figures in `indel_size_distribution.png`.

### Assembly-quality outlier: exclude A188 from parameter fitting

A188 (one of the 5 held-out lines) shows 19–20% indel fraction (vs.
11.5–12.7% for every other founder) and 60.5% of its indels at exactly
1bp (vs. ~41% for everyone else) — both in this fresh gVCF query and
independently in the pre-existing events table (every A188-containing
cross in `simval_events_results.tsv` shows indel_frac 0.159–0.167 vs.
0.115–0.117 for non-A188 crosses). This is corroborated by this
project's own prior reference-bias investigation
(`../../reference-bias/results/refbias_pav_mechanism.md`), which found
A188 has 282–1,682 assembly sequences and 6.9% scaffold content, versus
B73's 12 sequences and 0% scaffold content — an assembly-quality
artifact, not real biology. **Do not fit indel-model parameters on
A188.**

## Cassava calibration data (second organism, confirms the general-plant framing)

Measured 2026-09-11 from `grits_workdir/cassava/gvcfs/` (106
haplotype-resolved gVCFs, same PHGv2 `ASM_Start`/`ASM_End` format as the
maize founder gVCFs — the same method and script apply unchanged). 6
individuals (12 haplotypes: BGM_2098, COL1734, COL386, ECU72,
IITA_TMS_IBA000070, TMe_261) × all 18 chromosomes = 216
haplotype/chromosome pairs, run in parallel (12-way, sharded by
haplotype) rather than serially, per this project's standing
resource-check-before-parallelizing practice. Figure:
[`cassava_indel_size_distribution.png`](cassava_indel_size_distribution.png).

**The event/bp inversion is not maize-specific — it reproduces almost
exactly in cassava**, a highly heterozygous, clonally propagated
outcrosser structurally opposite to inbred maize:

| | maize (chr1, 8 founders×10 chroms) | cassava (12 haplotypes×18 chroms) |
|---|---|---|
| 1bp events | 41.5% of events, 0.07% of bp | 43.9% of events, 0.17% of bp |
| ≥4kb events | ~2.7% of events, ~94.8% of bp | ~1.3% of events, 88.1% of bp |
| ins:del ratio | 0.95–1.07 (most classes) | 0.99–1.03 (most classes) |

This is real support for §2.4's core design claim — the two-component
mixture *structure* (small/slippage-driven, large/retrotransposon-
driven) is general-plant biology, not something fit to maize's numbers
specifically.

**The indel-affected reference-bp fraction is measurably lower in
cassava than maize**: pooled mean **32.7%** (per-haplotype means
30.5–35.2%, per-pair range 10.9–55.3%) versus maize's 39.3% pooled mean.
Reported as found, not forced to match — a ~7 percentage-point gap
between an inbred-vs-reference comparison (maize) and a heterozygous-
individual-haplotype-vs-reference comparison (cassava) is plausible from
several non-exclusive causes (assembly quality, time since divergence
from the specific reference genotype, real biological turnover rate)
that this measurement alone can't distinguish — flagged as an open
question, not resolved here. No cassava analogue of maize's A188 outlier
has been identified in this 12-haplotype subset, but that is not a
systematic outlier scan (only ~11% of the 106 available gVCFs were
sampled) and shouldn't be read as "cassava has no outliers."

**Practical consequence for `PLAN.md` §2.7**: cassava calibration input
is no longer blocked — the earlier "on hold until the collaborator
supplies cassava data" note is stale as of this measurement. What's
still deferred is the *simulator-output* side of cassava validation (the
~72%-either-founder-covered acceptance check), which needs the
read-sampling/coverage layer (§2.5) actually built first.

## Why this matters for the model's current error budget

This is not a marginal correction. Existing project measurements:
`results/simval_snp_refcall_results.md` shows 40–62% of otherwise
comparable sites are indel-touching and currently excluded from
SNP+RefCall scoring entirely. The 2026-08-06 investigation recorded in
`docs/HANDOFF.md` (history) found that removing indels from scoring
moves held-out event-level error from 2.47% to 3.41% — i.e., indel-heavy
regions currently contribute a large, non-representative share of both
the model's easy and hard cases, and a simulator that cannot produce
realistic indel structure cannot produce representative training data
for this problem.
