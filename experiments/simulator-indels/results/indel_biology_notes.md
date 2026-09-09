# Indel biology notes — grounding for the simulator's indel model

Compiled 2026-09-09 to ground `simulate_alleles.py`'s planned indel
modeling in real biology, not an arbitrary parametric choice. Combines
general plant-genome literature (so the model stays defensible across
species, not maize-specific) with real calibration numbers measured
directly from this project's own maize founder gVCF data.

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

### Length distribution is bimodal, not monotone

Fine-grained histogram, CML103 chromosome 1 (432,887 indel events):

| length range | fraction | note |
|---|---|---|
| 1bp | 41.52% | dominant slippage mode |
| 2–3bp | 25.54% | |
| 4–7bp | 15.30% | |
| 8–15bp | 7.36% | |
| 16–31bp | 3.14% | |
| 32–63bp | 1.06% | |
| 64–127bp | 0.53% | **trough** — boundary between the two mechanisms |
| 128–4095bp | ~2.4% combined | transition zone |
| 4096–16383bp | 1.11–0.73% (two buckets) | **second mode** — matches LTR-retrotransposon scale |
| 16384bp+ | 0.86% combined, long tail to ~1Mb | |

Mean insertion/deletion length ~610bp / ~606bp (skewed high by the
large-indel component's long tail — the median is far below the mean).
Tail counts (chr1, CML103): ≥1kb → 15,628 events; ≥10kb → 6,450 events;
≥100kb → 267 events. A single geometric or power-law distribution fit to
the marginal statistics would systematically miss the 4–16kb mode
entirely, despite that mode carrying a disproportionate share of total
indel-affected base pairs.

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
