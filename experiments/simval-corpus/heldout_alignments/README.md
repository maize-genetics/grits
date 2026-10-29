# heldout_alignments

Realigns the 5 held-out (out-of-index) assemblies -- **Tx303, A188, EP1,
CML459, Ia453** -- against the organelle-including `maize_v2/B73.fa`
reference, at parity with `scripts/nam_alignments/`'s founder rebuild
(same reference, same reused ref-prep anchors, same AnchorWave parameters,
same no-fill MAF-to-gVCF convention).

These 5 assemblies are the project's only genuine held-out generalization
signal (see `HANDOFF.md`, "Goal" section). Their truth gVCFs were
previously built against the *old* B73 (pre organelle-trim); those are now
coordinate-incompatible with `rope_bwt_index_v2/` /
`panel_25founders_v2.vcf` and cannot score anything from the rebuilt
pipeline. This directory's scripts replace them.

Full context and decisions: `/home/zrm22/.claude/plans/ok-now-that-we-warm-minsky.md`.
Shared gotchas (AnchorWave swallowing `proali` failures, `.gz` double
extension, etc.) are documented once in `scripts/nam_alignments/README.md`
-- read that first, this file only covers what's different here.

## Run order

```
scripts/heldout_alignments/make_heldout_symlinks.sh
scripts/heldout_alignments/align_heldout_maize_v2.sh 5 10   # ~2.5-3h, all 5 concurrent
scripts/heldout_alignments/build_heldout_gvcfs.sh 5          # ~20 min
scripts/heldout_alignments/sort_heldout_gvcfs.sh 5           # ~few min, optional but recommended
```

Each script is independently resumable (skips any sample whose output
already exists and is non-empty) -- same caveat as the founder scripts:
delete any partial `.maf` before re-running `align_heldout_maize_v2.sh`
after a kill/crash, since MAFs grow incrementally during `proali` and can
look "done" to the skip check.

## What's different from `scripts/nam_alignments/`

- **5 samples, not 24**, sourced from three different directories under
  three different naming conventions (`non_index_asms/maize/`,
  `data/external_assemblies/`), not one panel directory. `make_heldout_symlinks.sh`
  normalizes them into `<Sample>.fa` symlinks first -- `align-assemblies`
  derives its output MAF name (and this project's convention passes the
  same string as `--sample-name`) from the assembly file's basename, so
  this is what turns e.g. `ep1.genome.fa` into a clean `EP1` sample rather
  than propagating the ugly original filename through the MAF, gVCF, and
  every downstream comparison.
- **No ref-prep step.** `align_heldout_maize_v2.sh` does not regenerate
  `ref.cds.fasta`/`B73.sam`/`B73_v2.gff3` -- it hard-requires the founder
  run's own copies under `data/maize_v2_rebuild/ref/` and fails loudly if
  they're missing. Reusing the exact same anchors as all 24 founders is the
  point; regenerating them would risk subtly different anchoring between the
  founder panel and the held-out samples, undermining the comparison this
  is for.
- **Outputs live under a separate tree, `data/maize_v2_heldout/`**, never
  under `data/maize_v2_rebuild/`. `merge-gvcfs --input-dir` walks
  recursively and picks up anything matching `*.g.vcf*` -- a held-out gVCF
  sitting under the panel tree would silently contaminate any future panel
  merge. These 5 gVCFs are truth data for scoring, not panel inputs, and
  must never be merged into `panel_25founders_v2.vcf`.
- **`sort_heldout_gvcfs.sh` combines normalize+sort in one script** (the
  founder pipeline splits these into `normalize_gvcf_contigs.sh` +
  `sort_gvcfs.sh` because the merge step needs the intermediate
  header-normalized-but-unsorted stage as a checkpoint). Not load-bearing
  for a `merge-gvcfs` validation here since these 5 are never merged with
  the founders -- done anyway so all `.g.vcf.gz` files in the project share
  one contig-order convention.

## Downstream usage

Each sample's sorted gVCF (`data/maize_v2_heldout/gvcf_sorted/<Sample>.g.vcf.gz`)
is a drop-in replacement for that sample's old truth gVCF wherever it's
consumed:

- `heldout_batch.py`'s work-list TSV third column (`sample<TAB>fasta<TAB>truth_gvcf`)
- `heldout_assembly_eval.py --truth-gvcf PATH`

Rerunning the actual held-out evaluation (refmap -> CRF+affinity inference
-> `bed-to-vcf` -> `compare_gvcf_truth.py`) against these new truth gVCFs,
`rope_bwt_index_v2/`, and `panel_25founders_v2.vcf` is deliberately **not**
done by this directory's scripts -- see the plan file's "Out of scope"
section. Before that rerun: `scripts/preflight_heldout.py`'s
`EXPECTED_CONTIGS` (currently chr1-10 only) and the index/panel paths
hardcoded in `scripts/nam_baseline.py:47-48` / `scripts/kmer_sweep.py:31-32`
need updating to point at the v2 index/panel first.
