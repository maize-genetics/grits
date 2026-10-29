# nam_alignments

Rebuilds the maize_v2 25-founder panel VCF against the organelle-including
B73 reference (`chr1..chr10` + `Pt` + `Mt`): AnchorWave alignment for each of
the 24 NAM founder assemblies, MAF → gVCF conversion, a synthesized B73
reference gVCF, and a merge into one panel VCF.

Full run history, decisions, and verification for the run that produced the
current panel (`data/maize_v2_rebuild/panel/panel_25founders_v2.vcf`, 13 GB,
160,581,050 records) live in `HANDOFF.md` at the project root, under
"2026-08-08: AnchorWave realignment, gVCFs, and new 25-founder panel VCF —
DONE". This README exists so the pipeline is understandable without that
file, once these scripts move into a git repo.

## Run order

```
scripts/nam_alignments/prepare_ref_gff.sh
scripts/nam_alignments/realign_maize_v2.sh 4 8          # ~18h at 4 concurrent x 8 threads
scripts/nam_alignments/build_maize_v2_gvcfs.sh 8         # ~20 min
scripts/nam_alignments/make_b73_ref_gvcf.sh
scripts/nam_alignments/normalize_gvcf_contigs.sh 8
scripts/nam_alignments/sort_gvcfs.sh 6
```

Then the merge itself — there is no wrapper script for this step, it was run
directly:

```
biokotlin-tools merge-gvcfs \
  --input-dir   <gvcf_sorted dir> \
  --output-file <panel output.vcf>
```

Each numbered script is independently resumable (skips any sample whose
output already exists and is non-empty), so re-running the same command
after an interruption picks up where it left off. Exception: MAFs are
written incrementally during AnchorWave's `proali` step, so a MAF from a
killed/interrupted run can look "done" to the skip check — delete any
partial `.maf` before re-running `realign_maize_v2.sh`.

## Environment

| Purpose | Path |
|---|---|
| JDK 21 (**required**) | `/programs/jdk-21.0.1` — system `java` is JDK 13 and fails |
| conda env (anchorwave, minimap2, bgzip, bcftools) | `/home/zrm22/mambaforge/envs/phgv2-conda` |
| PHG v2 CLI (`align-assemblies`) | `/local/workdir/zrm22/HackathonJun2026/DebugSim/phg/bin/phg` |
| biokotlin-tools (`maf-to-gvcf-converter`, `merge-gvcfs`) | `/local/workdir/zrm22/HackathonJun2026/biokotlin-tools/build/install/biokotlin-tools/bin/biokotlin-tools` |

## Hardcoded paths — needs parameterizing before this is repo-portable

- Reference/assembly location: `/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/`
- Output root: `/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_v2_rebuild`
- The 24-name `FOUNDERS` array, duplicated identically in `realign_maize_v2.sh`
  and `build_maize_v2_gvcfs.sh`
- The 12-accession RefSeq-seqid → chromosome-name map hardcoded in the awk
  script inside `prepare_ref_gff.sh` (specific to the `GCF_902167145.1`
  assembly's GFF)

## Gotchas (already paid for once — don't re-learn these)

- **`merge-gvcfs`'s contig-order validation reads the record/index data, not
  the `##contig` header text.** Rewriting headers alone (`normalize_gvcf_contigs.sh`)
  does *not* fix a "different order" validation failure — the gVCF bodies
  must be physically re-sorted (`sort_gvcfs.sh`, via `bcftools sort`). This
  is why both steps exist and run in that order; skipping the sort step
  reproduces the failure even with perfectly normalized headers.
- **`bcftools reheader -f <fai>` is non-deterministic** across separate
  process invocations — it can reconstruct `##contig` in a different order
  each time, which is exactly the wrong tool when the goal is a canonical
  order. Use a literal, pre-built contig block spliced into each file's
  header text instead (see `normalize_gvcf_contigs.sh`).
- **`maf-to-gvcf-converter` appends `.gz` itself.** Pass the plain `.g.vcf`
  output name, not `.g.vcf.gz`, or you get `.g.vcf.gz.gz`.
- **`align-assemblies` swallows `proali` failures** — a failed alignment
  still exits non-error and falls through to dot-plot generation. Verify by
  checking MAF existence/size, never the process exit code.
- **`merge-gvcfs --input-dir` walks recursively** and picks up anything
  matching `*.g.vcf*`/`*.gvcf*` (compressed or not). Keep the input
  directory limited to exactly the files meant to be merged.
- Organelle alignment is founder-specific: whether a given NAM founder's
  assembly happens to align to `Pt`/`Mt` varies per founder, so their gVCFs
  can legitimately declare different `##contig` sets. This is the actual
  root cause of the contig-order validation failure above — it never
  surfaced against the older, organelle-free reference.
