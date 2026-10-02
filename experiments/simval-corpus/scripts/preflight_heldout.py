#!/usr/bin/env python
"""
Preflight checks for heldout_assembly_eval.py -- run this BEFORE launching a
multi-hour AnchorWave + refmap + inference chain on a new assembly, so a
wiring/config mistake costs 30 seconds instead of half a day.

heldout_assembly_eval.py's own docstring admits it does NOT verify the sample
is genuinely out-of-index (its "CAUTION" paragraph: "this script does not
verify that for you"). This script makes that a real, enforced check, plus
the other things that would silently corrupt or waste a run:

  1. Index exclusion    -- sample name / assembly basename not already in
                            ropebwt_refMap/keyfile_fastas.txt.
  2. Chromosome scale   -- few large sequences spanning most of the genome
                            (not a fragmented hifiasm/canu contig-level
                            draft), and a sane maize genome size. This checks
                            SIZE, not NAME: traced build_truth_gvcf.sh's
                            AnchorWave -> biokotlin MAFToGVCF chain (by
                            decompiling biokotlin-1.0.0.jar) and confirmed the
                            truth gVCF's CHROM column always comes from the
                            REFERENCE (B73) side of the alignment
                            (AssemblyVariantInfo.getChr()) -- the query
                            assembly's own contig name is preserved only as a
                            separate ASM_Chr annotation, never used as the
                            comparison axis, and AlignAssemblies.kt has no
                            naming requirement on the query side either. So a
                            candidate's own contig-naming convention (chr1,
                            Chr1, or bare "1") has NO bearing on correctness --
                            an earlier version of this check required literal
                            "chr1".."chr10" names and would have falsely
                            rejected several genuinely good, officially
                            published chromosome-scale assemblies (e.g. the
                            European Flint panel, which uses bare "1".."10").
                            What DOES matter is genuine chromosome-scale
                            continuity, since a fragmented draft produces a
                            noisy/incomplete AnchorWave alignment -- that's a
                            size property, checked here via .fai lengths.
  3. Panel/index pairing -- the panel VCF is our OWN 25-founder
                            panel_25founders.vcf, never smm477's 24-founder
                            (B73-excluded) one (HANDOFF.md: confirmed
                            mismatch would silently corrupt scoring).
  4. Tool resolution     -- wgsim, sample (bed-to-vcf CLI), phg
                            (align-assemblies), biokotlin-tools
                            (maf-to-gvcf-converter), JDK 21, the ropebwt3
                            index files, and samtools/bcftools on PATH all
                            exist and are reachable, reusing the exact
                            constants heldout_assembly_eval.py itself uses
                            (imported from it, not re-hardcoded).

Usage:
    python preflight_heldout.py <assembly.fa> <sample_name> [--panel-vcf PATH]

Run it with the SAME python you intend to run heldout_assembly_eval.py with
(e.g. /home/zrm22/mambaforge/envs/phg-ml/bin/python -- see nam_baseline.py's
own usage docstring) since heldout_assembly_eval.py's run_comparison() step
launches compare_gvcf_truth.py via `PY = sys.executable`, i.e. whatever
interpreter ran the driver is reused for the comparator subprocess too.

Exits 0 ("PREFLIGHT OK") only if every check passes. Exits 1 with specific
per-check failure messages otherwise -- never silently continues past a
failed check.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../refmap-founder-eval/scripts"))  # nam_baseline.py lives in the refmap-founder-eval experiment
import nam_baseline as nb              # noqa: E402  (reuse REFMAP_ROOT/FMD/LIFT)
import heldout_assembly_eval as hae    # noqa: E402  (reuse WGSIM/SAMPLE_BIN/etc paths;
                                        # safe to import -- its numpy/pandas-importing
                                        # module level has no heavy side effects, and
                                        # torch is only imported inside main())

KEYFILE = nb.REFMAP_ROOT / "keyfile_fastas.txt"
OWN_PANEL_VCF = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/"
                      "maize_panel_vcf/panel_25founders.vcf")

# NAM founder FASTAs in ropebwt_refMap/fastas/ run ~2.16-2.35 Gb across
# chr1-chr10; pad generously so a legitimately more/less compact maize line
# doesn't false-positive, while still catching "wrong species" / truncation.
MIN_GENOME_SIZE = 1_800_000_000
MAX_GENOME_SIZE = 2_700_000_000

N_CHROMOSOMES = 10          # maize: 10 chromosomes
TOP_N_MIN_FRACTION = 0.90   # top N_CHROMOSOMES sequences must cover >=90% of the genome
MIN_CHROM_SIZE = 20_000_000  # each of the top N_CHROMOSOMES must exceed this floor
                              # (real maize chromosomes run ~150-300 Mb; 20 Mb is a
                              # generous floor that only excludes genuinely fragmented
                              # contigs, not a legitimately smaller real chromosome)
EXPECTED_CONTIGS = {f"chr{i}" for i in range(1, N_CHROMOSOMES + 1)}  # naming convention
                                                                       # check only (informational)


class PreflightError(Exception):
    pass


def load_keyfile_founders():
    founders = {}
    for line in KEYFILE.read_text().splitlines():
        if not line.strip():
            continue
        fasta_rel, name = line.split("\t")
        founders[name] = str(nb.REFMAP_ROOT / fasta_rel)
    return founders


def check_fasta_readable(assembly_fasta):
    assembly_fasta = Path(assembly_fasta)
    if not assembly_fasta.exists():
        raise PreflightError(f"assembly FASTA not found: {assembly_fasta}")
    size = assembly_fasta.stat().st_size
    if size < 1_000_000:
        raise PreflightError(
            f"assembly FASTA is suspiciously small ({size:,} bytes) -- likely "
            f"truncated or the wrong file: {assembly_fasta}")
    print(f"[OK] FASTA readable: {assembly_fasta} ({size:,} bytes)")


def check_index_exclusion(assembly_fasta, sample_name):
    if not KEYFILE.exists():
        raise PreflightError(f"index keyfile not found: {KEYFILE}")
    founders = load_keyfile_founders()
    if sample_name in founders:
        raise PreflightError(
            f"sample name {sample_name!r} IS in the index keyfile ({KEYFILE}) -- "
            f"this would be an in-panel run, not a held-out test. Rename the "
            f"sample, or confirm this is intentionally the in-panel smoke test "
            f"(nam_inpanel_smoketest.py), not this preflight's use case.")
    fasta_basename = Path(assembly_fasta).name
    keyfile_basenames = {Path(v).name for v in founders.values()}
    if fasta_basename in keyfile_basenames:
        raise PreflightError(
            f"assembly file basename {fasta_basename!r} matches a FASTA already "
            f"in the index keyfile -- even though the sample NAME differs, this "
            f"looks like the same underlying assembly file. Confirm this is a "
            f"distinct, genuinely out-of-index assembly.")
    print(f"[OK] index exclusion: {sample_name!r} / {fasta_basename!r} not in "
          f"{KEYFILE} ({len(founders)} founders checked)")


def check_chromosome_scale(assembly_fasta):
    """Size-based check: the assembly must be genuinely chromosome-scale (a
    handful of large sequences covering most of the genome), not a
    fragmented hifiasm/canu contig-level draft. Does NOT require any
    particular contig NAMING convention -- see the module docstring for why
    that would test the wrong side of the eventual truth-gVCF comparison."""
    assembly_fasta = Path(assembly_fasta)
    fai = Path(str(assembly_fasta) + ".fai")
    if not fai.exists():
        print(f"  building .fai index for {assembly_fasta} (samtools faidx)...")
        proc = subprocess.run(["samtools", "faidx", str(assembly_fasta)],
                               capture_output=True, text=True)
        if proc.returncode != 0:
            raise PreflightError(f"samtools faidx failed: {proc.stderr[-1000:]}")
    contig_sizes = {}
    for line in fai.read_text().splitlines():
        parts = line.split("\t")
        contig_sizes[parts[0]] = int(parts[1])

    total_size = sum(contig_sizes.values())
    ranked = sorted(contig_sizes.items(), key=lambda kv: kv[1], reverse=True)
    top_n = ranked[:N_CHROMOSOMES]
    top_n_size = sum(size for _name, size in top_n)
    top_n_fraction = (top_n_size / total_size) if total_size else 0.0

    if len(ranked) < N_CHROMOSOMES or any(size < MIN_CHROM_SIZE for _name, size in top_n):
        raise PreflightError(
            f"assembly does not look chromosome-scale: fewer than {N_CHROMOSOMES} "
            f"sequences exceed the {MIN_CHROM_SIZE:,} bp floor (found "
            f"{len(contig_sizes):,} sequences total, largest {ranked[0][1]:,} bp "
            f"if any). This looks like a fragmented contig-level draft (e.g. raw "
            f"hifiasm/canu output) rather than a scaffolded/chromosome-level "
            f"assembly -- AnchorWave alignment against this would likely be noisy "
            f"and incomplete. Top sequences: "
            f"{[(n, s) for n, s in ranked[:5]]}")

    if top_n_fraction < TOP_N_MIN_FRACTION:
        raise PreflightError(
            f"top {N_CHROMOSOMES} sequences cover only {top_n_fraction:.1%} of the "
            f"total assembly size ({top_n_size:,} / {total_size:,} bp) -- expected "
            f">={TOP_N_MIN_FRACTION:.0%}. This looks like a fragmented draft, not a "
            f"genuinely chromosome-scale assembly.")

    if not (MIN_GENOME_SIZE <= total_size <= MAX_GENOME_SIZE):
        raise PreflightError(
            f"total assembly size {total_size:,} bp is outside the sane maize "
            f"range [{MIN_GENOME_SIZE:,}, {MAX_GENOME_SIZE:,}] -- double check "
            f"this is the right assembly, not truncated, and not a different "
            f"species.")

    print(f"[OK] chromosome scale: top {N_CHROMOSOMES} sequences cover "
          f"{top_n_fraction:.1%} of {total_size:,} bp total "
          f"({len(contig_sizes):,} sequences overall)")

    # Informational only, never fatal: whether naming happens to match B73's
    # own chr1..chr10 convention. Doesn't affect the truth gVCF (which is
    # always named from the REFERENCE side, not the query), but useful
    # context if you're later staring at ASM_Chr annotations wondering why
    # they don't say "chr1".
    top_n_names = {name for name, _size in top_n}
    if top_n_names == EXPECTED_CONTIGS:
        print(f"  [info] top {N_CHROMOSOMES} sequence names match the "
              f"chr1..chr{N_CHROMOSOMES} convention exactly")
    else:
        print(f"  [info] top {N_CHROMOSOMES} sequence names do NOT match "
              f"chr1..chr{N_CHROMOSOMES} (this is fine -- naming has no bearing "
              f"on the truth-gVCF comparison): {sorted(top_n_names)}")


def check_panel_pairing(panel_vcf):
    panel_vcf = Path(panel_vcf)
    if panel_vcf.resolve() != OWN_PANEL_VCF.resolve():
        print(f"  [WARN] --panel-vcf ({panel_vcf}) is not the default 25-founder "
              f"panel ({OWN_PANEL_VCF}) -- confirm this override is intentional.")
    if not panel_vcf.exists():
        raise PreflightError(f"panel VCF not found: {panel_vcf}")
    proc = subprocess.run(["bcftools", "view", "-h", str(panel_vcf)],
                           capture_output=True, text=True)
    if proc.returncode != 0:
        raise PreflightError(f"could not read panel VCF header ({panel_vcf}): "
                              f"{proc.stderr[-500:]}")
    header_lines = [l for l in proc.stdout.splitlines() if l.startswith("#CHROM")]
    if not header_lines:
        raise PreflightError(f"panel VCF has no #CHROM header line: {panel_vcf}")
    samples = header_lines[0].split("\t")[9:]
    if len(samples) != 25 or "B73" not in samples:
        raise PreflightError(
            f"panel VCF {panel_vcf} has {len(samples)} sample column(s) "
            f"(expected 25, including B73) -- this looks like the wrong panel "
            f"(e.g. smm477's 24-founder, B73-excluded one; HANDOFF.md: 'do not "
            f"use... for anything'). Samples found: {samples}")
    print(f"[OK] panel pairing: {panel_vcf.name} has {len(samples)} founders "
          f"including B73")


def check_tools():
    checks = [
        ("wgsim", hae.WGSIM),
        ("sample (bed-to-vcf CLI)", hae.SAMPLE_BIN),
        ("build_truth_gvcf.sh", hae.BUILD_TRUTH_GVCF_SH),
        ("compare_gvcf_truth.py", hae.COMPARE_SCRIPT),
        ("affinity checkpoint", hae.AFFINITY_CKPT),
        ("ropebwt3 index .fmd", nb.FMD),
        ("ropebwt3 index .lift", nb.LIFT),
        ("ropebwt3 binary", nb.BIN),
        ("JDK 21 java (build_truth_gvcf.sh default)",
         Path("/programs/jdk-21.0.1/bin/java")),
        ("phg CLI (align-assemblies, build_truth_gvcf.sh default)",
         Path("/local/workdir/zrm22/HackathonJun2026/DebugSim/phg/bin/phg")),
        ("biokotlin-tools (maf-to-gvcf-converter, build_truth_gvcf.sh default)",
         Path("/local/workdir/zrm22/HackathonJun2026/biokotlin-tools/build/install/"
              "biokotlin-tools/bin/biokotlin-tools")),
    ]
    missing = [(label, path) for label, path in checks if not Path(path).exists()]
    if missing:
        lines = "\n".join(f"  - {label}: {path}" for label, path in missing)
        raise PreflightError(f"missing required tool(s)/path(s):\n{lines}")
    for tool in ("samtools", "bcftools"):
        if shutil.which(tool) is None:
            raise PreflightError(f"{tool!r} not found on PATH")
    print(f"[OK] all {len(checks)} required tool paths resolve; samtools/bcftools "
          f"on PATH")


def check_interpreter():
    """compare_gvcf_truth.py is launched via `PY = sys.executable` in
    run_comparison() -- i.e. whatever interpreter ran heldout_assembly_eval.py
    also runs the comparator subprocess. Confirm THIS interpreter (the one
    running preflight_heldout.py right now) has what both stages need,
    since that's the same interpreter that would run the real driver."""
    try:
        import numpy, pandas  # noqa: F401
    except ImportError as e:
        raise PreflightError(
            f"the python running this preflight check ({sys.executable}) is "
            f"missing {e.name} -- this is the same interpreter "
            f"heldout_assembly_eval.py's own comparator subprocess reuses "
            f"(`PY = sys.executable`), so run both with an env that has numpy/"
            f"pandas/torch (e.g. /home/zrm22/mambaforge/envs/phg-ml/bin/python, "
            f"per nam_baseline.py's own usage docstring).")
    print(f"[OK] interpreter {sys.executable} has numpy/pandas importable")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("assembly_fasta")
    ap.add_argument("sample_name")
    ap.add_argument("--panel-vcf", default=str(OWN_PANEL_VCF))
    args = ap.parse_args()

    checks = [
        ("FASTA readable", lambda: check_fasta_readable(args.assembly_fasta)),
        ("index exclusion", lambda: check_index_exclusion(args.assembly_fasta, args.sample_name)),
        ("chromosome scale", lambda: check_chromosome_scale(args.assembly_fasta)),
        ("panel pairing", lambda: check_panel_pairing(args.panel_vcf)),
        ("tool resolution", check_tools),
        ("interpreter", check_interpreter),
    ]

    failures = []
    for label, fn in checks:
        try:
            fn()
        except PreflightError as e:
            print(f"[FAIL] {label}: {e}")
            failures.append(label)
        except Exception as e:
            print(f"[FAIL] {label}: unexpected error: {type(e).__name__}: {e}")
            failures.append(label)

    print()
    if failures:
        print(f"PREFLIGHT FAILED ({len(failures)}/{len(checks)} checks failed): "
              f"{', '.join(failures)}")
        sys.exit(1)
    print(f"PREFLIGHT OK ({len(checks)}/{len(checks)} checks passed) -- safe to run:\n"
          f"  heldout_assembly_eval.py {args.assembly_fasta} {args.sample_name}")


if __name__ == "__main__":
    main()
