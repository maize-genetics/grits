#!/usr/bin/env python
"""
In-panel smoke test: run heldout_assembly_eval.py's full pipeline for every
NAM founder in our own 25-founder index, even though (unlike a genuine
held-out test) every one of them IS in the index. This is NOT a
generalization test -- it's a best-case ceiling / end-to-end wiring sanity
check, requested explicitly alongside the (separate, still-pending) genuine
out-of-index test.

Now a thin caller over scripts/heldout_batch.py (the generalized orchestrator
this pipeline also uses for genuine held-out assemblies, leave-one-out
founders, and diverged material): builds this run's work list (25 founders ->
their own keyfile FASTA, paired with each founder's existing truth gVCF via
--truth-gvcf where available -- only B73 has none, so it self-builds via
AnchorWave, no special-casing needed), writes it to
scratch/heldout_assembly_eval/nam_inpanel_worklist.tsv, and delegates to
heldout_batch.main() with this run's own label so its summary JSON
(results/nam_inpanel_smoketest_summary.json) and log directory
(results/nam_inpanel_smoketest_logs/) -- what build_inpanel_report.py already
parses -- are unchanged.

Usage:
    python nam_inpanel_smoketest.py [--depth N] [--founders A,B,C] [--parallel N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../refmap-founder-eval/scripts"))  # nam_baseline.py lives in the refmap-founder-eval experiment
import nam_baseline as nb    # noqa: E402  (reuse REFMAP_ROOT)
import heldout_batch as hb   # noqa: E402  (reuse the generalized orchestrator)

EXISTING_GVCF_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/"
                          "maize_panel_vcf/gvcf_input")
WORK_LIST = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/"
                  "heldout_assembly_eval/nam_inpanel_worklist.tsv")
LABEL = "nam_inpanel_smoketest"
DEFAULT_DEPTH = 250_000  # dry-run on Oh43 at this depth: ~19 min/founder end-to-end,
                          # real sensible result (concordance 0.881, ceiling 0.813) --
                          # 500k would push a 25-founder batch well past a day


def load_founders():
    """Founder name -> assembly FASTA path, from the SAME keyfile our own
    ropebwt3 index was built from (25 founders including B73) -- not
    smm477's 24-founder one."""
    founders = {}
    keyfile = nb.REFMAP_ROOT / "keyfile_fastas.txt"
    for line in keyfile.read_text().splitlines():
        if not line.strip():
            continue
        fasta_rel, name = line.split("\t")
        founders[name] = nb.REFMAP_ROOT / fasta_rel
    return founders


def write_work_list(founders):
    WORK_LIST.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for name in sorted(founders):
        fasta = founders[name]
        row = [name, str(fasta)]
        truth_gvcf = EXISTING_GVCF_DIR / f"{name}.g.vcf"
        if truth_gvcf.exists():
            row.append(str(truth_gvcf))
        lines.append("\t".join(row))
    WORK_LIST.write_text("\n".join(lines) + "\n")
    return WORK_LIST


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--founders", default=None,
                     help="Comma-separated subset (default: all 25 in the keyfile)")
    ap.add_argument("--parallel", type=int, default=3,
                     help="Concurrent founders (default 3). Each run is fully "
                          "isolated (own subprocess, own outdir keyed by sample "
                          "name), so this is safe. refmap uses 20 threads per run; "
                          "3 concurrent stays near the 62-core budget.")
    args = ap.parse_args()

    founders = load_founders()
    missing = []
    if args.founders:
        wanted = set(args.founders.split(","))
        missing = sorted(wanted - set(founders))
    if missing:
        raise SystemExit(f"unknown founder(s): {missing}")

    write_work_list(founders)
    print(f"Wrote work list for {len(founders)} founders -> {WORK_LIST}")

    batch_argv = [str(WORK_LIST), f"--depth={args.depth}", f"--parallel={args.parallel}",
                  f"--label={LABEL}"]
    if args.founders:
        batch_argv.append(f"--samples={args.founders}")
    hb.main(batch_argv)


if __name__ == "__main__":
    main()
