#!/usr/bin/env python
"""Re-run the comparison step (only) for the 5 held-out samples plus the two
Il14H methodology-comparison arms (no-fill-gaps and gap-filled-with-fixed-
comparator), against their EXISTING cached imputed VCFs -- companion to
`rerun_comparisons_fixed.py`, which does the same for the 25 in-panel
founders. Needed after the Phase 1 `compare_gvcf_truth.py` rewrite (see
`/home/zrm22/.claude/plans/ok-we-need-to-squishy-lovelace.md`): span-bug fix
+ unconditional panel-space truth projection.

Usage:
    python rerun_comparisons_heldout_il14h.py [--parallel N]
"""
import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRATCH = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/heldout_assembly_eval")
COMPARE_SCRIPT = Path("/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/"
                      "src/python/vcf_eval/compare_gvcf_truth.py")
CRF_SRC = Path("/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src")
PY = "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default/bin/python"
RESULTS = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results")

JOBS = {
    "Tx303": {
        "imputed": SCRATCH / "Tx303_250k" / "Tx303_imputed.vcf.gz",
        "truth": SCRATCH / "Tx303_250k" / "truth" / "gvcf" / "Zm-Tx303-REFERENCE-NAM-1.0.g.vcf.gz",
        "sample": "Tx303",
        "report": SCRATCH / "Tx303_250k" / "Tx303_comparison.txt",
    },
    "A188": {
        "imputed": SCRATCH / "A188_250k" / "A188_imputed.vcf.gz",
        "truth": SCRATCH / "A188_250k" / "truth" / "gvcf" / "Zm-A188-REFERENCE-KSU-1.0.g.vcf.gz",
        "sample": "A188",
        "report": SCRATCH / "A188_250k" / "A188_comparison.txt",
    },
    "EP1": {
        "imputed": SCRATCH / "EP1_250k" / "EP1_imputed.vcf.gz",
        "truth": SCRATCH / "EP1_250k" / "truth" / "gvcf" / "ep1.genome.g.vcf.gz",
        "sample": "EP1",
        "report": SCRATCH / "EP1_250k" / "EP1_comparison.txt",
    },
    "CML459": {
        "imputed": SCRATCH / "CML459_250k" / "CML459_imputed.vcf.gz",
        "truth": SCRATCH / "CML459_250k" / "truth" / "gvcf" / "CML459.chromosomes.v1.g.vcf.gz",
        "sample": "CML459",
        "report": SCRATCH / "CML459_250k" / "CML459_comparison.txt",
    },
    "Ia453": {
        "imputed": SCRATCH / "Ia453_250k" / "Ia453_imputed.vcf.gz",
        "truth": SCRATCH / "Ia453_250k" / "truth" / "gvcf" / "Zm-Ia453-REFERENCE-FL-1.0.g.vcf.gz",
        "sample": "Ia453",
        "report": SCRATCH / "Ia453_250k" / "Ia453_comparison.txt",
    },
    "Il14H_nofill": {
        "imputed": SCRATCH / "Il14H_250k" / "Il14H_imputed.vcf.gz",
        "truth": Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/"
                       "il14h_nofill_truth/Il14H.g.vcf.gz"),
        "sample": "Il14H",
        "report": RESULTS / "il14h_nofill_comparison.txt",
    },
    "Il14H_gapfilled_fixed": {
        "imputed": SCRATCH / "Il14H_250k" / "Il14H_imputed.vcf.gz",
        "truth": Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/"
                       "maize_panel_vcf/gvcf_input/Il14H.g.vcf"),
        "sample": "Il14H",
        "report": RESULTS / "il14h_gapfilled_comparison_fixed.txt",
    },
}


def run_one(name, job):
    import os
    cmd = [PY, str(COMPARE_SCRIPT), f"--imputed-vcf={job['imputed']}",
           f"--truth-gvcf={job['truth']}", f"--sample={job['sample']}",
           "--truth-ploidy-expand=2", "--partial-credit"]
    env = {**os.environ, "PYTHONPATH": str(CRF_SRC)}
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    job["report"].write_text(proc.stdout + "\n" + proc.stderr)
    ok = proc.returncode == 0
    print(f"[{name}] {'OK' if ok else 'FAILED'} in {dt:.1f}s -> {job['report']}")
    if not ok:
        print(proc.stderr[-2000:])
    return name, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=7)
    args = ap.parse_args()

    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_one, name, job): name for name, job in JOBS.items()}
        for fut in as_completed(futures):
            results.append(fut.result())

    ok = sum(1 for _, o in results if o)
    print(f"\nDone: {ok}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
