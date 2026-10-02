#!/usr/bin/env python
"""Re-run scripts/diag_error_mechanism_breakdown.py for all 30 samples (25
in-panel founders + 5 held-out) with the Phase 1 fixed comparator -- see
/home/zrm22/.claude/plans/ok-we-need-to-squishy-lovelace.md.

Usage:
    python rerun_mechanism_breakdown_all.py [--parallel N]
"""
import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRATCH = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/heldout_assembly_eval")
DIAG_SCRIPT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scripts/diag_error_mechanism_breakdown.py")
CRF_SRC = Path("/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src")
PY = "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default/bin/python"
RESULTS = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/diag_error_mechanism_breakdown")
RESULTS.mkdir(parents=True, exist_ok=True)

EXISTING_GVCF_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_panel_vcf/gvcf_input")

FOUNDERS = ["B73", "B97", "CML103", "CML228", "CML247", "CML277", "CML322", "CML333",
            "CML52", "CML69", "HP301", "Il14H", "Ki11", "Ki3", "Ky21", "M162W", "M37W",
            "Mo18W", "Ms71", "NC350", "NC358", "Oh43", "Oh7B", "P39", "Tzi8"]

JOBS = {}
for f in FOUNDERS:
    outdir = SCRATCH / f"{f}_250k"
    truth = EXISTING_GVCF_DIR / f"{f}.g.vcf"
    if f == "B73":
        truth = outdir / "truth" / "gvcf" / "B73.g.vcf.gz"
    JOBS[f] = {
        "imputed": outdir / f"{f}_imputed.vcf.gz" if (outdir / f"{f}_imputed.vcf.gz").exists()
                   else outdir / f"{f}_imputed.vcf",
        "truth": truth,
        "sample": f,
        "report": RESULTS / f"{f}.log",
    }

HELDOUT = {
    "Tx303": "Zm-Tx303-REFERENCE-NAM-1.0.g.vcf.gz",
    "A188": "Zm-A188-REFERENCE-KSU-1.0.g.vcf.gz",
    "EP1": "ep1.genome.g.vcf.gz",
    "CML459": "CML459.chromosomes.v1.g.vcf.gz",
    "Ia453": "Zm-Ia453-REFERENCE-FL-1.0.g.vcf.gz",
}
for name, truth_name in HELDOUT.items():
    outdir = SCRATCH / f"{name}_250k"
    JOBS[name] = {
        "imputed": outdir / f"{name}_imputed.vcf.gz",
        "truth": outdir / "truth" / "gvcf" / truth_name,
        "sample": name,
        "report": RESULTS / f"{name}.log",
    }

JOBS["Il14H_nofill"] = {
    "imputed": SCRATCH / "Il14H_250k" / "Il14H_imputed.vcf.gz",
    "truth": Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/"
                   "il14h_nofill_truth/Il14H.g.vcf.gz"),
    "sample": "Il14H",
    "report": RESULTS / "Il14H_nofill.log",
}


def run_one(name, job):
    import os
    cmd = [PY, str(DIAG_SCRIPT), f"--imputed-vcf={job['imputed']}",
           f"--truth-gvcf={job['truth']}", f"--sample={job['sample']}",
           "--truth-ploidy-expand=2"]
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
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--only", default=None,
                     help="comma-separated subset of JOBS keys to run (default: all 31)")
    args = ap.parse_args()

    jobs = JOBS
    if args.only:
        wanted = set(args.only.split(","))
        jobs = {k: v for k, v in JOBS.items() if k in wanted}
        missing = wanted - set(jobs)
        if missing:
            raise SystemExit(f"--only names not found in JOBS: {missing}")

    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_one, name, job): name for name, job in jobs.items()}
        for fut in as_completed(futures):
            results.append(fut.result())

    ok = sum(1 for _, o in results if o)
    print(f"\nDone: {ok}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
