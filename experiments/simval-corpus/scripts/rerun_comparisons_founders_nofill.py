#!/usr/bin/env python
"""Phase 2 of the panel-rebuild plan: re-score the 24 real founders against
the freshly-built, single-lineage, no-fill-gaps truth gVCFs from
build_founder_truth_gvcfs.sh (data/founder_truth_gvcfs_nofill/<founder>.g.vcf.gz),
instead of smm477's production gVCFs.

Writes to a SEPARATE report location (does not touch the existing
{founder}_250k/{founder}_comparison.txt smm477-gap-filled reports) so the
Phase 1 -> Phase 2 delta can be measured directly: same imputed predictions,
only the truth side's lineage/fill-policy differs.

Usage:
    python rerun_comparisons_founders_nofill.py [--parallel N]
"""
import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRATCH = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/heldout_assembly_eval")
TRUTH_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/founder_truth_gvcfs_nofill")
COMPARE_SCRIPT = Path("/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/"
                      "src/python/vcf_eval/compare_gvcf_truth.py")
CRF_SRC = Path("/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src")
PY = "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default/bin/python"
OUT_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/founder_nofill_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOUNDERS = ["B97", "CML103", "CML228", "CML247", "CML277", "CML322", "CML333",
            "CML52", "CML69", "HP301", "Il14H", "Ki11", "Ki3", "Ky21", "M162W",
            "M37W", "Mo18W", "Ms71", "NC350", "NC358", "Oh43", "Oh7B", "P39", "Tzi8"]


def run_one(founder):
    import os
    outdir = SCRATCH / f"{founder}_250k"
    imputed = outdir / f"{founder}_imputed.vcf.gz"
    if not imputed.exists():
        imputed = outdir / f"{founder}_imputed.vcf"
    truth = TRUTH_DIR / f"{founder}.g.vcf.gz"
    if not imputed.exists() or not truth.exists():
        return founder, False, f"missing input: imputed={imputed.exists()} truth={truth.exists()}"

    cmd = [PY, str(COMPARE_SCRIPT), f"--imputed-vcf={imputed}",
           f"--truth-gvcf={truth}", f"--sample={founder}",
           "--truth-ploidy-expand=2", "--partial-credit"]
    env = {**os.environ, "PYTHONPATH": str(CRF_SRC)}
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    report = OUT_DIR / f"{founder}_comparison.txt"
    report.write_text(proc.stdout + "\n" + proc.stderr)
    ok = proc.returncode == 0
    print(f"[{founder}] {'OK' if ok else 'FAILED'} in {dt:.1f}s -> {report}")
    if not ok:
        return founder, False, proc.stderr[-2000:]
    return founder, True, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()

    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_one, f): f for f in FOUNDERS}
        for fut in as_completed(futures):
            results.append(fut.result())

    ok = sum(1 for _, o, _ in results if o)
    print(f"\nDone: {ok}/{len(results)} succeeded")
    for name, o, err in results:
        if not o:
            print(f"  FAILED {name}: {err}")


if __name__ == "__main__":
    main()
