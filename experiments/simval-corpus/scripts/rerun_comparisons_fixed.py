#!/usr/bin/env python
"""Re-run ONLY the truth-comparison step for all 25 NAM in-panel founders,
against their EXISTING cached imputed VCFs from the original
nam_inpanel_smoketest.py batch. Both fixes (chr9/chr10 contig-ordering,
deletion-interior <DEL> propagation) live entirely in
vcf_eval/compare_gvcf_truth.py -- reads/refmap/windowing/inference/BED/
imputed-VCF generation are all unaffected, so there is no need to redo any
of that (which was the ~19-min/founder cost); only the comparison itself
(~2 min/founder) needs to be redone.

Overwrites each founder's {founder}_250k/{founder}_comparison.txt in place
(same path build_inpanel_report.py already expects), then
build_inpanel_report.py + render_inpanel_report.py can be rerun unchanged
to regenerate the artifact data/HTML.

Usage:
    python rerun_comparisons_fixed.py [--parallel N]
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../refmap-founder-eval/scripts"))  # nam_baseline.py lives in the refmap-founder-eval experiment
import nam_baseline as nb  # noqa: E402  (reuse REFMAP_ROOT for the founder list)

SCRATCH = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/heldout_assembly_eval")
EXISTING_GVCF_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/"
                          "maize_panel_vcf/gvcf_input")
COMPARE_SCRIPT = Path("/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/"
                       "src/python/vcf_eval/compare_gvcf_truth.py")
CRF_SRC = Path("/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src")
PY = "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default/bin/python"
DEPTH = 250_000


def load_founders():
    founders = {}
    keyfile = nb.REFMAP_ROOT / "keyfile_fastas.txt"
    for line in keyfile.read_text().splitlines():
        if not line.strip():
            continue
        _fasta_rel, name = line.split("\t")
        founders[name] = True
    return list(founders.keys())


def truth_gvcf_for(founder):
    real = EXISTING_GVCF_DIR / f"{founder}.g.vcf"
    if real.exists():
        return real
    self_built = SCRATCH / f"{founder}_{DEPTH // 1000}k" / "truth" / "gvcf" / f"{founder}.g.vcf.gz"
    if self_built.exists():
        return self_built
    raise FileNotFoundError(f"no truth gVCF found for {founder} (checked {real} and {self_built})")


def run_one(founder):
    outdir = SCRATCH / f"{founder}_{DEPTH // 1000}k"
    imputed_vcf = outdir / f"{founder}_imputed.vcf"
    if not imputed_vcf.exists():
        return {"founder": founder, "ok": False, "error": f"missing imputed VCF: {imputed_vcf}"}
    try:
        truth_gvcf = truth_gvcf_for(founder)
    except FileNotFoundError as e:
        return {"founder": founder, "ok": False, "error": str(e)}

    cmd = [PY, str(COMPARE_SCRIPT), f"--imputed-vcf={imputed_vcf}",
           f"--truth-gvcf={truth_gvcf}", f"--sample={founder}",
           "--truth-ploidy-expand=2", "--partial-credit"]
    import os
    env = {**os.environ, "PYTHONPATH": str(CRF_SRC)}
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    report = outdir / f"{founder}_comparison.txt"
    report.write_text(proc.stdout + "\n" + proc.stderr)
    ok = proc.returncode == 0
    print(f"[{founder}] {'OK' if ok else 'FAILED'} in {dt:.1f}s")
    if not ok:
        return {"founder": founder, "ok": False, "error": proc.stderr[-2000:]}
    return {"founder": founder, "ok": True, "seconds": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()

    founders = load_founders()
    print(f"Re-running comparisons for {len(founders)} founders, parallel={args.parallel}")

    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_one, f): f for f in founders}
        for fut in as_completed(futures):
            results.append(fut.result())

    ok = sum(r["ok"] for r in results)
    print(f"\nDone: {ok}/{len(results)} succeeded")
    failed = [(r["founder"], r.get("error")) for r in results if not r["ok"]]
    if failed:
        print("FAILED:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
