#!/usr/bin/env python
"""
Rough genotype-level (SNP+RefCall) check of the region-buffer-fix checkpoint
(diploid-indel-v3-regionfix-fullscale) vs the v3-K25 baseline, scoped to
OUT-of-index data only at 0.1x -- the specific regime the region-buffer fix
targets (real held-out decode needs a much higher switch rate than in-panel
data, which restoring the simulator's intended crossover rate is meant to
provide). Fork a45bea03197fcd883 already found a real founder-level IDX
regression for this checkpoint and never tested OUT/MIX; this script fills
that gap at the genotype level.

15 manifest rows (OUT-INBRED/OUT-HYB/OUT-RIL2 x 5 each) x 2 checkpoints = 30
jobs, via simval_eval_indel.py --stage all. ThreadPoolExecutor + per-row JSON
cache (status=="ok" required to count as cached), matching
run_indel_snprc_sweep.py's established pattern.
"""
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPTS = Path(__file__).parent
PY = "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default/bin/python"
MANIFEST = Path("/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv")
OUT_ROOT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/regionfix_out_snprc")
RESULTS_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/regionfix_out_snprc")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

CKPTS = {
    "regionfix": ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
                  "diploid-indel-v3-regionfix-fullscale/d-epoch=04-val_pair_acc=0.4190.ckpt"),
    "baseline": ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
                 "diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt"),
}
DEPTH = "0.1"
TARGET_CLASSES = ["OUT-INBRED", "OUT-HYB", "OUT-RIL2"]
KIND_OF = {"OUT-INBRED": "inbred", "OUT-HYB": "hybrid", "OUT-RIL2": "ril2"}
# All three target classes are heldout, but keep this explicit/general rather
# than hardcoding "heldout" at the call site, in case this script is ever
# reused for a MIX follow-up.
CLASS_OF = {"OUT-INBRED": "heldout", "OUT-HYB": "heldout", "OUT-RIL2": "heldout"}


def load_rows():
    rows = []
    with open(MANIFEST) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["dataset_id"] in TARGET_CLASSES and row["coverage"] == DEPTH:
                rows.append(row)
    return rows


def run_one(model, row, gpu):
    ds, ind = row["dataset_id"], row["individual"]
    row_key = f"{model}__{ds}__{ind}__{DEPTH}x"
    json_path = RESULTS_DIR / f"{row_key}.json"
    if json_path.exists():
        cached = json.loads(json_path.read_text())
        if cached.get("status") == "ok":
            return row_key, cached, "cached"

    outdir = OUT_ROOT / f"{model}__{ds}__{ind}__{DEPTH}x"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(SCRIPTS / "simval_eval_indel.py"), "--stage", "all", "--sample", ind,
           "--r1", row["r1_path"], "--r2", row["r2_path"],
           "--truth-h1", row["truth_h1"], "--truth-h2", row["truth_h2"],
           "--outdir", str(outdir), "--out-json", str(outdir / "result.json"),
           "--dataset-id", ds, "--coverage", DEPTH, "--dataset-class", CLASS_OF[ds],
           "--kind", KIND_OF[ds], "--ckpt", CKPTS[model]]
    log_path = outdir / "driver.log"
    t0 = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        result = {"row_key": row_key, "status": "failed", "wall_seconds": round(dt, 1),
                  "log": str(log_path)}
    else:
        out = json.loads((outdir / "result.json").read_text())
        result = {"row_key": row_key, "status": "ok", "wall_seconds": round(dt, 1),
                  **out.get("score", {})}
    json_path.write_text(json.dumps(result, indent=1))
    return row_key, result, "ran"


def main():
    rows = load_rows()
    print(f"{len(rows)} manifest rows matched (expect {len(TARGET_CLASSES) * 5})", flush=True)

    jobs = [(model, row) for row in rows for model in CKPTS]
    print(f"{len(jobs)} total jobs", flush=True)

    # GPU 1 confirmed idle (0 MiB / 0%) at launch time; GPU 0 had 2349 MiB
    # resident but 0% util (stale, not actively in use) -- use GPU 1 to stay
    # clear of it.
    gpu = "1"

    results = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(run_one, m, r, gpu): (m, r) for m, r in jobs}
        for fut in as_completed(futs):
            m, r = futs[fut]
            try:
                row_key, result, status = fut.result()
            except Exception as e:
                row_key = f"{m}__{r['dataset_id']}__{r['individual']}__{DEPTH}x"
                result = {"row_key": row_key, "status": "exception", "error": str(e)}
                status = "error"
            results[row_key] = result
            err = result.get("error_rate")
            print(f"[{status:>6}] {row_key:<40} status={result.get('status')} "
                  f"error_rate={err}", flush=True)

    (RESULTS_DIR / "_sweep_summary.json").write_text(json.dumps(results, indent=1))
    n_ok = sum(1 for r in results.values() if r.get("status") == "ok")
    print(f"\nDONE: {n_ok}/{len(jobs)} rows succeeded", flush=True)

    print("\n=== SUMMARY: mean error_rate by dataset_id x model ===", flush=True)
    for ds in TARGET_CLASSES:
        for model in CKPTS:
            errs = [r["error_rate"] for k, r in results.items()
                    if r.get("status") == "ok" and k.startswith(f"{model}__{ds}__")
                    and r.get("error_rate") is not None]
            if errs:
                mean_err = sum(errs) / len(errs)
                print(f"{ds:<12} {model:<10} n={len(errs)}  mean_error_rate={mean_err:.4f}",
                      flush=True)
            else:
                print(f"{ds:<12} {model:<10} n=0  (no successful rows)", flush=True)

    for model in CKPTS:
        errs = [r["error_rate"] for k, r in results.items()
                if r.get("status") == "ok" and k.startswith(f"{model}__")
                and r.get("error_rate") is not None]
        if errs:
            print(f"OVERALL      {model:<10} n={len(errs)}  mean_error_rate={sum(errs)/len(errs):.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
