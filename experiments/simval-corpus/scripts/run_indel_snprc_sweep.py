#!/usr/bin/env python
"""
Driver: 40-row 0.1x SNP+RefCall sweep for Branch B (diploid-indel-v3-collapse-fullscale)
across IDX/OUT/MIX x INBRED/HYB/RIL2, via simval_eval_indel.py. Plus the 15
baseline (old-model) rows not already cached in results/simval_results.tsv
(the *-RIL2 classes, new this session) via simval_eval_one.py --kind ril2.
ThreadPoolExecutor + per-row JSON cache, matching simval_snp_refcall_rescore.py's
established pattern.
"""
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPTS = Path(__file__).parent
PY = "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default/bin/python"
MANIFEST = Path("/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv")
OUT_ROOT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/simval_eval_indel")
RESULTS_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/simval_indel_sweep")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INDEL_CKPT = ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
              "diploid-indel-v3-collapse-fullscale/d-epoch=04-val_pair_acc=0.6542.ckpt")
DEPTH = "0.1"

TARGET_CLASSES = ["IDX-INBRED", "IDX-HYB", "IDX-RIL2", "OUT-INBRED", "OUT-HYB", "OUT-RIL2",
                  "MIX-HYB", "MIX-RIL2"]
KIND_OF = {"IDX-INBRED": "inbred", "IDX-HYB": "hybrid", "IDX-RIL2": "ril2",
           "OUT-INBRED": "inbred", "OUT-HYB": "hybrid", "OUT-RIL2": "ril2",
           "MIX-HYB": "hybrid", "MIX-RIL2": "ril2"}
CLASS_OF = {"IDX-INBRED": "indexed", "IDX-HYB": "indexed", "IDX-RIL2": "indexed",
            "OUT-INBRED": "heldout", "OUT-HYB": "heldout", "OUT-RIL2": "heldout",
            "MIX-HYB": "mixed", "MIX-RIL2": "mixed"}
BASELINE_GAP = {"IDX-RIL2", "OUT-RIL2", "MIX-RIL2"}  # not already in simval_results.tsv


def load_rows():
    rows = []
    with open(MANIFEST) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["dataset_id"] in TARGET_CLASSES and row["coverage"] == DEPTH:
                rows.append(row)
    return rows


def run_one(model, row):
    ds, ind = row["dataset_id"], row["individual"]
    row_key = f"{model}__{ds}__{ind}__{DEPTH}x"
    json_path = RESULTS_DIR / f"{row_key}.json"
    if json_path.exists():
        cached = json.loads(json_path.read_text())
        # Only a genuinely successful prior run is safe to skip -- a
        # "failed" row's JSON must NOT count as cached-and-done, or a retry
        # after fixing the root cause would just re-report the same old
        # failure forever without ever actually re-running the row. (Found
        # this bug via the GPU-memory-leak incident: the first sweep left
        # ~50 "failed" result JSONs on disk, and this exact check would have
        # silently skipped every one of them on the retry.)
        if cached.get("status") == "ok":
            return row_key, cached, "cached"

    outdir = OUT_ROOT / f"{model}__{ds}__{ind}__{DEPTH}x"
    outdir.mkdir(parents=True, exist_ok=True)
    script = "simval_eval_indel.py" if model == "collapse" else "simval_eval_one.py"
    cmd = [PY, str(SCRIPTS / script), "--stage", "all", "--sample", ind,
           "--r1", row["r1_path"], "--r2", row["r2_path"],
           "--truth-h1", row["truth_h1"], "--truth-h2", row["truth_h2"],
           "--outdir", str(outdir), "--out-json", str(outdir / "result.json"),
           "--dataset-id", ds, "--coverage", DEPTH, "--dataset-class", CLASS_OF[ds],
           "--kind", KIND_OF[ds]]
    if model == "collapse":
        cmd += ["--ckpt", INDEL_CKPT]
    log_path = outdir / "driver.log"
    t0 = time.time()
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"  # checked nvidia-smi at relaunch time: GPU 0
                                        # idle (0%), GPU 1 busy (100%, 49985MiB) --
                                        # opposite of the coordinator's earlier
                                        # observation, so use CURRENT state, not that
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
    print(f"{len(rows)} manifest rows matched (expect {len(TARGET_CLASSES) * 5})")

    jobs = []
    for row in rows:
        jobs.append(("collapse", row))
        if row["dataset_id"] in BASELINE_GAP:
            jobs.append(("baseline", row))
    print(f"{len(jobs)} total jobs ({sum(1 for m, _ in jobs if m == 'collapse')} collapse, "
          f"{sum(1 for m, _ in jobs if m == 'baseline')} baseline)")

    results = {}
    # Reduced from 10 -> 5 after the first run's GPU-memory-leak incident
    # (fixed now, but staying conservative on the retry until confirmed).
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(run_one, m, r): (m, r) for m, r in jobs}
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
            print(f"[{status:>6}] {row_key:<45} status={result.get('status')} "
                  f"error_rate={err}", flush=True)

    (RESULTS_DIR / "_sweep_summary.json").write_text(json.dumps(results, indent=1))
    n_ok = sum(1 for r in results.values() if r.get("status") == "ok")
    print(f"\nDONE: {n_ok}/{len(jobs)} rows succeeded")


if __name__ == "__main__":
    main()
