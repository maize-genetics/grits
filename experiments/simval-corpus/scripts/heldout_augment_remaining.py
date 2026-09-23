#!/usr/bin/env python
"""Finish the remaining Option B (heldout-augment) OUT rows after the prior
driver process died mid-sweep, leaving 3 RIL2 rows as live orphans. Handles
the other 12 rows (10 HYB/INBRED that failed pre-null-state-fix, 2 RIL2 that
stalled after windowing) on GPU 1 so as not to race the 3 still-live GPU 0
orphans (CML459xIa453, A188xEP1, EP1xIa453)."""
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
OUT_ROOT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/heldout_augment_out_snprc")
RESULTS_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/heldout_augment_out_snprc")

HELDOUT_CKPT = ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
                "diploid-indel-v3-heldout-augment-fullscale/d-epoch=02-val_pair_acc=0.7314.ckpt")
DEPTH = "0.1"
SKIP = {("OUT-RIL2", "CML459xIa453"), ("OUT-RIL2", "A188xEP1"), ("OUT-RIL2", "EP1xIa453")}
TARGET_CLASSES = ["OUT-INBRED", "OUT-HYB", "OUT-RIL2"]
KIND_OF = {"OUT-INBRED": "inbred", "OUT-HYB": "hybrid", "OUT-RIL2": "ril2"}
CLASS_OF = {"OUT-INBRED": "heldout", "OUT-HYB": "heldout", "OUT-RIL2": "heldout"}


def load_rows():
    rows = []
    with open(MANIFEST) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["dataset_id"] in TARGET_CLASSES and row["coverage"] == DEPTH:
                if (row["dataset_id"], row["individual"]) in SKIP:
                    continue
                rows.append(row)
    return rows


def run_one(row):
    ds, ind = row["dataset_id"], row["individual"]
    row_key = f"heldout_augment__{ds}__{ind}__{DEPTH}x"
    json_path = RESULTS_DIR / f"{row_key}.json"
    if json_path.exists():
        cached = json.loads(json_path.read_text())
        if cached.get("status") == "ok":
            return row_key, cached, "cached"

    outdir = OUT_ROOT / row_key
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(SCRIPTS / "simval_eval_indel.py"), "--stage", "all", "--sample", ind,
           "--r1", row["r1_path"], "--r2", row["r2_path"],
           "--truth-h1", row["truth_h1"], "--truth-h2", row["truth_h2"],
           "--outdir", str(outdir), "--out-json", str(outdir / "result.json"),
           "--dataset-id", ds, "--coverage", DEPTH, "--dataset-class", CLASS_OF[ds],
           "--kind", KIND_OF[ds], "--ckpt", HELDOUT_CKPT]
    log_path = outdir / "driver.log"
    t0 = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
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
    print(f"{len(rows)} rows to process (skipping {len(SKIP)} live orphans)", flush=True)
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(run_one, r): r for r in rows}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                row_key, result, status = fut.result()
            except Exception as e:
                row_key = f"heldout_augment__{r['dataset_id']}__{r['individual']}__{DEPTH}x"
                result, status = {"status": "exception", "error": str(e)}, "error"
            err = result.get("error_rate")
            print(f"[{status:>6}] {row_key:<45} status={result.get('status')} error_rate={err}", flush=True)
    print("REMAINING DONE", flush=True)


if __name__ == "__main__":
    main()
