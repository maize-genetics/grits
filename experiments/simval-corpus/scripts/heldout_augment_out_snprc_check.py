#!/usr/bin/env python
"""
Genotype-level (SNP+RefCall) check of the Option B checkpoint
(diploid-indel-v3-heldout-augment-fullscale, synthetic leave-one-out
held-out training) vs the v3-K25 baseline, scoped to OUT-of-index data at
0.1x -- the same scope/method as regionfix_out_snprc_check.py, reused
verbatim apart from the checkpoint under test.

NOTE on why this is genotype-level, not founder-level pair_acc: pair_acc
(eval_common.py's score_ibd_adjusted_accuracy) requires true_lo/true_hi --
a literal known PANEL FOUNDER ID ground truth. That's well-defined for IDX
samples (they ARE panel founders) but fundamentally undefined for OUT
samples (genuinely held-out assemblies, not equal to any panel member).
This is why every OUT/MIX accuracy check this session (region-fix's OUT
sweep, Branch B's genotype sweep) has used genotype-level SNP+RefCall
error_rate instead -- it's the only accuracy metric that's actually
meaningful for held-out individuals, and it's what this script computes
too, for direct comparability with those two prior results.

Baseline's 15 OUT-0.1x rows are reused from the already-cached
regionfix_out_snprc/baseline__*.json results (same checkpoint, same rows,
same depth -- no need to recompute) rather than re-run.
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
OUT_ROOT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/heldout_augment_out_snprc")
RESULTS_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/heldout_augment_out_snprc")
BASELINE_CACHE_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/regionfix_out_snprc")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

HELDOUT_CKPT = ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
                "diploid-indel-v3-heldout-augment-fullscale/d-epoch=02-val_pair_acc=0.7314.ckpt")

DEPTH = "0.1"
TARGET_CLASSES = ["OUT-INBRED", "OUT-HYB", "OUT-RIL2"]
KIND_OF = {"OUT-INBRED": "inbred", "OUT-HYB": "hybrid", "OUT-RIL2": "ril2"}
CLASS_OF = {"OUT-INBRED": "heldout", "OUT-HYB": "heldout", "OUT-RIL2": "heldout"}


def load_rows():
    rows = []
    with open(MANIFEST) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["dataset_id"] in TARGET_CLASSES and row["coverage"] == DEPTH:
                rows.append(row)
    return rows


def load_cached_baseline():
    results = {}
    for ds in TARGET_CLASSES:
        for row in load_rows():
            if row["dataset_id"] != ds:
                continue
            ind = row["individual"]
            f = BASELINE_CACHE_DIR / f"baseline__{ds}__{ind}__{DEPTH}x.json"
            d = json.loads(f.read_text())
            assert d.get("status") == "ok", f"cached baseline row not ok: {f}"
            results[f"baseline__{ds}__{ind}__{DEPTH}x"] = d
    return results


def run_one(row, gpu):
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

    baseline_results = load_cached_baseline()
    print(f"loaded {len(baseline_results)} cached baseline rows", flush=True)

    gpu = "0"  # both GPUs idle at launch time

    results = dict(baseline_results)
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(run_one, r, gpu): r for r in rows}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                row_key, result, status = fut.result()
            except Exception as e:
                row_key = f"heldout_augment__{r['dataset_id']}__{r['individual']}__{DEPTH}x"
                result = {"row_key": row_key, "status": "exception", "error": str(e)}
                status = "error"
            results[row_key] = result
            err = result.get("error_rate")
            print(f"[{status:>6}] {row_key:<45} status={result.get('status')} "
                  f"error_rate={err}", flush=True)

    (RESULTS_DIR / "_sweep_summary.json").write_text(json.dumps(results, indent=1))
    n_ok = sum(1 for r in results.values() if r.get("status") == "ok")
    print(f"\nDONE: {n_ok}/{len(results)} rows ok", flush=True)

    print("\n=== SUMMARY 2: mean error_rate by dataset_id x model ===", flush=True)
    for ds in TARGET_CLASSES:
        for model in ("baseline", "heldout_augment"):
            errs = [r["error_rate"] for k, r in results.items()
                    if r.get("status") == "ok" and k.startswith(f"{model}__{ds}__")
                    and r.get("error_rate") is not None]
            if errs:
                mean_err = sum(errs) / len(errs)
                print(f"{ds:<12} {model:<16} n={len(errs)}  mean_error_rate={mean_err:.4f}",
                      flush=True)
            else:
                print(f"{ds:<12} {model:<16} n=0  (no successful rows)", flush=True)


if __name__ == "__main__":
    main()
