#!/usr/bin/env python
"""
Genotype-level (SNP+RefCall) 0.1x check for any checkpoint (OUT or IDX) --
heldout_augment_out_snprc_check.py generalized to take --tag/--ckpt/--gpu,
so the full-recipe retrains (diploid-indel-v3-{heldout-augment,lowrate}-
fullrecipe) are scored with exactly the same rows, depth and pipeline as
every earlier OUT result, against the same cached baseline rows.

Resumable: rows whose results/<tag>_out_snprc/<row>.json is status=ok are
skipped.
"""
import argparse
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
WORK = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
BASELINE_CACHE_DIR = WORK / "results/regionfix_out_snprc"

DEPTH = "0.1"
SCOPES = {"out": ["OUT-INBRED", "OUT-HYB", "OUT-RIL2"],
          "idx": ["IDX-INBRED", "IDX-HYB", "IDX-RIL2"]}
KIND_OF = {"INBRED": "inbred", "HYB": "hybrid", "RIL2": "ril2"}
CLASS_OF = {"OUT": "heldout", "IDX": "indexed"}


def load_rows(classes):
    with open(MANIFEST) as f:
        return [r for r in csv.DictReader(f, delimiter="\t")
                if r["dataset_id"] in classes and r["coverage"] == DEPTH]


def load_cached_baseline(rows):
    results = {}
    for row in rows:
        key = f"baseline__{row['dataset_id']}__{row['individual']}__{DEPTH}x"
        f = BASELINE_CACHE_DIR / f"{key}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        assert d.get("status") == "ok", f"cached baseline row not ok: {key}"
        results[key] = d
    return results


def run_one(row, tag, ckpt, gpu, out_root, results_dir):
    ds, ind = row["dataset_id"], row["individual"]
    row_key = f"{tag}__{ds}__{ind}__{DEPTH}x"
    json_path = results_dir / f"{row_key}.json"
    if json_path.exists():
        cached = json.loads(json_path.read_text())
        if cached.get("status") == "ok":
            return row_key, cached, "cached"

    outdir = out_root / row_key
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(SCRIPTS / "simval_eval_indel.py"), "--stage", "all", "--sample", ind,
           "--r1", row["r1_path"], "--r2", row["r2_path"],
           "--truth-h1", row["truth_h1"], "--truth-h2", row["truth_h2"],
           "--outdir", str(outdir), "--out-json", str(outdir / "result.json"),
           "--dataset-id", ds, "--coverage", DEPTH,
           "--dataset-class", CLASS_OF[ds.split("-")[0]],
           "--kind", KIND_OF[ds.split("-")[1]], "--ckpt", ckpt]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    t0 = time.time()
    with open(outdir / "driver.log", "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    dt = round(time.time() - t0, 1)
    if proc.returncode != 0:
        result = {"row_key": row_key, "status": "failed", "wall_seconds": dt,
                  "log": str(outdir / "driver.log")}
    else:
        out = json.loads((outdir / "result.json").read_text())
        result = {"row_key": row_key, "status": "ok", "wall_seconds": dt, **out.get("score", {})}
    json_path.write_text(json.dumps(result, indent=1))
    return row_key, result, "ran"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--scope", choices=sorted(SCOPES), default="out")
    args = p.parse_args()
    classes = SCOPES[args.scope]

    out_root = WORK / "scratch" / f"{args.tag}_{args.scope}_snprc"
    results_dir = WORK / "results" / f"{args.tag}_{args.scope}_snprc"
    out_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(classes)
    print(f"{len(rows)} manifest rows matched (expect {len(classes) * 5}); ckpt={args.ckpt}",
          flush=True)
    results = load_cached_baseline(rows)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, r, args.tag, args.ckpt, args.gpu, out_root, results_dir): r
                for r in rows}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                row_key, result, status = fut.result()
            except Exception as e:
                row_key = f"{args.tag}__{r['dataset_id']}__{r['individual']}__{DEPTH}x"
                result, status = {"row_key": row_key, "status": "exception", "error": str(e)}, "error"
            results[row_key] = result
            print(f"[{status:>6}] {row_key:<55} status={result.get('status')} "
                  f"error_rate={result.get('error_rate')} snprc={result.get('snprc_error_rate')}",
                  flush=True)

    (results_dir / "_sweep_summary.json").write_text(json.dumps(results, indent=1))
    print(f"\nDONE: {sum(r.get('status') == 'ok' for r in results.values())}/{len(results)} rows ok")
    for ds in classes:
        for model in ("baseline", args.tag):
            errs = [r["error_rate"] for k, r in results.items()
                    if k.startswith(f"{model}__{ds}__") and r.get("status") == "ok"
                    and r.get("error_rate") is not None]
            mean = f"{sum(errs) / len(errs):.4f}" if errs else "n/a"
            print(f"{ds:<12} {model:<28} n={len(errs)}  mean_error_rate={mean}", flush=True)


if __name__ == "__main__":
    main()
