#!/usr/bin/env python
"""
Diagnostic 3 (plan section: stratified error by variant class). Re-scores 8
representative rows (one individual per dataset type, at 0.5x) using the
CACHED imputed VCFs the real 200-row batch already produced -- no
alignment, no bed-to-vcf, just compare_gvcf_truth_diploid.py's comparator
with --class-breakdown (Diagnostic 3's new per-class counters) against the
same cached, bgzipped+indexed *_imputed.autosomes.vcf.gz each row's real
score stage left behind. Whole genome (no --region), matching the
published matrix's own scope exactly -- these cached VCFs already paid the
bed-to-vcf cost once; only the comparator pass is repeated here.

Usage: simval_class_breakdown.py [--parallel 6]
"""
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import simval_paths as P  # noqa: E402

PY = "/home/zrm22/mambaforge/envs/phg-ml/bin/python"
COMPARATOR = Path(__file__).parent / "compare_gvcf_truth_diploid.py"

# One individual per dataset type, all at 0.5x -- chosen to share B73/Tx303
# where possible so results are easy to eyeball against each other.
ROWS = [
    ("IDX-INBRED", "B73"),
    ("IDX-HYB", "B73xOh43"),
    ("IDX-RIL", "B73xOh43"),
    ("OUT-INBRED", "Tx303"),
    ("OUT-HYB", "Tx303xA188"),
    ("OUT-RIL", "Tx303xA188"),
    ("MIX-HYB", "B73xTx303"),
    ("MIX-RIL", "B73xTx303"),
]
COVERAGE = "0.5"


def load_manifest_row(dataset_id, individual, coverage):
    import csv
    with open(P.MANIFEST) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["dataset_id"] == dataset_id and r["individual"] == individual and r["coverage"] == coverage:
                return r
    raise KeyError(f"no manifest row for {dataset_id}/{individual}/{coverage}x")


def run_one(dataset_id, individual):
    row_key = f"{dataset_id}__{individual}__{COVERAGE}x"
    outdir = P.SCRATCH_ROOT / row_key
    imputed_gz = next(outdir.glob("*_imputed.autosomes.vcf.gz"), None)
    if imputed_gz is None:
        return {"dataset_id": dataset_id, "individual": individual, "status": "no_cached_vcf"}

    mrow = load_manifest_row(dataset_id, individual, COVERAGE)
    log_path = P.RESULTS_DIR / "simval_logs" / f"classbreak_{row_key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [PY, str(COMPARATOR),
           "--imputed-vcf", str(imputed_gz), "--sample", individual,
           "--truth-gvcf-h1", mrow["truth_h1"], "--truth-gvcf-h2", mrow["truth_h2"],
           "--partial-credit", "--class-breakdown"]
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if proc.returncode != 0:
        return {"dataset_id": dataset_id, "individual": individual, "status": "failed",
                "log": str(log_path)}

    text = log_path.read_text()
    result = {"dataset_id": dataset_id, "individual": individual, "status": "ok",
              "wall_seconds": round(dt, 2)}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("SNP", "INS", "DEL", "HOMREF", "HET_MIXED")):
            parts = line.split()
            cls, tot, mis, err_pct = parts[0], int(parts[1]), int(parts[2]), float(parts[3].rstrip("%"))
            result[f"{cls}_total"] = tot
            result[f"{cls}_mismatch"] = mis
            result[f"{cls}_error_pct"] = err_pct
        elif line.startswith("allele_GT_concordance"):
            result["overall_error_pct"] = round(100 * (1 - float(line.split()[-1])), 4)
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=6)
    args = ap.parse_args()

    print(f"Re-scoring {len(ROWS)} cached rows with --class-breakdown "
          f"(parallel={args.parallel}) ...", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {pool.submit(run_one, d, i): (d, i) for d, i in ROWS}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"  -> {r}", flush=True)

    out_path = P.RESULTS_DIR / "simval_class_breakdown.json"
    out_path.write_text(json.dumps(results, indent=1))
    print(f"\nWrote {out_path}")

    print(f"\n{'dataset':12s} {'indiv':14s} {'overall%':>9s} "
          f"{'SNP%':>7s} {'INS%':>7s} {'DEL%':>7s} {'HOMREF%':>8s}")
    for r in sorted(results, key=lambda x: x.get("dataset_id", "")):
        if r.get("status") != "ok":
            print(f"{r.get('dataset_id',''):12s} {r.get('individual',''):14s}  {r.get('status')}")
            continue
        print(f"{r['dataset_id']:12s} {r['individual']:14s} {r.get('overall_error_pct', float('nan')):>8.3f}% "
              f"{r.get('SNP_error_pct', float('nan')):>6.3f}% {r.get('INS_error_pct', float('nan')):>6.3f}% "
              f"{r.get('DEL_error_pct', float('nan')):>6.3f}% {r.get('HOMREF_error_pct', float('nan')):>7.3f}%")


if __name__ == "__main__":
    main()
