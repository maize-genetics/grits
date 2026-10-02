#!/usr/bin/env python
"""
Corpus-wide driver for the full-corpus affinity-model evaluation: runs every
row of the maize simulated-validation corpus's manifest.tsv through
simval_eval_one.py, in two independently-tunable parallelism stages, and
appends results to a single TSV. Modeled directly on scripts/heldout_batch.py
(resumable via per-row result.json existence, per-row log files, subprocess
isolation).

Two-stage parallelism, NOT one pool per row: refmap ("align") uses 20
threads/run against a ~64-core budget (default --parallel-align=3, same
reasoning as heldout_batch.py's own default); bed-to-vcf + the genotype
comparator ("score") is dominated by a single-threaded pass over a
panel-VCF-sized record stream, effectively ~1 core + streaming memory each
(default --parallel-score=10). Splitting each row's work into two separate
subprocess calls (not one opaque per-row call) is what lets these run at
different concurrency levels against the SAME 64-core machine at once.

Usage:
    simval_batch.py [--dataset ID[,ID...]] [--coverage C[,C...]] [--kind K]
        [--samples A,B,C] [--limit N] [--dry-run] [--force]
        [--drop-idx N] [--region R]
        [--parallel-align 3] [--parallel-score 10] [--threads-per-align 20]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import simval_paths as P  # noqa: E402

PY = "/home/zrm22/mambaforge/envs/phg-ml/bin/python"
EVAL_ONE = Path(__file__).parent / "simval_eval_one.py"

RESULTS_HEADER = [
    "dataset_id", "class", "kind", "individual", "coverage", "realized_coverage",
    "error_rate", "partial_error_rate", "compared_sites", "excluded_no_info",
    "excluded_partial_info", "imputed_records", "gt_allele_matches", "gt_allele_mismatches",
    "matchable_ceiling", "align_dropped_idx", "align_het_scale", "align_het_scale_diagnostic",
    "align_prep_fastq_s", "align_refmap_s", "align_window_s", "align_inference_s",
    "align_write_bed_s", "align_align_total_s", "bed_to_vcf_s", "compare_s",
    "score_total_s", "wall_seconds", "row_key",
]


def load_manifest(path):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def filter_rows(rows, args):
    out = rows
    if args.dataset:
        wanted = set(args.dataset.split(","))
        out = [r for r in out if r["dataset_id"] in wanted]
    if args.coverage:
        wanted = set(args.coverage.split(","))
        out = [r for r in out if r["coverage"] in wanted]
    if args.kind:
        wanted = set(args.kind.split(","))
        out = [r for r in out if r["kind"] in wanted]
    if args.samples:
        wanted = set(args.samples.split(","))
        out = [r for r in out if r["individual"] in wanted]
    if args.limit:
        out = out[: args.limit]
    return out


def run_row(row, args, sem_align, sem_score, results_lock):
    dataset_id, klass, kind = row["dataset_id"], row["class"], row["kind"]
    individual, coverage = row["individual"], row["coverage"]
    row_key = f"{dataset_id}__{individual}__{coverage}x"
    outdir = P.SCRATCH_ROOT / row_key
    outdir.mkdir(parents=True, exist_ok=True)
    P.LOG_DIR.mkdir(parents=True, exist_ok=True)
    combined_json = outdir / "result.json"
    align_json = outdir / "align_timings.json"
    score_json = outdir / "score_result.json"
    log_path = P.LOG_DIR / f"{row_key}.log"

    if combined_json.exists() and not args.force:
        cached = json.loads(combined_json.read_text())
        return {"row_key": row_key, "status": "cached", **cached}

    if args.dry_run:
        return {"row_key": row_key, "status": "dry_run"}

    env = {**os.environ, "TMPDIR": str(P.TMPDIR)}
    cmd_common = [
        PY, str(EVAL_ONE), "--sample", individual,
        "--outdir", str(outdir), "--out-json", str(outdir / "stage_out.json"),
        "--r1", row["r1_path"], "--r2", row["r2_path"],
        "--truth-h1", row["truth_h1"], "--truth-h2", row["truth_h2"],
        "--threads", str(args.threads_per_align),
        "--panel-vcf", str(P.PANEL_VCF_V2), "--ckpt", str(P.CKPT_DIPLOID),
        "--dataset-id", dataset_id, "--coverage", coverage,
        "--dataset-class", klass, "--kind", kind,
    ]
    if args.drop_idx is not None and args.drop_idx != -1:
        cmd_common += ["--drop-idx", str(args.drop_idx)]
    if args.region:
        cmd_common += ["--region", args.region]
    if args.no_cleanup:
        cmd_common += ["--no-cleanup"]

    t0 = time.time()
    with open(log_path, "a") as logf:
        logf.write(f"\n=== {row_key} : batch run started ===\n")
        logf.flush()

        need_align = args.force or not (align_json.exists() and (outdir / "bed").exists())
        if need_align:
            with sem_align:
                proc = subprocess.run(cmd_common + ["--stage", "align"],
                                       stdout=logf, stderr=subprocess.STDOUT, env=env)
            if proc.returncode != 0:
                return {"row_key": row_key, "status": "align_failed",
                        "returncode": proc.returncode, "log": str(log_path)}

        with sem_score:
            proc = subprocess.run(cmd_common + ["--stage", "score"],
                                   stdout=logf, stderr=subprocess.STDOUT, env=env)
        if proc.returncode != 0:
            return {"row_key": row_key, "status": "score_failed",
                    "returncode": proc.returncode, "log": str(log_path)}
    dt = time.time() - t0

    align_data = json.loads(align_json.read_text()) if align_json.exists() else {}
    score_data = json.loads(score_json.read_text()) if score_json.exists() else {}
    result = {
        "dataset_id": dataset_id, "class": klass, "kind": kind,
        "individual": individual, "coverage": coverage,
        "realized_coverage": row.get("realized_coverage"),
        **{f"align_{k}": v for k, v in align_data.items()},
        **score_data,
        "wall_seconds": round(dt, 2),
        "row_key": row_key,
    }
    combined_json.write_text(json.dumps(result, indent=1))

    with results_lock:
        write_header = not P.RESULTS_TSV.exists()
        P.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(P.RESULTS_TSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULTS_HEADER, extrasaction="ignore", delimiter="\t")
            if write_header:
                w.writeheader()
            w.writerow(result)

    print(f"[{row_key}] OK in {dt:.1f}s  error_rate={result.get('error_rate')}", flush=True)
    return {"row_key": row_key, "status": "ok", **result}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None, help="comma-separated dataset_id filter")
    ap.add_argument("--coverage", default=None, help="comma-separated coverage filter")
    ap.add_argument("--kind", default=None, help="comma-separated kind filter")
    ap.add_argument("--samples", default=None, help="comma-separated individual filter")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--drop-idx", type=int, default=P.FIXED_DROP_IDX,
                     help="fixed founder-drop index (defaults to simval_paths.FIXED_DROP_IDX, "
                          "pinned from the pilot reference run; pass --drop-idx=-1 to instead "
                          "let each row pick its own hits-based drop, e.g. for a one-off "
                          "reference run that determines a new fixed index)")
    ap.add_argument("--region", default=None)
    ap.add_argument("--no-cleanup", action="store_true")
    ap.add_argument("--parallel-align", type=int, default=3)
    ap.add_argument("--parallel-score", type=int, default=10)
    ap.add_argument("--threads-per-align", type=int, default=20)
    args = ap.parse_args()

    P.SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    P.TMPDIR.mkdir(parents=True, exist_ok=True)
    P.LOG_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(P.MANIFEST)
    rows = filter_rows(rows, args)
    print(f"Running simval batch over {len(rows)} row(s) "
          f"(parallel_align={args.parallel_align}, parallel_score={args.parallel_score}, "
          f"drop_idx={args.drop_idx}, region={args.region})")

    sem_align = threading.Semaphore(args.parallel_align)
    sem_score = threading.Semaphore(args.parallel_score)
    results_lock = threading.Lock()

    t_batch0 = time.time()
    outcomes = []
    max_workers = max(args.parallel_align + args.parallel_score, 1)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_row, row, args, sem_align, sem_score, results_lock): row
                   for row in rows}
        for fut in as_completed(futures):
            outcomes.append(fut.result())
    makespan_s = time.time() - t_batch0

    ok = sum(1 for o in outcomes if o["status"] in ("ok", "cached"))
    failed = [o for o in outcomes if o["status"] not in ("ok", "cached", "dry_run")]
    summed_wall = sum(o.get("wall_seconds", 0) for o in outcomes if o["status"] == "ok")

    print(f"\nDone: {ok}/{len(outcomes)} succeeded (incl. cached)")
    if failed:
        print(f"Failed ({len(failed)}):")
        for o in failed:
            print(f"  {o['row_key']}: {o['status']} (log: {o.get('log')})")

    print(f"Batch makespan: {makespan_s:.1f}s ({makespan_s/3600:.2f}h)")
    print(f"Summed per-row wall time (this run's new rows only): {summed_wall:.1f}s "
          f"({summed_wall/3600:.2f}h)")

    summary = {
        "n_rows": len(rows), "ok": ok, "failed": len(failed),
        "makespan_s": round(makespan_s, 1), "summed_new_wall_s": round(summed_wall, 1),
        "args": vars(args),
    }
    (P.RESULTS_DIR / "simval_batch_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
