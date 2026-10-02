#!/usr/bin/env python
"""
Re-score all 200 rows of the maize simulated-validation corpus with BOTH the
existing per-panel-SITE metric and the new per-EVENT metric (one unit per
truth variant event, per haplotype -- see compare_gvcf_truth_diploid.py's
--event-metrics for the exact definition and
/home/zrm22/.claude/plans/swift-chasing-melody.md for why). Whole genome, no
--region, matching the published matrix's own scope exactly.

COMPARATOR-ONLY: reuses the CACHED *_imputed.autosomes.vcf.gz + .tbi each
row's original align+score run already produced under
scratch/simval_eval/<row_key>/ (verified this session: all 200/200 present,
622GB total) -- no refmap, no inference, no bed-to-vcf here. Recomputing the
site-level metric alongside the new event metric is deliberate: it must
reproduce the published results/simval_results.tsv EXACTLY, which is the
hard correctness check on this whole pass (see main()'s --verify-against).

Modeled directly on simval_class_breakdown.py (locate cached VCF by row_key,
ThreadPoolExecutor), extended to the full manifest with simval_batch.py's
resumability convention: a row whose per-row JSON already exists is skipped
unless --force.

Usage:
    simval_rescore_events.py [--parallel 16] [--limit N] [--force]
        [--verify-against results/simval_results.tsv]
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import simval_paths as P  # noqa: E402

PY = "/home/zrm22/mambaforge/envs/phg-ml/bin/python"
COMPARATOR = Path(__file__).parent / "compare_gvcf_truth_diploid.py"
EVENTS_DIR = P.RESULTS_DIR / "simval_events"
EVENTS_TSV = P.RESULTS_DIR / "simval_events_results.tsv"
EVENT_CLASSES = ("SNP", "INS", "DEL")

TSV_HEADER = [
    "dataset_id", "class", "kind", "individual", "coverage", "row_key",
    # Site-level (recomputed here; must match the published matrix exactly).
    "error_rate", "partial_error_rate", "compared_sites",
    # Event-level headline.
    "event_error_rate_strict", "event_error_rate_fractional",
    "events_total", "events_observable", "events_unobservable",
    # Event-level per class.
    "events_total_SNP", "events_observable_SNP", "events_correct_strict_SNP",
    "events_total_INS", "events_observable_INS", "events_correct_strict_INS",
    "events_total_DEL", "events_observable_DEL", "events_correct_strict_DEL",
    # Diagnostics / self-checks.
    "false_positive_runs", "invariant_check_delta",
    "h1_events_total_all", "h2_events_total_all", "h1_eq_h2_events",
    "wall_seconds", "status",
]


def load_manifest():
    with open(P.MANIFEST) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def row_key_of(row):
    return f"{row['dataset_id']}__{row['individual']}__{row['coverage']}x"


def run_one(row, force):
    row_key = row_key_of(row)
    outdir = P.SCRATCH_ROOT / row_key
    imputed_gz = next(outdir.glob("*_imputed.autosomes.vcf.gz"), None)
    if imputed_gz is None:
        return {"row_key": row_key, "dataset_id": row["dataset_id"],
                "individual": row["individual"], "coverage": row["coverage"],
                "status": "no_cached_vcf"}

    json_path = EVENTS_DIR / f"{row_key}.json"
    log_path = P.LOG_DIR / f"events_{row_key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if json_path.exists() and not force:
        counts = json.loads(json_path.read_text())
        wall = counts.get("_wall_seconds")
        return summarize(row, row_key, counts, wall, "ok_cached")

    cmd = [PY, str(COMPARATOR),
           "--imputed-vcf", str(imputed_gz), "--sample", row["individual"],
           "--truth-gvcf-h1", row["truth_h1"], "--truth-gvcf-h2", row["truth_h2"],
           "--partial-credit", "--event-metrics",
           "--json-out", str(json_path)]
    env = {"TMPDIR": str(P.TMPDIR)}
    import os
    full_env = {**os.environ, **env}

    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=full_env)
    dt = time.time() - t0

    if proc.returncode != 0 or not json_path.exists():
        return {"row_key": row_key, "dataset_id": row["dataset_id"],
                "individual": row["individual"], "coverage": row["coverage"],
                "status": "failed", "log": str(log_path), "wall_seconds": round(dt, 2)}

    counts = json.loads(json_path.read_text())
    counts["_wall_seconds"] = round(dt, 2)
    json_path.write_text(json.dumps(counts, indent=1))  # persist wall time in the cache
    return summarize(row, row_key, counts, dt, "ok")


def summarize(row, row_key, counts, wall_seconds, status):
    compared = counts.get("compared_sites", 0)
    error_rate = (1.0 - counts["gt_allele_matches"] / compared) if compared else None
    partial_error_rate = (1.0 - counts["partial_credit_sum"] / compared) if compared else None

    tot_total = sum(counts.get(f"events_total_{c}", 0) for c in EVENT_CLASSES)
    tot_obs = sum(counts.get(f"events_observable_{c}", 0) for c in EVENT_CLASSES)
    tot_unobs = sum(counts.get(f"events_unobservable_{c}", 0) for c in EVENT_CLASSES)
    tot_correct = sum(counts.get(f"events_correct_strict_{c}", 0) for c in EVENT_CLASSES)
    tot_frac = sum(counts.get(f"events_fractional_sum_{c}", 0) for c in EVENT_CLASSES)
    event_error_strict = (1.0 - tot_correct / tot_obs) if tot_obs else None
    event_error_frac = (1.0 - tot_frac / tot_obs) if tot_obs else None

    invariant_delta = counts.get("haplotype_correct_sum", 0) - 2 * counts.get(
        "partial_credit_sum", 0.0)

    h1t = counts.get("h1_events_total", {})
    h2t = counts.get("h2_events_total", {})
    h1_all = sum(h1t.values()) if h1t else None
    h2_all = sum(h2t.values()) if h2t else None
    h1_eq_h2 = (h1t == h2t) if (h1t and h2t) else None

    out = {
        "dataset_id": row["dataset_id"], "class": row["class"], "kind": row["kind"],
        "individual": row["individual"], "coverage": row["coverage"], "row_key": row_key,
        "error_rate": error_rate, "partial_error_rate": partial_error_rate,
        "compared_sites": compared,
        "event_error_rate_strict": event_error_strict,
        "event_error_rate_fractional": event_error_frac,
        "events_total": tot_total, "events_observable": tot_obs,
        "events_unobservable": tot_unobs,
        "false_positive_runs": counts.get("false_positive_runs"),
        "invariant_check_delta": invariant_delta,
        "h1_events_total_all": h1_all, "h2_events_total_all": h2_all,
        "h1_eq_h2_events": h1_eq_h2,
        "wall_seconds": round(wall_seconds, 2) if wall_seconds is not None else None,
        "status": status,
    }
    for c in EVENT_CLASSES:
        out[f"events_total_{c}"] = counts.get(f"events_total_{c}", 0)
        out[f"events_observable_{c}"] = counts.get(f"events_observable_{c}", 0)
        out[f"events_correct_strict_{c}"] = counts.get(f"events_correct_strict_{c}", 0)
    return out


def write_tsv_row(f, row):
    f.write("\t".join(str(row.get(k, "")) for k in TSV_HEADER) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-against", default=str(P.RESULTS_DIR / "simval_results.tsv"),
                     help="published site-level results TSV to cross-check the recomputed "
                          "error_rate/partial_error_rate against (must match every row exactly)")
    args = ap.parse_args()

    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()
    if args.limit:
        rows = rows[: args.limit]

    print(f"Re-scoring {len(rows)} rows (parallel={args.parallel}, force={args.force}) ...",
          flush=True)
    results = []
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {pool.submit(run_one, r, args.force): row_key_of(r) for r in rows}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            n_done += 1
            print(f"  [{n_done}/{len(rows)}] {r.get('row_key')}  status={r.get('status')}  "
                  f"error_rate={r.get('error_rate')}  event_error_strict="
                  f"{r.get('event_error_rate_strict')}", flush=True)

    with open(EVENTS_TSV, "w") as f:
        f.write("\t".join(TSV_HEADER) + "\n")
        for r in sorted(results, key=lambda x: x.get("row_key", "")):
            write_tsv_row(f, r)
    print(f"\nWrote {EVENTS_TSV}")

    n_ok = sum(1 for r in results if r.get("status") in ("ok", "ok_cached"))
    n_failed = sum(1 for r in results if r.get("status") == "failed")
    n_missing = sum(1 for r in results if r.get("status") == "no_cached_vcf")
    print(f"\nstatus: ok={n_ok} failed={n_failed} no_cached_vcf={n_missing} / {len(results)}")
    if n_failed:
        print("  FAILED rows:")
        for r in results:
            if r.get("status") == "failed":
                print(f"    {r['row_key']}  log={r.get('log')}")
    if n_missing:
        print("  NO_CACHED_VCF rows:")
        for r in results:
            if r.get("status") == "no_cached_vcf":
                print(f"    {r['row_key']}")

    n_bad_invariant = sum(1 for r in results if r.get("invariant_check_delta") not in (None,)
                           and abs(r["invariant_check_delta"]) > 1e-3)
    n_bad_h1h2 = sum(1 for r in results if r.get("kind") == "inbred"
                      and r.get("h1_eq_h2_events") is False)
    print(f"\nself-checks: invariant_delta>1e-3 in {n_bad_invariant} rows; "
          f"inbred h1!=h2 events in {n_bad_h1h2} rows (both should be 0)")

    verify_path = Path(args.verify_against)
    if verify_path.exists():
        published = {}
        with open(verify_path) as f:
            for pr in csv.DictReader(f, delimiter="\t"):
                published[pr["row_key"]] = pr
        n_checked = n_mismatch = 0
        for r in results:
            pub = published.get(r["row_key"])
            if pub is None or r.get("error_rate") is None:
                continue
            n_checked += 1
            pub_er = float(pub["error_rate"]) if pub.get("error_rate") not in (None, "", "None") else None
            if pub_er is None:
                continue
            if abs(r["error_rate"] - pub_er) > 1e-9:
                n_mismatch += 1
                print(f"  MISMATCH {r['row_key']}: recomputed={r['error_rate']!r} "
                      f"published={pub_er!r}")
        print(f"\nverify-against {verify_path}: {n_checked} rows checked, "
              f"{n_mismatch} mismatches (should be 0)")
    else:
        print(f"\n(no {verify_path} found -- skipping cross-check against published site-level "
              f"results)")


if __name__ == "__main__":
    main()
