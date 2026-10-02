#!/usr/bin/env python
"""
SNP + RefCall re-scoring of the 40 refmap rows at the 0.1x coverage rung of
the maize simulated-validation corpus (see
/home/zrm22/.claude/plans/snp-refcall-rescore-simval-0.1x.md).

Computes a second error rate alongside the published all-sites one: a site
scores iff BOTH the truth call and the imputed call classify to {HOMREF, SNP}
-- any site where either side actually CALLED an indel is skipped (not
filtered on panel record shape; see compare_gvcf_truth_diploid.py's
SCORED_CLASSES / snp_refcall_metrics docstring for why that distinction
matters). refmap only -- no chain_* arms, no re-alignment.

COMPARATOR-ONLY: reuses the CACHED *_imputed.autosomes.vcf.gz + .tbi each
row's original align+score run already produced under
scratch/simval_eval/<row_key>/ (verified this session: 40/40 present) -- no
refmap, no inference, no bed-to-vcf here. Recomputing the all-sites
error_rate/partial_error_rate/compared_sites alongside the new metric is
deliberate: it must reproduce the published results/simval_results.tsv
EXACTLY on all 40 rows, which is the hard correctness check on this pass
(see main()'s --verify-against).

Modeled directly on simval_rescore_events.py (locate cached VCF by row_key,
ThreadPoolExecutor, per-row JSON cache, --json-out instead of stdout regex,
--force, --verify-against).

Usage:
    simval_snp_refcall_rescore.py [--parallel 12] [--limit N] [--force]
        [--verify-against results/simval_results.tsv]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import simval_paths as P  # noqa: E402

PY = "/home/zrm22/mambaforge/envs/phg-ml/bin/python"
COMPARATOR = Path(__file__).parent / "compare_gvcf_truth_diploid.py"
SNPRC_DIR = P.RESULTS_DIR / "simval_snp_refcall"
SNPRC_TSV = P.RESULTS_DIR / "simval_snp_refcall_results.tsv"
SNPRC_MD = P.RESULTS_DIR / "simval_snp_refcall_results.md"
SCORED_CLASSES = ("HOMREF", "SNP")
VARIANT_CLASSES = ("HOMREF", "SNP", "INS", "DEL", "HET_MIXED")

TSV_HEADER = [
    "dataset_id", "class", "kind", "individual", "coverage", "row_key",
    # SNP+RefCall headline.
    "snprc_compared_sites", "snprc_error_rate", "snprc_partial_error_rate",
    "snprc_class_total_HOMREF", "snprc_class_mismatch_HOMREF",
    "snprc_class_total_SNP", "snprc_class_mismatch_SNP",
    # Published all-sites, joined in for comparison (also the regression check).
    "error_rate", "partial_error_rate", "compared_sites",
    "published_error_rate", "published_partial_error_rate", "published_compared_sites",
    "regression_match",
    # Drop-reason table (5x5 pairclass_<truth>_<imputed> counts).
    *[f"pairclass_{t}_{i}" for t in VARIANT_CLASSES for i in VARIANT_CLASSES],
    "imputed_missing_gt", "excluded_no_info",
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

    json_path = SNPRC_DIR / f"{row_key}.json"
    log_path = P.LOG_DIR / f"snprc_{row_key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if json_path.exists() and not force:
        counts = json.loads(json_path.read_text())
        wall = counts.get("_wall_seconds")
        return summarize(row, row_key, counts, wall, "ok_cached")

    cmd = [PY, str(COMPARATOR),
           "--imputed-vcf", str(imputed_gz), "--sample", row["individual"],
           "--truth-gvcf-h1", row["truth_h1"], "--truth-gvcf-h2", row["truth_h2"],
           "--partial-credit", "--class-breakdown", "--snp-refcall-metrics",
           "--json-out", str(json_path)]
    full_env = {**os.environ, "TMPDIR": str(P.TMPDIR)}

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

    snprc_compared = counts.get("snprc_compared_sites", 0)
    snprc_error_rate = (
        1.0 - counts["snprc_gt_allele_matches"] / snprc_compared) if snprc_compared else None
    snprc_partial_error_rate = (
        1.0 - counts["snprc_partial_credit_sum"] / snprc_compared) if snprc_compared else None

    out = {
        "dataset_id": row["dataset_id"], "class": row["class"], "kind": row["kind"],
        "individual": row["individual"], "coverage": row["coverage"], "row_key": row_key,
        "snprc_compared_sites": snprc_compared,
        "snprc_error_rate": snprc_error_rate,
        "snprc_partial_error_rate": snprc_partial_error_rate,
        "snprc_class_total_HOMREF": counts.get("snprc_class_total_HOMREF", 0),
        "snprc_class_mismatch_HOMREF": counts.get("snprc_class_mismatch_HOMREF", 0),
        "snprc_class_total_SNP": counts.get("snprc_class_total_SNP", 0),
        "snprc_class_mismatch_SNP": counts.get("snprc_class_mismatch_SNP", 0),
        "error_rate": error_rate, "partial_error_rate": partial_error_rate,
        "compared_sites": compared,
        "imputed_missing_gt": counts.get("imputed_missing_gt", 0),
        "excluded_no_info": counts.get("excluded_no_info", 0),
        "wall_seconds": round(wall_seconds, 2) if wall_seconds is not None else None,
        "status": status,
    }
    for t in VARIANT_CLASSES:
        for i in VARIANT_CLASSES:
            out[f"pairclass_{t}_{i}"] = counts.get(f"pairclass_{t}_{i}", 0)
    return out


def write_tsv_row(f, row):
    f.write("\t".join(str(row.get(k, "")) for k in TSV_HEADER) + "\n")


def write_md(results):
    ok = [r for r in results if r.get("status") in ("ok", "ok_cached")
          and r.get("snprc_error_rate") is not None]
    lines = ["# SNP + RefCall re-score -- simulated-validation 0.1x rung (refmap only)", ""]
    lines.append("Indel calls (either side) excluded; see the plan for the exact "
                  "definition. Not comparable to the published all-sites metric -- "
                  "always read alongside it.")
    lines.append("")
    lines.append("| dataset_id | kind | individual | snprc_error% | snprc_n | "
                  "all-sites_error% | all-sites_n | regression |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for r in sorted(ok, key=lambda x: x["row_key"]):
        lines.append(
            f"| {r['dataset_id']} | {r['kind']} | {r['individual']} | "
            f"{100 * r['snprc_error_rate']:.3f} | {r['snprc_compared_sites']} | "
            f"{100 * r['error_rate']:.3f} | {r['compared_sites']} | "
            f"{r.get('regression_match', 'n/a')} |")
    SNPRC_MD.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parallel", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-against", default=str(P.RESULTS_DIR / "simval_results.tsv"),
                     help="published site-level results TSV to cross-check the recomputed "
                          "all-sites error_rate/partial_error_rate/compared_sites against "
                          "(must match every row exactly)")
    args = ap.parse_args()

    SNPRC_DIR.mkdir(parents=True, exist_ok=True)
    rows = [r for r in load_manifest() if r["coverage"] == "0.1"]
    if args.limit:
        rows = rows[: args.limit]

    print(f"Re-scoring {len(rows)} rows at coverage=0.1 (parallel={args.parallel}, "
          f"force={args.force}) ...", flush=True)
    results = []
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {pool.submit(run_one, r, args.force): row_key_of(r) for r in rows}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            n_done += 1
            print(f"  [{n_done}/{len(rows)}] {r.get('row_key')}  status={r.get('status')}  "
                  f"snprc_error_rate={r.get('snprc_error_rate')}  "
                  f"error_rate={r.get('error_rate')}", flush=True)

    # Verification item 1: exact regression against published all-sites results, joined
    # in per-row (not a separate pass) so the TSV carries the check result directly.
    published = {}
    verify_path = Path(args.verify_against)
    if verify_path.exists():
        with open(verify_path) as f:
            for pr in csv.DictReader(f, delimiter="\t"):
                published[pr["row_key"]] = pr

    n_checked = n_mismatch = 0
    for r in results:
        pub = published.get(r["row_key"])
        if pub is None or r.get("error_rate") is None:
            r["published_error_rate"] = r["published_partial_error_rate"] = ""
            r["published_compared_sites"] = ""
            r["regression_match"] = "no_published_row"
            continue
        pub_er = float(pub["error_rate"]) if pub.get("error_rate") not in (None, "", "None") else None
        pub_per = (float(pub["partial_error_rate"])
                   if pub.get("partial_error_rate") not in (None, "", "None") else None)
        pub_cs = int(pub["compared_sites"]) if pub.get("compared_sites") not in (None, "") else None
        r["published_error_rate"] = pub_er
        r["published_partial_error_rate"] = pub_per
        r["published_compared_sites"] = pub_cs
        n_checked += 1
        match = (pub_er is not None and abs(r["error_rate"] - pub_er) <= 1e-9
                 and pub_cs is not None and r["compared_sites"] == pub_cs)
        if pub_per is not None and r.get("partial_error_rate") is not None:
            match = match and abs(r["partial_error_rate"] - pub_per) <= 1e-9
        r["regression_match"] = "MATCH" if match else "MISMATCH"
        if not match:
            n_mismatch += 1
            print(f"  MISMATCH {r['row_key']}: recomputed error_rate={r['error_rate']!r} "
                  f"compared_sites={r['compared_sites']!r}  vs published "
                  f"error_rate={pub_er!r} compared_sites={pub_cs!r}")

    with open(SNPRC_TSV, "w") as f:
        f.write("\t".join(TSV_HEADER) + "\n")
        for r in sorted(results, key=lambda x: x.get("row_key", "")):
            write_tsv_row(f, r)
    print(f"\nWrote {SNPRC_TSV}")
    write_md(results)
    print(f"Wrote {SNPRC_MD}")

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

    print(f"\nregression check (item 1): {n_checked} rows checked against "
          f"{verify_path}, {n_mismatch} mismatches (should be 0)")

    # Verification items 2-4: partition identity / class-sum identity / directional
    # sanity, computed here from the per-row TSV fields so a bad row is visible
    # immediately rather than requiring a separate script.
    n_bad_partition = n_bad_classsum = n_bad_directional = 0
    for r in results:
        if r.get("status") not in ("ok", "ok_cached"):
            continue
        pairclass_sum = sum(r.get(f"pairclass_{t}_{i}", 0)
                             for t in VARIANT_CLASSES for i in VARIANT_CLASSES)
        if pairclass_sum != r.get("compared_sites"):
            n_bad_partition += 1
            print(f"  BAD PARTITION {r['row_key']}: sum(pairclass_*)={pairclass_sum} "
                  f"!= compared_sites={r.get('compared_sites')}")
        scored_sum = sum(r.get(f"pairclass_{t}_{i}", 0)
                          for t in SCORED_CLASSES for i in SCORED_CLASSES)
        if scored_sum != r.get("snprc_compared_sites"):
            n_bad_partition += 1
            print(f"  BAD SCORED-SUM {r['row_key']}: sum(scored pairclass)={scored_sum} "
                  f"!= snprc_compared_sites={r.get('snprc_compared_sites')}")
        class_sum = r.get("snprc_class_total_HOMREF", 0) + r.get("snprc_class_total_SNP", 0)
        if class_sum != r.get("snprc_compared_sites"):
            n_bad_classsum += 1
            print(f"  BAD CLASS-SUM {r['row_key']}: HOMREF+SNP totals={class_sum} "
                  f"!= snprc_compared_sites={r.get('snprc_compared_sites')}")
        # Plan item 4 expects a strict drop on every row, EXCEPT the B73
        # self-alignment negative control: confirmed (2026-08-18, this
        # session) to have class_total_{SNP,INS,DEL,HET_MIXED} == 0
        # genome-wide (its own truth gVCF selects the reference allele at
        # every site), so there is nothing for the SNP+RefCall filter to
        # drop -- equality there is correct, not a bug.
        no_true_indel_or_snp = (
            sum(r.get(f"pairclass_{t}_{i}", 0)
                for t in VARIANT_CLASSES if t != "HOMREF" for i in VARIANT_CLASSES) == 0)
        if (r.get("snprc_compared_sites") is not None and r.get("compared_sites") is not None
                and not (r["snprc_compared_sites"] < r["compared_sites"])
                and not no_true_indel_or_snp):
            n_bad_directional += 1
            print(f"  BAD DIRECTIONAL {r['row_key']}: snprc_compared_sites="
                  f"{r['snprc_compared_sites']} not < compared_sites={r['compared_sites']}")

    print(f"\nself-checks: partition/scored-sum identity failures={n_bad_partition}, "
          f"class-sum identity failures={n_bad_classsum}, "
          f"directional-sanity failures={n_bad_directional} (all should be 0)")


if __name__ == "__main__":
    main()
