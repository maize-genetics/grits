#!/usr/bin/env python
"""
SNP+RefCall error-rate scoring for the 3 IDX-RIL2 filter-sweep tables (see
/home/zrm22/.claude/plans/wondrous-discovering-octopus.md). Every row's
align stage (refmap -> window -> diploid-affinity CRF decode -> founder-
path BED) is already done, across 5 pairs x 5 coverages x 3 filter tags
(unfiltered-bin, hitfrac0.3-bin, hitfrac0.5-bin) = 75 rows, under
scratch/simval_eval/IDX-RIL2__<individual>__<coverage>x__<tag>/bed/. This
script does ONLY the two remaining stages, per row:

  1. VCF prep -- bed_to_vcf -> filter_to_autosomes -> bgzip+index. This is
     exactly the first half of simval_eval_one.do_score(), extracted
     inline rather than calling do_score() itself: do_score() would also
     immediately run the comparator once with the WRONG (non-SNP) flags,
     wasting a full second ~35-40min comparator pass per row.
  2. SNP+RefCall comparator -- reuses simval_snp_refcall_rescore.py's own
     proven subprocess call to compare_gvcf_truth_diploid.py verbatim
     (same flags, same JSON-cache-first check, same TMPDIR override).

Both stages are resumable by file/JSON existence, matching every other
script in this pipeline -- safe to re-invoke after an interruption; only
whatever wasn't finished gets redone.

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src \
      /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/run_ril2_snp_scoring.py \
        [--parallel 12] [--force]
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
import heldout_assembly_eval as hae  # noqa: E402
import simval_eval_one as seo  # noqa: E402
import simval_paths as P  # noqa: E402

PY = "/home/zrm22/mambaforge/envs/phg-ml/bin/python"
COMPARATOR = Path(__file__).parent / "compare_gvcf_truth_diploid.py"
SNPRC_DIR = P.RESULTS_DIR / "ril2_snprc"
SNPRC_TSV = P.RESULTS_DIR / "ril2_snprc_results.tsv"

PAIRS = ["B73xOh43", "B73xCML103", "Oh43xIl14H", "B97xCML103", "Il14HxB97"]
COVERAGES = ["0.01", "0.1", "0.5", "1.0", "2.0"]
TAGS = ["unfiltered-bin", "hitfrac0.3-bin", "hitfrac0.5-bin"]

TSV_HEADER = [
    "individual", "coverage", "tag", "row_key",
    "snprc_compared_sites", "snprc_error_rate", "snprc_partial_error_rate",
    "snprc_class_total_HOMREF", "snprc_class_mismatch_HOMREF",
    "snprc_class_total_SNP", "snprc_class_mismatch_SNP",
    "error_rate", "partial_error_rate", "compared_sites",
    "wall_seconds", "status",
]


def load_manifest_row(individual, coverage):
    with open(P.MANIFEST) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["dataset_id"] == "IDX-RIL2" and row["individual"] == individual \
                    and row["coverage"] == coverage:
                return row
    raise KeyError(f"no manifest row for IDX-RIL2/{individual}/{coverage}x")


def prep_vcf(outdir, sample, panel_vcf):
    """First half of simval_eval_one.do_score() -- bed_to_vcf ->
    filter_to_autosomes -> bgzip+index. Returns the final .vcf.gz path.
    Each step is skip-if-exists, same as do_score()'s own logic."""
    bed_dir = outdir / "bed"
    imputed_vcf = outdir / f"{sample}_imputed.vcf"
    filtered_vcf = outdir / f"{sample}_imputed.autosomes.vcf"
    imputed_vcf_gz = filtered_vcf.with_suffix(filtered_vcf.suffix + ".gz")

    if imputed_vcf_gz.exists():
        return imputed_vcf_gz
    if not filtered_vcf.exists():
        if not imputed_vcf.exists():
            hae.bed_to_vcf(bed_dir, panel_vcf, imputed_vcf)
        seo.filter_to_autosomes(imputed_vcf, filtered_vcf)
        imputed_vcf.unlink(missing_ok=True)
    imputed_vcf_gz = hae.bgzip_and_index_vcf(filtered_vcf)
    filtered_vcf.unlink(missing_ok=True)
    return imputed_vcf_gz


def run_one(individual, coverage, tag, force):
    row_key = f"IDX-RIL2__{individual}__{coverage}x__{tag}"
    outdir = P.SCRATCH_ROOT / row_key
    sample = f"{individual}_ril2_{tag}"
    json_path = SNPRC_DIR / f"{row_key}.json"
    log_path = P.LOG_DIR / f"ril2_snprc_{row_key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    SNPRC_DIR.mkdir(parents=True, exist_ok=True)

    if json_path.exists() and not force:
        counts = json.loads(json_path.read_text())
        return summarize(individual, coverage, tag, row_key, counts,
                          counts.get("_wall_seconds"), "ok_cached")

    t0 = time.time()
    try:
        mrow = load_manifest_row(individual, coverage)
        imputed_vcf_gz = prep_vcf(outdir, sample, str(P.PANEL_VCF_V2))
    except Exception as e:
        return {"individual": individual, "coverage": coverage, "tag": tag,
                "row_key": row_key, "status": f"vcf_prep_failed: {e}",
                "wall_seconds": round(time.time() - t0, 2)}
    t_vcf = time.time()

    cmd = [PY, str(COMPARATOR),
           "--imputed-vcf", str(imputed_vcf_gz), "--sample", sample,
           "--truth-gvcf-h1", mrow["truth_h1"], "--truth-gvcf-h2", mrow["truth_h2"],
           "--partial-credit", "--class-breakdown", "--snp-refcall-metrics",
           "--json-out", str(json_path)]
    full_env = {**os.environ, "TMPDIR": str(P.TMPDIR)}

    with open(log_path, "w") as logf:
        logf.write(f"vcf_prep_seconds={t_vcf - t0:.2f}\ncmd={' '.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=full_env)
    dt = time.time() - t0

    if proc.returncode != 0 or not json_path.exists():
        return {"individual": individual, "coverage": coverage, "tag": tag,
                "row_key": row_key, "status": "comparator_failed",
                "log": str(log_path), "wall_seconds": round(dt, 2)}

    counts = json.loads(json_path.read_text())
    counts["_wall_seconds"] = round(dt, 2)
    counts["_vcf_prep_seconds"] = round(t_vcf - t0, 2)
    json_path.write_text(json.dumps(counts, indent=1))
    return summarize(individual, coverage, tag, row_key, counts, dt, "ok")


def summarize(individual, coverage, tag, row_key, counts, wall_seconds, status):
    compared = counts.get("compared_sites", 0)
    error_rate = (1.0 - counts["gt_allele_matches"] / compared) if compared else None
    partial_error_rate = (1.0 - counts["partial_credit_sum"] / compared) if compared else None

    snprc_compared = counts.get("snprc_compared_sites", 0)
    snprc_error_rate = (
        1.0 - counts["snprc_gt_allele_matches"] / snprc_compared) if snprc_compared else None
    snprc_partial_error_rate = (
        1.0 - counts["snprc_partial_credit_sum"] / snprc_compared) if snprc_compared else None

    return {
        "individual": individual, "coverage": coverage, "tag": tag, "row_key": row_key,
        "snprc_compared_sites": snprc_compared,
        "snprc_error_rate": snprc_error_rate,
        "snprc_partial_error_rate": snprc_partial_error_rate,
        "snprc_class_total_HOMREF": counts.get("snprc_class_total_HOMREF", 0),
        "snprc_class_mismatch_HOMREF": counts.get("snprc_class_mismatch_HOMREF", 0),
        "snprc_class_total_SNP": counts.get("snprc_class_total_SNP", 0),
        "snprc_class_mismatch_SNP": counts.get("snprc_class_mismatch_SNP", 0),
        "error_rate": error_rate, "partial_error_rate": partial_error_rate,
        "compared_sites": compared,
        "wall_seconds": round(wall_seconds, 2) if wall_seconds is not None else None,
        "status": status,
    }


def write_tsv(results):
    P.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SNPRC_TSV, "w") as f:
        f.write("\t".join(TSV_HEADER) + "\n")
        for r in sorted(results, key=lambda x: x["row_key"]):
            f.write("\t".join(str(r.get(k, "")) for k in TSV_HEADER) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    jobs = [(individual, coverage, tag)
            for tag in TAGS for coverage in COVERAGES for individual in PAIRS]
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"Scoring {len(jobs)} IDX-RIL2 rows (parallel={args.parallel}, force={args.force}) ...",
          flush=True)
    results = []
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {pool.submit(run_one, i, c, t, args.force): (i, c, t) for i, c, t in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            n_done += 1
            print(f"  [{n_done}/{len(jobs)}] {r.get('row_key')}  status={r.get('status')}  "
                  f"snprc_error_rate={r.get('snprc_error_rate')}  "
                  f"wall_seconds={r.get('wall_seconds')}", flush=True)
            write_tsv(results)  # incremental -- safe to inspect mid-run

    n_ok = sum(1 for r in results if r.get("status") in ("ok", "ok_cached"))
    print(f"\nDone: {n_ok}/{len(jobs)} ok. Wrote {SNPRC_TSV}")


if __name__ == "__main__":
    main()
