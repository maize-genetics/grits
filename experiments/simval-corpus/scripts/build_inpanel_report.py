#!/usr/bin/env python
"""
Parse nam_inpanel_smoketest.py's output (summary JSON + each founder's
compare_gvcf_truth.py report) into one JSON blob the HTML artifact report
consumes directly (kept separate from HTML generation so the report's
design can be iterated on without re-running the pipeline).

Usage:
    python build_inpanel_report.py [--out PATH]
"""
import argparse
import json
import re
from pathlib import Path

RESULTS_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results")
SUMMARY_JSON = RESULTS_DIR / "nam_inpanel_smoketest_summary.json"
SCRATCH = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/heldout_assembly_eval")
DEFAULT_OUT = RESULTS_DIR / "nam_inpanel_report_data.json"

METRIC_KEYS = [
    "imputed_records", "truth_records", "compared_sites", "excluded_no_info",
    "gt_allele_matches", "gt_allele_mismatches", "allele_GT_concordance",
    "partial_allele_concordance", "matchable_variant_sites",
    "unmatchable_variant_sites", "matchable_ceiling_fraction",
]
NUM_RE = re.compile(r"^\s*([A-Za-z_]+)\s+([0-9.]+)")


def parse_comparison_report(path):
    metrics = {}
    for line in path.read_text().splitlines():
        m = NUM_RE.match(line)
        if m and m.group(1) in METRIC_KEYS:
            val = m.group(2)
            metrics[m.group(1)] = float(val) if "." in val else int(val)
    # Error rate is the primary reported metric now -- concordance clusters
    # so close to 100% (99.2-100%) that it hides real per-founder signal at
    # any reasonable display precision; 1-concordance puts that signal in
    # the leading digits instead of the trailing ones.
    if "allele_GT_concordance" in metrics:
        metrics["error_rate"] = 1.0 - metrics["allele_GT_concordance"]
    if "partial_allele_concordance" in metrics:
        metrics["partial_error_rate"] = 1.0 - metrics["partial_allele_concordance"]
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    summary = json.loads(SUMMARY_JSON.read_text())
    depth = summary["depth"]
    rows = []
    for r in summary["results"]:
        name = r["founder"]
        row = {"founder": name, "ok": r["ok"], "seconds": r.get("seconds", 0.0),
               "reused_truth_gvcf": r.get("reused_truth_gvcf", False)}
        if r["ok"]:
            report_path = SCRATCH / f"{name}_{depth // 1000}k" / f"{name}_comparison.txt"
            if report_path.exists():
                row.update(parse_comparison_report(report_path))
            else:
                row["ok"] = False
                row["error"] = f"comparison report missing: {report_path}"
        else:
            row["error"] = r.get("skip_reason", "pipeline failed -- see log")
        rows.append(row)

    # ascending error rate = best (lowest-error) founder first
    rows.sort(key=lambda r: r.get("error_rate", 1e9))

    ok_rows = [r for r in rows if r["ok"]]
    n = len(ok_rows)
    # matchable_ceiling_fraction is legitimately absent for a founder with zero
    # truth variant sites (e.g. B73 vs. its own self-aligned truth gVCF --
    # there is nothing to have a "ceiling" over); exclude those from the mean
    # rather than crash or silently coerce to 0/1.
    ceiling_rows = [r for r in ok_rows if "matchable_ceiling_fraction" in r]
    agg = {
        "n_founders": len(rows),
        "n_ok": n,
        "n_failed": len(rows) - n,
        "depth": depth,
        "mean_error_rate": sum(r["error_rate"] for r in ok_rows) / n if n else None,
        "mean_partial_error_rate": sum(r["partial_error_rate"] for r in ok_rows) / n if n else None,
        "mean_concordance": sum(r["allele_GT_concordance"] for r in ok_rows) / n if n else None,
        "mean_partial_concordance": sum(r["partial_allele_concordance"] for r in ok_rows) / n if n else None,
        "mean_ceiling": (sum(r["matchable_ceiling_fraction"] for r in ceiling_rows) / len(ceiling_rows)
                         if ceiling_rows else None),
        "n_ceiling_na": n - len(ceiling_rows),
        "total_seconds": sum(r["seconds"] for r in rows),
        # Partial credit only differs from strict credit when a heterozygous
        # mispredict happens. Bug #3's fix (homo_scale=0, forcing the model's
        # heterozygous penalty to full strength on these 100%-homozygous-by-
        # construction samples) means that never happens here -- so this
        # metric is currently degenerate with error_rate on every founder in
        # this dataset. Kept in the report anyway (not dropped): it becomes
        # informative again the moment a genuinely heterozygous/diploid
        # sample goes through this same pipeline.
        "partial_metric_note": (
            "Partial error rate is currently identical to strict error rate "
            "on every founder here: these samples are 100% homozygous by "
            "construction, and the homo_scale=0 fix means no heterozygous "
            "mispredict ever occurs, so partial credit never differs from "
            "strict credit. Kept in the report because it becomes "
            "meaningful again on a genuinely heterozygous/diploid sample."
        ),
    }

    out = {"aggregate": agg, "founders": rows}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
