#!/usr/bin/env python
"""
Build the report JSON for the FIRST genuine held-out/generalization test: 5
maize assemblies confirmed excluded from our 25-founder ropebwt3 index
(Tx303, A188, EP1, CML459, Ia453), picked for germplasm-type diversity and
quality-checked against MaizeGDB/the primary literature this session (see
the plan file for the full survey). Unlike nam_inpanel_report_data.json
(in-panel, best-case ceiling), every result here is a real generalization
number.

Parses each assembly's compare_gvcf_truth.py report
(scratch/heldout_assembly_eval/<name>_250k/<name>_comparison.txt) the same
way build_inpanel_report.py does, adds the germplasm-type/provenance
metadata gathered this session, and pulls in the in-panel aggregate
(results/nam_inpanel_report_data.json) for direct contrast -- this pair of
runs is the "second real run" the in-panel report's own roadmap section
said would seed a proper comparison view.

Usage:
    python build_heldout5_report.py [--out PATH]
"""
import argparse
import json
import re
from pathlib import Path

RESULTS_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results")
SUMMARY_JSON = RESULTS_DIR / "heldout5_fullgenome_summary.json"
SCRATCH = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/heldout_assembly_eval")
INPANEL_REPORT_DATA = RESULTS_DIR / "nam_inpanel_report_data.json"
MECHANISM_DIR = RESULTS_DIR / "diag_error_mechanism_breakdown"
DEFAULT_OUT = RESULTS_DIR / "heldout5_report_data.json"

MECHANISM_KEYS = ["MATCH_REFBLOCK", "MATCH_VARIANT", "REFBLOCK_FALSE_POSITIVE",
                  "VARIANT_REF_BIAS", "VARIANT_OTHER_WRONG", "VARIANT_UNMATCHABLE"]
MECHANISM_RE = re.compile(r"^\s*([A-Za-z_]+)\s+([0-9,]+)")

TRUTH_CLASSES = ["HOMREF", "SNP", "INS", "DEL", "HET_MIXED"]
STRAT_TOTAL_RE = re.compile(
    r"^\s*STRAT_TOTAL\s+(\w+)\s+\(all outcomes\)\s+([\d,]+)\s+error_rate=([\d.]+)")
STRAT_RE = re.compile(r"^\s*STRAT\s+(\w+)\s+(\w+)\s+([\d,]+)\s*$")
CONFUSION_CELL_RE = re.compile(r"^\s*CONFUSION_CELL\s+(\w+)\s+(\w+)\s+(\d+)")

# ROUND 3: event-level (contiguous-run) tracking, added from
# diag_error_mechanism_breakdown.py -- see that script's module docstring
# and the plan file's "ROUND 3" section for why this is a meaningfully
# different (and non-trivial-to-get-right) unit from the site-level rate.
EVENT_MISMATCH_RE = re.compile(r"^\s*EVENT_MISMATCH\s+([\d,]+)")
GT_MATCHES_RE = re.compile(r"^\s*gt_allele_matches\s+([\d,]+)")
# ROUND 4: "ignore indels" event-level tracking (note the trailing \b-style
# guard via requiring whitespace right after -- EVENT_MISMATCH_RE above does
# NOT accidentally match this line, since '_NOINDEL' isn't whitespace).
EVENT_MISMATCH_NOINDEL_RE = re.compile(r"^\s*EVENT_MISMATCH_NOINDEL\s+([\d,]+)")

# ROUND 3: whole-genome-weighted error rate. Panel sites are only 5.41% of
# the actual maize genome (115,354,779 / 2,131,846,805bp, chr1-10 only, from
# B73.fa.fai) -- and are NOT a uniform sample: they exist ONLY where some
# founder differs from B73, i.e. exactly the positions where a wrong
# founder-path choice is punished. Re-weighting the SAME numerator
# (gt_allele_mismatches, unchanged) over the true genome size instead of
# compared_sites answers "what fraction of the actual genome is wrong,"
# assuming the excluded 94.6% (never a panel site, so never checked either
# way) would score correct if it were ever checked -- an UPPER BOUND on
# correctness, not a direct measurement, especially for held-out samples
# which can carry private variants at non-panel positions no founder path
# can express (see the plan file's caveat).
CHROM_LENGTHS = {
    "chr1": 308452471, "chr2": 243675191, "chr3": 238017767, "chr4": 250330460,
    "chr5": 226353449, "chr6": 181357234, "chr7": 185808916, "chr8": 182411202,
    "chr9": 163004744, "chr10": 152435371,
}
GENOME_SIZE = sum(CHROM_LENGTHS.values())  # 2,131,846,805

METRIC_KEYS = [
    "imputed_records", "truth_records", "compared_sites", "excluded_no_info",
    "gt_allele_matches", "gt_allele_mismatches", "allele_GT_concordance",
    "partial_allele_concordance", "matchable_variant_sites",
    "unmatchable_variant_sites", "matchable_ceiling_fraction",
]
NUM_RE = re.compile(r"^\s*([A-Za-z_]+)\s+([0-9.]+)")

# Germplasm-type / provenance metadata gathered this session (candidate
# survey: header-count structural check + MaizeGDB/literature cross-
# reference). See the plan file for full sourcing.
METADATA = {
    "Tx303": {
        "type": "NAM founder (tropical/Gulf Coast background)",
        "provenance": "Hufford et al. 2021, 26-genome NAM PanAnd release",
        "maizegdb_status": "confirmed: Zm-Tx303-REFERENCE-NAM-1.0",
        "note": "required pick",
    },
    "A188": {
        "type": "Reid Yellow Dent-derived (transformation line)",
        "provenance": "KSU chromosome-level assembly",
        "maizegdb_status": "confirmed: Zm-A188-REFERENCE-KSU-1.0",
        "note": "BUSCO 97.25%, published scaffold N50 11.61 Mb",
    },
    "EP1": {
        "type": "European Flint (Spanish landrace Lizargarate)",
        "provenance": "TUM European Flint reference panel",
        "maizegdb_status": "confirmed: Zm-EP1-REFERENCE-TUM-1.0",
        "note": "BUSCO 97.3% complete; most genetically distant heterotic "
                "pool of the 5 from our all-dent/tropical 25-founder panel",
    },
    "CML459": {
        "type": "CIMMYT highland tropical/subtropical",
        "provenance": "CIMMYT/Buckler pangenome sequencing",
        "maizegdb_status": "not confirmed on MaizeGDB",
        "note": "126 sequences, already chr-named, chromosome-scale by "
                "header-count structural check",
    },
    "Ia453": {
        "type": "Sweet corn (sh2/shrunken2 mutant)",
        "provenance": "Hu et al., \"...evolution of modern sweet corn\"",
        "maizegdb_status": "confirmed: Zm-Ia453-REFERENCE-FL-1.0",
        "note": "fetched fresh from MaizeGDB this session (not in the "
                "supplied candidate directory) -- the genuine sweet-corn "
                "pick, replacing the QPM stand-in (K0326Y) originally "
                "short-listed",
    },
}


def parse_comparison_report(path):
    metrics = {}
    for line in path.read_text().splitlines():
        m = NUM_RE.match(line)
        if m and m.group(1) in METRIC_KEYS:
            val = m.group(2)
            metrics[m.group(1)] = float(val) if "." in val else int(val)
    if "allele_GT_concordance" in metrics:
        metrics["error_rate"] = 1.0 - metrics["allele_GT_concordance"]
    if "partial_allele_concordance" in metrics:
        metrics["partial_error_rate"] = 1.0 - metrics["partial_allele_concordance"]
    if "matchable_variant_sites" in metrics and "unmatchable_variant_sites" in metrics:
        total = metrics["matchable_variant_sites"] + metrics["unmatchable_variant_sites"]
        if total:
            metrics["matchable_ceiling_fraction"] = metrics["matchable_variant_sites"] / total
    if "gt_allele_mismatches" in metrics:
        metrics["whole_genome_error_rate"] = metrics["gt_allele_mismatches"] / GENOME_SIZE
    return metrics


def parse_events(path):
    """Parse EVENT_MISMATCH + gt_allele_matches directly from a
    diag_error_mechanism_breakdown.py log (both fields live in the same
    file, so this recomputes the rate itself rather than trust a
    pre-formatted string, and avoids any cross-file inconsistency). Also
    parses ROUND 4's EVENT_MISMATCH_NOINDEL; its matching denominator
    (by_class[HOMREF][MATCH] + by_class[SNP][MATCH]) is filled in by the
    caller from an already-parsed parse_stratification() result, since that
    data already lives there and there's no reason to re-derive it here."""
    if not path.exists():
        return None
    event_mismatch = None
    event_mismatch_noindel = None
    gt_matches = None
    for line in path.read_text().splitlines():
        m = EVENT_MISMATCH_NOINDEL_RE.match(line)
        if m:
            event_mismatch_noindel = int(m.group(1).replace(",", ""))
            continue
        m = EVENT_MISMATCH_RE.match(line)
        if m:
            event_mismatch = int(m.group(1).replace(",", ""))
            continue
        m = GT_MATCHES_RE.match(line)
        if m:
            gt_matches = int(m.group(1).replace(",", ""))
    if event_mismatch is None or gt_matches is None:
        return None
    denom = event_mismatch + gt_matches
    return {
        "event_mismatch": event_mismatch,
        "event_mismatch_noindel": event_mismatch_noindel,
        "gt_allele_matches": gt_matches,
        "event_error_rate": event_mismatch / denom if denom else None,
    }


def combined_noindel_panel_site_rate(stratification):
    """ROUND 4, panel-site framing: combine HOMREF + SNP classes from an
    already-parsed parse_stratification() dict into one site-count-weighted
    error rate (NOT a naive average of the two class rates -- HOMREF and SNP
    have very different site counts). Returns None if either class total is
    unavailable."""
    if not stratification:
        return None
    by_class = stratification.get("by_class", {})
    homref = by_class.get("HOMREF", {})
    snp = by_class.get("SNP", {})
    if "total" not in homref or "total" not in snp:
        return None
    total = homref["total"] + snp["total"]
    if not total:
        return None
    mismatches = homref["total"] * homref["error_rate"] + snp["total"] * snp["error_rate"]
    return mismatches / total


def add_noindel_event_rate(events, stratification):
    """Fill in event_error_rate_noindel on an already-parsed parse_events()
    dict, using the HOMREF+SNP match counts from an already-parsed
    parse_stratification() dict. Mutates and returns `events` for
    convenience; no-ops if either input is missing what it needs."""
    if not events or not stratification or events.get("event_mismatch_noindel") is None:
        return events
    by_class = stratification.get("by_class", {})
    noindel_matches = (by_class.get("HOMREF", {}).get("outcomes", {}).get("MATCH", 0)
                        + by_class.get("SNP", {}).get("outcomes", {}).get("MATCH", 0))
    denom = events["event_mismatch_noindel"] + noindel_matches
    events["event_error_rate_noindel"] = (events["event_mismatch_noindel"] / denom
                                           if denom else None)
    return events


def parse_stratification(path):
    """Parse the STRAT_TOTAL / STRAT / CONFUSION_CELL lines added to
    diag_error_mechanism_breakdown.py this session: for each TRUTH variant
    class (HOMREF/SNP/INS/DEL/HET_MIXED), its own site count and error rate,
    plus the truth-class x imputed-prediction-class confusion matrix. This
    is what actually answers whether SNP accuracy is meaningfully better
    than the overall rate, once ref-block sites (which dominate the genome)
    are structurally separated out rather than pooled in."""
    if not path.exists():
        return None
    by_class = {}
    confusion = {}
    for line in path.read_text().splitlines():
        m = STRAT_TOTAL_RE.match(line)
        if m:
            cls, count_s, err_s = m.groups()
            by_class.setdefault(cls, {})["total"] = int(count_s.replace(",", ""))
            by_class.setdefault(cls, {})["error_rate"] = float(err_s)
            continue
        m = STRAT_RE.match(line)
        if m:
            cls, outcome, count_s = m.groups()
            by_class.setdefault(cls, {}).setdefault("outcomes", {})[outcome] = \
                int(count_s.replace(",", ""))
            continue
        m = CONFUSION_CELL_RE.match(line)
        if m:
            truth_cls, imp_cls, count_s = m.groups()
            confusion.setdefault(truth_cls, {})[imp_cls] = int(count_s)
    if not by_class:
        return None
    return {"by_class": by_class, "confusion": confusion}


def parse_mechanism_breakdown(path):
    """Parse diag_error_mechanism_breakdown.py's log output: raw counts per
    mechanism, plus each mechanism's own share of compared_sites (so these
    sum to the sample's overall error rate) -- see the plan file's Finding 1
    table for how this is used."""
    if not path.exists():
        return None
    counts = {}
    compared_sites = None
    for line in path.read_text().splitlines():
        m = MECHANISM_RE.match(line)
        if not m:
            continue
        key, val = m.group(1), int(m.group(2).replace(",", ""))
        if key == "compared_sites":
            compared_sites = val
        if key in MECHANISM_KEYS:
            counts[key] = val
    if not counts or not compared_sites:
        return None
    mismatch_total = sum(counts.get(k, 0) for k in
                          ["REFBLOCK_FALSE_POSITIVE", "VARIANT_REF_BIAS",
                           "VARIANT_OTHER_WRONG", "VARIANT_UNMATCHABLE"])
    return {
        "counts": counts,
        "compared_sites": compared_sites,
        # each mechanism's share of the OVERALL error rate (pp of compared_sites)
        "error_rate_share": {k: counts.get(k, 0) / compared_sites for k in
                              ["REFBLOCK_FALSE_POSITIVE", "VARIANT_REF_BIAS",
                               "VARIANT_OTHER_WRONG", "VARIANT_UNMATCHABLE"]},
        # each mechanism's share of mismatches only (sums to 100%)
        "mismatch_share": {k: (counts.get(k, 0) / mismatch_total if mismatch_total else None)
                            for k in ["REFBLOCK_FALSE_POSITIVE", "VARIANT_REF_BIAS",
                                      "VARIANT_OTHER_WRONG", "VARIANT_UNMATCHABLE"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    summary = json.loads(SUMMARY_JSON.read_text())
    depth = summary["depth"]
    rows = []
    for r in summary["results"]:
        name = r["founder"]
        row = {"assembly": name, "ok": r["ok"], "seconds": r.get("seconds", 0.0),
               **METADATA.get(name, {})}
        if r["ok"]:
            report_path = SCRATCH / f"{name}_{depth // 1000}k" / f"{name}_comparison.txt"
            if report_path.exists():
                row.update(parse_comparison_report(report_path))
            else:
                row["ok"] = False
                row["error"] = f"comparison report missing: {report_path}"
            mechanism = parse_mechanism_breakdown(MECHANISM_DIR / f"{name}.log")
            if mechanism:
                row["mechanism"] = mechanism
            stratification = parse_stratification(MECHANISM_DIR / f"{name}.log")
            if stratification:
                row["stratification"] = stratification
                row["error_rate_noindel"] = combined_noindel_panel_site_rate(stratification)
            events = parse_events(MECHANISM_DIR / f"{name}.log")
            if events:
                row["events"] = add_noindel_event_rate(events, stratification)
        else:
            row["error"] = r.get("skip_reason", "pipeline failed -- see log")
        rows.append(row)

    rows.sort(key=lambda r: r.get("error_rate", 1e9))

    ok_rows = [r for r in rows if r["ok"]]
    n = len(ok_rows)
    events_rows = [r for r in ok_rows if r.get("events", {}).get("event_error_rate") is not None]
    noindel_events_rows = [r for r in ok_rows
                            if r.get("events", {}).get("event_error_rate_noindel") is not None]
    noindel_panel_rows = [r for r in ok_rows if r.get("error_rate_noindel") is not None]
    agg = {
        "n_assemblies": len(rows),
        "n_ok": n,
        "n_failed": len(rows) - n,
        "depth": depth,
        "mean_error_rate": sum(r["error_rate"] for r in ok_rows) / n if n else None,
        "mean_concordance": sum(r["allele_GT_concordance"] for r in ok_rows) / n if n else None,
        "mean_ceiling": sum(r["matchable_ceiling_fraction"] for r in ok_rows) / n if n else None,
        "mean_whole_genome_error_rate": (sum(r["whole_genome_error_rate"] for r in ok_rows) / n
                                          if n else None),
        "mean_event_error_rate": (sum(r["events"]["event_error_rate"] for r in events_rows)
                                   / len(events_rows) if events_rows else None),
        # ROUND 4: "ignore indels"
        "mean_error_rate_noindel": (sum(r["error_rate_noindel"] for r in noindel_panel_rows)
                                     / len(noindel_panel_rows) if noindel_panel_rows else None),
        "mean_event_error_rate_noindel": (
            sum(r["events"]["event_error_rate_noindel"] for r in noindel_events_rows)
            / len(noindel_events_rows) if noindel_events_rows else None),
        "total_seconds": sum(r["seconds"] for r in rows),
    }

    inpanel_agg = None
    if INPANEL_REPORT_DATA.exists():
        inpanel_data = json.loads(INPANEL_REPORT_DATA.read_text())
        inpanel_agg = inpanel_data["aggregate"]

    inpanel_mechanism = parse_mechanism_breakdown(MECHANISM_DIR / "Oh43.log")

    # 2026-08-05, ROUND 2 (this retracts the "2.74x FILL_GAPS methodology
    # artifact" reported after round 1): what looked like a denominator/
    # methodology asymmetry between in-panel and held-out scoring was
    # actually a single comparator bug (Bug A). compare_gvcf_truth.py's
    # truth-record span was computed ONLY from a literal INFO/END= field.
    # Our own maf-to-gvcf-converter output never writes END= on variant
    # records (only on <NON_REF> ref blocks); smm477's production gVCFs
    # always do. So every deletion in every truth gVCF we built ourselves
    # collapsed to 1bp and its interior silently fell into "no truth info"
    # -- ~37% of the genome -- regardless of any FILL_GAPS setting. Fixed:
    # end = pos + len(ref) - 1, maxed against a literal END= when present.
    # Directly confirmed on request: smm477's gVCFs were NEVER gap-filled to
    # begin with -- a fresh no-fill-gaps rebuild of Il14H from a shared MAF
    # is byte-identical to smm477's original (0 differing rows of
    # POS/REF/ALT/GT out of 40,866,096). Post-fix, exclusion rates are
    # comparable everywhere without any FILL_GAPS distinction: in-panel
    # 1.3-1.9%, held-out 1.4-3.4%. See the plan file's "Outcome" section for
    # the full story, including why the planned panel rebuild (Phase 3) was
    # skipped as a result (no independent second alignment lineage exists on
    # this machine for the 24 real founders -- their truth gVCFs were
    # already inputs to the panel's own construction via merge-gvcfs, so
    # re-scoring them against ANY same-lineage truth source is tautological,
    # ~100.000% concordance for all 25 founders regardless of which
    # same-lineage file is used).
    #
    # Il14H's own "no-fill" comparison below is the one exception that
    # remains genuinely informative: its truth gVCF comes from a MAF at
    # aligned-donors/ (dated May 2025) that is a DIFFERENT, independently
    # confirmed file from the one that built the panel (shared_files/,
    # dated Sep 2024; different size, different checksum) -- so it is NOT
    # the same tautology as the other 24 founders, and its error rate is
    # real signal about founder-path recovery, not circular verification.
    il14h_corrected_path = RESULTS_DIR / "il14h_nofill_comparison.txt"
    il14h_gapfilled_fixed_path = RESULTS_DIR / "il14h_gapfilled_comparison_fixed.txt"
    inpanel_corrected = None
    if il14h_corrected_path.exists():
        il14h_nofill = parse_comparison_report(il14h_corrected_path)
        heldout_mean = agg.get("mean_error_rate")
        inpanel_corrected = {
            "sample": "Il14H",
            "nofill_error_rate": il14h_nofill.get("error_rate"),
            "nofill_ceiling": il14h_nofill.get("matchable_ceiling_fraction"),
            "nofill_whole_genome_error_rate": il14h_nofill.get("whole_genome_error_rate"),
            "corrected_fold_change": (heldout_mean / il14h_nofill["error_rate"]
                                       if heldout_mean and il14h_nofill.get("error_rate") else None),
        }
        if il14h_gapfilled_fixed_path.exists():
            il14h_gapfilled_fixed = parse_comparison_report(il14h_gapfilled_fixed_path)
            inpanel_corrected["gap_filled_error_rate_TAUTOLOGICAL"] = \
                il14h_gapfilled_fixed.get("error_rate")
    inpanel_corrected_mechanism = parse_mechanism_breakdown(MECHANISM_DIR / "Il14H_nofill.log")
    inpanel_corrected_stratification = parse_stratification(MECHANISM_DIR / "Il14H_nofill.log")
    inpanel_stratification = parse_stratification(MECHANISM_DIR / "Oh43.log")
    if inpanel_corrected is not None:
        il14h_events = parse_events(MECHANISM_DIR / "Il14H_nofill.log")
        if il14h_events:
            inpanel_corrected["events"] = add_noindel_event_rate(
                il14h_events, inpanel_corrected_stratification)
        inpanel_corrected["nofill_error_rate_noindel"] = \
            combined_noindel_panel_site_rate(inpanel_corrected_stratification)

    denominator_note = (
        "RETRACTED (2026-08-05, round 2): the '2.74x FILL_GAPS methodology "
        "artifact' reported after round 1 of this investigation was itself "
        "a misdiagnosis. It was Bug A the whole time -- compare_gvcf_truth.py "
        "only trusted a literal INFO/END= field for a truth record's span, "
        "which our own maf-to-gvcf-converter output never writes on variant "
        "records (only smm477's production gVCFs do), so every deletion in "
        "every truth gVCF we built ourselves collapsed to 1bp regardless of "
        "any gap-filling setting. Directly confirmed: smm477's gVCFs were "
        "never gap-filled at all (a fresh no-fill rebuild of Il14H from a "
        "shared MAF is byte-identical to smm477's original). Post-fix, "
        "exclusion rates are comparable everywhere without any FILL_GAPS "
        "distinction. The planned panel rebuild (Phase 3) was skipped as a "
        "result -- there is no independent second alignment lineage "
        "available for the 24 real founders (their truth gVCFs were already "
        "inputs to the panel's own construction), so re-scoring them is "
        "tautological regardless of which same-lineage truth file is used. "
        "See the plan file's 'Outcome' section for the full story."
    )

    round3_note = (
        "2026-08-06, ROUND 3: the user asked a precise, falsifiable question -- "
        "when scoring a truth ref-block/indel, is a partial disagreement (e.g. "
        "99/100bp correct) counted as 1bp of error or the whole block? Verified "
        "directly on real data: it is per-panel-site partial credit, NOT "
        "whole-block -- a real 246bp truth ref-block with exactly 1 real "
        "disagreement scores compared_sites=12, gt_allele_mismatches=1, not "
        "246. No comparator bug exists here. But two real, previously "
        "undocumented effects fully explain why the panel-site error rate "
        "'feels too high': (a) panel sites are only 5.41% of the genome "
        "(115,354,779 / 2,131,846,805bp) and exist ONLY where some founder "
        "differs from B73 -- an adversarially-selected hardest subset, not a "
        "uniform sample (random founder choice already scores ~43% error on "
        "this subset). (b) ~35% of compared sites are synthetic per-base "
        "positions inside a truth deletion's interior, so one missed large "
        "structural variant counts as one mismatch per interior site -- a "
        "single ~21kb deletion can cost 1,000+ counted errors for one "
        "biological wrong call. Two new, complementary metrics added below: "
        "whole_genome_error_rate (mismatches / true genome size -- an UPPER "
        "BOUND on correctness, not a direct measurement, since held-out "
        "samples can carry private variants at non-panel positions no "
        "founder path can express) and event_error_rate (contiguous "
        "mismatch RUNS counted once each, rather than once per base -- "
        "matches still counted per-site, since there's no comparator trap "
        "in a long correct stretch the way there is in a long wrong one; an "
        "earlier version collapsed BOTH match and mismatch runs into "
        "'events' and used their ratio, which converges to ~50% regardless "
        "of quality since alternating runs always split ~50/50 in COUNT -- "
        "caught before publishing, see the plan file). Neither replaces the "
        "panel-site rate; both are additional, differently-scoped views of "
        "the SAME underlying model performance."
    )

    out = {"aggregate": agg, "assemblies": rows, "inpanel_comparison": inpanel_agg,
           "inpanel_mechanism": inpanel_mechanism,
           "inpanel_stratification_TAUTOLOGICAL": inpanel_stratification,
           "denominator_note": denominator_note,
           "round3_note": round3_note,
           "genome_size": GENOME_SIZE,
           "inpanel_corrected": inpanel_corrected,
           "inpanel_corrected_mechanism": inpanel_corrected_mechanism,
           "inpanel_corrected_stratification": inpanel_corrected_stratification}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(agg, indent=2))
    if inpanel_agg:
        print(f"\nFor contrast, in-panel mean error rate: "
              f"{inpanel_agg['mean_error_rate']:.4%} "
              f"(this held-out mean: {agg['mean_error_rate']:.4%})")
    if inpanel_corrected:
        print(f"\nMethodology-matched in-panel baseline (Il14H, no-fill-gaps, "
              f"genuinely independent lineage): {inpanel_corrected['nofill_error_rate']:.4%} "
              f"-- true fold-change: {inpanel_corrected['corrected_fold_change']:.2f}x")
    if agg.get("mean_whole_genome_error_rate") is not None:
        print(f"\nROUND 3 -- whole-genome-weighted mean (upper bound): "
              f"{agg['mean_whole_genome_error_rate']:.4%} "
              f"(vs. panel-site mean {agg['mean_error_rate']:.4%})")
    if agg.get("mean_event_error_rate") is not None:
        print(f"ROUND 3 -- event-level mean (independent wrong decisions, not bases): "
              f"{agg['mean_event_error_rate']:.4%}")
    if inpanel_corrected and inpanel_corrected.get("events", {}).get("event_error_rate") is not None:
        print(f"ROUND 3 -- Il14H no-fill event-level: "
              f"{inpanel_corrected['events']['event_error_rate']:.4%}")


if __name__ == "__main__":
    main()
