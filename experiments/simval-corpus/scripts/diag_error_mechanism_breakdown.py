#!/usr/bin/env python
"""
Genome-wide breakdown of WHY each mismatch happens, not just that it does.

Prompted by the held-out generalization result (mean 16.63% error vs. 0.72%
in-panel) looking suspiciously high. Two narrow (chr1, single-window)
`bcftools`-loop spot-checks this session found real signal in two different
directions -- reference-calling bias at truth-variant sites (Tx303: 44% of
sampled mismatches), and false-positive non-ref calls at truth-ref-block
sites (also Tx303: 12.8%) -- but a second sample (EP1) showed much weaker
versions of both despite a WORSE overall error rate, meaning the narrow
spot-checks can't say which mechanism actually dominates each sample's real,
genome-wide number. This script answers that properly: every comparable
site, genome-wide, all 10 chromosomes, bucketed into:

  MATCH_REFBLOCK           truth says hom-ref here, imputed agrees
  MATCH_VARIANT            truth is a real variant, imputed matches it
  REFBLOCK_FALSE_POSITIVE  truth says hom-ref, imputed predicted non-ref
                           (a pure false positive -- REF is always a valid,
                           always-available answer, so this is never a
                           panel-completeness issue, only a model one)
  VARIANT_REF_BIAS         truth is a real variant, the panel's own ALT
                           list DID include the true allele, but imputed
                           predicted REF anyway (the "hedges toward
                           reference" pattern from the spot-checks)
  VARIANT_OTHER_WRONG      truth is a real variant, the panel's ALT list DID
                           include the true allele, but imputed predicted a
                           DIFFERENT (non-ref, still-wrong) allele
  VARIANT_UNMATCHABLE      truth is a real variant and the panel's own
                           REF+ALT set at this exact site does NOT include
                           the true allele at all -- structurally impossible
                           for ANY founder-path choice, regardless of model
                           quality. NOTE: this is a stricter, deeper check
                           than compare_gvcf_truth.py's own "matchable
                           ceiling" (which only checks whether the panel has
                           ANY record at this position at all, not whether
                           that record's allele set contains the truth
                           allele) -- a 200-site spot-check this session
                           found zero instances of this category, so it's
                           expected to be rare, but tracked here for
                           completeness.
  NO_PANEL_RECORD          no imputed record at this exact position at all
                           (equivalent to compare_gvcf_truth.py's own
                           excluded_no_info from the OTHER direction -- a
                           truth variant with nothing in the panel to even
                           compare against). Cross-check: summed across a
                           full run this should reconcile with
                           unmatchable_variant_sites in the standard report.

Reuses compare_gvcf_truth.py's own TruthCursor / iter_truth_records /
partition_truth_by_contig / iter_imputed_tsv machinery UNCHANGED -- this is
deliberately not a reimplementation, so there's no risk of the diagnostic
disagreeing with the real report due to a second, subtly different
comparison path. The one thing it needs that the base module doesn't expose
is which underlying truth record (ref-block vs. real variant) a resolved
comparison came from -- pulled from TruthCursor's own `_cur.is_ref_block`
right after calling `resolve()` (before the cursor can advance past it),
which is accessing the class's "private" state but is the same object this
script imports, not a fork of it.

STRATIFICATION BY VARIANT CLASS (added to directly answer: "does SNP
accuracy actually improve over the overall rate, the way it should since
SNPs are widely shared across haplotypes?"). Every compared site's TRUTH
allele is additionally classified via compare_gvcf_truth.classify_alleles()
into HOMREF / SNP / INS / DEL / HET_MIXED, and each site's outcome (MATCH /
FALSE_POSITIVE / REF_BIAS / OTHER_WRONG / UNMATCHABLE) is tallied per class
-- printed as "STRAT <class> <outcome> <count>" lines. Keying on the TRUTH
class (not the imputed side) puts HOMREF in its own row structurally: ~50%+
of genome-wide compared sites are hom-ref-truth, so pooling classes naively
(e.g. "does the imputed REF/ALT look SNP-shaped") would let those trivially
easy sites swamp any real SNP-vs-indel signal. A second table, "CONFUSION
<truth_class> <imputed_class> <count>", cross-tabulates truth class against
whatever class the model's own prediction resolves to -- this is what
actually shows whether the model's mistakes at real SNP sites are
concentrated in a particular class of wrong answer.

EVENT-LEVEL (contiguous-run) TRACKING (added after a user pushed back on the
18.6% held-out mean feeling too high, and asked specifically whether a truth
ref-block/indel is scored as a whole-block match/mismatch or per-position --
see the plan file's "ROUND 3" section). Verified directly: compare_gvcf_truth
already gives full per-panel-site partial credit, no whole-block bug exists.
But ~35% of all compared sites are synthetic per-base positions inside a
truth deletion's interior (by design, mirroring merge-gvcfs' own panel
representation -- established in Phase 0/1), so ONE missed large structural
variant is counted as one mismatch PER interior panel site it covers -- a
single ~21kb deletion can cost >1,000 counted "errors" for what is
biologically one wrong call. This adds an alternative unit: collapse
consecutive contiguous MISMATCHING sites (in the ordered stream of
actually-compared sites; excluded/missing-GT sites don't break continuity --
they simply have no outcome to break with) into one "event" per run, giving
EVENT_MISMATCH (a count of independent wrong DECISIONS, not wrong bases).

IMPORTANT: the denominator for the resulting rate is EVENT_MISMATCH +
gt_allele_matches (raw per-site match count), NOT EVENT_MISMATCH +
"match-event count". An earlier version of this also collapsed MATCH runs
into match-events and used (EVENT_MISMATCH + match_events) as the
denominator -- that is a real, near-invisible statistical trap: in any
alternating match/mismatch sequence, the number of match RUNS and mismatch
RUNS are always within 1 of each other BY CONSTRUCTION (runs alternate), no
matter how long each run actually is. That made the "collapsed" rate
converge to ~50% regardless of true quality (verified: Tx303 chr1 gave
49.9999% under that formula) -- a broken metric, caught only by cross-
checking against an independent Explore-agent computation before trusting
it. Matches should stay counted per-site (there's no reason to discount 13M
individually-correct trivial predictions down to a handful of "runs" the way
there IS reason to discount one biologically-single wrong SV call counted
thousands of times) -- only mismatches get collapsed. Printed as
"EVENT_MISMATCH" (and diagnostic-only "EVENT_TOTAL", the raw alternating-run
count, kept for the mean-run-length sanity check, NOT used in the final rate).

ROUND 4: "what do the error rates look like if we ignore indels?" -- a
second, parallel run-tracker ("EVENT_MISMATCH_NOINDEL") applies the exact
same event-collapsing logic but restricted to truth_class in {HOMREF, SNP}.
An indel-class site (INS/DEL/HET_MIXED) is SKIPPED entirely by this second
tracker -- it neither starts nor breaks a run, the same treatment the main
tracker already gives excluded/missing-GT sites -- so an indel site sitting
between two SNP mismatches does not artificially split them into two
events. The matching denominator needs no new tracking: it's already
available as by_class["HOMREF"]["MATCH"] + by_class["SNP"]["MATCH"].

Usage:
    python diag_error_mechanism_breakdown.py --imputed-vcf V --truth-gvcf G
        --sample NAME [--truth-ploidy-expand 2]
"""
import argparse
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "test_crf_relatedness" / "src"))
# Also try the direct known path in case relative layout differs
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src")

from python.vcf_eval.accuracy import gt_to_allele_multiset, write_query_tsv, sort_tsv  # noqa: E402
from python.vcf_eval.compare_gvcf_truth import (  # noqa: E402
    TruthCursor,
    classify_alleles,
    iter_imputed_tsv,
    iter_truth_records,
    partition_truth_by_contig,
)

TRUTH_CLASSES = ["HOMREF", "SNP", "INS", "DEL", "HET_MIXED"]
OUTCOMES = ["MATCH", "FALSE_POSITIVE", "REF_BIAS", "OTHER_WRONG", "UNMATCHABLE"]


def breakdown(imputed_tsv, truth_gvcf, truth_ploidy_expand):
    counts = {
        "MATCH_REFBLOCK": 0, "MATCH_VARIANT": 0,
        "REFBLOCK_FALSE_POSITIVE": 0,
        "VARIANT_REF_BIAS": 0, "VARIANT_OTHER_WRONG": 0, "VARIANT_UNMATCHABLE": 0,
        "NO_PANEL_RECORD": 0,
        # cross-check totals, same fields compare_gvcf_truth.py itself reports
        "imputed_records": 0, "excluded_no_info": 0, "compared_sites": 0,
        "gt_allele_matches": 0, "gt_allele_mismatches": 0,
        "truth_missing_gt": 0, "imputed_missing_gt": 0,
        # variant-class stratification: counts["by_class"][truth_class][outcome]
        "by_class": defaultdict(lambda: defaultdict(int)),
        # confusion matrix: counts["confusion"][truth_class][imputed_class]
        "confusion": defaultdict(lambda: defaultdict(int)),
        # event-level (contiguous-run) tracking -- see module docstring
        "EVENT_TOTAL": 0, "EVENT_MISMATCH": 0,
        # ROUND 4: same event tracking, but restricted to non-indel truth
        # classes (HOMREF/SNP) -- indel-class sites are skipped entirely
        # (don't update this second tracker's run state), same treatment as
        # excluded/missing-GT sites already get for the main tracker, so an
        # indel site sitting between two SNP mismatches does NOT split them
        # into separate events here.
        "EVENT_MISMATCH_NOINDEL": 0,
    }
    run_chrom = None
    run_is_match = None
    run_chrom_noindel = None
    run_is_match_noindel = None

    with tempfile.TemporaryDirectory() as td:
        contig_files = partition_truth_by_contig(truth_gvcf, Path(td))
        cursors = {}

        def get_cursor(chrom):
            cur = cursors.get(chrom)
            if cur is None:
                path = contig_files.get(chrom)
                records = iter_truth_records(str(path)) if path is not None else iter(())
                cur = TruthCursor(records)
                cursors[chrom] = cur
            return cur

        for chrom, pos, ref, alt, info, gt in iter_imputed_tsv(imputed_tsv):
            counts["imputed_records"] += 1
            cur = get_cursor(chrom)
            resolved = cur.resolve(chrom, pos, truth_ploidy_expand)
            if resolved is None:
                counts["excluded_no_info"] += 1
                counts["NO_PANEL_RECORD"] += 1
                continue

            t_alt, t_gt = resolved
            is_ref_block = cur._cur is not None and cur._cur.is_ref_block  # see module docstring

            t_alleles = gt_to_allele_multiset(ref, t_alt, t_gt)
            i_alleles = gt_to_allele_multiset(ref, alt, gt)
            if t_alleles is None:
                counts["truth_missing_gt"] += 1
                continue
            if i_alleles is None:
                counts["imputed_missing_gt"] += 1
                continue

            counts["compared_sites"] += 1
            is_match = t_alleles == i_alleles
            if is_match:
                counts["gt_allele_matches"] += 1
            else:
                counts["gt_allele_mismatches"] += 1

            truth_class = classify_alleles(t_alleles, ref)
            imputed_class = classify_alleles(i_alleles, ref)
            counts["confusion"][truth_class][imputed_class] += 1

            # Event-level tracking: a new "event" starts whenever the
            # outcome changes from the previous ACTUALLY-COMPARED site (or
            # we've moved to a new contig) -- excluded/missing-GT sites are
            # invisible here, they don't break a run since they were never
            # part of the compared-site sequence to begin with.
            if chrom != run_chrom or is_match != run_is_match:
                counts["EVENT_TOTAL"] += 1
                if not is_match:
                    counts["EVENT_MISMATCH"] += 1
                run_chrom, run_is_match = chrom, is_match

            # ROUND 4: same tracking, restricted to non-indel truth classes.
            # An indel-class site (INS/DEL/HET_MIXED) is skipped entirely --
            # it neither starts nor breaks a run here, exactly like an
            # excluded site is invisible to the main tracker above.
            if truth_class in ("HOMREF", "SNP"):
                if chrom != run_chrom_noindel or is_match != run_is_match_noindel:
                    if not is_match:
                        counts["EVENT_MISMATCH_NOINDEL"] += 1
                    run_chrom_noindel, run_is_match_noindel = chrom, is_match

            if is_ref_block:
                if is_match:
                    counts["MATCH_REFBLOCK"] += 1
                    counts["by_class"][truth_class]["MATCH"] += 1
                else:
                    counts["REFBLOCK_FALSE_POSITIVE"] += 1
                    counts["by_class"][truth_class]["FALSE_POSITIVE"] += 1
                continue

            # Real variant (exact-match or deletion-interior synthetic).
            if is_match:
                counts["MATCH_VARIANT"] += 1
                counts["by_class"][truth_class]["MATCH"] += 1
                continue

            panel_alts = [] if alt in {".", ""} else alt.split(",")
            panel_allele_set = {ref, *panel_alts}
            truth_available = all(a in panel_allele_set for a in t_alleles)
            ref_alleles = tuple([ref] * len(i_alleles)) if i_alleles else (ref,)
            predicted_ref = i_alleles == ref_alleles

            if not truth_available:
                counts["VARIANT_UNMATCHABLE"] += 1
                counts["by_class"][truth_class]["UNMATCHABLE"] += 1
            elif predicted_ref:
                counts["VARIANT_REF_BIAS"] += 1
                counts["by_class"][truth_class]["REF_BIAS"] += 1
            else:
                counts["VARIANT_OTHER_WRONG"] += 1
                counts["by_class"][truth_class]["OTHER_WRONG"] += 1

        for cur in cursors.values():
            cur.flush()

    return counts


def print_report(sample, counts):
    print(f"=== {sample} ===")
    for k in ["imputed_records", "excluded_no_info", "compared_sites",
              "gt_allele_matches", "gt_allele_mismatches",
              "truth_missing_gt", "imputed_missing_gt"]:
        print(f"  {k:26s} {counts[k]:>12,}")
    print()
    print("  --- match breakdown ---")
    for k in ["MATCH_REFBLOCK", "MATCH_VARIANT"]:
        print(f"  {k:26s} {counts[k]:>12,}")
    print("  --- mismatch mechanism breakdown ---")
    mismatch_total = counts["gt_allele_mismatches"]
    for k in ["REFBLOCK_FALSE_POSITIVE", "VARIANT_REF_BIAS",
              "VARIANT_OTHER_WRONG", "VARIANT_UNMATCHABLE"]:
        pct = f"({counts[k] / mismatch_total:.1%} of all mismatches)" if mismatch_total else ""
        print(f"  {k:26s} {counts[k]:>12,}  {pct}")
    print(f"  NO_PANEL_RECORD (excluded, not a mismatch) {counts['NO_PANEL_RECORD']:>12,}")

    # cross-check against the standard report's own arithmetic
    computed_concordance = (counts["gt_allele_matches"] / counts["compared_sites"]
                             if counts["compared_sites"] else None)
    print(f"\n  allele_GT_concordance (should match compare_gvcf_truth.py's own report "
          f"for this sample): {computed_concordance}")
    print()

    print("  --- event-level (contiguous-run) tracking ---")
    print(f"  EVENT_TOTAL               {counts['EVENT_TOTAL']:>12,}  "
          f"(diagnostic only -- all alternating runs, NOT used in the rate below)")
    print(f"  EVENT_MISMATCH            {counts['EVENT_MISMATCH']:>12,}  "
          f"(independent wrong-decision events)")
    event_denom = counts["EVENT_MISMATCH"] + counts["gt_allele_matches"]
    event_error_rate = counts["EVENT_MISMATCH"] / event_denom if event_denom else None
    if event_error_rate is not None:
        print(f"  event_error_rate          {event_error_rate:.6f}  "
              f"(= EVENT_MISMATCH / (EVENT_MISMATCH + gt_allele_matches) -- "
              f"mismatches counted once per contiguous run, matches counted per-site)")
    if counts["EVENT_MISMATCH"]:
        mean_mismatch_run_len = counts["gt_allele_mismatches"] / counts["EVENT_MISMATCH"]
        print(f"  mean_mismatch_run_length  {mean_mismatch_run_len:.2f}  "
              f"(gt_allele_mismatches / EVENT_MISMATCH)")
    print()

    print("  --- ROUND 4: event-level tracking, indels excluded (truth class in {HOMREF, SNP}) ---")
    print(f"  EVENT_MISMATCH_NOINDEL    {counts['EVENT_MISMATCH_NOINDEL']:>12,}")
    noindel_matches = (counts["by_class"].get("HOMREF", {}).get("MATCH", 0)
                        + counts["by_class"].get("SNP", {}).get("MATCH", 0))
    noindel_denom = counts["EVENT_MISMATCH_NOINDEL"] + noindel_matches
    event_error_rate_noindel = (counts["EVENT_MISMATCH_NOINDEL"] / noindel_denom
                                 if noindel_denom else None)
    if event_error_rate_noindel is not None:
        print(f"  event_error_rate_noindel  {event_error_rate_noindel:.6f}  "
              f"(= EVENT_MISMATCH_NOINDEL / (EVENT_MISMATCH_NOINDEL + "
              f"by_class[HOMREF][MATCH] + by_class[SNP][MATCH]))")
    print()

    print("  --- variant-class stratification (truth class x outcome) ---")
    print("  STRAT_CLASS   OUTCOME          COUNT        ERROR_RATE (of this class's own sites)")
    for cls in TRUTH_CLASSES:
        row = counts["by_class"].get(cls, {})
        cls_total = sum(row.values())
        if cls_total == 0:
            continue
        for outcome in OUTCOMES:
            n = row.get(outcome, 0)
            if n == 0 and outcome not in ("MATCH",):
                continue
            print(f"  STRAT {cls:10s} {outcome:16s} {n:>12,}")
        cls_matches = row.get("MATCH", 0)
        cls_error_rate = 1.0 - (cls_matches / cls_total)
        print(f"  STRAT_TOTAL {cls:10s} {'(all outcomes)':16s} {cls_total:>12,}  "
              f"error_rate={cls_error_rate:.6f}")
    print()

    print("  --- confusion matrix (truth class x imputed prediction's own class) ---")
    print("  CONFUSION_HEADER  " + "  ".join(f"{c:>10s}" for c in TRUTH_CLASSES))
    for truth_cls in TRUTH_CLASSES:
        row = counts["confusion"].get(truth_cls, {})
        if not row:
            continue
        cells = "  ".join(f"{row.get(imp_cls, 0):>10,}" for imp_cls in TRUTH_CLASSES)
        print(f"  CONFUSION {truth_cls:10s} " + cells)
        for imp_cls in TRUTH_CLASSES:
            n = row.get(imp_cls, 0)
            if n:
                print(f"  CONFUSION_CELL {truth_cls} {imp_cls} {n}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imputed-vcf", required=True)
    ap.add_argument("--truth-gvcf", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--truth-ploidy-expand", type=int, default=2)
    ap.add_argument("--region", default=None)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw = str(td / "imputed.raw.tsv")
        sorted_tsv = str(td / "imputed.sorted.tsv")
        write_query_tsv(args.imputed_vcf, raw, args.sample, args.region)
        sort_tsv(raw, sorted_tsv)
        counts = breakdown(sorted_tsv, args.truth_gvcf, args.truth_ploidy_expand)

    print_report(args.sample, counts)


if __name__ == "__main__":
    main()
