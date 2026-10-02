#!/usr/bin/env python3
"""
Dual-TruthCursor genotype comparator for individuals with two genuinely
independent haplotype truth gVCFs (hybrid/RIL rows in the maize simulated-
validation corpus, where truth_h1 != truth_h2).

Why not just merge the two truth gVCFs into one diploid VCF and reuse
compare_gvcf_truth.py unmodified? Because TruthCursor.resolve() (in that
module) does UNCONDITIONAL panel-space projection keyed on a SINGLE-ALLELE
GT at a SPECIFIC record's own position -- it is not a general-purpose VCF
GT reader. Feeding it a genuinely diploid GT (e.g. "0/1") hits its
literal-passthrough guard (the "already-multi-allelic truth GT" branch),
which only compares at the record's own POS, never spans the record's
END. That silently breaks every ref-block (collapses a whole-chromosome
block to matching at 1bp only), every deletion interior, and every indel
(compares literal truth ALT against the imputed side's symbolic <INS>/
<DEL>, guaranteed mismatch).

The fix used here is simpler and exactly correct: run TWO independent,
single-haplotype TruthCursors -- one per truth file -- side by side. Each
one only ever sees a single-allele GT, so it always takes the SAME
projection path compare_gvcf_truth.py's own validated single-cursor use
already exercises (--truth-ploidy-expand=1 semantics, per-cursor). The two
per-haplotype allele calls are combined into a diploid multiset AFTER
projection, at the caller level, mirroring exactly what
--truth-ploidy-expand=2 does when duplicating one file -- which is also
this module's own self-check (assert_equivalence_h1_eq_h2 below).

Reused, unmodified, from compare_gvcf_truth.py (imported as `cgt`):
TruthCursor, iter_truth_records, partition_truth_by_contig,
iter_imputed_tsv. Reused from vcf_eval.accuracy: gt_to_allele_multiset,
allele_multiset_score, extract_allele_frequency, initialize_counts,
sort_tsv, write_query_tsv, update_frequency_bins.

Usage:
  python compare_gvcf_truth_diploid.py \\
      --truth-gvcf-h1 parentA.g.vcf.gz --truth-gvcf-h2 parentB.g.vcf.gz \\
      --imputed-vcf imputed.vcf.gz --sample <name> --partial-credit

Points at the grits-perfscore-worktree checkout (branch
perf-allele-multiset-score, not yet merged -- PR pending) so this pipeline
picks up the allele_multiset_score Counter-overhead fix. Switch back to
CRF_REPO_ROOT = .../test_crf_relatedness once that branch is merged.
"""

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict

CRF_REPO_ROOT = Path("/local/workdir/zrm22/HackathonJun2026/grits-perfscore-worktree")
CRF_SRC = CRF_REPO_ROOT / "src"
sys.path.insert(0, str(CRF_SRC))

from python.vcf_eval import compare_gvcf_truth as cgt  # noqa: E402
from python.vcf_eval.accuracy import (  # noqa: E402
    allele_multiset_score,
    extract_allele_frequency,
    gt_to_allele_multiset,
    initialize_counts,
    sort_tsv,
    update_frequency_bins,
    write_query_tsv,
)


def _get_cursor(chrom: str, contig_files: Dict[str, Path], cache: Dict[str, "cgt.TruthCursor"],
                 cursor_cls=None):
    cursor_cls = cursor_cls or cgt.TruthCursor
    cur = cache.get(chrom)
    if cur is None:
        path = contig_files.get(chrom)
        records = cgt.iter_truth_records(str(path)) if path is not None else iter(())
        cur = cursor_cls(records)
        cache[chrom] = cur
    return cur


#: Event-level metrics (one unit per truth VARIANT EVENT, per haplotype --
#: NOT per panel site). Added to answer a follow-up to the chain-error
#: investigation: the headline error_rate is deletion-span-weighted (see
#: memory/simval_full_corpus_eval.md's "What counts as one error" section --
#: a single 1.16Mb deletion can cost 96,537 scored units), which is a
#: legitimate "fraction of the genome reconstructed" measure but not the
#: "how many of this individual's real variants did we get right" measure a
#: variant-caller audience expects. This adds that second measure, computed
#: in the SAME pass as the existing site metric (both use the identical
#: comparator loop and cursors), so the two are guaranteed comparable and
#: the recomputed site metric doubles as a correctness check on this pass.
EVENT_CLASSES = ("SNP", "INS", "DEL")


def classify_truth_record(rec: "cgt.TruthRecord"):
    """Classify a truth RECORD (not a query position) as one variant EVENT
    (SNP/INS/DEL) or None if it is not an event -- a <NON_REF> ref block, a
    GT that selects the reference allele (index 0) or is a no-call, or an
    allele/ref containing 'N' (assembly gap). Mirrors the same branching
    TruthCursor.resolve() uses to decide allele identity (see that
    function's docstring for the full rule), but classifies once per RECORD,
    independent of query position -- this is what makes "one event" possible
    at all, since resolve() itself only ever answers per-position queries.
    Deliberately does not attempt the "already-multi-allelic GT" guard
    branch resolve() has (not expected on these haploid single-assembly
    gVCFs; if it ever fires, this returns None, which undercounts events by
    the same guard resolve() already applies -- consistent, not silently
    wrong)."""
    if rec.is_ref_block:
        return None
    gt = rec.gt
    if gt in (".", "") or "/" in gt or "|" in gt:
        return None
    try:
        idx = int(gt)
    except ValueError:
        return None
    if idx == 0:
        return None
    alts = [] if rec.alt in {".", ""} else rec.alt.split(",")
    if idx - 1 >= len(alts):
        return None
    allele = alts[idx - 1]
    if allele == cgt.NON_REF:
        return None
    ref = rec.ref
    if "N" in ref or "N" in allele:
        return None
    if len(allele) == len(ref):
        return "SNP"
    if len(allele) > len(ref):
        return "INS"
    return "DEL"


class EventTruthCursor(cgt.TruthCursor):
    """A TruthCursor that ALSO tallies per-event (not per-site) outcomes as
    records expire, in addition to the base class's per-site matchable-
    ceiling bookkeeping (which it leaves untouched, via super() calls).

    One record = one event (see classify_truth_record). As the base class's
    _expire_current() finalizes a record when the query cursor moves past
    its span, this subclass finalizes that same record's event outcome
    first (reading self._cur before super() mutates it), from however many
    of the record's compared sites note_site() was told were correct.

    "Strict" event correctness requires EVERY compared site inside the
    event's span to be correct on this haplotype; "fractional" is the mean
    of correct/compared per event -- reported alongside because strict is
    harsh on very long deletions (one bad boundary window fails a
    96,000-site event)."""

    def __init__(self, records):
        self.event_stats = {k: defaultdict(int) for k in (
            "events_total", "events_observable", "events_unobservable",
            "events_correct_strict", "events_wrong_strict")}
        self.event_stats["events_fractional_sum"] = defaultdict(float)
        self._cur_event_class = None
        self._cur_compared_sites = 0
        self._cur_correct_sites = 0
        super().__init__(records)  # calls self._advance() (overridden below)

    def _advance(self) -> None:
        super()._advance()
        self._cur_event_class = (
            classify_truth_record(self._cur) if self._cur is not None else None)
        self._cur_compared_sites = 0
        self._cur_correct_sites = 0

    def _finalize_event(self) -> None:
        cls = self._cur_event_class
        if cls is None:
            return
        self.event_stats["events_total"][cls] += 1
        if self._cur_compared_sites == 0:
            self.event_stats["events_unobservable"][cls] += 1
            return
        self.event_stats["events_observable"][cls] += 1
        if self._cur_correct_sites == self._cur_compared_sites:
            self.event_stats["events_correct_strict"][cls] += 1
        else:
            self.event_stats["events_wrong_strict"][cls] += 1
        self.event_stats["events_fractional_sum"][cls] += (
            self._cur_correct_sites / self._cur_compared_sites)

    def _expire_current(self) -> None:
        self._finalize_event()  # self._cur still the about-to-expire record
        super()._expire_current()  # base bookkeeping, then self._advance()

    def note_site(self, correct: bool) -> None:
        """Called once per compared site, per haplotype, by the caller (who
        alone knows whether THIS site was actually compared -- excluded
        sites never reach here). No-op when the currently active record
        isn't a variant event."""
        if self._cur_event_class is not None:
            self._cur_compared_sites += 1
            if correct:
                self._cur_correct_sites += 1


def _aggregate_event_stats(cursors: Dict[str, "cgt.TruthCursor"]) -> Dict[str, Dict[str, float]]:
    agg = {k: defaultdict(int) for k in (
        "events_total", "events_observable", "events_unobservable",
        "events_correct_strict", "events_wrong_strict")}
    agg["events_fractional_sum"] = defaultdict(float)
    for cur in cursors.values():
        stats = getattr(cur, "event_stats", None)
        if stats is None:
            continue
        for key, per_class in stats.items():
            for cls, v in per_class.items():
                agg[key][cls] += v
    return agg


#: Per-variant-class breakdown (Diagnostic 3, plan section on stratified
#: error): classify_alleles() (compare_gvcf_truth.py) was written but never
#: called anywhere -- wiring it up here, not in the shared module, since
#: only this diploid comparator needs it for this investigation. Classified
#: by the TRUTH side's own multiset (HOMREF/SNP/INS/DEL/HET_MIXED, see that
#: function's docstring) so "how much of this row's error is deletion-span
#: sites vs. SNP sites" is answerable directly, independent of what the
#: imputed side happened to predict.
VARIANT_CLASSES = ("HOMREF", "SNP", "INS", "DEL", "HET_MIXED")

#: SNP + RefCall re-scoring (plan: snp-refcall-rescore-simval-0.1x). A site
#: scores iff BOTH the truth side and the imputed side classify_alleles() to
#: HOMREF or SNP -- i.e. neither side actually CALLED an indel there. This is
#: a call-based filter, not a panel-shape filter: see the plan's "Why a
#: panel-shape filter is wrong" section for why filtering on the panel
#: record's ALT list instead would silently reintroduce the bias this is
#: meant to remove.
SCORED_CLASSES = ("HOMREF", "SNP")


def compare_diploid(imputed_tsv: str, truth_h1: str, truth_h2: str,
                     phase_sensitive: bool = False, partial_credit: bool = False,
                     class_breakdown: bool = False, event_metrics: bool = False,
                     snp_refcall_metrics: bool = False) -> Dict:
    counts = initialize_counts()
    counts["excluded_no_info"] = 0
    counts["excluded_partial_info"] = 0  # one haplotype had truth info, the other didn't
    if class_breakdown:
        for cls in VARIANT_CLASSES:
            counts[f"class_total_{cls}"] = 0
            counts[f"class_mismatch_{cls}"] = 0
            counts[f"class_partial_sum_{cls}"] = 0.0
    if snp_refcall_metrics:
        counts["snprc_compared_sites"] = 0
        counts["snprc_partial_credit_sum"] = 0.0
        counts["snprc_gt_allele_matches"] = 0
        counts["snprc_gt_allele_mismatches"] = 0
    if event_metrics:
        counts["haplotype_correct_sum"] = 0
        counts["false_positive_runs"] = 0
        fp_active, fp_chrom = False, None
    cursor_cls = EventTruthCursor if event_metrics else cgt.TruthCursor

    same_source = str(truth_h1) == str(truth_h2)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dir1 = td / "h1"
        dir1.mkdir()
        contig_files_1 = cgt.partition_truth_by_contig(str(truth_h1), dir1)
        if same_source:
            # Every inbred row: partitioning once and reading it via two
            # independent cursor caches is exactly equivalent to (and half
            # the I/O of) partitioning twice, since the two cursors never
            # share mutable state -- each get_cursor() call opens its own
            # fresh iter_truth_records() generator.
            contig_files_2 = contig_files_1
        else:
            dir2 = td / "h2"
            dir2.mkdir()
            contig_files_2 = cgt.partition_truth_by_contig(str(truth_h2), dir2)

        cursors1: Dict[str, "cgt.TruthCursor"] = {}
        cursors2: Dict[str, "cgt.TruthCursor"] = {}

        # TruthCursor.resolve() is forward-only per contig -- it already
        # silently assumes non-decreasing pos within a contig (whether or not
        # main() skipped the external sort). One dict lookup/site turns a
        # violation of that assumption into a loud, immediate failure instead
        # of silent wrong "no truth info" results for the out-of-order tail.
        _last_pos: Dict[str, int] = {}

        for chrom, pos, ref, alt, info, gt in cgt.iter_imputed_tsv(imputed_tsv):
            last = _last_pos.get(chrom, -1)
            if pos < last:
                raise RuntimeError(
                    f"imputed_tsv is not position-sorted within {chrom}: "
                    f"saw {pos} after {last}. Comparator requires coordinate-"
                    f"sorted input (see main()'s indexed-input sort skip).")
            _last_pos[chrom] = pos
            counts["imputed_records"] += 1
            c1 = _get_cursor(chrom, contig_files_1, cursors1, cursor_cls)
            c2 = _get_cursor(chrom, contig_files_2, cursors2, cursor_cls)
            r1 = c1.resolve(chrom, pos, 1)
            r2 = c2.resolve(chrom, pos, 1)

            if r1 is None or r2 is None:
                counts["excluded_no_info"] += 1
                if (r1 is None) != (r2 is None):
                    counts["excluded_partial_info"] += 1
                continue

            t1_alt, t1_gt = r1
            t2_alt, t2_gt = r2
            counts["truth_records"] += 1
            counts["site_key_matches"] += 1

            a1 = gt_to_allele_multiset(ref, t1_alt, t1_gt, phase_sensitive=phase_sensitive)
            a2 = gt_to_allele_multiset(ref, t2_alt, t2_gt, phase_sensitive=phase_sensitive)
            if a1 is None or a2 is None:
                counts["truth_missing_gt"] += 1
                continue
            t_alleles = (a1 + a2) if phase_sensitive else tuple(sorted(a1 + a2))

            i_alleles = gt_to_allele_multiset(ref, alt, gt, phase_sensitive=phase_sensitive)
            if i_alleles is None:
                counts["imputed_missing_gt"] += 1
                continue

            counts["compared_sites"] += 1
            ac, an = extract_allele_frequency(info)
            if ac is not None and an is not None and an > 0:
                update_frequency_bins(t_alleles, i_alleles, ac, an, alt, counts,
                                       partial_credit, phase_sensitive)

            # Hoisted out of the class_breakdown/snp_refcall_metrics blocks
            # below: at whole-genome scale (~160M compared sites) these two
            # pure functions of (t_alleles, i_alleles, ref, phase_sensitive)
            # were each being recomputed 2-3x per site for IDENTICAL inputs
            # (classify_alleles(t_alleles, ref) once for class_breakdown's
            # `cls` and again for snp_refcall_metrics' `t_cls`;
            # allele_multiset_score once for partial_credit_sum, again for
            # class_partial_sum, again for snprc_partial_credit_sum) --
            # confirmed via cProfile as the dominant real cost of this
            # comparator (the per-site Python loop, not bcftools/sort).
            # Values are identical either way since both functions are pure.
            score = None
            if partial_credit:
                score = allele_multiset_score(t_alleles, i_alleles, phase_sensitive)
                counts["partial_credit_sum"] += score

            if event_metrics:
                # Per-haplotype site correctness, derived the same way
                # allele_multiset_score's multiset-intersection logic would,
                # just resolved per haplotype instead of pooled -- x/y are
                # each haplotype's own single truth allele (a1/a2 before
                # pooling into t_alleles), i_alleles is the imputed pair.
                # Phase-insensitive by construction (same as everywhere else
                # in this comparator): x==y in i_alleles is a set/count
                # membership test, not a positional one.
                x, y = a1[0], a2[0]
                if x == y:
                    n = sum(1 for v in i_alleles if v == x)
                    h1_ok, h2_ok = n >= 1, n >= 2
                else:
                    h1_ok, h2_ok = (x in i_alleles), (y in i_alleles)
                c1.note_site(h1_ok)
                c2.note_site(h2_ok)
                counts["haplotype_correct_sum"] += int(h1_ok) + int(h2_ok)

                # False-positive run tracking (approximate -- see module
                # docstring / plan): a maximal run of consecutive COMPARED
                # sites, same chrom, where truth is homozygous-reference on
                # both haplotypes but the imputed call is non-reference.
                truth_homref = all(v == ref for v in t_alleles)
                imputed_nonref = any(v != ref for v in i_alleles)
                if truth_homref and imputed_nonref:
                    if not (fp_active and fp_chrom == chrom):
                        counts["false_positive_runs"] += 1
                    fp_active, fp_chrom = True, chrom
                else:
                    fp_active, fp_chrom = False, None

            is_match = t_alleles == i_alleles
            if is_match:
                counts["gt_allele_matches"] += 1
            else:
                counts["gt_allele_mismatches"] += 1

            t_cls = None
            if class_breakdown or snp_refcall_metrics:
                t_cls = cgt.classify_alleles(t_alleles, ref)

            if class_breakdown:
                cls = t_cls
                counts[f"class_total_{cls}"] = counts.get(f"class_total_{cls}", 0) + 1
                if not is_match:
                    counts[f"class_mismatch_{cls}"] = counts.get(f"class_mismatch_{cls}", 0) + 1
                if partial_credit:
                    counts[f"class_partial_sum_{cls}"] = counts.get(
                        f"class_partial_sum_{cls}", 0.0) + score

            if snp_refcall_metrics:
                i_cls = cgt.classify_alleles(i_alleles, ref)
                counts[f"pairclass_{t_cls}_{i_cls}"] = counts.get(
                    f"pairclass_{t_cls}_{i_cls}", 0) + 1
                if t_cls in SCORED_CLASSES and i_cls in SCORED_CLASSES:
                    counts["snprc_compared_sites"] += 1
                    counts["snprc_partial_credit_sum"] += (
                        score if score is not None
                        else allele_multiset_score(t_alleles, i_alleles, phase_sensitive))
                    counts["snprc_gt_allele_matches" if is_match
                           else "snprc_gt_allele_mismatches"] += 1
                    counts[f"snprc_class_total_{t_cls}"] = counts.get(
                        f"snprc_class_total_{t_cls}", 0) + 1
                    if not is_match:
                        counts[f"snprc_class_mismatch_{t_cls}"] = counts.get(
                            f"snprc_class_mismatch_{t_cls}", 0) + 1

        def _flush_totals(cursors):
            m = u = 0
            for cur in cursors.values():
                cur.flush()
                m += cur.matchable_variant_sites
                u += cur.unmatchable_variant_sites
            return m, u

        m1, u1 = _flush_totals(cursors1)
        if same_source and not event_metrics:
            # Micro-optimization: m2/u2 are mathematically identical to
            # m1/u1 here (cursors2 replays the SAME file independently, so
            # flushing it would produce the same matchable/unmatchable
            # counts) -- skip the redundant flush(). NOT safe under
            # event_metrics: flush() also finalizes each cursor's OWN
            # trailing per-contig event (via _expire_current), which is a
            # real side effect on cursors2.event_stats, not just a count to
            # borrow -- skipping it there would leave h2's last event per
            # contig unfinalized, silently breaking the h1==h2 event-count
            # invariant this mode relies on for inbred rows. Always flush
            # both independently when event_metrics is on.
            m2, u2 = m1, u1
        else:
            m2, u2 = _flush_totals(cursors2)
        counts["matchable_variant_sites_h1"] = m1
        counts["unmatchable_variant_sites_h1"] = u1
        counts["matchable_variant_sites_h2"] = m2
        counts["unmatchable_variant_sites_h2"] = u2
        counts["matchable_variant_sites"] = m1 + m2
        counts["unmatchable_variant_sites"] = u1 + u2

        if event_metrics:
            h1_agg = _aggregate_event_stats(cursors1)
            h2_agg = _aggregate_event_stats(cursors2)
            for key in h1_agg:
                for cls in EVENT_CLASSES:
                    counts[f"{key}_{cls}"] = h1_agg[key].get(cls, 0) + h2_agg[key].get(cls, 0)
            # Per-haplotype breakdown, kept separately (not summed away above)
            # for the IDX-INBRED structural check: h1 and h2 event totals
            # must be IDENTICAL per class when truth_h1==truth_h2, since the
            # two cursors replay the same file independently -- any
            # divergence would mean per-haplotype bookkeeping crossed wires.
            counts["h1_events_total"] = {cls: h1_agg["events_total"].get(cls, 0)
                                          for cls in EVENT_CLASSES}
            counts["h2_events_total"] = {cls: h2_agg["events_total"].get(cls, 0)
                                          for cls in EVENT_CLASSES}

    return counts


def assert_equivalence_h1_eq_h2(imputed_tsv: str, truth_path: str,
                                 phase_sensitive: bool = False,
                                 partial_credit: bool = True) -> Dict:
    """Correctness self-check (run on the pilot, not every row): when
    truth_h1 == truth_h2, this dual-cursor comparator must reproduce
    compare_gvcf_truth.compare(..., truth_ploidy_expand=2) EXACTLY. Returns
    an empty dict if every checked count matches; otherwise {key: (ours,
    theirs)} for every mismatch."""
    a = compare_diploid(imputed_tsv, truth_path, truth_path, phase_sensitive, partial_credit)
    b = cgt.compare(imputed_tsv, truth_path, truth_ploidy_expand=2,
                     phase_sensitive=phase_sensitive, partial_credit=partial_credit)
    keys = ["imputed_records", "truth_records", "site_key_matches", "excluded_no_info",
            "truth_missing_gt", "imputed_missing_gt", "compared_sites",
            "gt_allele_matches", "gt_allele_mismatches", "partial_credit_sum"]
    return {k: (a[k], b[k]) for k in keys if a[k] != b[k]}


def print_report(counts: Dict) -> None:
    print("gVCF-aware DIPLOID truth compare (dual-cursor, two independent haplotype truths)")
    for k in ["imputed_records", "truth_records", "site_key_matches",
              "excluded_no_info", "excluded_partial_info", "truth_missing_gt",
              "imputed_missing_gt", "compared_sites", "gt_allele_matches",
              "gt_allele_mismatches"]:
        print(f"  {k:24s} {counts.get(k, 0)}")

    if counts["compared_sites"] > 0:
        concord = counts["gt_allele_matches"] / counts["compared_sites"]
        # Same key/format as compare_gvcf_truth.py's print_report -- a single
        # regex in simval_eval_one.py parses either comparator's output.
        print(f"  allele_GT_concordance       {concord:.6f}")
        if counts["partial_credit_sum"]:
            frac = counts["partial_credit_sum"] / counts["compared_sites"]
            print(f"  partial_allele_concordance  {frac:.6f}")
    else:
        print("  allele_GT_concordance       NA (no comparable sites)")

    print(f"\n  matchable_variant_sites_h1   {counts['matchable_variant_sites_h1']}")
    print(f"  unmatchable_variant_sites_h1 {counts['unmatchable_variant_sites_h1']}")
    print(f"  matchable_variant_sites_h2   {counts['matchable_variant_sites_h2']}")
    print(f"  unmatchable_variant_sites_h2 {counts['unmatchable_variant_sites_h2']}")

    if any(f"class_total_{cls}" in counts for cls in VARIANT_CLASSES):
        print("\n  per-variant-class breakdown (classified by the TRUTH side's own multiset):")
        print(f"    {'class':10s} {'total':>12s} {'mismatches':>12s} {'error%':>8s} "
              f"{'share of all mismatches':>24s}")
        total_mismatch = counts.get("gt_allele_mismatches", 0) or 1
        for cls in VARIANT_CLASSES:
            tot = counts.get(f"class_total_{cls}", 0)
            mis = counts.get(f"class_mismatch_{cls}", 0)
            err_pct = 100 * mis / tot if tot else 0.0
            share = 100 * mis / total_mismatch
            print(f"    {cls:10s} {tot:>12d} {mis:>12d} {err_pct:>7.3f}% {share:>23.3f}%")

    if any(f"events_total_{cls}" in counts for cls in EVENT_CLASSES):
        print("\n  EVENT-LEVEL metrics (one unit per truth VARIANT EVENT, per haplotype --"
              " NOT per panel site; is_phased=False throughout, see module docstring)")
        tot_total = sum(counts.get(f"events_total_{c}", 0) for c in EVENT_CLASSES)
        tot_obs = sum(counts.get(f"events_observable_{c}", 0) for c in EVENT_CLASSES)
        tot_unobs = sum(counts.get(f"events_unobservable_{c}", 0) for c in EVENT_CLASSES)
        tot_correct = sum(counts.get(f"events_correct_strict_{c}", 0) for c in EVENT_CLASSES)
        tot_frac = sum(counts.get(f"events_fractional_sum_{c}", 0) for c in EVENT_CLASSES)
        print(f"    events_total={tot_total}  events_observable={tot_obs}  "
              f"events_unobservable={tot_unobs}  (no panel site fell in this event's span)")
        if tot_obs > 0:
            strict_err = 1.0 - tot_correct / tot_obs
            frac_err = 1.0 - tot_frac / tot_obs
            print(f"    event_error_rate_strict      {strict_err:.6f}"
                  "  (ANY compared site inside the event wrong -> whole event wrong)")
            print(f"    event_error_rate_fractional  {frac_err:.6f}"
                  "  (mean of correct/compared sites per event)")
        else:
            print("    event_error_rate_strict/fractional  NA (no observable events)")

        print(f"    {'class':6s} {'total':>10s} {'observable':>11s} {'unobservable':>13s} "
              f"{'correct_strict':>15s} {'err_strict%':>12s}")
        for c in EVENT_CLASSES:
            tot = counts.get(f"events_total_{c}", 0)
            obs = counts.get(f"events_observable_{c}", 0)
            unobs = counts.get(f"events_unobservable_{c}", 0)
            corr = counts.get(f"events_correct_strict_{c}", 0)
            err = 100 * (1 - corr / obs) if obs else 0.0
            print(f"    {c:6s} {tot:>10d} {obs:>11d} {unobs:>13d} {corr:>15d} {err:>11.3f}%")

        if "false_positive_runs" in counts:
            print(f"    false_positive_runs (approximate)  {counts['false_positive_runs']}")

        if counts.get("compared_sites"):
            invariant_delta = counts.get("haplotype_correct_sum", 0) - 2 * counts.get(
                "partial_credit_sum", 0.0)
            print(f"    invariant check (haplotype_correct_sum - 2*partial_credit_sum): "
                  f"{invariant_delta:.6f}  (should be ~0)")

        h1t, h2t = counts.get("h1_events_total"), counts.get("h2_events_total")
        if h1t is not None and h2t is not None:
            print(f"    h1_events_total {h1t}  h2_events_total {h2t}"
                  "  (identical iff truth_h1==truth_h2, i.e. an inbred row)")

    if "snprc_compared_sites" in counts:
        print("\n  SNP + RefCall metrics (sites where BOTH truth and imputed classify to "
              "{HOMREF, SNP} -- any site where either side called an indel is skipped)")
        snprc_compared = counts["snprc_compared_sites"]
        print(f"    snprc_compared_sites        {snprc_compared}  "
              f"(of {counts.get('compared_sites', 0)} all-sites compared)")
        if snprc_compared > 0:
            snprc_concord = counts["snprc_gt_allele_matches"] / snprc_compared
            print(f"    snprc_error_rate             {1.0 - snprc_concord:.6f}")
            snprc_partial = counts["snprc_partial_credit_sum"] / snprc_compared
            print(f"    snprc_partial_error_rate     {1.0 - snprc_partial:.6f}")
        else:
            print("    snprc_error_rate/partial_error_rate  NA (no comparable sites)")
        for cls in SCORED_CLASSES:
            tot = counts.get(f"snprc_class_total_{cls}", 0)
            mis = counts.get(f"snprc_class_mismatch_{cls}", 0)
            err_pct = 100 * mis / tot if tot else 0.0
            print(f"    snprc_class_total_{cls:6s}   {tot:>12d}   mismatches {mis:>12d}   "
                  f"error% {err_pct:>7.3f}")
        print("    pairclass_<truth>_<imputed> drop-reason table:")
        for key in sorted(k for k in counts if k.startswith("pairclass_")):
            print(f"      {key:28s} {counts[key]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imputed-vcf", required=True)
    ap.add_argument("--truth-gvcf-h1", required=True)
    ap.add_argument("--truth-gvcf-h2", required=True)
    ap.add_argument("--sample", required=True, help="Sample column to score in --imputed-vcf")
    ap.add_argument("--region", default=None)
    ap.add_argument("--phase-sensitive", action="store_true")
    ap.add_argument("--partial-credit", action="store_true")
    ap.add_argument("--class-breakdown", action="store_true",
                     help="also report the per-variant-class (SNP/INS/DEL/HOMREF) error breakdown")
    ap.add_argument("--event-metrics", action="store_true",
                     help="also report EVENT-level accuracy (one unit per truth variant event, "
                          "per haplotype, not per panel site) alongside the site-level metric")
    ap.add_argument("--snp-refcall-metrics", action="store_true",
                     help="also report a SNP+RefCall-only error rate: a site is skipped if "
                          "either the truth or imputed side classify_alleles() to INS/DEL/"
                          "HET_MIXED (i.e. either side actually called an indel there)")
    ap.add_argument("--json-out", default=None,
                     help="write the full counts dict as JSON to this path (for driver scripts "
                          "consuming event-metrics output; avoids stdout regex parsing)")
    args = ap.parse_args()

    if args.snp_refcall_metrics and args.event_metrics:
        ap.error("--snp-refcall-metrics and --event-metrics cannot be combined: "
                  "EventTruthCursor.note_site is driven off the scored (all-sites) path, "
                  "so gating it on the SNP+RefCall subset would silently distort event "
                  "observability")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw = str(td / "imputed.raw.tsv")
        write_query_tsv(args.imputed_vcf, raw, args.sample, args.region)

        # bcftools query streams records in the source VCF's own file order.
        # A bgzipped VCF with a .tbi/.csi sidecar is REQUIRED to be coordinate-
        # sorted (tabix/bcftools index refuse to index an unsorted file), so
        # for every indexed input in this corpus the external re-sort below
        # is pure redundant I/O over the full genome-wide TSV (measured at
        # ~120s of a ~2000-2400s row here -- not the dominant cost, but real
        # and free to skip). Fall back to the safe sort for any unindexed
        # input (e.g. a plain text .vcf test fixture) where this guarantee
        # doesn't hold. compare_diploid()'s own position-monotonicity guard
        # (below) makes this self-verifying rather than a silent assumption.
        indexed = (Path(args.imputed_vcf + ".tbi").exists()
                   or Path(args.imputed_vcf + ".csi").exists())
        if indexed:
            imputed_tsv = raw
        else:
            imputed_tsv = str(td / "imputed.sorted.tsv")
            sort_tsv(raw, imputed_tsv)

        counts = compare_diploid(imputed_tsv, args.truth_gvcf_h1, args.truth_gvcf_h2,
                                  args.phase_sensitive, args.partial_credit,
                                  class_breakdown=args.class_breakdown,
                                  event_metrics=args.event_metrics,
                                  snp_refcall_metrics=args.snp_refcall_metrics)

    print_report(counts)

    if args.json_out:
        # initialize_counts() (vcf_eval/accuracy.py, shared/unmodified) also
        # accumulates af_bin_true_alt_counts/af_bin_imp_alt_counts -- one raw
        # value APPENDED PER COMPARED SITE, for that module's own R^2-by-
        # frequency-bin reporting. At whole-genome scale (~160M compared
        # sites) that serializes to ~2GB of JSON per row -- confirmed
        # directly on the IDX-INBRED/B73 smoke run. None of the new
        # event/site metrics here use those two fields (or the never-
        # populated "examples" list) -- excluded from the JSON dump, not
        # from `counts` itself, so print_report()'s own R^2 section above
        # (which DOES read them) is unaffected.
        _BULKY_KEYS = {"examples", "af_bin_true_alt_counts", "af_bin_imp_alt_counts"}
        counts_for_json = {k: v for k, v in counts.items() if k not in _BULKY_KEYS}
        Path(args.json_out).write_text(json.dumps(counts_for_json, indent=1))


if __name__ == "__main__":
    main()
