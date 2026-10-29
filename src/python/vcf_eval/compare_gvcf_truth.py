#!/usr/bin/env python3
"""
gVCF-aware truth comparison for held-out-assembly evaluation.

Compares a dense imputed VCF (e.g. GRITS founder-path BED composed through
`bed-to-vcf` against a founder reference panel) against a held-out assembly's
own truth gVCF, WITHOUT converting the gVCF to a dense VCF and WITHOUT
treating positions absent from the truth gVCF as reference.

At each imputed site, the truth record covering that position (if any) is
PROJECTED into the panel's own representation before comparing -- literal
alleles for SNPs, `<DEL>`/`<INS>` for indels -- rather than compared to the
imputed side's REF/ALT/GT at all. This mirrors exactly what `merge-gvcfs`
itself does when it builds a founder's own column in the panel VCF from that
founder's own gVCF (reverse-engineered from `javap -c` on
`biokotlin.util.MergeGVCFUtilsKt`, and independently verified this session at
5-nines agreement against real founder gVCF/panel-column pairs -- see
`scripts/verify_panel_projection.py` and
`/home/zrm22/.claude/plans/ok-we-need-to-squishy-lovelace.md`). Truth's own
selected allele A (from its single haploid GT), against its own record REF R,
projects as:
  A == R (GT index 0, incl. <NON_REF> blocks)     -> panel-space REF here
  len(A) == len(R)   (SNP; MNP never observed)     -> literal allele A
  len(A) >  len(R)   (insertion)                   -> `<INS>` (anchor only)
  len(A) <  len(R)   (deletion), A == R[:len(A)]   -> REF at the anchor (POS),
                                                       `<DEL>` at every
                                                       interior position
                                                       (POS, END]
  len(A) <  len(R)   (deletion), A != R[:len(A)]   -> `<DEL>` for the WHOLE
                                                       span [POS, END]
                                                       INCLUDING the anchor --
                                                       a complex
                                                       substitution+deletion
                                                       where even the
                                                       retained base differs
                                                       from R; only observed
                                                       on very large (>100kb)
                                                       records, confirmed 3/3
                                                       on real data
  R or A contains 'N' (assembly gap/masked base)   -> no opinion at all, not
                                                       even the anchor (merge-
                                                       gvcfs leaves the
                                                       sample "." there)

This is a genuine correction, not merely a workaround: an EARLIER version of
this module tried to patch the exact-position/deletion-interior cases
one-by-one by looking at what the IMPUTED side predicted (see git history /
the plan file's "Bug B"/"Finding 4" sections) -- that approach was
representation-asymmetric (truth compared as literal, imputed as symbolic,
reconciled ad hoc) and never handled insertions or complex-deletion anchors
correctly. Projecting truth into panel space UNCONDITIONALLY, independent of
whatever the imputed side happens to predict, is simpler and correct by
construction: both sides of every comparison now speak the panel's own
representation.

`END=` handling: this module's own truth-record span used to trust ONLY a
literal `END=` field in the truth gVCF's INFO column. Our own
`maf-to-gvcf-converter` output does NOT write `END=` on variant records
(only on `<NON_REF>` ref blocks) -- confirmed this session -- so every
variant record's span silently collapsed to 1bp and its true interior fell
into "no truth info", regardless of representation. `end = pos + len(ref) - 1`
(maxed against a literal `END=` when present, so gVCFs that DO carry it,
e.g. smm477's production founder gVCFs, still work) fixes this for both
lineages.

This is deliberately different from vcf_eval/accuracy.py's --missing-as-ref,
which assumes absence-from-imputed means reference; here absence-from-TRUTH
must never be assumed to mean reference. Allele-comparison, MAF-binning and
partial-credit scoring are reused directly from vcf_eval.accuracy so both
scripts stay consistent.

Also reports the "matchable ceiling": among truth VARIANT sites, how many
coincide with a site the imputed VCF actually carries (i.e. some founder in
the panel has that allele) vs. sites no founder can represent -- this
separates model error from founder-panel incompleteness.

Usage:
  PYTHONPATH=src python src/python/vcf_eval/compare_gvcf_truth.py \\
      --truth-gvcf Oh43.g.vcf --imputed-vcf imputed.vcf.gz \\
      --sample Oh43 --truth-ploidy-expand 2 --partial-credit
"""

import argparse
import gzip
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, TextIO, Tuple

from python.vcf_eval.accuracy import (
    allele_multiset_score,
    extract_allele_frequency,
    gt_to_allele_multiset,
    initialize_counts,
    sort_tsv,
    update_frequency_bins,
    write_query_tsv,
)

NON_REF = "<NON_REF>"
SYMBOLIC_DEL = "<DEL>"
SYMBOLIC_INS = "<INS>"


def classify_alleles(alleles: Tuple[str, ...], ref: str) -> str:
    """Classify a resolved allele multiset (as returned by
    gt_to_allele_multiset) into one variant class, relative to `ref` (the
    REF string the multiset was built against -- always the imputed/panel
    site's own REF, per this module's convention).

    HOMREF   -- every allele == ref
    SNP      -- every non-ref allele is a literal base string the same
                length as ref (this project's data has no real MNPs; a
                literal allele of equal length is treated as SNP-like)
    INS      -- every non-ref allele is the symbolic <INS>, or a literal
                allele longer than ref
    DEL      -- every non-ref allele is the symbolic <DEL>, or a literal
                allele shorter than ref
    HET_MIXED -- non-ref alleles disagree in class (e.g. one INS, one DEL) --
                only possible on the IMPUTED side in principle (this
                project's truth is always a single haploid GT duplicated to
                diploid, so truth multisets are always homogeneous); kept
                for correctness/generality, not expected to fire on truth.
    """
    classes = set()
    for a in alleles:
        if a == ref:
            classes.add("REF")
        elif a == SYMBOLIC_INS:
            classes.add("INS")
        elif a == SYMBOLIC_DEL:
            classes.add("DEL")
        elif len(a) > len(ref):
            classes.add("INS")
        elif len(a) < len(ref):
            classes.add("DEL")
        else:
            classes.add("SNP")

    non_ref = classes - {"REF"}
    if not non_ref:
        return "HOMREF"
    if len(non_ref) == 1:
        return next(iter(non_ref))
    return "HET_MIXED"


class TruthRecord(NamedTuple):
    chrom: str
    pos: int
    end: int
    ref: str
    alt: str
    gt: str
    is_ref_block: bool


def _open_maybe_gz(path: str) -> TextIO:
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def iter_truth_records(path: str) -> Iterator[TruthRecord]:
    """Stream a (g)VCF's data records in file order. Assumes position-sorted
    per contig, as PHGv2-produced gVCFs are by construction."""
    with _open_maybe_gz(path) as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue
            chrom, pos_s, _id, ref, alt, _qual, _filt, info = parts[0:8]
            gt = parts[9].split(":", 1)[0]
            try:
                pos = int(pos_s)
            except ValueError:
                continue
            # Default span from REF length (correct for gVCFs that never
            # write END= on variant records, e.g. our own maf-to-gvcf-converter
            # output); a literal END= (e.g. smm477's production gVCFs, or any
            # <NON_REF> ref block) can only extend it, never shrink it below
            # what REF's own length implies.
            end = pos + len(ref) - 1
            for field in info.split(";"):
                if field.startswith("END="):
                    try:
                        end = max(end, int(field[4:]))
                    except ValueError:
                        pass
                    break
            is_ref_block = alt == NON_REF
            yield TruthRecord(chrom, pos, end, ref, alt, gt, is_ref_block)


class TruthCursor:
    """Forward-only cursor over a truth gVCF's records for the CURRENT
    contig, resolving truth state at increasing query positions and
    tracking which truth VARIANT records were ever matched by an imputed
    site (for the matchable-ceiling stat)."""

    def __init__(self, records: Iterator[TruthRecord]):
        self._it = records
        self._cur: Optional[TruthRecord] = None
        self._cur_matched = False
        self.matchable_variant_sites = 0
        self.unmatchable_variant_sites = 0
        self._advance()

    def _advance(self) -> None:
        try:
            self._cur = next(self._it)
        except StopIteration:
            self._cur = None
        self._cur_matched = False

    def _expire_current(self) -> None:
        if self._cur is not None and not self._cur.is_ref_block:
            if self._cur_matched:
                self.matchable_variant_sites += 1
            else:
                self.unmatchable_variant_sites += 1
        self._advance()

    def resolve(self, chrom: str, pos: int,
                truth_ploidy_expand: int) -> Optional[Tuple[str, str]]:
        """Return (alt_field, gt_field) usable with gt_to_allele_multiset
        (paired with the IMPUTED side's own REF, which is always the correct
        reference base at this exact position -- both sides are anchored to
        the same B73 coordinate system), or None if this position carries no
        truth information at all.

        Unconditionally projects the covering truth record into the panel's
        own representation -- see module docstring for the full rule and
        why this replaced the earlier imputed-side-dependent patches.

        NOTE: a TruthCursor is now always scoped to a SINGLE contig (see
        `compare()`'s per-contig partitioning) -- there is deliberately no
        cross-contig catch-up/skip logic here anymore. An earlier version
        tried to handle contig transitions inline using plain Python string
        comparison on contig names (`self._cur.chrom < chrom`), which is
        WRONG for double-digit chromosome numbers: "chr9" > "chr10" as a
        string even though chr9 comes first genomically. Combined with real
        gVCFs not necessarily storing contigs in numeric file order (PHGv2's
        MAFToGVCF writes them in whatever order its sequence dictionary
        iterates, observed in practice as chr1, chr10, chr2, chr3, ..., chr9
        for one real file), that logic caused an entire out-of-numeric-order
        contig (chr10) to be silently discarded before its queries ever
        arrived, then permanently reported as 100% "no truth info" --
        confirmed empirically on real data. Splitting the truth gVCF into
        per-contig streams up front removes the need for this comparison
        entirely; every TruthCursor here only ever sees positions from the
        one contig it was built for, so the two guards below (originally
        meant to catch a contig mismatch) will not trigger.
        """
        while self._cur is not None and self._cur.chrom == chrom and self._cur.end < pos:
            self._expire_current()

        if self._cur is None or self._cur.chrom != chrom or self._cur.pos > pos:
            return None  # gap: no truth record covers this position

        rec = self._cur
        gt = rec.gt
        n = max(truth_ploidy_expand, 1)

        if gt in (".", ""):
            return None  # explicit no-call: this record has no opinion here

        if "/" in gt or "|" in gt:
            # Already-multi-allelic truth GT -- not expected for these
            # haploid single-assembly gVCFs, but guard rather than
            # misclassify: compare literally, only at the record's own
            # exact position.
            if rec.pos == pos:
                self._cur_matched = True
                return rec.alt, gt
            return None

        idx = int(gt)
        if idx == 0:
            # Hom-ref here -- true for every <NON_REF> ref block, and for any
            # real variant record whose GT happens to select index 0.
            self._cur_matched = True
            return NON_REF, "/".join(["0"] * n)

        alts = [] if rec.alt in {".", ""} else rec.alt.split(",")
        if idx - 1 >= len(alts):
            return None
        allele = alts[idx - 1]
        if allele == NON_REF:
            return None  # shouldn't occur at idx > 0, guard rather than misclassify

        ref = rec.ref
        if "N" in ref or "N" in allele:
            return None  # assembly gap/masked base -- no confident call either direction

        if len(allele) == len(ref):
            # SNP (MNP never observed in practice) -- single position only
            if rec.pos != pos:
                return None
            self._cur_matched = True
            return allele, "/".join(["1"] * n)

        if len(allele) > len(ref):
            # insertion -- anchor position only
            if rec.pos != pos:
                return None
            self._cur_matched = True
            return SYMBOLIC_INS, "/".join(["1"] * n)

        # deletion: "clean" (retained prefix matches REF's own prefix) ->
        # anchor reads REF, interior reads <DEL>. NOT clean (the retained
        # base itself differs from REF -- a complex substitution+deletion,
        # only observed on very large >100kb records) -> <DEL> for the
        # WHOLE span, anchor included.
        clean = allele == ref[: len(allele)]
        if clean and rec.pos == pos:
            self._cur_matched = True
            return NON_REF, "/".join(["0"] * n)
        self._cur_matched = True
        return SYMBOLIC_DEL, "/".join(["1"] * n)

    def flush(self) -> None:
        """Expire all remaining records so trailing variant records are
        counted in the matchable-ceiling stats. Call once after the last
        query position has been resolved."""
        while self._cur is not None:
            self._expire_current()


def partition_truth_by_contig(truth_gvcf: str, tmp_dir: Path) -> Dict[str, Path]:
    """One streaming pass through the truth gVCF, writing each record's raw
    line into a PER-CONTIG temp file (tmp_dir/<contig>.truth.tsv).

    This is what makes the comparison correct regardless of the truth
    file's own on-disk contig order (see TruthCursor.resolve()'s docstring
    for the chr9/chr10 bug this replaces): each contig's records are still
    written out in the same relative (position-sorted) order they appeared
    in the source file, just routed into their own stream so a later
    per-contig TruthCursor never has to reason about OTHER contigs at all.
    """
    handles: Dict[str, TextIO] = {}
    with _open_maybe_gz(truth_gvcf) as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            chrom = line.split("\t", 1)[0]
            fh = handles.get(chrom)
            if fh is None:
                fh = open(tmp_dir / f"{chrom}.truth.tsv", "w")
                handles[chrom] = fh
            fh.write(line if line.endswith("\n") else line + "\n")
    for fh in handles.values():
        fh.close()
    return {c: tmp_dir / f"{c}.truth.tsv" for c in handles}


def next_or_none(it):
    try:
        return next(it)
    except StopIteration:
        return None


def iter_imputed_tsv(path: str) -> Iterator[Tuple[str, int, str, str, str, str]]:
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            chrom, pos_s, ref, alt, info, gt = parts[:6]
            try:
                pos = int(pos_s)
            except ValueError:
                continue
            yield chrom, pos, ref, alt, info, gt


def compare(imputed_tsv: str, truth_gvcf: str, truth_ploidy_expand: int,
            phase_sensitive: bool, partial_credit: bool) -> Dict:
    counts = initialize_counts()
    counts["excluded_no_info"] = 0

    with tempfile.TemporaryDirectory() as td:
        contig_files = partition_truth_by_contig(truth_gvcf, Path(td))
        cursors: Dict[str, TruthCursor] = {}

        def get_cursor(chrom: str) -> TruthCursor:
            cur = cursors.get(chrom)
            if cur is None:
                path = contig_files.get(chrom)
                records = iter_truth_records(str(path)) if path is not None else iter(())
                cur = TruthCursor(records)
                cursors[chrom] = cur
            return cur

        for chrom, pos, ref, alt, info, gt in iter_imputed_tsv(imputed_tsv):
            counts["imputed_records"] += 1
            truth_cursor = get_cursor(chrom)
            resolved = truth_cursor.resolve(chrom, pos, truth_ploidy_expand)
            if resolved is None:
                counts["excluded_no_info"] += 1
                continue
            t_alt, t_gt = resolved
            counts["truth_records"] += 1
            counts["site_key_matches"] += 1

            t_alleles = gt_to_allele_multiset(ref, t_alt, t_gt, phase_sensitive=phase_sensitive)
            i_alleles = gt_to_allele_multiset(ref, alt, gt, phase_sensitive=phase_sensitive)

            if t_alleles is None:
                counts["truth_missing_gt"] += 1
                continue
            if i_alleles is None:
                counts["imputed_missing_gt"] += 1
                continue

            counts["compared_sites"] += 1
            ac, an = extract_allele_frequency(info)
            # extract_allele_frequency reads AC/AN from the IMPUTED site's INFO
            # (the panel VCF carries these); fall back gracefully if absent.
            if ac is not None and an is not None and an > 0:
                update_frequency_bins(t_alleles, i_alleles, ac, an, alt, counts,
                                       partial_credit, phase_sensitive)

            if partial_credit:
                counts["partial_credit_sum"] += allele_multiset_score(
                    t_alleles, i_alleles, phase_sensitive)

            if t_alleles == i_alleles:
                counts["gt_allele_matches"] += 1
            else:
                counts["gt_allele_mismatches"] += 1

        matchable_total = 0
        unmatchable_total = 0
        for truth_cursor in cursors.values():
            truth_cursor.flush()
            matchable_total += truth_cursor.matchable_variant_sites
            unmatchable_total += truth_cursor.unmatchable_variant_sites
        counts["matchable_variant_sites"] = matchable_total
        counts["unmatchable_variant_sites"] = unmatchable_total

    return counts


def print_report(counts: Dict) -> None:
    print("gVCF-aware truth compare (imputed site vs truth variant/ref-block/no-info)")
    for k in ["imputed_records", "truth_records", "site_key_matches",
              "excluded_no_info", "truth_missing_gt", "imputed_missing_gt",
              "compared_sites", "gt_allele_matches", "gt_allele_mismatches"]:
        print(f"  {k:24s} {counts[k]}")

    if counts["compared_sites"] > 0:
        concord = counts["gt_allele_matches"] / counts["compared_sites"]
        print(f"  allele_GT_concordance       {concord:.6f}")
        if counts["partial_credit_sum"]:
            frac = counts["partial_credit_sum"] / counts["compared_sites"]
            print(f"  partial_allele_concordance  {frac:.6f}")
    else:
        print("  allele_GT_concordance       NA (no comparable sites)")

    mv = counts["matchable_variant_sites"]
    uv = counts["unmatchable_variant_sites"]
    total_v = mv + uv
    print("\nMATCHABLE CEILING (truth variant sites vs. founder panel coverage)")
    print(f"  matchable_variant_sites   {mv}")
    print(f"  unmatchable_variant_sites {uv}  (no founder in panel carries this allele)")
    if total_v > 0:
        print(f"  matchable_ceiling_fraction {mv / total_v:.6f}")
    else:
        print("  matchable_ceiling_fraction NA (no truth variant sites)")

    freq_bins = counts["af_bins"]
    print("\nACCURACY OF HOMOZYGOUS TRUTH SITES BY TRUTH ALLELE FREQUENCY\n")
    print("LOW\tHIGH\tACCURACY\tCOUNT")
    for i in range(len(freq_bins) - 1):
        n = counts["af_bin_total"][i]
        acc = counts["af_bin_correct"][i] / n if n else None
        print(freq_bins[i], freq_bins[i + 1], acc, n, sep="\t")

    print("\nR2 OF ALL TRUTH SITES BY TRUTH ALLELE FREQUENCY\n")
    print("LOW\tHIGH\tR2")
    for i in range(len(freq_bins) - 1):
        true_vals = counts["af_bin_true_alt_counts"][i]
        pred_vals = counts["af_bin_imp_alt_counts"][i]
        n = len(true_vals)
        if n > 1 and len(set(true_vals)) > 1 and len(set(pred_vals)) > 1:
            import numpy as np
            r2 = float(np.corrcoef(true_vals, pred_vals)[0, 1]) ** 2
        else:
            r2 = None
        print(freq_bins[i], freq_bins[i + 1] if i + 1 < len(freq_bins) else 1.0, r2, sep="\t")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imputed-vcf", required=True, help="Dense imputed VCF (.vcf/.vcf.gz)")
    ap.add_argument("--truth-gvcf", required=True,
                     help="Held-out assembly's truth gVCF (.g.vcf/.g.vcf.gz, PHGv2 format "
                          "with <NON_REF> reference blocks and END= INFO field)")
    ap.add_argument("--sample", required=True, help="Sample column to score in --imputed-vcf")
    ap.add_argument("--region", default=None, help="e.g. chr1:1000000-2000000")
    ap.add_argument("--truth-ploidy-expand", type=int, default=2,
                     help="Replicate the (haploid) truth GT this many times before comparing "
                          "(default 2: score a single held-out assembly as homozygous diploid, "
                          "matching a diploid imputed call)")
    ap.add_argument("--phase-sensitive", action="store_true")
    ap.add_argument("--partial-credit", action="store_true")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw = str(td / "imputed.raw.tsv")
        sorted_tsv = str(td / "imputed.sorted.tsv")
        write_query_tsv(args.imputed_vcf, raw, args.sample, args.region)
        sort_tsv(raw, sorted_tsv)
        counts = compare(sorted_tsv, args.truth_gvcf, args.truth_ploidy_expand,
                          args.phase_sensitive, args.partial_credit)

    print_report(counts)


if __name__ == "__main__":
    main()
