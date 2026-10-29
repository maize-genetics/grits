#!/usr/bin/env python3
"""
Allele-aware GT comparison between two VCFs using bcftools query.

Rules implemented:
- Match sites by (CHROM, POS, REF). ALT may differ.
- Compare genotypes by mapping GT allele indices to actual allele strings (REF/ALT),
  then comparing the allele multiset (phase-insensitive by default).
- Treat missing genotypes (./., .|., .) as missing. Missing does not count as match/mismatch.
- Stream-friendly: can optionally write a per-site TSV for *every* site without storing in memory.

Requires:
- bcftools in PATH

Example:
  ./vcf_gt_allele_compare.py --truth truth.vcf.gz --imputed imp.vcf.gz -s SAMPLE \
    --out-sites allele_compare_sites.tsv

Notes:
- If your VCFs are very large, the --out-sites file can be huge.
  Use --only-mismatches or --only-matched-sites to reduce output.
"""

import argparse
import subprocess
import sys
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Iterator
from enum import Enum
from typing import NamedTuple
import numpy as np

class Key(NamedTuple):
    chr: str
    pos: int
    ref: str

class VariantData(NamedTuple):
    alt: str
    info: str
    gt: str

MISSING_GTS = {"./.", ".|.", "."}

class SiteKind(Enum):
    MATCH = "MATCH"
    MISMATCH_ALLELE = "MISMATCH_ALLELE"
    MISSING_IN_IMPUTED_SITE = "MISSING_IN_IMPUTED_SITE"
    EXTRA_IN_IMPUTED_SITE = "EXTRA_IN_IMPUTED_SITE"
    MISMATCH_GT = "MISMATCH_GT"
    TRUTH_MISSING_GT = "TRUTH_MISSING_GT"
    MATCH_ALLELE = "MATCH_ALLELE"
    BOTH_MISSING_GT = "BOTH_MISSING_GT"
    GT_UNPARSEABLE = "GT_UNPARSEABLE"
    IMPUTED_MISSING_GT = "IMPUTED_MISSING_GT"

def write_query_tsv(vcf: str, out_tsv: str, sample: Optional[str], region: Optional[str]) -> None:
    """
    Extract CHROM, POS, REF, ALT, INFO, GT from a VCF using bcftools query.
    ALT may include commas; we keep it as one field.
    """
    fmt = r"%CHROM\t%POS\t%REF\t%ALT\t%INFO\t[%GT]\n"
    cmd = ["bcftools", "query", "-f", fmt]
    if sample:
        cmd += ["-s", sample]
    if region:
        cmd += ["-r", region]
    cmd += [vcf]

    with open(out_tsv, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            sys.stderr.write(p.stderr)
            raise SystemExit(p.returncode)


def sort_tsv(in_tsv: str, out_tsv: str) -> None:
    """
    Sort by CHROM (version sort), POS (numeric), REF.
    """
    with open(out_tsv, "w", encoding="utf-8") as out_f:
        subprocess.run(
            ["sort", "-k1,1V", "-k2,2n", "-k3,3", in_tsv],
            env={**os.environ, "LC_ALL": "C"},
            stdout=out_f,
            check=True,
        )


def parse_gt(gt: str) -> Optional[List[str]]:
    """
    Return list of allele-index strings from GT (e.g., ['0','1']).
    Return None if missing/unknown.
    Supports diploid (0/1, 1|0) and haploid-like (0, 1) encodings.
    """
    gt = gt.strip()
    if not gt or gt in MISSING_GTS:
        return None

    # Determine separator
    sep = "/" if "/" in gt else ("|" if "|" in gt else None)
    if sep is None:
        # haploid-like genotype such as "0" or "1"
        if gt == ".":
            return None
        return [gt]

    parts = gt.split(sep)
    if any(p == "." or p == "" for p in parts):
        return None
    return parts


def gt_to_allele_multiset(
    ref: str,
    alt_field: str,
    gt: str,
    phase_sensitive: bool = False,
) -> Optional[Tuple[str, ...]]:
    """
    If phase_sensitive=True, preserve within-genotype order; otherwise sort alleles to ignore phase.
    Returns None for missing/unparseable GT.
    """
    gt_parts = parse_gt(gt)
    if gt_parts is None:
        return None

    alts = [] if alt_field in {".", ""} else alt_field.split(",")

    allele_strings: List[str] = []
    for a in gt_parts:
        try:
            idx = int(a)
        except ValueError:
            return None

        if idx == 0:
            allele_strings.append(ref)
        else:
            j = idx - 1
            if j < 0 or j >= len(alts):
                # GT refers to an ALT index that doesn't exist for this record
                return None
            allele_strings.append(alts[j])

    if phase_sensitive:
        return tuple(allele_strings)
    return tuple(sorted(allele_strings))


def allele_multiset_score(
        truth_alleles: Tuple[str, ...],
        imputed_alleles: Tuple[str, ...],
        phase_sensitive: bool = False
) -> float:
    """Return fractional concordance based on allele overlap.

    If phase_sensitive=True, compare alleles position by position.
    If phase_sensitive=False, use multiset intersection (current behavior).
    """
    if not truth_alleles:
        return 0.0

    if phase_sensitive:
        # Position-wise comparison
        matches = sum(1 for t, i in zip(truth_alleles, imputed_alleles) if t == i)
        return matches / float(len(truth_alleles))
    else:
        # Multiset intersection without Counter: genotypes are tiny (usually
        # ploidy 2-4), so a plain list scan avoids Counter's dict/ABC
        # construction overhead, which dominates this hot path at genome scale.
        remaining = list(imputed_alleles)
        inter = 0
        for a in truth_alleles:
            if a in remaining:
                remaining.remove(a)
                inter += 1
        return inter / float(len(truth_alleles))


def iter_records(path: str) -> Iterator[Tuple[Key, VariantData]]:
    """
    Yield (Key=(CHROM,POS,REF), VariantData=(ALT, INFO, GT)) from a sorted TSV.
    """
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            chrom, pos_s, ref, alt, info, gt = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            try:
                pos = int(pos_s)
            except ValueError:
                continue
            yield (chrom, pos, ref), (alt, info, gt)


def compare_sorted(
        truth_sorted: str,
        imputed_sorted: str,
        max_report: int,
        phase_sensitive: bool,
        partial_credit: bool,
        out_fh=None,
        only_mismatches: bool = False,
        only_matched_sites: bool = False,
        missing_as_ref: bool = False,
) -> Dict:
    """Main comparison function - orchestrates the comparison process."""

    # Initialize counters and bins
    counts = initialize_counts()

    # Create output writer
    writer = OutputWriter(out_fh, only_mismatches, only_matched_sites)

    # Stream through both files
    t_iter = iter_records(truth_sorted)
    i_iter = iter_records(imputed_sorted)

    t = next_or_none(t_iter)
    i = next_or_none(i_iter)

    while t is not None or i is not None:
        if t is not None and (i is None or t[0] < i[0]):
            # Truth site missing in imputed
            handle_missing_in_imputed(t, counts, writer, missing_as_ref, phase_sensitive, partial_credit)
            t = next_or_none(t_iter)

        elif i is not None and (t is None or i[0] < t[0]):
            # Extra site in imputed
            handle_extra_in_imputed(i, counts, writer, max_report)
            i = next_or_none(i_iter)

        else:
            # Sites match - this is where the main comparison happens
            handle_matched_sites(t, i, counts, writer, max_report, phase_sensitive, partial_credit, missing_as_ref)
            t = next_or_none(t_iter)
            i = next_or_none(i_iter)

    return counts


def initialize_counts() -> Dict:
    """Initialize the counts dictionary."""
    NBINS = 20  # freq_bins = [i / 20 for i in range(21)] gives 20 bins
    return {
        "truth_records": 0,
        "imputed_records": 0,
        "site_key_matches": 0,
        "missing_in_imputed_sites": 0,
        "extra_in_imputed_sites": 0,
        "both_missing_gt": 0,
        "truth_missing_gt": 0,
        "imputed_missing_gt": 0,
        "gt_unparseable": 0,
        "compared_sites": 0,
        "gt_allele_matches": 0,
        "gt_allele_mismatches": 0,
        "partial_credit_sum": 0.0,
        "examples": [],

        # Frequency bins - initialize here
        "af_bins": [i / 20 for i in range(21)],  # Move here too
        "af_bin_total": [0] * NBINS,
        "af_bin_correct": [0] * NBINS,

        # Heterozygous frequency bins
        "het_bin_total": [0] * NBINS,
        "het_bin_correct": [0] * NBINS,

        # R2 frequency bins
        "af_bin_true_alt_counts": [[] for _ in range(NBINS)],  # For R²
        "af_bin_imp_alt_counts": [[] for _ in range(NBINS)],  # For R²
    }


class OutputWriter:
    """Handles writing output with filtering logic."""

    def __init__(self, out_fh, only_mismatches: bool, only_matched_sites: bool):
        self.out_fh = out_fh
        self.only_mismatches = only_mismatches
        self.only_matched_sites = only_matched_sites

    def write_site(self, kind: SiteKind, key: Key, t_variant: VariantData, i_variant: VariantData):
        if not self.out_fh:
            return

        unmatched_sites = {SiteKind.MISSING_IN_IMPUTED_SITE, SiteKind.EXTRA_IN_IMPUTED_SITE}
        mismatch_sites = {SiteKind.MISMATCH_ALLELE, SiteKind.MISSING_IN_IMPUTED_SITE, SiteKind.EXTRA_IN_IMPUTED_SITE}

        if self.only_matched_sites and kind in unmatched_sites:
            return
        if self.only_mismatches and kind not in mismatch_sites:
            return

        chrom, pos, ref = key
        t_alt, t_info, t_gt = t_variant
        i_alt, i_info, i_gt = i_variant
        self.out_fh.write(
            f"{kind.value}\t{chrom}\t{pos}\t{ref}\t"
            f"{t_alt if t_alt is not None else '.'}\t{t_info if t_info is not None else '.'}\t{t_gt if t_gt is not None else '.'}\t"
            f"{i_alt if i_alt is not None else '.'}\t{i_info if i_info is not None else '.'}\t{i_gt if i_gt is not None else '.'}\n"
        )


def handle_missing_in_imputed(t, counts, writer, missing_as_ref, phase_sensitive, partial_credit):
    """Handle sites present in truth but missing in imputed."""
    key, t_variant = t
    t_alt, t_info, t_gt = t_variant
    i_variant = (None, None, None)
    counts["truth_records"] += 1

    if missing_as_ref:
        # Treat missing site as reference genotype
        truth_parts = parse_gt(t_gt)
        if truth_parts is None:
            counts["truth_missing_gt"] += 1
            writer.write_site(SiteKind.TRUTH_MISSING_GT, key, t_variant, i_variant)
        else:
            ploidy = len(truth_parts)
            ref_alleles = tuple([key[2]] * ploidy)
            t_alleles = gt_to_allele_multiset(key[2], t_alt, t_gt, phase_sensitive)

            counts["compared_sites"] += 1

            # Extract allele frequency and update bins
            ac, an = extract_allele_frequency(t_info)
            update_frequency_bins(t_alleles, ref_alleles, ac, an, t_alt, counts, partial_credit, phase_sensitive)

            if partial_credit:
                counts["partial_credit_sum"] += allele_multiset_score(t_alleles, ref_alleles, phase_sensitive)

            if t_alleles == ref_alleles:
                counts["gt_allele_matches"] += 1
                writer.write_site(SiteKind.MATCH_ALLELE, key, t_variant, i_variant)
            else:
                counts["gt_allele_mismatches"] += 1
                writer.write_site(SiteKind.MISMATCH_ALLELE, key, t_variant, i_variant)
    else:
        counts["missing_in_imputed_sites"] += 1
        writer.write_site(SiteKind.MISSING_IN_IMPUTED_SITE, key, t_variant, i_variant)


def handle_extra_in_imputed(i, counts, writer, max_report):
    """Handle sites present in imputed but missing in truth."""
    key, i_variant = i
    t_variant = (None, None, None)
    counts["imputed_records"] += 1
    counts["extra_in_imputed_sites"] += 1
    writer.write_site(SiteKind.EXTRA_IN_IMPUTED_SITE, key, t_variant, i_variant)

    if len(counts["examples"]) < max_report:
        counts["examples"].append((SiteKind.EXTRA_IN_IMPUTED_SITE, key, t_variant, i_variant))


def handle_matched_sites(t, i, counts, writer, max_report, phase_sensitive, partial_credit, missing_as_ref):
    """Handle sites where truth and imputed keys match."""
    key, t_variant = t
    t_alt, t_info, t_gt = t_variant
    _, i_variant = i
    i_alt, i_info, i_gt = i_variant

    counts["truth_records"] += 1
    counts["imputed_records"] += 1
    counts["site_key_matches"] += 1

    t_alleles = gt_to_allele_multiset(key[2], t_alt, t_gt, phase_sensitive=phase_sensitive)
    i_alleles = gt_to_allele_multiset(key[2], i_alt, i_gt, phase_sensitive=phase_sensitive)

    # Handle missing/unparseable genotypes
    if not handle_missing_genotypes(t_alleles, i_alleles, key, t_variant, i_variant, counts, writer, missing_as_ref,
                                    phase_sensitive, partial_credit):
        return  # Early return if genotypes were missing/unparseable

    # Both genotypes are valid - do the comparison
    counts["compared_sites"] += 1

    # Extract allele frequency info with missing_as_ref logic
    ac, an = extract_allele_frequency(t_info)

    # Update frequency bins
    update_frequency_bins(t_alleles, i_alleles, ac, an, t_alt, counts, partial_credit, phase_sensitive)

    # Update overall counts
    if partial_credit:
        counts["partial_credit_sum"] += allele_multiset_score(t_alleles, i_alleles, phase_sensitive)

    if t_alleles == i_alleles:
        counts["gt_allele_matches"] += 1
        writer.write_site(SiteKind.MATCH_ALLELE, key, t_variant, i_variant)
    else:
        counts["gt_allele_mismatches"] += 1
        writer.write_site(SiteKind.MISMATCH_ALLELE, key, t_variant, i_variant)
        if len(counts["examples"]) < max_report:
            counts["examples"].append((SiteKind.MISMATCH_ALLELE, key, t_variant, i_variant))


def extract_allele_frequency(t_info: str) -> tuple:
    """Extract AC and AN from INFO field."""
    ac = None
    an = None

    for field in t_info.split(";"):
        if field.startswith("AC="):
            try:
                ac = int(field.split("=")[1])
            except ValueError:
                pass
        elif field.startswith("AN="):
            try:
                an = int(field.split("=")[1])
            except ValueError:
                pass

    return ac, an


def update_frequency_bins(t_alleles, i_alleles, ac, an, t_alt, counts, partial_credit, phase_sensitive):
    """Update frequency bins based on allele frequencies."""
    if ac is None or an is None or an == 0:
        print(f"Skipping frequency bin update: ac={ac}, an={an}")
        return

    alt_freq = ac / an
    ref_freq = 1 - alt_freq
    maf = min(alt_freq, ref_freq)
    af = alt_freq if t_alt in t_alleles else ref_freq
    NBINS = len(counts["af_bin_total"])

    # Calculate ALT allele counts for this site
    if not t_alt or t_alt == ".":
        truth_alt_count = 0
        imputed_alt_count = 0
    else:
        alt_alleles = t_alt.split(",")
        truth_alt_count = sum(1 for allele in t_alleles if allele in alt_alleles)
        imputed_alt_count = sum(1 for allele in i_alleles if allele in alt_alleles)

    # Homozygous sites
    if len(set(t_alleles)) == 1:
        bin_idx = min(int(af * 20), NBINS - 1)

        counts["af_bin_total"][bin_idx] += 1
        if partial_credit:
            counts["af_bin_correct"][bin_idx] += allele_multiset_score(t_alleles, i_alleles, phase_sensitive)
        elif t_alleles == i_alleles:
            counts["af_bin_correct"][bin_idx] += 1
        counts["af_bin_true_alt_counts"][min(int(af * 20), NBINS - 1)].append(truth_alt_count)
        counts["af_bin_imp_alt_counts"][min(int(af * 20), NBINS - 1)].append(imputed_alt_count)

    # Heterozygous sites
    else:
        bin_idx = min(int(maf * 20), NBINS - 1)

        counts["het_bin_total"][bin_idx] += 1
        counts["het_bin_correct"][bin_idx] += allele_multiset_score(t_alleles, i_alleles, phase_sensitive)
        counts["af_bin_true_alt_counts"][min(int(maf * 20), NBINS - 1)].append(truth_alt_count)
        counts["af_bin_imp_alt_counts"][min(int(maf * 20), NBINS - 1)].append(imputed_alt_count)


def handle_missing_genotypes(t_alleles, i_alleles, key, t_variant, i_variant,
                             counts, writer, missing_as_ref, phase_sensitive, partial_credit):
    """Handle cases where one or both genotypes are missing/unparseable. Returns False if handled."""
    t_alt, t_info, t_gt = t_variant
    i_alt, i_info, i_gt = i_variant

    t_missing = parse_gt(t_gt) is None
    i_missing = parse_gt(i_gt) is None

    if t_alleles is None and i_alleles is None:
        if t_missing and i_missing:
            counts["both_missing_gt"] += 1
            writer.write_site(SiteKind.BOTH_MISSING_GT, key, t_variant, i_variant)
        else:
            counts["gt_unparseable"] += 1
            writer.write_site(SiteKind.GT_UNPARSEABLE, key, t_variant, i_variant)
        return False

    elif t_alleles is None:
        if t_missing:
            counts["truth_missing_gt"] += 1
            writer.write_site(SiteKind.TRUTH_MISSING_GT, key, t_variant, i_variant)
        else:
            counts["gt_unparseable"] += 1
            writer.write_site(SiteKind.GT_UNPARSEABLE, key, t_variant, i_variant)
        return False

    elif i_alleles is None:
        if i_missing and missing_as_ref:
            # Reuse the missing_in_imputed logic - construct the truth record format
            t_record = (key, t_variant)
            handle_missing_in_imputed(t_record, counts, writer, missing_as_ref, phase_sensitive, partial_credit)
            return False
        else:
            if i_missing:
                counts["imputed_missing_gt"] += 1
                writer.write_site(SiteKind.IMPUTED_MISSING_GT, key, t_variant, i_variant)
            else:
                counts["gt_unparseable"] += 1
                writer.write_site(SiteKind.GT_UNPARSEABLE, key, t_variant, i_variant)
            return False

    return True  # Both genotypes are valid


def next_or_none(it):
    """Helper function to get next item or None if exhausted."""
    try:
        return next(it)
    except StopIteration:
        return None


def main():
    ap = argparse.ArgumentParser(
        description="Allele-aware GT comparison between two VCFs. Match by (CHROM,POS,REF); ALT may differ; ./. treated as missing."
    )
    ap.add_argument("--truth", required=True, help="Truth VCF (.vcf/.vcf.gz)")
    ap.add_argument("--imputed", required=True, help="Imputed VCF (.vcf/.vcf.gz)")
    ap.add_argument("-s", "--samples", help="Comma-separated list of sample names (do not add spaces)")
    ap.add_argument("-r", "--region", default=None, help="Optional region restriction, e.g. chr1:1000000-2000000")
    ap.add_argument("--phase-sensitive", action="store_true",
                    help="If set, treat phased order as meaningful (0|1 != 1|0). Default ignores phase.")
    ap.add_argument("--partial-credit", action="store_true",
                    help="If set, report fractional concordance based on allele overlap (e.g., 0.5 for one allele correct in diploids).")
    ap.add_argument("--max-report", type=int, default=20, help="Max example diffs printed to stdout (default 20)")
    ap.add_argument("--out-sites", default=None, help="Write per-site comparison to this TSV file")
    ap.add_argument("--only-mismatches", action="store_true",
                    help="When --out-sites is set, write only mismatches + missing/extra site lines")
    ap.add_argument("--only-matched-sites", action="store_true",
                    help="When --out-sites is set, suppress EXTRA/MISSING site lines; only write matched keys")
    ap.add_argument("--missing-as-ref", action="store_true",
                   help="Treat missing imputed sites/genotypes as reference calls")
    args = ap.parse_args()

    out_fh = None
    if args.out_sites:
        out_fh = open(args.out_sites, "w")
        out_fh.write("TYPE\tCHROM\tPOS\tREF\tTRUTH_ALT\tTRUTH_INFO\tTRUTH_GT\tIMPUTED_ALT\tIMPUTED_INFO\tIMPUTED_GT\n")

    if args.samples:
        samples = [s.strip() for s in args.samples.split(',')]
    else:
        samples = []

    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            t_raw = str(td / "truth.raw.tsv")
            i_raw = str(td / "imputed.raw.tsv")
            t_sorted = str(td / "truth.sorted.tsv")
            i_sorted = str(td / "imputed.sorted.tsv")

            res = initialize_counts()

            for sample in samples:
                write_query_tsv(args.truth, t_raw, sample, args.region)
                write_query_tsv(args.imputed, i_raw, sample, args.region)

                sort_tsv(t_raw, t_sorted)
                sort_tsv(i_raw, i_sorted)

                counts = compare_sorted(
                    t_sorted,
                    i_sorted,
                    args.max_report,
                    args.phase_sensitive,
                    args.partial_credit,
                    out_fh=out_fh,
                    only_mismatches=args.only_mismatches,
                    only_matched_sites=args.only_matched_sites,
                    missing_as_ref=args.missing_as_ref,
                )

                res["truth_records"] += counts["truth_records"]
                res["imputed_records"] += counts["imputed_records"]
                res["site_key_matches"] += counts["site_key_matches"]
                res["missing_in_imputed_sites"] += counts["missing_in_imputed_sites"]
                res["extra_in_imputed_sites"] += counts["extra_in_imputed_sites"]
                res["both_missing_gt"] += counts["both_missing_gt"]
                res["truth_missing_gt"] += counts["truth_missing_gt"]
                res["imputed_missing_gt"] += counts["imputed_missing_gt"]
                res["gt_unparseable"] += counts["gt_unparseable"]
                res["compared_sites"] += counts["compared_sites"]
                res["gt_allele_matches"] += counts["gt_allele_matches"]
                res["gt_allele_mismatches"] += counts["gt_allele_mismatches"]
                res["partial_credit_sum"] += counts["partial_credit_sum"]
                res["examples"] = res["examples"] + counts["examples"]
                res["af_bin_total"]  = np.add(counts["af_bin_total"], res["af_bin_total"])
                res["af_bin_correct"] = np.add(counts["af_bin_correct"], res["af_bin_correct"])
                res["het_bin_total"] = np.add(counts["het_bin_total"], res["het_bin_total"])
                res["het_bin_correct"] = np.add(counts["het_bin_correct"], res["het_bin_correct"])
                res["af_bin_true_alt_counts"] = [
                    res_bin + counts_bin
                    for res_bin, counts_bin in zip(res["af_bin_true_alt_counts"], counts["af_bin_true_alt_counts"])
                ]
                res["af_bin_imp_alt_counts"] = [
                    res_bin + counts_bin
                    for res_bin, counts_bin in zip(res["af_bin_imp_alt_counts"], counts["af_bin_imp_alt_counts"])
                ]


        # Summary
        print("VCF allele-aware GT compare (key = CHROM,POS,REF; ALT may differ; ./. missing)")
        for k in [
            "truth_records",
            "imputed_records",
            "site_key_matches",
            "missing_in_imputed_sites",
            "extra_in_imputed_sites",
            "both_missing_gt",
            "truth_missing_gt",
            "imputed_missing_gt",
            "gt_unparseable",
            "compared_sites",
            "gt_allele_matches",
            "gt_allele_mismatches",
        ]:
            print(f"  {k:24s} {res[k]}")

        if res["compared_sites"] > 0:
            concord = res["gt_allele_matches"] / res["compared_sites"]
            print(f"  allele_GT_concordance       {concord:.6f}")
            if args.partial_credit:
                frac = res["partial_credit_sum"] / res["compared_sites"]
                print(f"  partial_allele_concordance  {frac:.6f}")
        else:
            print("  allele_GT_concordance       NA (no comparable sites)")
            if args.partial_credit:
                print("  partial_allele_concordance  NA (no comparable sites)")

        identical_strict = (
            res["missing_in_imputed_sites"] == 0
            and res["extra_in_imputed_sites"] == 0
            and res["gt_allele_mismatches"] == 0
            and res["truth_missing_gt"] == 0
            and res["imputed_missing_gt"] == 0
            and res["both_missing_gt"] == 0
            and res["gt_unparseable"] == 0
        )
        print(f"  IDENTICAL_STRICT            {str(identical_strict).upper()}")

        freq_bins = res["af_bins"]

        print("\nACCURACY OF HOMOZYGOUS TRUTH SITES BY TRUTH ALLELE FREQUENCY\n")
        print("LOW\tHIGH\tACCURACY\tCOUNT")
        for i in range(len(freq_bins)-1):
            low = freq_bins[i]
            high = freq_bins[i + 1]

            n = res["af_bin_total"][i]
            if n!=0:
                acc = res["af_bin_correct"][i] / n
            else: acc = None

            print(low, high, acc, n, sep='\t')

        print("\nACCURACY OF HETEROZYGOUS TRUTH SITES BY MINOR ALLELE FREQUENCY\n")
        print("LOW\tHIGH\tACCURACY\tCOUNT")
        # for het sites, only need to iterate through minor AF (<=0.5)
        for i in range(len(freq_bins)-10):
            low = res["af_bins"][i]
            high = res["af_bins"][i + 1] if i + 1 < len(res["af_bins"]) else 1.0

            n = res["het_bin_total"][i]
            if n!=0:
                acc = res["het_bin_correct"][i] / n
            else: acc = None

            print(low, high, acc, n, sep='\t')

        print("\nR2 OF ALL TRUTH SITES BY TRUTH ALLELE FREQUENCY\n")
        print("LOW\tHIGH\tR2")
        # iterate through MAF bins
        for i in range(len(freq_bins) - 1):
            low = res["af_bins"][i]
            high = res["af_bins"][i + 1] if i + 1 < len(res["af_bins"]) else 1.0

            true_vals = res["af_bin_true_alt_counts"][i]
            pred_vals = res["af_bin_imp_alt_counts"][i]
            n = len(true_vals)

            if n > 1 and len(set(true_vals)) > 1 and len(set(pred_vals)) > 1:
                corr_matrix = np.corrcoef(true_vals, pred_vals)
                r = corr_matrix[0, 1]
                r2 = r ** 2
            else:
                r2 = None

            print(low, high, r2, sep='\t')

#        if res["examples"]:
#            print("\nExamples:")
#            print("  TYPE\tCHROM\tPOS\tREF\tTRUTH_ALT\tTRUTH_GT\tIMPUTED_ALT\tIMPUTED_GT")
#            for kind, key, t_alt, t_gt, i_alt, i_gt in res["examples"]:
#                chrom, pos, ref = key
#                print(f"  {kind}\t{chrom}\t{pos}\t{ref}\t{t_alt or '.'}\t{t_gt or '.'}\t{i_alt or '.'}\t{i_gt or '.'}")

        if args.out_sites:
            print(f"\nWrote per-site output to: {args.out_sites}")
            if args.only_mismatches:
                print("  (mode: only mismatches + missing/extra sites)")
            if args.only_matched_sites:
                print("  (mode: only matched sites)")

    finally:
        if out_fh:
            out_fh.close()


if __name__ == "__main__":
    main()