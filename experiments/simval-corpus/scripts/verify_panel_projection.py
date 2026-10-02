#!/usr/bin/env python
"""
Phase 0 validator for the panel-rebuild plan (see
`/home/zrm22/.claude/plans/ok-we-need-to-squishy-lovelace.md`).

Confirms, empirically, the exact rule `merge-gvcfs` uses to turn a founder's
own gVCF into that founder's column in the merged panel VCF -- reverse
engineered this session from `javap -c` on `MergeGVCFUtilsKt.class` and spot
checks, and load-bearing for the whole rebuild plan (Phase 1's `resolve()`
rewrite reimplements exactly this rule; Phase 3's panel QC re-runs this same
check against the rebuilt panel). If this script doesn't project a founder's
gVCF onto its own panel column at ~100%, the rule is wrong and nothing
downstream can be trusted.

Rule (one founder gVCF record, 1-based POS, REF, GT-selected allele A, END
from INFO or POS if absent):
  A == REF (GT index 0, incl. <NON_REF> blocks) -> every panel site in
                                                    [POS, END] reads REF there
  A is a SNP  (len(A) == len(REF))    -> panel site at POS is literal A
  A is an insertion (len(A) > len(REF)) -> panel site at POS is "<INS>"
  A is a deletion (len(A) < len(REF)) -> if A == REF[:len(A)] ("clean"
                                          deletion, the retained prefix is
                                          unchanged): panel site at POS
                                          reads REF there, (POS, END] is
                                          "<DEL>". If A != REF[:len(A)] (a
                                          complex substitution+deletion --
                                          seen on real data at very large
                                          (>100kb) deletion records where the
                                          retained base itself differs from
                                          REF's first base): the WHOLE span
                                          [POS, END] is "<DEL>", anchor
                                          included -- confirmed 3/3 on real
                                          examples this session.
  Either REF or the selected allele containing 'N' (an assembly gap/masked
  base, not a real call) -> no opinion at all, not even the anchor --
  confirmed on real data this session (merge-gvcfs leaves the sample "."
  there rather than emitting any allele).

Implemented as a forward-only two-pointer merge over (a) the panel's data
lines for one sample column and (b) that founder's own gVCF data lines --
same algorithmic shape as `TruthCursor.resolve()` in compare_gvcf_truth.py
(deliberately: Phase 1 replaces `resolve()` with this exact rule), so this
runs in O(panel_rows + gvcf_rows) per founder with no large in-memory maps.

Usage (chr1-only, using this session's pre-extracted TSVs at
scratch/panel_projection_chr1/, produced via a chr1-only `awk` filter over
the plain-text panel + founder gVCFs, ~20s total):

    python scripts/verify_panel_projection.py \\
        --panel scratch/panel_projection_chr1/panel_chr1.tsv \\
        --panel-header data/maize_panel_vcf/panel_25founders.vcf \\
        --gvcf-dir scratch/panel_projection_chr1 \\
        --founders B97,CML103,...,Tzi8

`--panel`/founder gVCF files may be full VCF/gVCF (with `#`-comment headers)
or already data-only TSVs -- both handled transparently.
"""
import argparse
import sys
from pathlib import Path

NON_REF = "<NON_REF>"
SYMBOLIC_INS = "<INS>"
SYMBOLIC_DEL = "<DEL>"


def data_lines(path):
    with open(path, "r") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            yield line.rstrip("\n")


def panel_sample_columns(panel_header_vcf):
    with open(panel_header_vcf, "r") as fh:
        for line in fh:
            if line.startswith("#CHROM"):
                return line.rstrip("\n").split("\t")[9:]
            if not line.startswith("#"):
                break
    raise SystemExit(f"no #CHROM line found in {panel_header_vcf}")


class GvcfCursor:
    """One founder's own gVCF, forward-only, one record 'live' at a time."""

    def __init__(self, path):
        self._it = data_lines(path)
        self._chrom = None
        self._pos = None
        self._end = None
        self._kind = None  # "REF" | "SNP" | "INS" | "DEL" | "DEL_WHOLE" | "SKIP"
        self._allele = None  # literal allele string, only meaningful for SNP
        self._advance()

    def _advance(self):
        line = next(self._it, None)
        if line is None:
            self._chrom = None
            return
        parts = line.split("\t")
        chrom, pos_s, ref, alt_field = parts[0], parts[1], parts[3], parts[4]
        info = parts[7]
        gt = parts[9].split(":", 1)[0]
        pos = int(pos_s)

        end = pos
        for field in info.split(";"):
            if field.startswith("END="):
                try:
                    end = int(field[4:])
                except ValueError:
                    pass
                break

        if gt in (".", ""):
            # no-call record: skip it, this gVCF has no opinion here
            self._chrom, self._pos, self._end = chrom, pos, pos - 1  # zero-span, never matches
            self._kind = "SKIP"
            return

        idx = int(gt)
        self._chrom, self._pos, self._end = chrom, pos, max(end, pos)
        if idx == 0:
            self._kind = "REF"
            return
        alts = [] if alt_field in {".", ""} else alt_field.split(",")
        if idx - 1 >= len(alts):
            self._kind = "SKIP"
            return
        allele = alts[idx - 1]
        if allele == NON_REF:
            self._kind = "SKIP"
            return
        if "N" in ref or "N" in allele:
            # assembly gap / masked base -- not a real call either direction
            self._kind = "SKIP"
            return
        if len(allele) == len(ref):
            self._kind = "SNP"
            self._allele = allele
        elif len(allele) > len(ref):
            self._kind = "INS"
        else:
            # deletion -- "clean" iff the retained prefix matches REF's own
            # prefix; otherwise a complex substitution+deletion where even
            # the anchor differs from REF (real data: only ever seen on
            # very large, >100kb records)
            if allele == ref[: len(allele)]:
                self._kind = "DEL"
            else:
                self._kind = "DEL_WHOLE"

    def expected_at(self, chrom, pos):
        """Advance past exhausted records, return expected panel-space
        allele class at (chrom, pos): "REF" | "SNP:<allele>" | "INS" | "DEL"
        | None (no opinion / not covered)."""
        while self._chrom is not None and (self._chrom, self._end) < (chrom, pos) and self._chrom == chrom:
            self._advance()
        # skip past exhausted records on earlier chroms too
        while self._chrom is not None and self._chrom != chrom and self._chrom < chrom:
            self._advance()
        if self._chrom != chrom or self._pos > pos or self._end < pos:
            return None
        if self._kind == "SKIP":
            return None
        if self._kind == "REF":
            return "REF"
        if self._kind == "INS":
            return "INS" if pos == self._pos else None
        if self._kind == "DEL":
            return "REF" if pos == self._pos else "DEL"
        if self._kind == "DEL_WHOLE":
            return "DEL"  # anchor included -- retained base differs from REF
        if self._kind == "SNP":
            return f"SNP:{self._allele}" if pos == self._pos else None
        return None


def compare_founder(panel_path, sample_idx, gvcf_path, sample, max_examples=8):
    cur = GvcfCursor(gvcf_path)
    agree = disagree = uncovered = 0
    examples = []
    for line in data_lines(panel_path):
        parts = line.split("\t")
        chrom = parts[0]
        pos = int(parts[1])
        ref = parts[3]
        alt_field = parts[4]
        gt = parts[9 + sample_idx]

        expected = cur.expected_at(chrom, pos)
        if expected is None:
            uncovered += 1
            continue

        if gt in (".", ""):
            actual = None
        else:
            gi = int(gt)
            actual = ref if gi == 0 else ([] if alt_field in {".", ""} else alt_field.split(","))[gi - 1]

        if expected == "REF":
            ok = actual == ref
        elif expected == "INS":
            ok = actual == SYMBOLIC_INS
        elif expected == "DEL":
            ok = actual == SYMBOLIC_DEL
        elif expected.startswith("SNP:"):
            ok = actual == expected[4:]
        else:
            ok = False

        if ok:
            agree += 1
        else:
            disagree += 1
            if len(examples) < max_examples:
                examples.append((chrom, pos, expected, actual))

    total = agree + disagree
    frac = agree / total if total else float("nan")
    print(f"=== {sample} ===")
    print(f"  agree      {agree:>12,}")
    print(f"  disagree   {disagree:>12,}")
    print(f"  uncovered  {uncovered:>12,}  (gVCF has no opinion at this panel site -- expected/common)")
    print(f"  agreement_fraction (agree/(agree+disagree))  {frac:.8f}")
    for chrom, pos, expected, actual in examples:
        print(f"    DISAGREE {chrom}:{pos}  expected={expected!r}  panel={actual!r}")
    print()
    return agree, disagree, uncovered


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--panel-header", default=None)
    ap.add_argument("--gvcf-dir", required=True)
    ap.add_argument("--founders", required=True)
    ap.add_argument("--gvcf-suffix", default="_chr1.tsv")
    args = ap.parse_args()

    header_source = args.panel_header or args.panel
    sample_names = panel_sample_columns(header_source)
    print(f"panel has {len(sample_names)} sample columns: {sample_names}\n")

    founders = args.founders.split(",")
    missing = [f for f in founders if f not in sample_names]
    if missing:
        raise SystemExit(f"founders not in panel header: {missing}")

    totals = {"agree": 0, "disagree": 0, "uncovered": 0}
    worst = (None, 1.0)
    for f in founders:
        idx = sample_names.index(f)
        gvcf_path = Path(args.gvcf_dir) / f"{f}{args.gvcf_suffix}"
        if not gvcf_path.exists():
            print(f"=== {f} === SKIPPED (no file at {gvcf_path})\n")
            continue
        agree, disagree, uncovered = compare_founder(args.panel, idx, str(gvcf_path), f)
        totals["agree"] += agree
        totals["disagree"] += disagree
        totals["uncovered"] += uncovered
        frac = agree / (agree + disagree) if (agree + disagree) else 1.0
        if frac < worst[1]:
            worst = (f, frac)

    total_checked = totals["agree"] + totals["disagree"]
    overall = totals["agree"] / total_checked if total_checked else float("nan")
    print("=== OVERALL ===")
    print(f"  agree      {totals['agree']:>12,}")
    print(f"  disagree   {totals['disagree']:>12,}")
    print(f"  uncovered  {totals['uncovered']:>12,}")
    print(f"  agreement_fraction  {overall:.8f}")
    print(f"  worst founder: {worst[0]}  ({worst[1]:.8f})")

    gate = 0.99999
    if overall < gate or worst[1] < gate:
        print(f"\nFAIL: agreement below gate ({gate})")
        sys.exit(1)
    print(f"\nPASS: agreement >= gate ({gate})")


if __name__ == "__main__":
    main()
