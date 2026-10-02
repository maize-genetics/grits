#!/usr/bin/env python
"""
Unit tests for the Phase 1 rewrite of `TruthCursor.resolve()` /
`iter_truth_records()` in
`test_crf_relatedness/src/python/vcf_eval/compare_gvcf_truth.py` (see
`/home/zrm22/.claude/plans/ok-we-need-to-squishy-lovelace.md`).

Not wired into the repo's own pytest harness (`test_crf_relatedness/tests/`)
because that harness's existing `from src.python.vcf_eval.accuracy import *`
import convention is already broken independent of this change (confirmed:
`ModuleNotFoundError: No module named 'src'` under `pixi run pytest`, even
for the pre-existing `test_accuracy.py`) -- not something this session's
scope covers fixing. Run directly instead:

    python scripts/test_compare_gvcf_truth_projection.py

Exits 0 and prints "ALL PASS" if every assertion holds, else raises
AssertionError with the failing case.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "test_crf_relatedness" / "src"))
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src")

from python.vcf_eval.compare_gvcf_truth import (  # noqa: E402
    NON_REF,
    SYMBOLIC_DEL,
    SYMBOLIC_INS,
    TruthCursor,
    TruthRecord,
    classify_alleles,
    iter_truth_records,
)


def cursor_from_records(records):
    return TruthCursor(iter(records))


def rec(pos, end, ref, alt, gt, is_ref_block=False):
    return TruthRecord("chr1", pos, end, ref, alt, gt, is_ref_block)


PLOIDY = 2


def check(name, actual, expected):
    assert actual == expected, f"{name}: expected {expected!r}, got {actual!r}"
    print(f"  ok: {name}")


def test_snp():
    cur = cursor_from_records([rec(100, 100, "A", "T,<NON_REF>", "1")])
    check("snp exact-pos match", cur.resolve("chr1", 100, PLOIDY), ("T", "1/1"))


def test_ref_block():
    cur = cursor_from_records([rec(100, 200, "A", NON_REF, "0", is_ref_block=True)])
    check("ref block, at start", cur.resolve("chr1", 100, PLOIDY), (NON_REF, "0/0"))
    check("ref block, mid-span (same cursor)", cur.resolve("chr1", 150, PLOIDY), (NON_REF, "0/0"))


def test_variant_record_gt0():
    # a non-ref-block variant record whose GT nonetheless selects index 0
    cur = cursor_from_records([rec(100, 100, "A", "T,<NON_REF>", "0")])
    check("variant record, GT=0", cur.resolve("chr1", 100, PLOIDY), (NON_REF, "0/0"))


def test_clean_deletion_anchor_and_interior():
    # AAAA -> A  (a 3bp clean deletion: retained prefix "A" == REF[:1])
    cur = cursor_from_records([rec(100, 103, "AAAA", "A,<NON_REF>", "1")])
    check("clean deletion, anchor", cur.resolve("chr1", 100, PLOIDY), (NON_REF, "0/0"))
    check("clean deletion, interior", cur.resolve("chr1", 101, PLOIDY), (SYMBOLIC_DEL, "1/1"))
    check("clean deletion, interior end", cur.resolve("chr1", 103, PLOIDY), (SYMBOLIC_DEL, "1/1"))


def test_complex_deletion_whole_span():
    # REF="CACC...", ALT="T" -- retained base T != REF[0] C: the real
    # >100kb-record pattern found on chr1 this session (e.g. Oh43 @
    # chr1:150979913). Whole span, including the anchor, is <DEL>.
    ref = "CACCAGATG"
    cur = cursor_from_records([rec(100, 108, ref, "T,<NON_REF>", "1")])
    check("complex deletion, anchor is DEL too", cur.resolve("chr1", 100, PLOIDY), (SYMBOLIC_DEL, "1/1"))
    check("complex deletion, interior", cur.resolve("chr1", 104, PLOIDY), (SYMBOLIC_DEL, "1/1"))
    check("complex deletion, last position", cur.resolve("chr1", 108, PLOIDY), (SYMBOLIC_DEL, "1/1"))


def test_insertion_anchor_only():
    cur = cursor_from_records([rec(100, 100, "G", "GCTT,<NON_REF>", "1")])
    check("insertion, anchor", cur.resolve("chr1", 100, PLOIDY), (SYMBOLIC_INS, "1/1"))


def test_n_masked_no_opinion():
    # assembly gap / masked base -- confirmed real pattern (NC350 @
    # chr1:252972782, REF="NCTTAAG", ALT="N"): no opinion anywhere in the
    # record's span, not even the anchor.
    cur = cursor_from_records([rec(100, 106, "NCTTAAG", "N,<NON_REF>", "1")])
    check("N-masked, anchor -> no info", cur.resolve("chr1", 100, PLOIDY), None)
    check("N-masked, interior -> no info", cur.resolve("chr1", 103, PLOIDY), None)


def test_missing_gt():
    cur = cursor_from_records([rec(100, 100, "A", "T,<NON_REF>", ".")])
    check("missing truth GT -> no info", cur.resolve("chr1", 100, PLOIDY), None)


def test_gap_no_record():
    cur = cursor_from_records([rec(200, 200, "A", "T,<NON_REF>", "1")])
    check("position before any record -> no info", cur.resolve("chr1", 100, PLOIDY), None)


def test_multi_record_advance():
    # confirms _expire_current/_advance correctly walks forward across
    # multiple records in one contig
    cur = cursor_from_records([
        rec(100, 149, "A", NON_REF, "0", is_ref_block=True),
        rec(150, 150, "C", "G,<NON_REF>", "1"),
        rec(151, 250, "A", NON_REF, "0", is_ref_block=True),
    ])
    check("record 1 (ref block)", cur.resolve("chr1", 120, PLOIDY), (NON_REF, "0/0"))
    check("record 2 (SNP)", cur.resolve("chr1", 150, PLOIDY), ("G", "1/1"))
    check("record 3 (ref block)", cur.resolve("chr1", 200, PLOIDY), (NON_REF, "0/0"))


def test_matchable_ceiling_bookkeeping():
    # a real variant record that's queried (matched) should count toward
    # matchable_variant_sites once expired; an unqueried one toward
    # unmatchable_variant_sites. Ref blocks never count either way.
    cur = cursor_from_records([
        rec(100, 100, "A", "T,<NON_REF>", "1"),   # will be queried
        rec(200, 200, "C", "G,<NON_REF>", "1"),   # will NOT be queried
        rec(300, 350, "A", NON_REF, "0", is_ref_block=True),
    ])
    cur.resolve("chr1", 100, PLOIDY)
    cur.resolve("chr1", 300, PLOIDY)  # skips past the unqueried record 200 too
    cur.flush()
    check("matchable_variant_sites", cur.matchable_variant_sites, 1)
    check("unmatchable_variant_sites", cur.unmatchable_variant_sites, 1)


def test_end_computed_from_ref_length_when_no_END():
    # the actual Bug A fix: iter_truth_records must derive `end` from REF
    # length when INFO carries no literal END= (our own maf-to-gvcf-converter
    # output never writes END= on variant records).
    lines = [
        "chr1\t100\t.\tAA\tA,<NON_REF>\t.\t.\tASM_Chr=chr1\tGT:AD\t1:0,30\n",
    ]
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
        f.write("".join(lines))
        path = f.name
    records = list(iter_truth_records(path))
    check("end computed from len(REF)-1 when END= absent", records[0].end, 101)


def test_end_uses_literal_END_when_larger():
    lines = [
        "chr1\t100\t.\tAA\tA,<NON_REF>\t.\t.\tEND=500;ASM_Chr=chr1\tGT:AD\t1:0,30\n",
    ]
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
        f.write("".join(lines))
        path = f.name
    records = list(iter_truth_records(path))
    check("end = max(literal END=, computed) when END= is larger", records[0].end, 500)


def test_classify_alleles():
    check("homref", classify_alleles(("A", "A"), "A"), "HOMREF")
    check("snp hom", classify_alleles(("T", "T"), "A"), "SNP")
    check("snp het", classify_alleles(("A", "T"), "A"), "SNP")
    check("ins hom (symbolic)", classify_alleles((SYMBOLIC_INS, SYMBOLIC_INS), "A"), "INS")
    check("del hom (symbolic)", classify_alleles((SYMBOLIC_DEL, SYMBOLIC_DEL), "A"), "DEL")
    check("ins hom (literal)", classify_alleles(("ATT", "ATT"), "A"), "INS")
    check("del hom (literal)", classify_alleles(("A", "A"), "ATT"), "DEL")  # "A" != ref "ATT", shorter -> DEL
    check("del het (literal, non-ref)", classify_alleles(("A", "AT"), "ATT"), "DEL")
    check("mixed ins/del", classify_alleles((SYMBOLIC_INS, SYMBOLIC_DEL), "A"), "HET_MIXED")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"{t.__name__}:")
        t()
    print(f"\nALL PASS ({len(tests)} test functions)")


if __name__ == "__main__":
    main()
