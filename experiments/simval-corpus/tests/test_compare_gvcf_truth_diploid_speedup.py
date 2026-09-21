#!/usr/bin/env python
"""
Regression coverage for the speed investigation of compare_gvcf_truth_diploid.py
(user request: "can we speed up the VCF comparisons"). cProfile on real data
showed the per-site Python loop in compare_diploid(), not bcftools/sort, is
the dominant cost -- and that classify_alleles()/allele_multiset_score() were
each being recomputed 2-3x per site for IDENTICAL inputs under the flag
combination the production driver (simval_snp_refcall_rescore.py) always
uses (--partial-credit --class-breakdown --snp-refcall-metrics). Deduping
those calls was verified bit-identical against the pre-change version on
real chr1:1-5,000,000 data (306,286 compared sites, exact JSON diff) before
this test file was written; this test locks in the *new* behavior that real
data alone doesn't exercise: the position-monotonicity guard that makes
main()'s indexed-input sort skip self-verifying rather than a silent
assumption (see compare_diploid()'s `_last_pos` check and main()'s
`indexed` branch).

Not wired into the repo's own pytest harness for the same reason as
test_compare_gvcf_truth_projection.py (see that file's docstring) -- run
directly:

    python experiments/simval-corpus/tests/test_compare_gvcf_truth_diploid_speedup.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import compare_gvcf_truth_diploid as d  # noqa: E402


def _gvcf(path, block_end=1000):
    path.write_text(
        f"##fileformat=VCFv4.2\nchr1\t1\t.\tA\t<NON_REF>\t.\t.\tEND={block_end}\tGT\t0\n"
    )


def test_raises_on_out_of_order_positions():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tsv = td / "imputed.tsv"
        tsv.write_text("chr1\t200\tA\tT\tAC=1;AN=2\t0/1\nchr1\t100\tA\tT\tAC=1;AN=2\t0/1\n")
        gvcf1 = td / "h1.g.vcf"
        _gvcf(gvcf1)
        try:
            d.compare_diploid(str(tsv), str(gvcf1), str(gvcf1), False, False)
        except RuntimeError as e:
            assert "not position-sorted" in str(e), f"wrong error: {e}"
            print("  ok: raises RuntimeError on out-of-order positions")
            return
        raise AssertionError("expected RuntimeError, none raised")


def test_accepts_sorted_input():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tsv = td / "imputed.tsv"
        tsv.write_text("chr1\t100\tA\tT\tAC=1;AN=2\t0/1\nchr1\t200\tA\tT\tAC=1;AN=2\t0/1\n")
        gvcf1 = td / "h1.g.vcf"
        _gvcf(gvcf1)
        counts = d.compare_diploid(str(tsv), str(gvcf1), str(gvcf1), False, False)
        assert counts["imputed_records"] == 2
        print("  ok: sorted input processes normally")


def test_flags_combo_matches_naive_independent_computation():
    """The production driver always passes --partial-credit --class-breakdown
    --snp-refcall-metrics together -- exactly the combination whose redundant
    classify_alleles()/allele_multiset_score() calls were deduped. Build a
    site where truth is heterozygous SNP/DEL-ish (h1 SNP, h2 ref) and check
    every one of the counters those three flags populate lands on the value
    an independent, unoptimized re-derivation would give.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tsv = td / "imputed.tsv"
        # imputed calls het SNP (0/1 -> ref A, alt T)
        tsv.write_text("chr1\t100\tA\tT\tAC=1;AN=2\t0/1\n")
        h1 = td / "h1.g.vcf"
        h1.write_text("##fileformat=VCFv4.2\nchr1\t100\t.\tA\tT,<NON_REF>\t.\t.\t.\tGT\t1\n")
        h2 = td / "h2.g.vcf"
        h2.write_text("##fileformat=VCFv4.2\nchr1\t1\t.\tA\t<NON_REF>\t.\t.\tEND=1000\tGT\t0\n")

        counts = d.compare_diploid(str(tsv), str(h1), str(h2), False, True,
                                    class_breakdown=True, snp_refcall_metrics=True)

        # Independent re-derivation, not reusing any of compare_diploid's
        # internal dedup: truth = {T, A} (h1 SNP T, h2 ref A), imputed = {A, T}.
        from python.vcf_eval.accuracy import allele_multiset_score  # noqa: E402
        expected_score = allele_multiset_score(("A", "T"), ("A", "T"), False)

        assert counts["compared_sites"] == 1
        assert counts["gt_allele_matches"] == 1
        assert counts["partial_credit_sum"] == expected_score
        assert counts["class_total_SNP"] == 1
        assert counts["class_partial_sum_SNP"] == expected_score
        assert counts["snprc_compared_sites"] == 1
        assert counts["snprc_partial_credit_sum"] == expected_score
        assert counts["pairclass_SNP_SNP"] == 1
        print("  ok: dedup'd class_breakdown+snp_refcall_metrics+partial_credit "
              "counters match an independent re-derivation")


if __name__ == "__main__":
    test_raises_on_out_of_order_positions()
    test_accepts_sorted_input()
    test_flags_combo_matches_naive_independent_computation()
    print("ALL PASS")
