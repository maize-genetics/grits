import pytest
import io
import random
from collections import Counter
from src.python.vcf_eval.accuracy import *


def _allele_multiset_score_counter_reference(truth_alleles, imputed_alleles, phase_sensitive=False):
    """Old Counter-based implementation, kept here only as a reference oracle
    for equivalence-testing the current list-based allele_multiset_score."""
    if not truth_alleles:
        return 0.0
    if phase_sensitive:
        matches = sum(1 for t, i in zip(truth_alleles, imputed_alleles) if t == i)
        return matches / float(len(truth_alleles))
    t = Counter(truth_alleles)
    i = Counter(imputed_alleles)
    inter = sum((t & i).values())
    return inter / float(len(truth_alleles))


@pytest.fixture
def truth_vcf():
    return "data/truth.vcf"

@pytest.fixture
def imputed_vcf():
    return "data/imputed.vcf"


def test_write_query_tsv():
    pass

def test_sort_tsv():
    pass

def test_parse_gt():
    # Missing/empty genotypes
    assert parse_gt(".") is None
    assert parse_gt("./.") is None
    assert parse_gt(".|.") is None
    assert parse_gt("") is None  # if your function handles empty strings

    # Homozygous genotypes
    assert parse_gt("0/0") == ['0', '0']  # homozygous reference
    assert parse_gt("1/1") == ['1', '1']  # homozygous alternate
    assert parse_gt("2/2") == ['2', '2']  # homozygous second alternate

    # Heterozygous genotypes
    assert parse_gt("0/1") == ['0', '1']  # het ref/alt
    assert parse_gt("1/0") == ['1', '0']  # het alt/ref (if order matters)
    assert parse_gt("1/2") == ['1', '2']  # het between alternates

    # Phased genotypes (if your function handles them)
    assert parse_gt("0|1") == ['0', '1']
    assert parse_gt("1|0") == ['1', '0']
    assert parse_gt("1|2") == ['1', '2']

    # Partially missing genotypes
    assert parse_gt("./1") is None  # or however your function handles this
    assert parse_gt("1/.") is None
    assert parse_gt(".|1") is None
    assert parse_gt("1|.") is None

    # Higher ploidy (if supported)
    assert parse_gt("0/1/2") == ['0', '1', '2']  # triploid

    # Edge cases
    assert parse_gt("10/11") == ['10', '11']  # multi-digit alleles

def test_gt_to_allele_multiset():
    assert gt_to_allele_multiset(ref="A", alt_field="T", gt="0/0", phase_sensitive=False) == ("A", "A")
    assert gt_to_allele_multiset(ref="A", alt_field="T", gt="1/1", phase_sensitive=False) == ("T", "T")
    assert gt_to_allele_multiset(ref="A", alt_field="T", gt="0/1", phase_sensitive=False) == ("A", "T")
    assert gt_to_allele_multiset(ref="A", alt_field="T", gt="1/0", phase_sensitive=False) == ("A", "T")
    assert gt_to_allele_multiset(ref="A", alt_field="T", gt="1/0", phase_sensitive=True) == ("T", "A")
    assert gt_to_allele_multiset(ref="C", alt_field="G", gt="1|0", phase_sensitive=True) == ("G", "C")

    # Higher ploidy
    assert gt_to_allele_multiset("A", "T,C", "0/1/2") == ("A", "C", "T")  # sorted
    assert gt_to_allele_multiset(ref="A", alt_field="T,C", gt="2/1/0", phase_sensitive=False) == ("A", "C", "T")
    assert gt_to_allele_multiset("A", "T,C", "2/1/0", phase_sensitive=True) == ("C", "T", "A")  # preserves order

    # Edge case: no alternates (monomorphic site)
    result = gt_to_allele_multiset("A", ".", "0/0")
    assert result == ("A", "A")

    result = gt_to_allele_multiset("A", "", "0/0")
    assert result == ("A", "A")

    # Missing genotypes should return None
    assert gt_to_allele_multiset("A", "T", ".") is None
    assert gt_to_allele_multiset("A", "T", "./.") is None
    assert gt_to_allele_multiset("A", "T", ".|.") is None
    assert gt_to_allele_multiset("A", "T", "./1") is None
    assert gt_to_allele_multiset("A", "T", "1/.") is None

def test_allele_multiset_score():
    # Perfect matches
    assert allele_multiset_score(("A", "T"), ("A", "T")) == 1.0
    assert allele_multiset_score(("A", "T"), ("A", "T"), phase_sensitive=True) == 1.0
    assert allele_multiset_score(("A", "T"), ("T", "A")) == 1.0  # order shouldn't matter
    assert allele_multiset_score(("G", "G"), ("G", "G")) == 1.0  # homozygous match
    assert allele_multiset_score(("G", "G"), ("G", "G"), phase_sensitive=True) == 1.0  # homozygous match

    # Complete mismatches
    assert allele_multiset_score(("A", "T"), ("C", "G")) == 0.0
    assert allele_multiset_score(("A", "A"), ("T", "T")) == 0.0  # homozygous mismatch
    assert allele_multiset_score(("A", "T"), ("T", "A"), phase_sensitive=True) == 0.0  # order matters
    assert allele_multiset_score(("A", "T"), ("T", "C"), phase_sensitive=True) == 0.0  # order matters

    # Partial matches (diploid)
    assert allele_multiset_score(("A", "T"), ("A", "C")) == 0.5  # one allele matches
    assert allele_multiset_score(("A", "T"), ("A", "C"), phase_sensitive=True) == 0.5  # one allele matches
    assert allele_multiset_score(("A", "T"), ("C", "T")) == 0.5  # one allele matches
    assert allele_multiset_score(("A", "T"), ("C", "T"), phase_sensitive=True) == 0.5  # one allele matches
    assert allele_multiset_score(("A", "A"), ("A", "T")) == 0.5  # one of two A's matches
    assert allele_multiset_score(("A", "A"), ("A", "T"), phase_sensitive=True) == 0.5  # one of two A's matches

    # Different ploidy scenarios

    # Haploid truth vs diploid imputed
    assert allele_multiset_score(("A",), ("A", "A")) == 1.0  # 1 match out of 1 truth allele
    assert allele_multiset_score(("A",), ("A", "A"), phase_sensitive=True) == 1.0
    assert allele_multiset_score(("A",), ("A", "T")) == 1.0  # 1 match out of 1 truth allele
    assert allele_multiset_score(("A",), ("A", "T"), phase_sensitive=True) == 1.0
    assert allele_multiset_score(("A",), ("T", "A"), phase_sensitive=True) == 0.0
    assert allele_multiset_score(("A",), ("T", "C")) == 0.0  # 0 matches out of 1 truth allele
    assert allele_multiset_score(("A",), ("T", "C"), phase_sensitive=True) == 0.0

    # Diploid truth vs haploid imputed
    assert allele_multiset_score(("A", "T"), ("A",)) == 0.5  # 1 match out of 2 truth alleles
    assert allele_multiset_score(("A", "T"), ("A",), phase_sensitive=True) == 0.5
    assert allele_multiset_score(("A", "T"), ("T",), phase_sensitive=True) == 0.0 # I think should be 0.5
    assert allele_multiset_score(("A", "T"), ("C",)) == 0.0  # 0 matches out of 2 truth alleles
    assert allele_multiset_score(("A", "A"), ("A",)) == 0.5  # 1 match out of 2 truth alleles

    # Triploid scenarios
    assert allele_multiset_score(("A", "T", "C"), ("A", "T", "C")) == 1.0  # perfect match
    assert allele_multiset_score(("A", "T", "C"), ("A", "T", "G")) == 2.0 / 3.0  # 2 out of 3 match
    assert allele_multiset_score(("A", "T", "C"), ("A", "G", "T"), phase_sensitive=True) == 1.0 / 3.0
    assert allele_multiset_score(("A", "T", "C"), ("A", "G", "G")) == 1.0 / 3.0  # 1 out of 3 match
    assert allele_multiset_score(("A", "T", "C"), ("G", "G", "G")) == 0.0  # 0 out of 3 match

    # Triploid truth vs diploid imputed
    assert allele_multiset_score(("A", "A", "T"), ("A", "A")) == 2.0 / 3.0  # 2 A matches out of 3 truth
    assert allele_multiset_score(("A", "T", "C"), ("A", "T")) == 2.0 / 3.0  # 2 matches out of 3 truth

    # Complex multiset scenarios (testing Counter intersection logic)
    assert allele_multiset_score(("A", "A", "T"), ("A", "T", "T")) == 2.0 / 3.0  # 1 A + 1 T match
    assert allele_multiset_score(("A", "A", "A"), ("A", "A", "T")) == 2.0 / 3.0  # 2 A's match
    assert allele_multiset_score(("A", "T", "T"), ("T", "C", "C")) == 1.0 / 3.0  # 1 T matches

    # Edge case: empty truth (should return 0.0)
    assert allele_multiset_score((), ("A", "T")) == 0.0
    assert allele_multiset_score((), ()) == 0.0

    # Edge case: empty imputed but non-empty truth
    assert allele_multiset_score(("A", "T"), ()) == 0.0

    # Single allele cases
    assert allele_multiset_score(("A",), ("A",)) == 1.0
    assert allele_multiset_score(("A",), ("T",)) == 0.0

    # Complex allele strings (indels)
    assert allele_multiset_score(("ATG", "A"), ("ATG", "A")) == 1.0
    assert allele_multiset_score(("ATG", "A"), ("ATG", "C")) == 0.5
    assert allele_multiset_score(("AAG", "C"), ("ATG", "C")) == 0.5
    assert allele_multiset_score(("ATGCCC", "DEL"), ("ATGCCC", "DEL")) == 1.0

    # Many copies of same allele
    truth_many_a = ("A", "A", "A", "A")  # 4 copies of A
    imputed_some_a = ("A", "A", "T", "T")  # 2 copies of A
    assert allele_multiset_score(truth_many_a, imputed_some_a) == 0.5  # 2 out of 4 A's match

    # More complex multiset intersections
    truth = ("A", "A", "B", "B", "C")  # 2 A's, 2 B's, 1 C
    imputed = ("A", "B", "B", "C", "C")  # 1 A, 2 B's, 2 C's
    # Intersection: 1 A (min(2,1)), 2 B's (min(2,2)), 1 C (min(1,2)) = 4 total
    expected = 4.0 / 5.0  # 4 matches out of 5 truth alleles
    assert allele_multiset_score(truth, imputed) == expected

def test_allele_multiset_score_matches_counter_reference_fuzz():
    """The list-based allele_multiset_score must produce byte-identical
    results to the old Counter-based implementation across randomized
    inputs -- covers ploidy/alphabet/duplicate combinations the hand-picked
    cases above don't enumerate."""
    random.seed(1234)
    alphabets = [
        ["A", "T", "C", "G"],
        ["A", "T"],
        ["A", "T", "C", "G", "ATG", "DEL", "N"],
    ]
    for _ in range(5000):
        alphabet = random.choice(alphabets)
        truth_len = random.randint(0, 8)
        imputed_len = random.randint(0, 8)
        truth_alleles = tuple(random.choices(alphabet, k=truth_len))
        imputed_alleles = tuple(random.choices(alphabet, k=imputed_len))
        for phase_sensitive in (False, True):
            got = allele_multiset_score(truth_alleles, imputed_alleles, phase_sensitive)
            want = _allele_multiset_score_counter_reference(truth_alleles, imputed_alleles, phase_sensitive)
            assert got == want, (truth_alleles, imputed_alleles, phase_sensitive, got, want)

def test_allele_multiset_score_matches_counter_reference_high_ploidy():
    """Same equivalence check but skewed toward larger multisets (polyploid
    edge cases) and heavy duplication, where the O(n^2) list-remove approach
    is most likely to diverge from Counter's O(n) hashing if the logic were
    wrong (e.g. removing the same element twice, or picking the wrong count
    on ties)."""
    random.seed(5678)
    for _ in range(2000):
        n = random.randint(0, 12)
        m = random.randint(0, 12)
        # Small alphabet forces lots of duplicate/tie scenarios.
        alphabet = ["A", "T"]
        truth_alleles = tuple(random.choices(alphabet, k=n))
        imputed_alleles = tuple(random.choices(alphabet, k=m))
        got = allele_multiset_score(truth_alleles, imputed_alleles, phase_sensitive=False)
        want = _allele_multiset_score_counter_reference(truth_alleles, imputed_alleles, phase_sensitive=False)
        assert got == want, (truth_alleles, imputed_alleles, got, want)

def test_iter_records():
    pass

def test_compare_sorted():
    pass

def test_initialize_counts():
    NBINS=20
    assert initialize_counts() == {
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
        "af_bins": [i / 20 for i in range(21)],
        "af_bin_total": [0] * NBINS,
        "af_bin_correct": [0] * NBINS,
        "af_bin_true_alt_counts": [[] for _ in range(NBINS)],
        "af_bin_imp_alt_counts": [[] for _ in range(NBINS)],
        "het_bin_total": [0] * NBINS,
        "het_bin_correct": [0] * NBINS}

def test_write_sites():
    """Test basic site writing functionality"""
    output = io.StringIO()
    writer = OutputWriter(output, only_mismatches=False, only_matched_sites=False)

    # Test data
    key = ("chr1", 100, "A")
    t_variant = ("T", "AF=0.5", "0/1")
    i_variant = ("T", "AF=0.6", "1/1")

    writer.write_site(SiteKind.MATCH, key, t_variant, i_variant)

    assert output.getvalue() == "MATCH\tchr1\t100\tA\tT\tAF=0.5\t0/1\tT\tAF=0.6\t1/1\n"

    """Test writing with None values (converts to dots)"""
    key = ("chr1", 200, "G")
    t_variant = (None, None, None)  # Missing truth data
    i_variant = ("C", "AF=0.3", "./.")

    writer.write_site(SiteKind.EXTRA_IN_IMPUTED_SITE, key, t_variant, i_variant)

    assert output.getvalue().strip().split('\n')[-1] == "EXTRA_IN_IMPUTED_SITE\tchr1\t200\tG\t.\t.\t.\tC\tAF=0.3\t./."

    """Test writing with some None values"""
    key = ("chrX", 500, "AT")
    t_variant = ("A", None, "0/1")  # Missing info
    i_variant = ("A", "TYPE=DEL", None)  # Missing GT

    writer.write_site(SiteKind.MISMATCH_GT, key, t_variant, i_variant)

    assert output.getvalue().strip().split('\n')[-1] == "MISMATCH_GT\tchrX\t500\tAT\tA\t.\t0/1\tA\tTYPE=DEL\t."

    """Test with no filters - should write everything"""
    output = io.StringIO()
    writer = OutputWriter(output, only_mismatches=False, only_matched_sites=False)

    key = ("chr1", 100, "A")
    t_variant = ("T", "AF=0.5", "0/1")
    i_variant = ("T", "AF=0.6", "1/1")

    all_kinds = [
        SiteKind.MATCH,
        SiteKind.MISMATCH_ALLELE,
        SiteKind.MISSING_IN_IMPUTED_SITE,
        SiteKind.EXTRA_IN_IMPUTED_SITE,
        SiteKind.MISMATCH_GT,
        SiteKind.MATCH_ALLELE,
    ]

    for kind in all_kinds:
        writer.write_site(kind, key, t_variant, i_variant)

    result = output.getvalue()
    lines = result.strip().split('\n')
    assert len(lines) == len(all_kinds)

    """Test only_matched_sites=True filtering"""
    output = io.StringIO()
    writer = OutputWriter(output, only_mismatches=False, only_matched_sites=True)
    key = ("chr1", 100, "A")
    t_variant = ("T", "AF=0.5", "0/1")
    i_variant = ("T", "AF=0.6", "1/1")

    # Should write - not an unmatched site
    writer.write_site(SiteKind.MATCH, key, t_variant, i_variant)
    writer.write_site(SiteKind.MISMATCH_ALLELE, key, t_variant, i_variant)
    writer.write_site(SiteKind.MISMATCH_GT, key, t_variant, i_variant)

    # Should NOT write - unmatched sites
    writer.write_site(SiteKind.MISSING_IN_IMPUTED_SITE, key, t_variant, i_variant)
    writer.write_site(SiteKind.EXTRA_IN_IMPUTED_SITE, key, t_variant, i_variant)

    result = output.getvalue()
    lines = result.strip().split('\n')
    assert len(lines) == 3  # Only 3 lines should be written
    assert "MATCH" in lines[0]
    assert "MISMATCH_ALLELE" in lines[1]
    assert "MISMATCH_GT" in lines[2]

    """Test only_mismatches=True filtering"""
    output = io.StringIO()
    writer = OutputWriter(output, only_mismatches=True, only_matched_sites=False)

    key = ("chr1", 100, "A")
    t_variant = ("T", "AF=0.5", "0/1")
    i_variant = ("T", "AF=0.6", "1/1")

    # Should write - mismatch sites
    writer.write_site(SiteKind.MISMATCH_ALLELE, key, t_variant, i_variant)
    writer.write_site(SiteKind.MISSING_IN_IMPUTED_SITE, key, t_variant, i_variant)
    writer.write_site(SiteKind.EXTRA_IN_IMPUTED_SITE, key, t_variant, i_variant)

    # Should NOT write - matches
    writer.write_site(SiteKind.MATCH, key, t_variant, i_variant)
    writer.write_site(SiteKind.MATCH_ALLELE, key, t_variant, i_variant)

    result = output.getvalue()
    lines = result.strip().split('\n')
    assert len(lines) == 3  # Only 3 lines should be written
    assert "MISMATCH_ALLELE" in lines[0]
    assert "MISSING_IN_IMPUTED_SITE" in lines[1]
    assert "EXTRA_IN_IMPUTED_SITE" in lines[2]

    """Test both only_matched_sites=True and only_mismatches=True (conflicting)"""
    output = io.StringIO()
    writer = OutputWriter(output, only_mismatches=True, only_matched_sites=True)

    key = ("chr1", 100, "A")
    t_variant = ("T", "AF=0.5", "0/1")
    i_variant = ("T", "AF=0.6", "1/1")

    # Should write - mismatch but not unmatched
    writer.write_site(SiteKind.MISMATCH_ALLELE, key, t_variant, i_variant)

    # Should NOT write - unmatched (filtered by only_matched_sites)
    writer.write_site(SiteKind.MISSING_IN_IMPUTED_SITE, key, t_variant, i_variant)
    writer.write_site(SiteKind.EXTRA_IN_IMPUTED_SITE, key, t_variant, i_variant)

    # Should NOT write - matches (filtered by only_mismatches)
    writer.write_site(SiteKind.MATCH, key, t_variant, i_variant)

    result = output.getvalue()
    lines = result.strip().split('\n')
    assert len(lines) == 1  # Only MISMATCH_ALLELE should be written
    assert "MISMATCH_ALLELE" in lines[0]

    """Test with None output handle"""
    writer = OutputWriter(None, only_mismatches=False, only_matched_sites=False)

    key = ("chr1", 100, "A")
    t_variant = ("T", "AF=0.5", "0/1")
    i_variant = ("T", "AF=0.6", "1/1")

    # Should not crash, should do nothing
    writer.write_site(SiteKind.MATCH, key, t_variant, i_variant)
    # No assert needed - just checking it doesn't crash

    """Test with complex real-world data"""
    output = io.StringIO()
    writer = OutputWriter(output, only_mismatches=False, only_matched_sites=False)

    # Multi-allelic site
    key = ("chr2", 12345, "ATCG")
    t_variant = ("A,AT", "AF=0.3,0.2;AC=30,20;AN=100", "1/2")
    i_variant = ("A,ATCGGG", "AF=0.25,0.15;AC=25,15;AN=100", "2/1")

    writer.write_site(SiteKind.MISMATCH_ALLELE, key, t_variant, i_variant)

    assert output.getvalue() == "MISMATCH_ALLELE\tchr2\t12345\tATCG\tA,AT\tAF=0.3,0.2;AC=30,20;AN=100\t1/2\tA,ATCGGG\tAF=0.25,0.15;AC=25,15;AN=100\t2/1\n"

def test_handle_missing_in_imputed():
    pass

def test_handle_extra_in_imputed():
    pass

def test_handle_matched_sites():
    pass

def test_extract_allele_frequency():
    assert extract_allele_frequency(t_info="AN=10;AC=2") == (2, 10)
    assert extract_allele_frequency(t_info="AC=2;AN=10") == (2, 10)
    assert extract_allele_frequency(t_info="AN=10") == (None, 10)
    assert extract_allele_frequency(t_info="AC=2") == (2, None)
    assert extract_allele_frequency(t_info="AN=10AC=2") == (None, None)
    assert extract_allele_frequency(t_info=".") == (None, None)

def test_update_frequency_bins():
    pass

def test_handle_missing_genotypes():
    pass

def test_next_or_none():
    pass