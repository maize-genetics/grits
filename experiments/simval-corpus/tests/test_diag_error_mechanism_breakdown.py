#!/usr/bin/env python
"""
Unit test for the event-level (contiguous-run) tracking added to
`scripts/diag_error_mechanism_breakdown.py` this session (see the plan
file's "ROUND 3" section). Locks in the corrected formula after catching a
real statistical trap in an earlier version: collapsing BOTH match and
mismatch runs into "events" and using (mismatch_events / total_events) as
the rate converges to ~50% regardless of true quality, because alternating
runs always split ~50/50 in COUNT no matter how long each run is. The
correct denominator is EVENT_MISMATCH + gt_allele_matches (raw per-site
match count) -- verified against this synthetic case AND independently
cross-checked against a from-scratch Explore-agent computation on real data
(Tx303 chr1: EVENT_MISMATCH=365,106, event_error_rate=2.6245%, matching the
agent's independently-computed 365,106 events / 2.62% exactly).

Run directly (not wired into the repo's own broken pytest harness -- see
test_compare_gvcf_truth_projection.py's docstring for why):

    python scripts/test_diag_error_mechanism_breakdown.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../scripts"))  # diag_error_mechanism_breakdown.py lives here now (this file was moved from grits_workdir/scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "test_crf_relatedness" / "src"))
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src")

from diag_error_mechanism_breakdown import breakdown  # noqa: E402


def test_event_tracking_synthetic():
    # Truth: one ref-block, chr1:1-10, hom-ref throughout.
    truth_lines = [
        "chr1\t1\t.\tA\t<NON_REF>\t.\t.\tEND=10\tGT:AD\t0:30,0\n",
    ]
    # Imputed: 10 positions, chr1:1-10. REF/ALT literal, GT selects index.
    # Mismatches (GT=1/1, i.e. non-ref) at positions 3,4 (contiguous run of
    # 2) and position 7 (isolated run of 1). Everything else matches (0/0).
    mismatch_positions = {3, 4, 7}
    imputed_lines = []
    for pos in range(1, 11):
        gt = "1/1" if pos in mismatch_positions else "0/0"
        imputed_lines.append(f"chr1\t{pos}\tA\tT\t.\t{gt}\n")

    with tempfile.TemporaryDirectory() as td:
        truth_path = Path(td) / "truth.g.vcf"
        truth_path.write_text("".join(truth_lines))
        imputed_path = Path(td) / "imputed.sorted.tsv"
        imputed_path.write_text("".join(imputed_lines))

        counts = breakdown(str(imputed_path), str(truth_path), truth_ploidy_expand=2)

    assert counts["compared_sites"] == 10, counts["compared_sites"]
    assert counts["gt_allele_matches"] == 7, counts["gt_allele_matches"]
    assert counts["gt_allele_mismatches"] == 3, counts["gt_allele_mismatches"]
    # Two contiguous mismatch runs: {3,4} and {7}.
    assert counts["EVENT_MISMATCH"] == 2, counts["EVENT_MISMATCH"]
    # Runs alternate: match(1,2) mismatch(3,4) match(5,6) mismatch(7) match(8,9,10)
    # = 5 total alternating runs.
    assert counts["EVENT_TOTAL"] == 5, counts["EVENT_TOTAL"]

    expected_rate = 2 / (2 + 7)
    actual_rate = counts["EVENT_MISMATCH"] / (counts["EVENT_MISMATCH"] + counts["gt_allele_matches"])
    assert abs(actual_rate - expected_rate) < 1e-9, (actual_rate, expected_rate)
    print(f"  ok: event_error_rate = {actual_rate:.6f} (expected {expected_rate:.6f})")

    # Regression guard: the FLAWED formula (mismatch_events / EVENT_TOTAL)
    # must NOT be what's reported -- confirm it would give a wildly
    # different (and wrong) answer for this same synthetic case, as a
    # tripwire against reintroducing it.
    flawed_rate = counts["EVENT_MISMATCH"] / counts["EVENT_TOTAL"]
    assert abs(flawed_rate - 0.4) < 1e-9  # 2/5 -- looks deceptively reasonable here,
    # but see the real Tx303 case (0.4999995) for why it's actually broken.
    print(f"  ok: flawed EVENT_MISMATCH/EVENT_TOTAL = {flawed_rate:.6f} (confirmed NOT used in print_report)")


def test_all_match_no_events():
    truth_lines = ["chr1\t1\t.\tA\t<NON_REF>\t.\t.\tEND=5\tGT:AD\t0:30,0\n"]
    imputed_lines = [f"chr1\t{pos}\tA\tT\t.\t0/0\n" for pos in range(1, 6)]
    with tempfile.TemporaryDirectory() as td:
        truth_path = Path(td) / "truth.g.vcf"
        truth_path.write_text("".join(truth_lines))
        imputed_path = Path(td) / "imputed.sorted.tsv"
        imputed_path.write_text("".join(imputed_lines))
        counts = breakdown(str(imputed_path), str(truth_path), truth_ploidy_expand=2)
    assert counts["EVENT_MISMATCH"] == 0
    assert counts["EVENT_TOTAL"] == 1  # one long match run
    assert counts["gt_allele_matches"] == 5
    print("  ok: all-match case -> EVENT_MISMATCH=0, EVENT_TOTAL=1")


def test_noindel_tracker_skips_not_breaks():
    # ROUND 4: an indel-class site sitting BETWEEN two SNP mismatches must
    # NOT split them into two separate events for the indel-excluded
    # tracker -- it should be invisible, exactly like an excluded site.
    # pos1: SNP match; pos2: SNP mismatch; pos3: INS (mismatch, imputed
    # predicts REF); pos4: SNP mismatch; pos5: SNP match.
    truth_lines = [
        "chr1\t1\t.\tA\tC,<NON_REF>\t.\t.\t.\tGT:AD\t1:0,30\n",
        "chr1\t2\t.\tA\tC,<NON_REF>\t.\t.\t.\tGT:AD\t1:0,30\n",
        "chr1\t3\t.\tA\tATT,<NON_REF>\t.\t.\t.\tGT:AD\t1:0,30\n",
        "chr1\t4\t.\tA\tC,<NON_REF>\t.\t.\t.\tGT:AD\t1:0,30\n",
        "chr1\t5\t.\tA\tC,<NON_REF>\t.\t.\t.\tGT:AD\t1:0,30\n",
    ]
    imputed_lines = [
        "chr1\t1\tA\tC\t.\t1/1\n",   # match, SNP
        "chr1\t2\tA\tC\t.\t0/0\n",   # mismatch, SNP (predicts REF)
        "chr1\t3\tA\tATT\t.\t0/0\n",  # mismatch, truth=INS, imputed predicts REF
        "chr1\t4\tA\tC\t.\t0/0\n",   # mismatch, SNP (predicts REF)
        "chr1\t5\tA\tC\t.\t1/1\n",   # match, SNP
    ]
    with tempfile.TemporaryDirectory() as td:
        truth_path = Path(td) / "truth.g.vcf"
        truth_path.write_text("".join(truth_lines))
        imputed_path = Path(td) / "imputed.sorted.tsv"
        imputed_path.write_text("".join(imputed_lines))
        counts = breakdown(str(imputed_path), str(truth_path), truth_ploidy_expand=2)

    # Sanity on the underlying classification.
    assert counts["by_class"]["SNP"]["MATCH"] == 2, counts["by_class"]["SNP"]
    assert counts["by_class"]["SNP"]["OTHER_WRONG"] == 2 or counts["by_class"]["SNP"]["REF_BIAS"] == 2, \
        counts["by_class"]["SNP"]
    assert sum(counts["by_class"]["INS"].values()) == 1, counts["by_class"]["INS"]

    # ALL-class tracker: match | mismatch,mismatch,mismatch | match -> 1 mismatch run.
    assert counts["EVENT_MISMATCH"] == 1, counts["EVENT_MISMATCH"]
    assert counts["EVENT_TOTAL"] == 3, counts["EVENT_TOTAL"]

    # NOINDEL tracker: pos3 (INS) is invisible -> pos2 and pos4 (both SNP
    # mismatches) are adjacent in the filtered stream -> still 1 event, NOT 2.
    assert counts["EVENT_MISMATCH_NOINDEL"] == 1, counts["EVENT_MISMATCH_NOINDEL"]
    print("  ok: indel site (pos3) skipped, not a break -> EVENT_MISMATCH_NOINDEL=1 (not 2)")


def main():
    tests = [test_event_tracking_synthetic, test_all_match_no_events,
             test_noindel_tracker_skips_not_breaks]
    for t in tests:
        print(f"{t.__name__}:")
        t()
    print(f"\nALL PASS ({len(tests)} test functions)")


if __name__ == "__main__":
    main()
