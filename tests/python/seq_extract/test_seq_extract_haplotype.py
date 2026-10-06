"""Builder tests on a hand-made 40 bp world.

Reference R (B73_chr1). Founder A = R lowercased, identity map. Founder B = R with "GGGGG" inserted
before ref base 20 and ref bases 30..34 deleted. Every expected string below is written out by hand from
these definitions.
"""
import random

import numpy as np
import pytest

from python.seq_extract.assembly import AssemblySet
from python.seq_extract.haplotype import (BedInterval, HaplotypeBuilder, Window, group_bed, metadata_row,
                                          read_bed_dir, read_imputed_bed, read_windows, resolve_tiling)
from python.seq_extract.projection import OK, Projection, TableProvider, TruthMapProvider

R = "ACGTTGCAAC" "GTACCATGGT" "CCAGTAGCTA" "TTCGAGGCAT"
A = R.lower()
B = R[:20] + "GGGGG" + R[20:30] + R[35:]
B_MAP = np.array(list(range(20)) + [25 + i for i in range(10)] + [-1] * 5 + [35 + i for i in range(5)] + [40],
                 np.int64)


def _fa(path, recs, width=7):
    with open(path, "w") as f:
        for n, s in recs:
            f.write(f">{n}\n")
            for k in range(0, len(s), width):
                f.write(s[k:k + width] + "\n")


@pytest.fixture
def world(tmp_path):
    d = tmp_path / "fa"
    d.mkdir()
    _fa(d / "B73.fa", [("B73_chr1", R)])
    _fa(d / "A.fa", [("A_chr1", A)])
    _fa(d / "B.fa", [("B_chr1", B)])
    m = tmp_path / "maps"
    m.mkdir()
    np.save(m / "A_chr1.npy", np.arange(41, dtype=np.int64))
    np.save(m / "B_chr1.npy", B_MAP)
    np.save(m / "B73_chr1.npy", np.arange(41, dtype=np.int64))
    asm = AssemblySet.from_dir(d, "{asm}.fa", fai_dir=tmp_path / "fai", build_missing=True)
    yield asm, TruthMapProvider(m)
    asm.close()


def bed(*rows):
    return group_bed([BedInterval("chr1", *r) for r in rows])


def builder(world, rows, policy="N", **kw):
    asm, prov = world
    return HaplotypeBuilder(bed(*rows), asm, prov, policy, **kw)


def W(s, e, wid="w"):
    return Window(wid, "chr1", s, e)


def test_world_definitions():
    assert len(B) == 40 and B[20:25] == "GGGGG"
    present = np.flatnonzero(B_MAP[:40] >= 0)
    assert "".join(B[B_MAP[i]] for i in present) == R[:30] + R[35:]


def test_single_founder(world):
    b = builder(world, [(0, 40, "A", "A")])
    r1, r2 = b.build(W(5, 15), 1), b.build(W(5, 15), 2)
    assert r1.seq == r2.seq == A[5:15]
    assert r1.n_switches == 0 and r1.gap_bases == 0 and r1.failures == []
    (seg,) = r1.segments
    assert (seg.founder, seg.ref_start, seg.ref_end, seg.contig, seg.start, seg.end, seg.status) == \
        ("A", 5, 15, "A_chr1", 5, 15, OK)


def test_single_founder_with_insertion_inside(world):
    b = builder(world, [(0, 40, "B", "B")])
    assert b.build(W(15, 25), 1).seq == R[15:20] + "GGGGG" + R[20:25]


def test_switch_mid_window_het(world):
    b = builder(world, [(0, 10, "A", "B"), (10, 40, "B", "A")])
    r1, r2 = b.build(W(5, 15), 1), b.build(W(5, 15), 2)
    assert r1.seq == A[5:10] + R[10:15]
    assert r2.seq == R[5:10] + A[10:15]
    assert r1.n_switches == r2.n_switches == 1
    assert [s.founder for s in r1.segments] == ["A", "B"]


def test_multiple_switches(world):
    b = builder(world, [(0, 8, "A", "A"), (8, 12, "B", "A"), (12, 16, "A", "A"), (16, 40, "B", "A")])
    r1 = b.build(W(2, 24), 1)
    assert r1.seq == A[2:8] + R[8:12] + A[12:16] + R[16:20] + "GGGGG" + R[20:24]
    assert r1.n_switches == 3
    r2 = b.build(W(2, 24), 2)
    assert r2.seq == A[2:24] and r2.n_switches == 0 and len(r2.segments) == 1


def test_hom_vs_het(world):
    hom = builder(world, [(0, 40, "B", "B")])
    assert hom.build(W(0, 40), 1).seq == hom.build(W(0, 40), 2).seq
    het = builder(world, [(0, 40, "A", "B")])
    assert het.build(W(0, 40), 1).seq == A
    assert het.build(W(0, 40), 2).seq == B


def test_switch_at_insertion_point_drops_insertion(world):
    b = builder(world, [(0, 20, "A", "B"), (20, 40, "B", "B")])
    assert b.build(W(15, 25), 1).seq == A[15:20] + R[20:25]
    assert b.build(W(15, 25), 2).seq == R[15:20] + "GGGGG" + R[20:25]


def test_projection_across_deletion(world):
    b = builder(world, [(0, 40, "B", "B")])
    r = b.build(W(28, 37), 1)
    assert r.seq == R[28:30] + R[35:37]
    assert (r.segments[0].start, r.segments[0].end) == (25 + 8, 37)


def test_segment_entirely_deleted_is_failure_and_N(world):
    b = builder(world, [(0, 30, "A", "A"), (30, 35, "B", "A"), (35, 40, "A", "A")])
    r = b.build(W(25, 40), 1)
    assert r.seq == A[25:30] + "NNNNN" + A[35:40]
    assert len(r.failures) == 1 and r.failures[0].status == "deleted" and r.failed_bases == 5
    assert r.n_switches == 2


def test_gap_policy_N(world):
    b = builder(world, [(0, 10, "A", "A"), (20, 40, "B", "B")])
    r = b.build(W(5, 25), 1)
    assert r.seq == A[5:10] + "N" * 10 + R[20:25]
    assert r.gap_bases == 10 and r.n_switches == 1
    assert [s.founder for s in r.segments] == ["A", None, "B"]


def test_gap_policy_skip(world):
    b = builder(world, [(0, 10, "A", "A"), (20, 40, "B", "B")], "skip")
    r = b.build(W(5, 25), 1)
    assert r.seq == A[5:10] + R[20:25]
    assert r.gap_bases == 10


def test_gap_policy_extend_split_midpoint(world):
    b = builder(world, [(0, 10, "A", "A"), (20, 40, "B", "B")], "extend")
    r = b.build(W(5, 25), 1)
    assert r.seq == A[5:15] + R[15:20] + "GGGGG" + R[20:25]
    assert r.gap_bases == 0
    b = builder(world, [(0, 10, "A", "A"), (21, 40, "B", "B")], "extend")
    assert [(s.ref_start, s.ref_end) for s in b.build(W(0, 30), 1).segments] == [(0, 15), (15, 30)]


def test_gap_policy_extend_same_founder_merges(world):
    b = builder(world, [(0, 10, "B", "A"), (15, 40, "B", "A")], "extend")
    r = b.build(W(5, 25), 1)
    assert r.seq == R[5:20] + "GGGGG" + R[20:25]
    assert len(r.segments) == 1


@pytest.mark.parametrize("policy,before,after", [("N", "N" * 5, "N" * 5), ("skip", "", ""),
                                                 ("extend", A[0:5], R[30:30] + R[35:38])])
def test_window_before_first_and_after_last(world, policy, before, after):
    b = builder(world, [(10, 20, "A", "A"), (20, 28, "B", "B")], policy)
    rb, ra = b.build(W(0, 5), 1), b.build(W(33, 38), 1)
    assert rb.seq == before and ra.seq == after
    if policy != "extend":
        assert rb.gap_bases == 5 and ra.gap_bases == 5 and rb.segments[0].founder is None


def test_window_partially_before_first(world):
    b = builder(world, [(10, 40, "A", "A")])
    r = b.build(W(5, 15), 1)
    assert r.seq == "NNNNN" + A[10:15] and r.gap_bases == 5


def test_chromosome_without_intervals(world):
    asm, prov = world
    for policy in ("N", "extend"):
        b = HaplotypeBuilder({}, asm, prov, policy)
        r = b.build(W(0, 7), 1)
        assert r.seq == "N" * 7 and r.gap_bases == 7 and r.n_switches == 0


def test_reference_identity_shortcut_and_off(world):
    asm, _ = world
    rows = [(0, 40, "B73", "B73")]
    b = HaplotypeBuilder(bed(*rows), asm, TableProvider([]))
    assert b.build(W(3, 9), 1).seq == R[3:9]
    assert b.projection_queries([W(3, 9)]) == []
    b = HaplotypeBuilder(bed(*rows), asm, TableProvider([]), reference_identity=False)
    r = b.build(W(3, 9), 1)
    assert r.seq == "N" * 6 and r.failures[0].status == "not_in_table"


def test_table_provider_minus_strand_and_extract_error(world):
    asm, _ = world
    rows = [Projection("chr1", 0, 10, "A", "A_chr1", 30, 40, "-", OK),
            Projection("chr1", 10, 20, "B", "B_chr1", 35, 45, "+", OK)]
    b = HaplotypeBuilder(bed((0, 10, "A", "A"), (10, 20, "B", "B")), asm, TableProvider(rows))
    r = b.build(W(0, 20), 1)
    comp = {"a": "t", "c": "g", "g": "c", "t": "a"}
    assert r.seq[:10] == "".join(comp[c] for c in reversed(A[30:40]))
    assert r.seq[10:] == "N" * 10
    assert r.failures[0].status.startswith("extract_error")


def test_unknown_founder_assembly_is_failure(world):
    b = builder(world, [(0, 40, "Zz", "A")])
    r = b.build(W(0, 4), 1)
    assert r.seq == "NNNN" and r.failures[0].status == "no_map"


def test_upper_option(world):
    b = builder(world, [(0, 40, "A", "A")], upper=True)
    assert b.build(W(0, 10), 1).seq == R[:10]


def test_projection_queries_unique(world):
    b = builder(world, [(0, 10, "A", "B"), (10, 40, "B", "B")])
    qs = b.projection_queries([W(0, 20, "w1"), W(0, 20, "w2"), W(5, 15, "w3")])
    keys = [(q.ref_chrom, q.ref_start, q.ref_end, q.assembly) for q in qs]
    assert keys == [("chr1", 0, 10, "A"), ("chr1", 10, 20, "B"), ("chr1", 0, 20, "B"),
                    ("chr1", 5, 10, "A"), ("chr1", 10, 15, "B"), ("chr1", 5, 15, "B")]
    assert len({q.qid for q in qs}) == len(qs)


def test_metadata_row(world):
    b = builder(world, [(0, 10, "A", "A"), (20, 30, "B", "B"), (30, 35, "B", "B"), (35, 40, "A", "A")])
    r = b.build(W(5, 40, "gene1"), 1)
    row = metadata_row(r)
    assert row[:11] == ["gene1", 1, "chr1", 5, 40, len(r.seq), 3, 2, 10, 0, 0]
    assert row[11] == "A,B"
    assert "GAP:10-20" in row[12] and row[13] == "."


def test_build_all(world):
    b = builder(world, [(0, 40, "A", "B")])
    out = list(b.build_all([W(0, 5, "x"), W(5, 9, "y")]))
    assert [(a.window.wid, a.hap, c.hap) for a, c in out] == [("x", 1, 2), ("y", 1, 2)]


def test_invalid_args(world):
    with pytest.raises(ValueError, match="gap_policy"):
        builder(world, [], "bogus")
    with pytest.raises(ValueError, match="hap"):
        builder(world, [(0, 5, "A", "A")]).build(W(0, 5), 3)


# --------------------------------------------------------------------------- tiling

def test_resolve_tiling_policies():
    ivs = bed((10, 20, "A", "B"), (25, 30, "C", "B"), (30, 40, "C", "D"))["chr1"]
    inf = 1 << 62
    assert resolve_tiling(ivs, 1, "N") == [(0, 10, None), (10, 20, "A"), (20, 25, None), (25, 40, "C"),
                                           (40, inf, None)]
    assert resolve_tiling(ivs, 2, "skip") == [(0, 10, None), (10, 20, "B"), (20, 25, None), (25, 30, "B"),
                                              (30, 40, "D"), (40, inf, None)]
    assert resolve_tiling(ivs, 1, "extend") == [(0, 22, "A"), (22, inf, "C")]
    assert resolve_tiling(ivs, 2, "extend") == [(0, 30, "B"), (30, inf, "D")]
    assert resolve_tiling([], 1, "extend") == [(0, inf, None)]
    with pytest.raises(ValueError):
        resolve_tiling(ivs, 1, "x")


def test_tiling_random_invariants():
    rng = random.Random(5)
    for _ in range(200):
        rows, pos = [], rng.randrange(0, 5)
        for _ in range(rng.randrange(1, 8)):
            s = pos + rng.choice([0, 0, rng.randrange(1, 6)])
            e = s + rng.randrange(1, 9)
            rows.append(BedInterval("chr1", s, e, rng.choice("AB"), rng.choice("AB")))
            pos = e
        for pol in ("N", "skip", "extend"):
            t = resolve_tiling(group_bed(rows)["chr1"], 1, pol)
            assert t[0][0] == 0
            assert all(a[1] == b[0] for a, b in zip(t, t[1:]))
            assert all(a[2] != b[2] for a, b in zip(t, t[1:]))
            for iv in rows:
                covering = [x for x in t if x[0] <= iv.start < x[1]]
                assert covering[0][2] == iv.parent1


# --------------------------------------------------------------------------- inputs

def test_read_real_format_bed_and_dir(tmp_path):
    d = tmp_path / "bed"
    d.mkdir()
    (d / "S1_chr10_imputed.bed").write_text("chrom\tstart\tend\tparent1\tparent2\n"
                                            "chr10\t1280\t125697\tM162W\tOh43\nchr10\t127232\t665857\tOh43\tP39\n")
    (d / "S1_chr2_imputed.bed").write_text("chrom\tstart\tend\tparent1\tparent2\nchr2\t5\t9\tB73\tB73\n")
    ivs = read_imputed_bed(d / "S1_chr10_imputed.bed")
    assert ivs[0] == BedInterval("chr10", 1280, 125697, "M162W", "Oh43")
    by = read_bed_dir(d)
    assert sorted(by) == ["chr10", "chr2"] and len(by["chr10"]) == 2
    assert read_bed_dir(d / "S1_chr2_imputed.bed")["chr2"][0].parent1 == "B73"
    (d / "S2_chr1_imputed.bed").write_text("chr1\t0\t5\tA\tA\n")
    with pytest.raises(ValueError, match="exactly one sample"):
        read_bed_dir(d)
    assert list(read_bed_dir(d, "S2")) == ["chr1"]
    with pytest.raises(ValueError, match="no \\*_imputed"):
        read_bed_dir(d, "S3")


def test_bed_overlap_and_bad_rows(tmp_path):
    with pytest.raises(ValueError, match="overlapping"):
        group_bed([BedInterval("chr1", 0, 10, "A", "A"), BedInterval("chr1", 5, 15, "A", "A")], overlap="error")
    with pytest.raises(ValueError, match="overlap must"):
        group_bed([], overlap="x")
    p = tmp_path / "x.bed"
    for body, msg in [("chr1\t5\t5\tA\tA\n", "invalid"), ("chr1\t5\tx\tA\tA\n", "non-integer"),
                      ("chr1\t5\t9\tA\n", "columns")]:
        p.write_text(body)
        with pytest.raises(ValueError, match=msg):
            read_imputed_bed(p)


def test_bed_overlap_trim_real_style():
    st = {}
    g = group_bed([BedInterval("chr10", 2584576, 2719745, "CML333", "M162W"),
                   BedInterval("chr10", 2719744, 2746625, "M162W", "NC350"),
                   BedInterval("chr10", 2720000, 2730000, "X", "X"),
                   BedInterval("chr10", 2746620, 2750000, "A", "B")], stats=st)
    assert [(i.start, i.end, i.parent1) for i in g["chr10"]] == [
        (2584576, 2719745, "CML333"), (2719745, 2746625, "M162W"), (2746625, 2750000, "A")]
    assert st == {"n_trimmed_intervals": 2, "trimmed_bases": 1 + 10000 + 5, "n_dropped_intervals": 1}


def test_bed_dir_overlap_policy(tmp_path):
    d = tmp_path / "b"
    d.mkdir()
    (d / "S_chr1_imputed.bed").write_text("chrom\tstart\tend\tparent1\tparent2\nchr1\t0\t11\tA\tA\n"
                                          "chr1\t10\t20\tB\tB\n")
    st = {}
    assert read_bed_dir(d, stats=st)["chr1"][1].start == 11 and st["trimmed_bases"] == 1
    with pytest.raises(ValueError, match="overlapping"):
        read_bed_dir(d, overlap="error")


def test_trimmed_overlap_builds(world):
    b = builder(world, [(0, 11, "A", "A"), (10, 40, "B", "B")])
    assert b.build(W(5, 15), 1).seq == A[5:11] + R[11:15]


def test_bed_unsorted_and_prefixed_chrom():
    g = group_bed([BedInterval("B73_chr1", 10, 20, "A", "A"), BedInterval("chr1", 0, 10, "B", "B")])
    assert [(i.start, i.parent1) for i in g["chr1"]] == [(0, "B"), (10, "A")]


def test_read_windows(tmp_path):
    p = tmp_path / "w.tsv"
    p.write_text("wid\tchrom\tstart\tend\ng1\tB73_chr1\t0\t100\ng2\tchr2\t5\t9\n")
    assert read_windows(p) == [Window("g1", "chr1", 0, 100), Window("g2", "chr2", 5, 9)]
    for body, msg in [("g1\tchr1\t5\t5\n", "invalid"), ("g1\tchr1\t0\t5\ng1\tchr1\t0\t5\n", "duplicate"),
                      ("g1\tchr1\tx\t5\n", "bad window")]:
        p.write_text("wid\tchrom\tstart\tend\n" + body)
        with pytest.raises(ValueError, match=msg):
            read_windows(p)
    p.write_text("a\tb\n")
    with pytest.raises(ValueError, match="header"):
        read_windows(p)
