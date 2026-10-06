from pathlib import Path

import numpy as np
import pytest

from python.seq_extract.assembly import AssemblySet, fai_path_for
from python.seq_extract.fai import FastaIndexError, RegionError, index_fasta
from python.seq_extract.projection import (Projection, ProjectionQuery, TableProvider, TruthMapProvider,
                                           bare_chrom, identity_projections, read_projection_table,
                                           write_projection_queries, write_projection_table)

HEADER = "qid\tref_contig\tref_start\tref_end\tassembly\tcontig\tstart\tend\tstrand\tstatus\n"


def _fa(path, recs, width=10):
    with open(path, "w") as f:
        for n, s in recs:
            f.write(f">{n}\n")
            for k in range(0, len(s), width):
                f.write(s[k:k + width] + "\n")
    return path


# --------------------------------------------------------------------------- AssemblySet

@pytest.fixture
def asm_dir(tmp_path):
    d = tmp_path / "fa"
    d.mkdir()
    _fa(d / "Oh43_renamed.fa", [("Oh43_chr1", "ACGTACGTAC" * 3), ("Oh43_chr2", "TTTTGGGG")])
    _fa(d / "Oh43_v2_renamed.fa", [("Oh43_v2_chr1", "CCCCAAAA")])
    _fa(d / "B73_renamed.fa", [("B73_chr1", "GATTACA" * 5)])
    return d


def test_assembly_set_from_dir_glob(asm_dir, tmp_path):
    s = AssemblySet.from_dir(asm_dir, "{asm}_renamed.fa", fai_dir=tmp_path / "fai", build_missing=True)
    assert sorted(s.assemblies) == ["B73", "Oh43", "Oh43_v2"]
    assert s.assembly_of("Oh43_chr1") == "Oh43"
    assert s.assembly_of("Oh43_v2_chr1") == "Oh43_v2"
    assert s.fetch("Oh43_chr2", 2, 6) == "TTGG"
    assert s.fetch("Oh43_v2_chr1", 2, 6, "-") == "TTGG"
    assert s.length("B73_chr1") == 35
    assert (tmp_path / "fai" / "Oh43_renamed.fa.fai").exists()
    assert not (asm_dir / "Oh43_renamed.fa.fai").exists()
    assert s.fetch_many([("Oh43_chr1", 0, 4), ("B73_chr1", 0, 7, "-")]) == ["ACGT", "TGTAATC"]
    s.close()


def test_assembly_set_explicit_list_and_errors(asm_dir, tmp_path):
    s = AssemblySet.from_dir(asm_dir, "{asm}_renamed.fa", assemblies=["Oh43", "Mo17"], fai_dir=tmp_path / "f2")
    with pytest.raises(FastaIndexError, match="missing index"):
        s.get("Oh43")
    with pytest.raises(FileNotFoundError):
        s.get("Mo17")
    with pytest.raises(RegionError, match="unknown assembly"):
        s.get("Ki3")
    with pytest.raises(RegionError, match="known assembly"):
        s.assembly_of("Ki3_chr1")
    with pytest.raises(RegionError, match="known assembly"):
        s.assembly_of("Oh43_")
    with pytest.raises(ValueError, match="fai_dir"):
        AssemblySet({"a": "x.fa"}, build_missing=True)
    with pytest.raises(ValueError, match="asm"):
        AssemblySet.from_dir(asm_dir, "x.fa")
    with pytest.raises(FileNotFoundError):
        AssemblySet.from_dir(asm_dir, "{asm}.fasta")


def test_assembly_set_existing_fai_next_to_fasta(asm_dir):
    index_fasta(asm_dir / "B73_renamed.fa", fai_path_for(asm_dir / "B73_renamed.fa"))
    s = AssemblySet.from_dir(asm_dir, "{asm}_renamed.fa", assemblies=["B73"])
    assert s.fetch("B73_chr1", 0, 7) == "GATTACA"
    with pytest.raises(RegionError, match="unknown sequence"):
        s.fetch("B73_chr9", 0, 1)


def test_fai_path_for():
    assert fai_path_for("/a/b/x.fa") == Path("/a/b/x.fa.fai")
    assert fai_path_for("/a/b/x.fa", "/z") == Path("/z/x.fa.fai")


# --------------------------------------------------------------------------- TruthMapProvider

@pytest.fixture
def maps_dir(tmp_path):
    d = tmp_path / "maps"
    d.mkdir()
    # 10 ref bases; ref 3-5 deleted; 4 inserted bases before ref 7
    np.save(d / "A_chr1.npy", np.array([0, 1, 2, -1, -1, -1, 3, 8, 9, 10, 11], np.int64))
    np.save(d / "B73_chr1.npy", np.arange(11, dtype=np.int64))
    return d


@pytest.mark.parametrize("s,e,exp", [
    (0, 10, (0, 11)), (0, 3, (0, 3)), (2, 7, (2, 4)), (3, 6, None), (3, 7, (3, 4)), (6, 8, (3, 9)),
    (7, 8, (8, 9)), (9, 10, (10, 11)), (5, 7, (3, 4)),
])
def test_truth_map_projection(maps_dir, s, e, exp):
    p = TruthMapProvider(maps_dir)
    pr = p.project("chr1", s, e, "A")
    if exp is None:
        assert pr.status == "deleted" and not pr.ok
    else:
        assert pr.ok and (pr.contig, pr.start, pr.end, pr.strand) == ("A_chr1", *exp, "+")


def test_truth_map_statuses_and_prefix(maps_dir):
    p = TruthMapProvider(maps_dir)
    assert p.project("chr2", 0, 5, "A").status == "no_map"
    assert p.project("chr1", 0, 11, "A").status == "out_of_range"
    assert p.project("chr1", -1, 3, "A").status == "out_of_range"
    assert p.project("chr1", 4, 4, "A").status == "out_of_range"
    pr = p.project("B73_chr1", 2, 7, "B73")
    assert (pr.ref_chrom, pr.contig, pr.start, pr.end) == ("chr1", "B73_chr1", 2, 7)


def test_bare_chrom():
    assert bare_chrom("B73_chr1") == "chr1"
    assert bare_chrom("chr1") == "chr1"
    assert bare_chrom("B73_") == "B73_"
    assert bare_chrom("Ref_chr3", "Ref") == "chr3"


# --------------------------------------------------------------------------- TableProvider

def test_table_provider_roundtrip(tmp_path):
    rows = [Projection("chr1", 0, 10, "Oh43", "Oh43_chr1", 5, 16, "+", "ok", "a"),
            Projection("chr1", 10, 20, "Oh43", "Oh43_chr1", 16, 25, "-", "ok", "b"),
            Projection("chr2", 0, 10, "Mo17", None, None, None, "+", "deleted", "c")]
    p = tmp_path / "t.tsv"
    write_projection_table(p, rows)
    assert p.read_text().startswith(HEADER.rstrip("\n") + "\tmethod\tsnap\tsnap_shift\n")
    assert "B73_chr1" in p.read_text()
    back = read_projection_table(p)
    assert back == [Projection("B73_chr1", 0, 10, "Oh43", "Oh43_chr1", 5, 16, "+", "ok", "a"),
                    Projection("B73_chr1", 10, 20, "Oh43", "Oh43_chr1", 16, 25, "-", "ok", "b"),
                    Projection("B73_chr2", 0, 10, "Mo17", None, None, None, "+", "deleted", "c")]
    t = TableProvider(back)
    assert len(t) == 3
    assert t.project("chr1", 10, 20, "Oh43").strand == "-"
    assert t.project("B73_chr1", 0, 10, "Oh43").start == 5
    assert t.project("chr2", 0, 10, "Mo17").status == "deleted"
    miss = t.project("chr1", 0, 11, "Oh43")
    assert miss.status == "not_in_table" and not miss.ok


def test_table_provider_from_tsv_extra_columns(tmp_path):
    p = tmp_path / "t.tsv"
    p.write_text(HEADER.rstrip("\n") + "\textra\n" + "q\tchr1\t0\t5\tA\tA_chr1\t0\t5\t+\tok\tfoo\n\n")
    t = TableProvider.from_tsv(p)
    assert t.project("chr1", 0, 5, "A").contig == "A_chr1"


@pytest.mark.parametrize("body,msg", [
    ("q\tchr1\t0\t5\tA\t.\t.\t.\t.\tok\n", "missing"),
    ("q\tchr1\t0\t5\tA\tA_chr1\t5\t5\t+\tok\n", "invalid interval"),
    ("q\tchr1\tx\t5\tA\tA_chr1\t0\t5\t+\tok\n", "line 2"),
    ("q\tchr1\t0\t5\tA\n", "columns"),
])
def test_table_malformed(tmp_path, body, msg):
    p = tmp_path / "t.tsv"
    p.write_text(HEADER + body)
    with pytest.raises(ValueError, match=msg):
        read_projection_table(p)


def test_table_bad_header(tmp_path):
    p = tmp_path / "t.tsv"
    p.write_text("a\tb\n")
    with pytest.raises(ValueError, match="header"):
        read_projection_table(p)


def test_table_conflict_and_duplicate():
    a = Projection("chr1", 0, 5, "A", "A_chr1", 0, 5, "+", "ok")
    TableProvider([a, a])
    with pytest.raises(ValueError, match="conflicting"):
        TableProvider([a, Projection("chr1", 0, 5, "A", "A_chr1", 1, 6, "+", "ok")])


def test_queries_and_identity(tmp_path):
    qs = [ProjectionQuery("q0", "chr1", 0, 5, "A"), ProjectionQuery("q1", "chr2", 3, 9, "B")]
    p = tmp_path / "q.tsv"
    write_projection_queries(p, qs)
    lines = p.read_text().splitlines()
    assert lines[0] == "qid\tref_contig\tref_start\tref_end\tassembly"
    assert lines[2] == "q1\tB73_chr2\t3\t9\tB"
    ids = identity_projections(qs)
    assert ids[1] == Projection("chr2", 3, 9, "B", "B_chr2", 3, 9, "+", "ok", "q1")


def test_table_snap_columns_and_ok_clamped(tmp_path):
    p = tmp_path / "t.tsv"
    p.write_text(HEADER.rstrip("\n") + "\tmethod\tsnap\tsnap_shift\n"
                 "p0\tB73_chr1\t0\t50\tOh43\tOh43_chr1\t3\t60\t+\tok\tlocal,local\tsnap,snap_walk\t0,-8\n"
                 "p1\tB73_chr1\t50\t90\tOh43\tOh43_chr1\t60\t100\t-\tok_clamped\tlocal1,local1\tproj_lowid,snap\t0,-9006\n"
                 "p2\tB73_chr1\t90\t99\tOh43\t.\t.\t.\t.\tprojection_failed: projected end precedes projected start"
                 "\tmedian,local1\tsnap_walk,snap\t6688,0\n")
    rows = read_projection_table(p)
    assert rows[0].method == "local,local" and rows[0].snap == ("snap", "snap_walk") and rows[0].snap_shift == (0, -8)
    assert rows[0].ok and not rows[0].clamped
    assert rows[1].ok and rows[1].clamped and rows[1].strand == "-"
    assert not rows[2].ok and rows[2].status.startswith("projection_failed")
    t = TableProvider(rows)
    assert t.project("chr1", 50, 90, "Oh43").clamped
    out = tmp_path / "o.tsv"
    write_projection_table(out, rows)
    assert read_projection_table(out) == rows


def test_table_ok_clamped_requires_coordinates(tmp_path):
    p = tmp_path / "t.tsv"
    p.write_text(HEADER + "q\tchr1\t0\t5\tA\t.\t.\t.\t.\tok_clamped\n")
    with pytest.raises(ValueError, match="missing"):
        read_projection_table(p)
