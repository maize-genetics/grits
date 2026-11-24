import numpy as np
import pytest

from python.seq_extract.assembly import AssemblySet
from python.seq_extract.cli import main
from python.seq_extract.haplotype import BedInterval, HaplotypeBuilder, Window, group_bed
from python.seq_extract.projection import (OK, OK_CLAMPED, Projection, SpanCheckProvider, TableProvider,
                                           span_implausible)

MSG = "projection_failed: projected span implausibly long"


class Fixed:
    def __init__(self, pr):
        self.pr = pr

    def project(self, *a):
        return self.pr


def prj(ref_len, prj_len, status=OK, start=1000):
    return Projection("chr1", 0, ref_len, "A", "A_chr1", start, start + prj_len, "+", status)


def check(pr, **kw):
    return SpanCheckProvider(Fixed(pr), **kw).project("chr1", pr.ref_start, pr.ref_end, "A")


def test_rule_trips_when_both_excess_and_ratio():
    p = check(prj(8000, 60000))
    assert not p.ok and p.status == f"{MSG} (60000 bp for 8000 bp)"
    assert p.contig is None and p.start is None and p.end is None


@pytest.mark.parametrize("ref_len,prj_len", [(8000, 58000), (8000, 58001)])
def test_excess_boundary_is_strict(ref_len, prj_len):
    assert check(prj(ref_len, prj_len)).ok == (prj_len - ref_len <= 50000)


def test_long_plausible_insertion_passes():
    # 40 kb insertion: excess under 50 kb
    assert check(prj(8000, 48000)).ok
    # 80 kb excess but ratio only 1.8x on a long interval
    p = check(prj(100000, 180000))
    assert p.ok and p.status == OK


def test_ratio_boundary_is_strict():
    assert check(prj(30000, 90000)).ok              # exactly 3x: not > 3x
    assert not check(prj(30000, 90001)).ok


@pytest.mark.parametrize("kw", [dict(max_excess=0), dict(max_ratio=0), dict(max_excess=0, max_ratio=0)])
def test_disable_switches(kw):
    assert check(prj(8000, 600000), **kw).ok


def test_custom_thresholds_and_function():
    assert not check(prj(100, 400), max_excess=200, max_ratio=2.0).ok
    assert check(prj(100, 250), max_excess=200, max_ratio=2.0).ok
    assert span_implausible(8000, 60000) and not span_implausible(8000, 60000, 0, 3.0)
    with pytest.raises(ValueError):
        SpanCheckProvider(None, max_excess=-1)


def test_ok_clamped_also_checked_and_kept_when_plausible():
    p = check(prj(8000, 60000, OK_CLAMPED))
    assert not p.ok and p.status.startswith(MSG)
    p = check(prj(8000, 7000, OK_CLAMPED))
    assert p.ok and p.clamped


def test_failures_and_short_projections_pass_through():
    bad = Projection("chr1", 0, 8000, "A", status="deleted")
    assert check(bad) == bad
    assert check(prj(8000, 10)).ok


def test_applies_to_table_rows_already_ok():
    t = TableProvider([prj(8000, 70000)])
    assert t.project("chr1", 0, 8000, "A").ok
    assert not SpanCheckProvider(t).project("chr1", 0, 8000, "A").ok


def _world(tmp_path):
    rng = np.random.default_rng(0)
    R = "".join(rng.choice(list("ACGT"), 1000))
    A = "".join(rng.choice(list("ACGT"), 200000))
    d = tmp_path / "fa"
    d.mkdir()
    (d / "B73.fa").write_text(f">B73_chr1\n{R}\n")
    (d / "A.fa").write_text(f">A_chr1\n{A}\n")
    asm = AssemblySet.from_dir(d, "{asm}.fa", fai_dir=tmp_path / "fai", build_missing=True)
    rows = [Projection("chr1", 0, 500, "A", "A_chr1", 0, 100000, "+", OK),
            Projection("chr1", 500, 1000, "A", "A_chr1", 100000, 100600, "+", OK)]
    return R, A, d, asm, rows


def test_builder_integration_n_fills_and_records(tmp_path):
    R, A, d, asm, rows = _world(tmp_path)
    bed = group_bed([BedInterval("chr1", 0, 500, "A", "B73"), BedInterval("chr1", 500, 1000, "B73", "A")])
    b = HaplotypeBuilder(bed, asm, SpanCheckProvider(TableProvider(rows)))
    r1 = b.build(Window("w", "chr1", 0, 1000), 1)
    assert r1.seq == "N" * 500 + R[500:]
    assert len(r1.failures) == 1 and r1.failed_bases == 500
    assert r1.failures[0].status == f"{MSG} (100000 bp for 500 bp)"
    r2 = b.build(Window("w", "chr1", 0, 1000), 2)
    assert r2.seq == R[:500] + A[100000:100600] and r2.failures == []
    off = HaplotypeBuilder(bed, asm, SpanCheckProvider(TableProvider(rows), max_excess=0))
    assert off.build(Window("w", "chr1", 0, 1000), 1).seq == A[:100000] + R[500:]


def test_cli_span_options(tmp_path):
    R, A, d, asm, rows = _world(tmp_path)
    bd = tmp_path / "bed"
    bd.mkdir()
    (bd / "S_chr1_imputed.bed").write_text("chrom\tstart\tend\tparent1\tparent2\nchr1\t0\t1000\tA\tA\n")
    (tmp_path / "w.tsv").write_text("wid\tchrom\tstart\tend\ng\tchr1\t0\t500\n")
    (tmp_path / "t.tsv").write_text("qid\tref_contig\tref_start\tref_end\tassembly\tcontig\tstart\tend\tstrand\tstatus\n"
                                    "q\tB73_chr1\t0\t500\tA\tA_chr1\t0\t100000\t+\tok\n")
    base = ["haplotypes", "--fasta-dir", str(d), "--fai-dir", str(tmp_path / "fai"), "--build-fai", "--bed", str(bd), "--windows",
            str(tmp_path / "w.tsv"), "--provider", "table", "--table", str(tmp_path / "t.tsv")]
    main(base + ["--out-dir", str(tmp_path / "o1")])
    assert "implausibly long (100000 bp for 500 bp)" in (tmp_path / "o1" / "metadata.tsv").read_text()
    assert (tmp_path / "o1" / "hap1.fa").read_text().split("\n")[1] == "N" * 500
    assert "extract_error" not in (tmp_path / "o1" / "metadata.tsv").read_text()
    main(base + ["--max-span-ratio", "0", "--out-dir", str(tmp_path / "o2")])
    assert (tmp_path / "o2" / "hap1.fa").read_text().split("\n")[1] == A[:100000]
    main(base + ["--max-span-excess", "200000", "--out-dir", str(tmp_path / "o3")])
    assert (tmp_path / "o3" / "hap1.fa").read_text().split("\n")[1] == A[:100000]
