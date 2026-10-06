import os
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from python.seq_extract.fai import (FaiEntry, FastaFile, FastaIndexError, Region, RegionError, build_fai,
                                    index_fasta, read_fai, revcomp, write_fai)

SAMTOOLS = os.environ.get("SAMTOOLS", "/programs/samtools-1.20/bin/samtools")
HAVE_SAMTOOLS = shutil.which(SAMTOOLS) is not None or Path(SAMTOOLS).exists()
needs_samtools = pytest.mark.skipif(not HAVE_SAMTOOLS, reason="samtools not available")


def write_fasta(path, records, width, eol="\n", trailing_newline=True):
    with open(path, "w", newline="") as f:
        parts = []
        for name, seq in records:
            parts.append(f">{name}{eol}")
            for k in range(0, len(seq), width):
                parts.append(seq[k:k + width] + eol)
        txt = "".join(parts)
        if not trailing_newline:
            txt = txt[:-len(eol)]
        f.write(txt)
    return path


def rseq(n, seed, alphabet="ACGT"):
    r = random.Random(seed)
    return "".join(r.choice(alphabet) for _ in range(n))


def samtools_fai(fasta, tmp_path):
    cp = tmp_path / ("st_" + Path(fasta).name)
    shutil.copy(fasta, cp)
    subprocess.run([SAMTOOLS, "faidx", str(cp)], check=True, capture_output=True)
    return (Path(str(cp) + ".fai")).read_bytes()


def indexed(tmp_path, fasta):
    fai = tmp_path / "idx" / (Path(fasta).name + ".fai")
    index_fasta(fasta, fai)
    return FastaFile(fasta, fai)


RECS = [("s1", rseq(1, 1)), ("s2", rseq(59, 2)), ("s3", rseq(60, 3)), ("s4", rseq(61, 4)),
        ("s5", rseq(500, 5)), ("s6", rseq(7, 6)), ("s7", rseq(161, 7))]


# --------------------------------------------------------------------------- index building

@pytest.mark.parametrize("width", [1, 7, 60, 80])
def test_fai_matches_samtools(tmp_path, width):
    fa = write_fasta(tmp_path / f"w{width}.fa", RECS, width)
    ours = b"".join(e.line().encode() for e in build_fai(fa))
    if HAVE_SAMTOOLS:
        assert ours == samtools_fai(fa, tmp_path)
    assert [e.name for e in build_fai(fa)] == [r[0] for r in RECS]
    assert [e.length for e in build_fai(fa)] == [len(r[1]) for r in RECS]


@needs_samtools
@pytest.mark.parametrize("width", [1, 7, 60, 80])
def test_fai_matches_samtools_crlf(tmp_path, width):
    fa = write_fasta(tmp_path / f"crlf{width}.fa", RECS, width, eol="\r\n")
    ours = b"".join(e.line().encode() for e in build_fai(fa))
    assert ours == samtools_fai(fa, tmp_path)
    assert all(e.linewidth == e.linebases + 2 for e in build_fai(fa))


@needs_samtools
def test_fai_matches_samtools_no_trailing_newline(tmp_path):
    fa = write_fasta(tmp_path / "nonl.fa", RECS, 60, trailing_newline=False)
    assert b"".join(e.line().encode() for e in build_fai(fa)) == samtools_fai(fa, tmp_path)
    fa1 = tmp_path / "one.fa"
    fa1.write_text(">a\nACGT")
    assert b"".join(e.line().encode() for e in build_fai(fa1)) == samtools_fai(fa1, tmp_path)


@needs_samtools
def test_fai_matches_samtools_headers_descriptions_and_blank_trailers(tmp_path):
    fa = tmp_path / "desc.fa"
    fa.write_text(">a some description\tx\nACGTA\nCG\n\n>b\nAC\nGT\nA\n>empty\n>d\nAAAA\nAA\n\n\n")
    ours = b"".join(e.line().encode() for e in build_fai(fa))
    assert ours == samtools_fai(fa, tmp_path)
    assert "empty" not in [e.name for e in build_fai(fa)]


@needs_samtools
@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 17, 64, 1 << 20])
def test_fai_chunking_invariant(tmp_path, chunk):
    fa = write_fasta(tmp_path / "c.fa", RECS, 7)
    want = samtools_fai(fa, tmp_path)
    assert b"".join(e.line().encode() for e in build_fai(fa, chunk_bytes=chunk)) == want


def test_fai_lowercase_and_iupac_counted(tmp_path):
    fa = write_fasta(tmp_path / "lc.fa", [("x", "acgtNNnnRYKMacgt")], 5)
    (e,) = build_fai(fa)
    assert (e.length, e.linebases, e.linewidth) == (16, 5, 6)


def test_write_read_fai_roundtrip(tmp_path):
    fa = write_fasta(tmp_path / "r.fa", RECS, 13)
    ents = build_fai(fa)
    p = write_fai(ents, tmp_path / "sub" / "dir" / "r.fai")
    assert read_fai(p) == ents
    assert not Path(str(fa) + ".fai").exists()


def test_index_fasta_writes_only_where_asked(tmp_path):
    fa = write_fasta(tmp_path / "r.fa", RECS, 13)
    out = tmp_path / "elsewhere" / "x.fai"
    index_fasta(fa, out)
    assert out.exists() and not Path(str(fa) + ".fai").exists()


@pytest.mark.parametrize("text,msg", [
    (">a\nACG\nACGTA\n", "different line length"),
    (">a\nACGTA\nACG\nACGTA\n", "different line length"),
    (">a\nACGTA\n\nCG\n", "after a blank line"),
    ("xx\n>a\nAC\n", "before the first"),
    (">a\nAC GT\nAC\n", "whitespace"),
    (">a\nACGT\n>a\nAA\n", "duplicate"),
    (">\nACGT\n", "empty sequence name"),
    (">a\r\nACGT\nAC\r\n", "terminator"),
    (">a\nAC\rGT\n", "carriage return"),
])
def test_fai_malformed_rejected(tmp_path, text, msg):
    fa = tmp_path / "bad.fa"
    fa.write_bytes(text.encode())
    with pytest.raises(FastaIndexError, match=msg):
        build_fai(fa)


def test_gzip_rejected(tmp_path):
    import gzip
    fa = tmp_path / "z.fa.gz"
    with gzip.open(fa, "wt") as f:
        f.write(">a\nACGT\n")
    with pytest.raises(FastaIndexError, match="gzip"):
        build_fai(fa)


def test_empty_file(tmp_path):
    fa = tmp_path / "e.fa"
    fa.write_text("")
    assert build_fai(fa) == []


@pytest.mark.parametrize("line", ["a\t10\t3\t5\n", "a\tx\t3\t5\t6\n", "a\t10\t3\t0\t1\n", "a\t10\t3\t5\t4\n"])
def test_read_fai_malformed(tmp_path, line):
    p = tmp_path / "x.fai"
    p.write_text(line)
    with pytest.raises(FastaIndexError):
        read_fai(p)


def test_read_fai_duplicate(tmp_path):
    p = tmp_path / "x.fai"
    p.write_text("a\t10\t3\t5\t6\na\t10\t20\t5\t6\n")
    with pytest.raises(FastaIndexError, match="duplicate"):
        read_fai(p)


# --------------------------------------------------------------------------- extraction

@pytest.mark.parametrize("width", [1, 7, 60, 80])
@pytest.mark.parametrize("eol", ["\n", "\r\n"])
def test_extract_every_interval_small(tmp_path, width, eol):
    recs = [("a", rseq(23, 10)), ("b", rseq(1, 11)), ("c", rseq(9, 12))]
    fa = write_fasta(tmp_path / "s.fa", recs, width, eol=eol)
    with indexed(tmp_path, fa) as ff:
        for name, seq in recs:
            for s in range(len(seq)):
                for e in range(s + 1, len(seq) + 1):
                    assert ff.fetch(name, s, e) == seq[s:e]
                    assert ff.fetch(name, s, e, "-") == revcomp(seq[s:e])


@pytest.mark.parametrize("width", [1, 7, 60, 80])
def test_extract_random_and_edges(tmp_path, width):
    fa = write_fasta(tmp_path / "s.fa", RECS, width)
    r = random.Random(width)
    with indexed(tmp_path, fa) as ff:
        for name, seq in RECS:
            L = len(seq)
            assert ff.fetch(name, 0, 1) == seq[0]                     # file / record start
            assert ff.fetch(name, L - 1, L) == seq[-1]                # last base (ragged last line)
            assert ff.fetch(name, 0, L) == seq                        # whole record, many lines
            for _ in range(50):
                s = r.randrange(L)
                e = r.randrange(s + 1, L + 1)
                assert ff.fetch(name, s, e) == seq[s:e]
        assert ff.length("s5") == 500


def test_extract_last_record_without_trailing_newline(tmp_path):
    fa = write_fasta(tmp_path / "n.fa", RECS, 60, trailing_newline=False)
    with indexed(tmp_path, fa) as ff:
        name, seq = RECS[-1]
        assert ff.fetch(name, 0, len(seq)) == seq
        assert ff.fetch(name, len(seq) - 3, len(seq)) == seq[-3:]


def test_case_preserved_by_default_and_upper_option(tmp_path):
    seq = "ACGTacgtNNnnACGTRYkm"
    fa = write_fasta(tmp_path / "lc.fa", [("x", seq)], 6)
    with indexed(tmp_path, fa) as ff:
        assert ff.fetch("x", 0, len(seq)) == seq
        assert ff.fetch("x", 0, len(seq), upper=True) == seq.upper()
        assert ff.fetch("x", 4, 8, "-") == "acgt"
        assert ff.fetch("x", 4, 8, "-", upper=True) == "ACGT"
        assert ff.fetch_bytes("x", 0, 4) == b"ACGT"


def test_revcomp_iupac():
    assert revcomp("ACGTN") == "NACGT"
    assert revcomp("RYKMBVDHSWN") == "NWSDHBVKMRY"
    assert revcomp("acgtRyKm") == "kMrYacgt"
    assert revcomp("A-C*") == "*G-T"
    assert revcomp(b"AACG") == b"CGTT"
    assert revcomp("") == ""
    iupac = "ACGTRYKMBVDHSWNacgtrykmbvdhswn"
    assert revcomp(revcomp(iupac)) == iupac


@pytest.mark.parametrize("args,exc,msg", [
    (("nope", 0, 1), RegionError, "unknown sequence"),
    (("s2", -1, 5), RegionError, "out of range"),
    (("s2", 0, 60), RegionError, "out of range"),
    (("s2", 5, 5), RegionError, "empty"),
    (("s2", 6, 5), RegionError, "empty"),
    (("s2", 0, 5, "x"), RegionError, "strand"),
    (("s2", 0.5, 5), RegionError, "non-integer"),
])
def test_extract_errors(tmp_path, args, exc, msg):
    fa = write_fasta(tmp_path / "s.fa", RECS, 60)
    with indexed(tmp_path, fa) as ff:
        with pytest.raises(exc, match=msg):
            ff.fetch(*args)


def test_missing_index_is_clear_error(tmp_path):
    fa = write_fasta(tmp_path / "s.fa", RECS, 60)
    with pytest.raises(FastaIndexError, match="no index"):
        FastaFile(fa)


def test_stale_index_detected(tmp_path):
    fa = write_fasta(tmp_path / "s.fa", RECS, 60)
    fai = tmp_path / "s.fai"
    index_fasta(fa, fai)
    write_fasta(fa, RECS[:3], 60)
    with pytest.raises(FastaIndexError, match="ends past EOF|mismatch"):
        FastaFile(fa, fai)


def test_index_file_mismatch_detected_on_fetch(tmp_path):
    fa = write_fasta(tmp_path / "s.fa", RECS, 60)
    fai = tmp_path / "s.fai"
    index_fasta(fa, fai)
    write_fasta(fa, RECS, 59)
    ff = FastaFile(fa, fai)
    with pytest.raises(FastaIndexError, match="mismatch"):
        ff.fetch("s5", 0, 500)


def test_fetch_after_close(tmp_path):
    fa = write_fasta(tmp_path / "s.fa", RECS, 60)
    ff = indexed(tmp_path, fa)
    ff.close()
    with pytest.raises(ValueError):
        ff.fetch("s5", 0, 5)


def test_fetch_many_mixed_inputs(tmp_path):
    fa = write_fasta(tmp_path / "s.fa", RECS, 7)
    seqs = dict(RECS)
    with indexed(tmp_path, fa) as ff:
        out = ff.fetch_many([("s5", 10, 20), ("s7", 0, 161, "-"), Region("s1", 0, 1), Region("s4", 59, 61, "-")])
    assert out == [seqs["s5"][10:20], revcomp(seqs["s7"]), seqs["s1"], revcomp(seqs["s4"][59:61])]


def test_byte_span_arithmetic():
    e = FaiEntry("x", 100, 10, 7, 8)
    assert e.byte_span(0, 1) == (10, 11)
    assert e.byte_span(6, 7) == (16, 17)
    assert e.byte_span(7, 8) == (18, 19)
    assert e.byte_span(0, 14) == (10, 25)


@needs_samtools
@pytest.mark.parametrize("width", [1, 7, 60, 80])
def test_extract_matches_samtools_faidx(tmp_path, width):
    recs = [("chrA", rseq(1000, 21, "ACGTacgtN")), ("chrB", rseq(333, 22))]
    fa = write_fasta(tmp_path / "m.fa", recs, width)
    fai = tmp_path / "m.custom.fai"
    index_fasta(fa, fai)
    r = random.Random(3)
    ff = FastaFile(fa, fai)
    for _ in range(20):
        name, seq = recs[r.randrange(2)]
        s = r.randrange(len(seq))
        e = r.randrange(s + 1, len(seq) + 1)
        rc = r.random() < 0.5
        cmd = [SAMTOOLS, "faidx", "--fai-idx", str(fai), str(fa), f"{name}:{s + 1}-{e}"]
        if rc:
            cmd.insert(2, "-i")
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        st = "".join(out.splitlines()[1:])
        assert ff.fetch(name, s, e, "-" if rc else "+") == st
