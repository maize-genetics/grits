import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from python.seq_extract.cli import PLACEHOLDER_TAG, main
from python.seq_extract.fai import read_fai

SRC = Path(__file__).resolve().parents[3] / "src"
R = "ACGTTGCAAC" "GTACCATGGT" "CCAGTAGCTA" "TTCGAGGCAT"


def _fa(path, recs, width=7):
    with open(path, "w") as f:
        for n, s in recs:
            f.write(f">{n}\n")
            for k in range(0, len(s), width):
                f.write(s[k:k + width] + "\n")
    return path


def _read_fa(path):
    recs, name = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            name = line[1:]
            recs[name] = ""
        else:
            recs[name] += line
    return recs


@pytest.fixture
def world(tmp_path):
    d = tmp_path / "fa"
    d.mkdir()
    _fa(d / "B73.fa", [("B73_chr1", R)])
    _fa(d / "A.fa", [("A_chr1", R.lower())])
    m = tmp_path / "maps"
    m.mkdir()
    np.save(m / "A_chr1.npy", np.arange(41, dtype=np.int64))
    bd = tmp_path / "bed"
    bd.mkdir()
    (bd / "S_chr1_imputed.bed").write_text("chrom\tstart\tend\tparent1\tparent2\n"
                                           "chr1\t0\t10\tA\tB73\nchr1\t15\t40\tB73\tA\n")
    w = tmp_path / "win.tsv"
    w.write_text("wid\tchrom\tstart\tend\ng1\tchr1\t5\t20\ng2\tB73_chr1\t30\t40\n")
    return dict(fa=d, maps=m, bed=bd, win=w, tmp=tmp_path)


def test_faidx_requires_location(world, capsys):
    with pytest.raises(SystemExit):
        main(["faidx", str(world["fa"] / "A.fa")])
    assert not (world["fa"] / "A.fa.fai").exists()


def test_faidx_out_and_dir(world):
    t = world["tmp"]
    assert main(["faidx", str(world["fa"] / "A.fa"), "--out", str(t / "x" / "a.fai")]) == 0
    assert read_fai(t / "x" / "a.fai")[0].length == 40
    assert main(["faidx", str(world["fa"] / "A.fa"), str(world["fa"] / "B73.fa"), "--fai-dir", str(t / "f")]) == 0
    assert (t / "f" / "A.fa.fai").exists() and (t / "f" / "B73.fa.fai").exists()
    with pytest.raises(SystemExit):
        main(["faidx", str(world["fa"] / "A.fa"), str(world["fa"] / "B73.fa"), "--out", str(t / "y.fai")])
    assert main(["faidx", str(world["fa"] / "A.fa"), "--next-to-fasta"]) == 0
    assert (world["fa"] / "A.fa.fai").exists()


def test_extract_fasta_list(world):
    t = world["tmp"]
    q = t / "q.tsv"
    q.write_text("qid\tcontig\tstart\tend\tstrand\nq1\tA_chr1\t0\t4\t+\nq2\tB73_chr1\t0\t4\t-\n")
    out = t / "o.fa"
    rc = main(["extract", "--fasta", str(world["fa"] / "A.fa"), str(world["fa"] / "B73.fa"), "--fai-dir",
               str(t / "fai"), "--build-fai", "--queries", str(q), "--out", str(out)])
    assert rc == 0
    assert out.read_text() == ">q1\nacgt\n>q2\nACGT\n"
    main(["extract", "--fasta", str(world["fa"] / "A.fa"), "--fai-dir", str(t / "fai"), "--queries", str(q),
          "--out", str(out), "--upper", "--on-error", "skip"])
    assert out.read_text() == ">q1\nACGT\n"


def test_extract_wrap_and_errors(world):
    t = world["tmp"]
    q = t / "q.tsv"
    q.write_text("qid\tcontig\tstart\tend\tstrand\nq1\tB73_chr1\t0\t25\t+\n")
    out = t / "o.fa"
    main(["extract", "--fasta-dir", str(world["fa"]), "--fai-dir", str(t / "fai"), "--build-fai",
          "--queries", str(q), "--out", str(out), "--line-width", "10"])
    assert out.read_text() == f">q1\n{R[:10]}\n{R[10:20]}\n{R[20:25]}\n"
    q.write_text("qid\tcontig\tstart\tend\tstrand\nq1\tB73_chr1\t0\t99\t+\n")
    with pytest.raises(SystemExit, match="q1"):
        main(["extract", "--fasta-dir", str(world["fa"]), "--fai-dir", str(t / "fai"), "--queries", str(q)])
    with pytest.raises(SystemExit):
        main(["extract", "--fasta-dir", str(world["fa"]), "--build-fai", "--queries", str(q)])
    q.write_text("bad\theader\n")
    with pytest.raises(ValueError, match="header"):
        main(["extract", "--fasta-dir", str(world["fa"]), "--fai-dir", str(t / "fai"), "--queries", str(q)])


def _hap_args(world, *extra):
    t = world["tmp"]
    return ["haplotypes", "--fasta-dir", str(world["fa"]), "--fai-dir", str(t / "fai"), "--build-fai",
            "--bed", str(world["bed"]), "--windows", str(world["win"]), *extra]


def test_haplotypes_truth_provider(world):
    out = world["tmp"] / "out"
    assert main(_hap_args(world, "--provider", "truth", "--maps-dir", str(world["maps"]), "--out-dir", str(out),
                          "--prefix", "S.")) == 0
    h1, h2 = _read_fa(out / "S.hap1.fa"), _read_fa(out / "S.hap2.fa")
    k1 = [k for k in h1 if k.startswith("g1 ")][0]
    assert h1[k1] == R.lower()[5:10] + "NNNNN" + R[15:20]
    assert "switches=1" in k1 and "gap_bases=5" in k1
    assert list(h2.values())[0] == R[5:10] + "NNNNN" + R[15:20].lower()
    md = (out / "S.metadata.tsv").read_text().splitlines()
    assert md[0].split("\t")[:3] == ["wid", "hap", "chrom"] and len(md) == 5
    assert PLACEHOLDER_TAG not in "".join(md)


def test_haplotypes_gap_policy_extend(world):
    out = world["tmp"] / "out"
    main(_hap_args(world, "--provider", "truth", "--maps-dir", str(world["maps"]), "--out-dir", str(out),
                   "--gap-policy", "extend"))
    h1 = _read_fa(out / "hap1.fa")
    assert list(h1.values())[0] == R.lower()[5:12] + R[12:20]


def test_haplotypes_emit_queries_then_table(world):
    t = world["tmp"]
    q = t / "pq.tsv"
    assert main(_hap_args(world, "--emit-queries", str(q))) == 0
    lines = q.read_text().splitlines()
    assert lines[0] == "qid\tref_contig\tref_start\tref_end\tassembly"
    assert [l.split("\t")[1:] for l in lines[1:]] == [["B73_chr1", "5", "10", "A"], ["B73_chr1", "15", "20", "A"],
                                                      ["B73_chr1", "30", "40", "A"]]
    tab = t / "tab.tsv"
    tab.write_text("qid\tref_contig\tref_start\tref_end\tassembly\tcontig\tstart\tend\tstrand\tstatus\n"
                   "q0\tB73_chr1\t5\t10\tA\tA_chr1\t5\t10\t-\tok\n"
                   "q1\tB73_chr1\t15\t20\tA\t.\t.\t.\t.\tunmapped\n")
    out = t / "o2"
    main(_hap_args(world, "--provider", "table", "--table", str(tab), "--out-dir", str(out)))
    h1, h2 = _read_fa(out / "hap1.fa"), _read_fa(out / "hap2.fa")
    assert list(h1.values())[0] == "gttgc" + "NNNNN" + R[15:20]
    assert list(h2.values())[0] == R[5:10] + "NNNNN" + "NNNNN"
    md = (out / "metadata.tsv").read_text()
    assert "unmapped" in md and "not_in_table" in md


def test_haplotypes_identity_placeholder_labelled(world):
    out = world["tmp"] / "o3"
    main(_hap_args(world, "--provider", "identity-placeholder", "--out-dir", str(out)))
    assert all(PLACEHOLDER_TAG in k for k in _read_fa(out / "hap1.fa"))
    md = (out / "metadata.tsv").read_text().splitlines()
    assert md[0].startswith("# " + PLACEHOLDER_TAG)
    assert all(l.split("\t")[-2] == "identity-placeholder" for l in md[2:])


def test_haplotypes_missing_args(world):
    with pytest.raises(SystemExit):
        main(_hap_args(world, "--provider", "truth", "--out-dir", str(world["tmp"] / "o")))
    with pytest.raises(SystemExit):
        main(_hap_args(world, "--provider", "table", "--out-dir", str(world["tmp"] / "o")))
    with pytest.raises(SystemExit):
        main(_hap_args(world, "--provider", "truth", "--maps-dir", str(world["maps"])))


def test_module_entry_point(world):
    import os
    env = dict(os.environ, PYTHONPATH=str(SRC))
    r = subprocess.run([sys.executable, "-m", "python.seq_extract", "--help"], capture_output=True, text=True,
                       env=env)
    assert r.returncode == 0 and "haplotypes" in r.stdout
