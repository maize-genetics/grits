"""Cross-check against the shared simulated pangenome (experiments/seq-extract/simulate_pangenome.py).

By default a scaled-down fixture is regenerated with a fixed seed. Set SEQ_EXTRACT_FIXTURE=<dir> to run
the same checks against an existing (e.g. full-size) fixture directory instead; nothing is written there.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from python.seq_extract.assembly import AssemblySet
from python.seq_extract.cli import main as cli_main
from python.seq_extract.fai import build_fai
from python.seq_extract.haplotype import BedInterval, HaplotypeBuilder, Window, group_bed
from python.seq_extract.projection import TruthMapProvider

REPO = Path(__file__).resolve().parents[3]
SIM = REPO / "experiments" / "seq-extract" / "simulate_pangenome.py"
SAMTOOLS = os.environ.get("SAMTOOLS", "/programs/samtools-1.20/bin/samtools")
SMALL_ARGS = ["--seed", "7", "--chrom-lens", "40000", "30000", "--founders", "Oh43", "Mo17", "Ki3",
              "--large-rate", "1e-4", "--te-copies", "3", "--n-direct", "150", "--n-ref", "150", "--window", "3000"]


def read_fasta(path):
    out, name, buf = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line)
    if name is not None:
        out[name] = "".join(buf)
    return out


def read_tsv(path):
    with open(path) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        return [dict(zip(hdr, l.rstrip("\n").split("\t"))) for l in f if l.strip()]


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    given = os.environ.get("SEQ_EXTRACT_FIXTURE")
    if given:
        d = Path(given)
    else:
        if not SIM.exists():
            pytest.skip(f"simulator not found: {SIM}")
        d = tmp_path_factory.mktemp("sim")
        subprocess.run([sys.executable, str(SIM), "--out", str(d)] + SMALL_ARGS, check=True, capture_output=True)
    meta = json.loads((d / "meta.json").read_text())
    work = tmp_path_factory.mktemp("work")
    asm = AssemblySet.from_dir(d, "{asm}.fa", assemblies=["B73"] + meta["founders"], fai_dir=work / "fai",
                               build_missing=True)
    yield d, meta, asm, work
    asm.close()


def test_fixture_fai_matches_samtools(fx):
    d, meta, asm, work = fx
    if not Path(SAMTOOLS).exists():
        pytest.skip("samtools not available")
    for a in ["B73"] + meta["founders"][:1]:
        asm.get(a)
        cp = work / f"st_{a}.fa"
        shutil.copy(d / f"{a}.fa", cp)
        subprocess.run([SAMTOOLS, "faidx", str(cp)], check=True)
        assert asm.fai_path(a).read_bytes() == Path(str(cp) + ".fai").read_bytes()
    pg = build_fai(d / "pangenome.fa")
    assert {e.name: e.length for e in pg} == meta["lengths"]


def test_direct_queries_exact(fx):
    d, meta, asm, _ = fx
    exp = read_fasta(d / "direct_expected.fa")
    qs = read_tsv(d / "direct_queries.tsv")
    assert len(qs) == len(exp) > 0
    got = asm.fetch_many([(q["contig"], int(q["start"]), int(q["end"]), q["strand"]) for q in qs])
    bad = [q["qid"] for q, g in zip(qs, got) if g != exp[q["qid"]]]
    assert bad == []


def test_reference_queries_truth_provider_exact(fx):
    d, meta, asm, _ = fx
    exp = read_fasta(d / "ref_expected.fa")
    prov = TruthMapProvider(d / "maps")
    n_large = 0
    for q in read_tsv(d / "ref_queries.tsv"):
        pr = prov.project(q["ref_contig"], int(q["ref_start"]), int(q["ref_end"]), q["assembly"])
        assert pr.ok, q
        assert (pr.contig, pr.start, pr.end) == (q["true_contig"], int(q["true_start"]), int(q["true_end"])), q
        assert asm.fetch(pr.contig, pr.start, pr.end, pr.strand) == exp[q["qid"]], q["qid"]
        n_large += int(q["spans_large_indel"])
    assert n_large > 0


def test_cli_extract_byte_identical(fx, tmp_path):
    d, meta, asm, work = fx
    out = tmp_path / "d.fa"
    rc = cli_main(["extract", "--fasta-dir", str(d), "--pattern", "{asm}.fa", "--fai-dir", str(work / "fai"),
                   "--queries", str(d / "direct_queries.tsv"), "--out", str(out)])
    assert rc == 0
    assert out.read_bytes() == (d / "direct_expected.fa").read_bytes()


# --------------------------------------------------------------------------- stitching vs independent truth

def synthetic_tracks(rng, L, founders):
    """Per-base founder index (-1 = uncalled) for hap1/hap2, and the BED rows that encode them."""
    t1, t2 = np.full(L, -1, np.int64), np.full(L, -1, np.int64)
    rows = []
    pos = int(rng.integers(200, 1500))
    f1, f2 = int(rng.integers(len(founders))), int(rng.integers(len(founders)))
    while pos < L - 2500:
        ln = int(rng.integers(150, 4000))
        e = min(pos + ln, L - 2500)
        if rng.random() < 0.15:
            pos = e
            continue
        if rng.random() < 0.5:
            f1 = int(rng.integers(len(founders)))
        if rng.random() < 0.5:
            f2 = int(rng.integers(len(founders)))
        t1[pos:e], t2[pos:e] = f1, f2
        rows.append((pos, e, founders[f1], founders[f2]))
        pos = e
    return t1, t2, rows


def extend_track(t):
    t = t.copy()
    called = np.flatnonzero(t >= 0)
    if len(called) == 0:
        return t
    t[:called[0]] = t[called[0]]
    t[called[-1] + 1:] = t[called[-1]]
    i = called[0]
    while i < len(t):
        if t[i] >= 0:
            i += 1
            continue
        j = i
        while t[j] < 0:
            j += 1
        half = (j - i) // 2
        left, right = t[i - 1], t[j]
        t[i:i + half] = left
        t[i + half:j] = right
        i = j
    return t


def expected_seq(track, ws, we, founders, seqs, maps, chrom, policy):
    out = []
    k = ws
    while k < we:
        f = track[k]
        j = k
        while j < we and track[j] == f:
            j += 1
        if f < 0:
            if policy != "skip":
                out.append("N" * (j - k))
        else:
            fn = founders[f]
            mp = maps[fn][k:j]
            pres = mp[mp >= 0]
            out.append(seqs[f"{fn}_{chrom}"][pres[0]:pres[-1] + 1] if len(pres) else "N" * (j - k))
        k = j
    return "".join(out)


@pytest.mark.parametrize("policy", ["N", "skip", "extend"])
def test_stitching_matches_independent_truth(fx, policy):
    d, meta, asm, _ = fx
    rng = np.random.default_rng(123)
    founders = ["B73"] + meta["founders"]
    seqs = {}
    for a in founders:
        seqs.update(read_fasta(d / f"{a}.fa"))
    prov = TruthMapProvider(d / "maps")
    total = dict(switches=0, gaps=0, failures=0, windows=0, bases=0)
    for chrom in sorted({k.split("_", 1)[1] for k in meta["lengths"] if k.startswith("B73_")}):
        L = meta["lengths"][f"B73_{chrom}"]
        maps = {a: np.load(d / "maps" / f"{a}_{chrom}.npy") for a in founders}
        t1, t2, rows = synthetic_tracks(rng, L, founders)
        if policy == "extend":
            t1, t2 = extend_track(t1), extend_track(t2)
        bed = group_bed([BedInterval(chrom, *r) for r in rows])
        b = HaplotypeBuilder(bed, asm, prov, policy)
        wins = [Window("first", chrom, 0, rows[0][0] + 100), Window("before", chrom, 0, max(1, rows[0][0] - 50)),
                Window("after", chrom, rows[-1][1] + 10, L), Window("all", chrom, 0, L)]
        for k in range(60):
            ln = int(rng.integers(1, 9000))
            s = int(rng.integers(0, L - ln))
            wins.append(Window(f"w{k}", chrom, s, s + ln))
        for w in wins:
            for hap, t in ((1, t1), (2, t2)):
                r = b.build(w, hap)
                exp = expected_seq(t, w.start, w.end, founders, seqs, maps, chrom, policy)
                assert r.seq == exp, (chrom, w, hap, policy)
                total["switches"] += r.n_switches
                total["gaps"] += r.gap_bases
                total["failures"] += len(r.failures)
                total["windows"] += 1
                total["bases"] += len(r.seq)
    assert total["switches"] > 50
    if policy != "extend":
        assert total["gaps"] > 0
    print(f"\nstitching[{policy}]: {total}")
