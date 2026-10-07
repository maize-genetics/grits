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
from python.seq_extract.projection import Projection, SpanCheckProvider, TruthMapProvider, read_projection_table
from python.seq_extract.snap import SnapProvider

REPO = Path(__file__).resolve().parents[3]
SIM = REPO / "experiments" / "seq-extract" / "simulate_pangenome.py"
SAMTOOLS = os.environ.get("SAMTOOLS", "/programs/samtools-1.20/bin/samtools")
SMALL_ARGS = ["--seed", "7", "--chrom-lens", "60000", "40000", "--founders", "Oh43", "Mo17", "Ki3",
              "--large-rate", "3e-5", "--te-copies", "3", "--n-direct", "150", "--n-ref", "150", "--window", "8000"]


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


# --------------------------------------------------------------------------- end snapping

class PerturbedProvider:
    """Truth projection with each end moved by a random offset: a stand-in for an unsnapped lift. The offset
    mix follows the C --no-snap lift on the full fixture (65.6% exact, 32.5% 1-40 bp off, 1.6% > 250 bp)."""

    def __init__(self, base, asm, seed=0):
        self.base, self.asm, self.rng = base, asm, np.random.default_rng(seed)

    def _off(self):
        u = self.rng.random()
        if u < 0.6:
            return 0
        if u < 0.97:
            return int(self.rng.choice([-1, 1]) * self.rng.integers(1, 41))
        return int(self.rng.choice([-1, 1]) * self.rng.integers(250, 3000))

    def project(self, *a):
        pr = self.base.project(*a)
        if not pr.ok:
            return pr
        L = self.asm.length(pr.contig)
        s = min(max(pr.start + self._off(), 0), L - 2)
        e = min(max(pr.end + self._off(), s + 1), L)
        return Projection(pr.ref_chrom, pr.ref_start, pr.ref_end, pr.assembly, pr.contig, s, e, "+", "ok")


def test_snap_recovers_truth_from_perturbed_projections(fx):
    d, meta, asm, _ = fx
    truth = TruthMapProvider(d / "maps")
    sp = SnapProvider(PerturbedProvider(truth, asm, seed=3), asm, asm)
    n = ends_exact = intervals = 0
    labels = {}
    for q in read_tsv(d / "ref_queries.tsv"):
        args = (q["ref_contig"], int(q["ref_start"]), int(q["ref_end"]), q["assembly"])
        p = sp.project(*args)
        t = (int(q["true_start"]), int(q["true_end"]))
        n += 1
        ends_exact += (p.start == t[0]) + (p.end == t[1])
        intervals += (p.start, p.end) == t
        for lab in p.snap:
            labels[lab] = labels.get(lab, 0) + 1
    print(f"\nsnap vs truth (perturbed raw): ends {ends_exact}/{2 * n}, intervals {intervals}/{n}, labels {labels}")
    assert ends_exact >= 0.97 * 2 * n
    assert labels.get("snap_wide", 0) > 0


FULL_FIXTURE = Path(os.environ.get("SEQ_EXTRACT_FULL_FIXTURE",
                                   "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/seq_extract_fixture"))
C_TABLES = Path(os.environ.get("SEQ_EXTRACT_SNAP_TABLES",
                               "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/seq_extract_snapcheck"))


@pytest.mark.parametrize("version,prefix", [("v3", "fx_"), ("v4", "v4_fx_"), ("v5", "v5_fx_"), ("v5", "v5_allb_")])
def test_snap_identical_to_c_on_full_fixture(tmp_path, version, prefix):
    """Python snap over C `--no-snap --project-only` == C `--project-only`, per row (coords, labels, shifts).
    v3 tables come from ropebwt3 a4df980, v4 from 509b2bf, v5 from 877bace (raw runs with --max-span-excess=0;
    v5_allb = boundaries inside every >= 2 bp deletion and at every insertion of the fixture, 3 kb pieces)."""
    if not (C_TABLES / f"{prefix}nosnap.tsv").exists() or not (FULL_FIXTURE / "B73.fa").exists():
        pytest.skip("C ropebwt3 --project-only tables for the full fixture not available")
    asm = AssemblySet.from_dir(FULL_FIXTURE, "{asm}.fa", assemblies=["B73", "Oh43", "Mo17", "Ki3", "CML247", "Tx303"],
                               fai_dir=tmp_path / "fai", build_missing=True)
    sp = SpanCheckProvider(SnapProvider(None, asm, asm, snap_version=version))
    c = {r.qid: r for r in read_projection_table(C_TABLES / f"{prefix}snap.tsv")}
    raw = read_projection_table(C_TABLES / f"{prefix}nosnap.tsv")
    bad = []
    for r in raw:
        p, w = sp.check(sp.base.snap(r)), c[r.qid]
        if (p.status, p.contig, p.start, p.end, p.strand, p.snap, p.snap_shift) != \
                (w.status, w.contig, w.start, w.end, w.strand, w.snap, w.snap_shift):
            bad.append((r.qid, p, w))
    assert len(raw) >= 500 and bad == []
