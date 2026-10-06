#!/usr/bin/env python
"""Real-data timing/validation of python.seq_extract on the 25 renamed founder FASTAs.

1. checks every FASTA's sequences are named <Asm>_*, with <Asm>_chr1..chr10 present;
2. times N random fixed-length extractions across all assemblies: a cold-ish pass (the queried byte
   ranges are evicted from the page cache with posix_fadvise(DONTNEED) first, files opened fresh), then a
   warm pass over the same queries;
3. checks a subset base-for-base against `samtools faidx --fai-idx <our .fai>` (+ and -i);
4. runs the haplotype builder on one real imputed sample over random B73 windows with an IDENTITY
   PLACEHOLDER projection (code path / timing only; the sequences are not meaningful).
"""
import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from python.seq_extract.assembly import AssemblySet
from python.seq_extract.cli import main as cli_main
from python.seq_extract.fai import read_fai


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta-dir", required=True)
    ap.add_argument("--pattern", default="{asm}_renamed.fa")
    ap.add_argument("--fai-dir", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--length", type=int, default=8000)
    ap.add_argument("--n-samtools", type=int, default=100)
    ap.add_argument("--samtools", default="/programs/samtools-1.20/bin/samtools")
    ap.add_argument("--bed", help="imputed BED dir of one sample")
    ap.add_argument("--n-windows", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    asm = AssemblySet.from_dir(a.fasta_dir, a.pattern, fai_dir=a.fai_dir)
    names = {}
    for x in asm.assemblies:
        ents = read_fai(asm.fai_path(x))
        bad = [e.name for e in ents if not e.name.startswith(x + "_")]
        missing = [f"{x}_chr{i}" for i in range(1, 11) if f"{x}_chr{i}" not in {e.name for e in ents}]
        names[x] = {e.name: e.length for e in ents}
        print(f"names {x}: {len(ents)} seqs, bad_prefix={len(bad)}, missing_chr={missing}")
    print(f"{len(asm.assemblies)} assemblies")

    qs = []
    for _ in range(a.n):
        x = rng.choice(asm.assemblies)
        c = f"{x}_chr{rng.randint(1, 10)}"
        s = rng.randrange(0, names[x][c] - a.length)
        qs.append((c, s, s + a.length, rng.choice("+-")))

    for x in asm.assemblies:
        fd = os.open(asm.fastas[x], os.O_RDONLY)
        f = asm.get(x)
        for c, s, e, _ in qs:
            if c.startswith(x + "_") and asm.assembly_of(c) == x:
                b0, b1 = f.entries[c].byte_span(s, e)
                os.posix_fadvise(fd, b0 - b0 % 4096, b1 - b0 + 8192, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    asm.close()

    t0 = time.perf_counter()
    asm = AssemblySet.from_dir(a.fasta_dir, a.pattern, fai_dir=a.fai_dir)
    cold = asm.fetch_many(qs)
    t_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    warm = asm.fetch_many(qs)
    t_warm = time.perf_counter() - t0
    assert cold == warm and all(len(s) == a.length for s in cold)
    print(f"extract {a.n} x {a.length} bp across {len(asm.assemblies)} assemblies: cold-ish {t_cold:.3f}s "
          f"({1e3 * t_cold / a.n:.3f} ms/query, incl. opening files + reading .fai), warm {t_warm:.3f}s "
          f"({1e3 * t_warm / a.n:.3f} ms/query)")

    idx = rng.sample(range(a.n), a.n_samtools)
    mism = 0
    by = {}
    for i in idx:
        c, s, e, st = qs[i]
        by.setdefault((asm.assembly_of(c), st), []).append(i)
    for (x, st), ii in by.items():
        cmd = [a.samtools, "faidx", "--fai-idx", str(asm.fai_path(x)), "-n", "0"]
        if st == "-":
            cmd.append("-i")
        cmd += [str(asm.fastas[x])] + [f"{qs[i][0]}:{qs[i][1] + 1}-{qs[i][2]}" for i in ii]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        recs = [r for r in out.split(">") if r]
        assert len(recs) == len(ii)
        for i, r in zip(ii, recs):
            seq = "".join(r.splitlines()[1:])
            mism += seq != cold[i]
    print(f"samtools cross-check: {a.n_samtools - mism}/{a.n_samtools} identical")
    lower = sum(sum(ch.islower() for ch in s) for s in cold)
    nn = sum(s.upper().count("N") for s in cold)
    print(f"composition of extracted bases: lowercase={lower}, N={nn}, total={a.n * a.length}")

    if a.bed:
        b73 = names["B73"]
        win = work / "windows.tsv"
        with open(win, "w") as f:
            f.write("wid\tchrom\tstart\tend\n")
            for k in range(a.n_windows):
                chrom = f"chr{rng.randint(1, 10)}"
                ln = rng.randint(2000, 30000) + 10000
                s = rng.randrange(0, b73[f"B73_{chrom}"] - ln)
                f.write(f"g{k}\t{chrom}\t{s}\t{s + ln}\n")
        out = work / "haplotypes_identity_placeholder"
        t0 = time.perf_counter()
        cli_main(["haplotypes", "--fasta-dir", a.fasta_dir, "--pattern", a.pattern, "--fai-dir", a.fai_dir,
                  "--bed", a.bed, "--windows", str(win), "--provider", "identity-placeholder",
                  "--out-dir", str(out)])
        t = time.perf_counter() - t0
        rows = [l.rstrip("\n").split("\t") for l in open(out / "metadata.tsv") if not l.startswith("#")]
        hdr, rows = rows[0], rows[1:]
        col = {h: i for i, h in enumerate(hdr)}
        tot = lambda k: sum(int(r[col[k]]) for r in rows)
        print(f"haplotypes (IDENTITY PLACEHOLDER, not meaningful): {a.n_windows} windows x 2 in {t:.3f}s total "
              f"(incl. BED/window parsing); bases={tot('seq_len')}, founder segments={tot('n_segments')}, "
              f"switches={tot('n_switches')}, gap_bases={tot('gap_bases')}, failures={tot('n_failures')} -> {out}")


if __name__ == "__main__":
    sys.exit(main())
