#!/usr/bin/env python
"""Simulated pangenome with known coordinate maps, the shared fixture for the two sequence-extraction
implementations (ropebwt3 `extract` and the faidx-based extractor) and their cross-check.

A random reference "B73" (chromosomes chr1..chrN, with a repeated "TE" element inserted at several
places so some sequence is multi-copy) and F founders derived from it by SNPs, small indels (1-20 bp)
and large indels (0.5-5 kb; large insertions are either random sequence or a copy of the TE element).
Every edit is recorded, so the exact reference -> founder coordinate map is known.

Outputs (in --out):
  <Asm>.fa                one FASTA per assembly, sequences named <Asm>_<chr> (ropebwt3 --ref-prefix style)
  pangenome.fa            all assemblies concatenated (input for `ropebwt3 build`)
  maps/<Asm>_<chr>.npy    int64 [ref_len + 1]: founder position of each reference position, -1 where the
                          reference base is deleted in the founder; index ref_len = founder length
  direct_queries.tsv      qid, contig, start, end, strand (0-based half-open, + or -) on any assembly,
                          incl. edge cases; expected sequence in direct_expected.fa (revcomp for -)
  ref_queries.tsv         qid, ref_contig, ref_start, ref_end, assembly, true_contig, true_start, true_end,
                          spans_large_indel: a reference window and the founder interval it truly maps to
                          (first / last reference base present in the founder); expected sequence in
                          ref_expected.fa
  events.tsv              every simulated edit (assembly, chr, ref_pos, type, ref_len, alt_len)
  meta.json               parameters and seed
Deterministic for a given --seed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s):
    return s.translate(COMP)[::-1]


def rand_seq(rng, n):
    return "".join(rng.choice(list("ACGT"), n))


def derive(rng, ref, te, a):
    """Apply edits left to right; return founder sequence, map[ref_len+1], event list."""
    out, mp, ev = [], np.full(len(ref) + 1, -1, np.int64), []
    i, fpos = 0, 0
    n = len(ref)
    while i < n:
        u = rng.random()
        if u < a.large_rate and i + 6000 < n:                  # large indel
            L = int(rng.integers(500, 5001))
            if rng.random() < 0.5:                              # deletion of L reference bases
                ev.append(("large_del", i, L, 0))
                i += L
                continue
            ins = te if rng.random() < 0.5 else rand_seq(rng, L)
            ev.append(("large_ins", i, 0, len(ins)))
            out.append(ins); fpos += len(ins)
        elif u < a.large_rate + a.indel_rate:                  # small indel
            L = int(min(20, rng.geometric(0.3)))
            if rng.random() < 0.5 and i + L < n:
                ev.append(("del", i, L, 0))
                i += L
                continue
            ins = rand_seq(rng, L)
            ev.append(("ins", i, 0, L))
            out.append(ins); fpos += L
        b = ref[i]
        if rng.random() < a.snp_rate:
            b = rng.choice([x for x in "ACGT" if x != b])
            ev.append(("snp", i, 1, 1))
        mp[i] = fpos
        out.append(b); fpos += 1
        i += 1
    mp[n] = fpos
    return "".join(out), mp, ev


def write_fa(path, recs, width=60):
    with open(path, "w") as f:
        for name, s in recs:
            f.write(f">{name}\n")
            for k in range(0, len(s), width):
                f.write(s[k:k + width] + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--chrom-lens", type=int, nargs="+", default=[300_000, 200_000, 150_000])
    ap.add_argument("--founders", nargs="+", default=["Oh43", "Mo17", "Ki3", "CML247", "Tx303"])
    ap.add_argument("--snp-rate", type=float, default=0.01)
    ap.add_argument("--indel-rate", type=float, default=0.001)
    ap.add_argument("--large-rate", type=float, default=1e-5)
    ap.add_argument("--te-copies", type=int, default=6)
    ap.add_argument("--n-direct", type=int, default=500)
    ap.add_argument("--n-ref", type=int, default=500)
    ap.add_argument("--window", type=int, default=8000)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); (out / "maps").mkdir(parents=True, exist_ok=True)
    te = rand_seq(rng, 2000)
    chroms = {}
    for ci, L in enumerate(a.chrom_lens, 1):
        s = list(rand_seq(rng, L))
        for _ in range(a.te_copies):                             # multi-copy element in the reference
            p = int(rng.integers(0, L - len(te)))
            s[p:p + len(te)] = te
        chroms[f"chr{ci}"] = "".join(s)
    asm = {"B73": {c: (s, np.r_[np.arange(len(s)), len(s)].astype(np.int64)) for c, s in chroms.items()}}
    events = []
    for fn in a.founders:
        asm[fn] = {}
        for c, s in chroms.items():
            fs, mp, ev = derive(rng, s, te, a)
            asm[fn][c] = (fs, mp)
            events += [(fn, c) + e for e in ev]
    allrecs = []
    for an, cs in asm.items():
        recs = [(f"{an}_{c}", s) for c, (s, _) in cs.items()]
        write_fa(out / f"{an}.fa", recs)
        allrecs += recs
        for c, (_, mp) in cs.items():
            np.save(out / "maps" / f"{an}_{c}.npy", mp)
    write_fa(out / "pangenome.fa", allrecs)
    with open(out / "events.tsv", "w") as f:
        f.write("assembly\tchr\tref_pos\ttype\tref_len\talt_len\n")
        for e in events:
            f.write("\t".join(map(str, (e[0], e[1], e[3], e[2], e[4], e[5]))) + "\n")
    seqs = {n: s for n, s in allrecs}
    # direct queries: random + edge cases on every assembly
    dq = []
    names = list(seqs)
    for n in names:
        L = len(seqs[n])
        dq += [(n, 0, 1, "+"), (n, L - 1, L, "+"), (n, 0, min(L, a.window), "+"), (n, L - a.window, L, "-"),
               (n, 1000, 1001, "-")]
    while len(dq) < a.n_direct:
        n = names[int(rng.integers(len(names)))]
        L = len(seqs[n])
        ln = int(rng.choice([1, 2, 50, 151, 1000, a.window, 20000]))
        st = int(rng.integers(0, L - ln + 1))
        dq.append((n, st, st + ln, "+" if rng.random() < 0.7 else "-"))
    with open(out / "direct_queries.tsv", "w") as f, open(out / "direct_expected.fa", "w") as g:
        f.write("qid\tcontig\tstart\tend\tstrand\n")
        for k, (n, st, en, sd) in enumerate(dq):
            f.write(f"d{k}\t{n}\t{st}\t{en}\t{sd}\n")
            s = seqs[n][st:en]
            g.write(f">d{k}\n{revcomp(s) if sd == '-' else s}\n")
    # reference-coordinate queries with the true founder interval
    large = {}
    for e in events:
        if e[2] in ("large_del", "large_ins"):
            large.setdefault((e[0], e[1]), []).append(e[3])
    with open(out / "ref_queries.tsv", "w") as f, open(out / "ref_expected.fa", "w") as g:
        f.write("qid\tref_contig\tref_start\tref_end\tassembly\ttrue_contig\ttrue_start\ttrue_end\tspans_large_indel\n")
        k = 0
        while k < a.n_ref:
            c = list(chroms)[int(rng.integers(len(chroms)))]
            L = len(chroms[c])
            st = int(rng.integers(0, L - a.window))
            en = st + a.window
            fn = a.founders[int(rng.integers(len(a.founders)))]
            fs, mp = asm[fn][c]
            present = np.flatnonzero(mp[st:en] >= 0)
            if len(present) == 0:
                continue
            ts, te_ = int(mp[st + present[0]]), int(mp[st + present[-1]]) + 1
            span = any(st <= p < en for p in large.get((fn, c), []))
            f.write(f"r{k}\tB73_{c}\t{st}\t{en}\t{fn}\t{fn}_{c}\t{ts}\t{te_}\t{int(span)}\n")
            g.write(f">r{k}\n{fs[ts:te_]}\n")
            k += 1
    meta = dict(vars(a), te_len=len(te), n_events=len(events),
                lengths={n: len(s) for n, s in allrecs})
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(allrecs)} sequences, {len(events)} events, {len(dq)} direct + {a.n_ref} reference queries -> {out}")


if __name__ == "__main__":
    main()
