"""Gene-window eval, step 4: align window sequences to truth assemblies and score each alignment.

For every truth line, the unique sequences that must be compared with it (imputed haplotypes of the
samples whose truth involves that line, and all 25 founder window sequences for the panel ceiling) are
aligned with minimap2 (-x asm20 -c --cs --secondary=no) against a prebuilt index of the line's assembly.

Scoring one query (scores_<line>.tsv):
  * blocks = the primary alignment plus supplementary alignments on the same contig and strand whose
    target span lies within --chain-slop bp of the primary's; taken primary first, then by AS. A query
    base already claimed by an earlier block is not counted again (overlapping supplementaries).
  * from the cs tags: matches, mismatches, inserted bases, deleted bases (only inside a block's claimed
    query range); target gaps between consecutive blocks (in query order) count as deleted bases;
  * unaligned = non-N query bases claimed by no block;
  * N in the query (uncalled gaps, failed projections) is never an error: N bases are excluded from
    mismatches, insertions and unaligned and reported once as `query_N`; truth-side N mismatches are
    reported as `truth_N`, not errors.
  errors = mismatches + inserted + deleted + unaligned.
  The primary hit's contig (normalised to chrN) and span are kept for the mislocation and RIL checks.

Usage: python gw_align.py --stitch DIR --out-dir DIR [--lines ...] [--threads 3] [--rescore-only]
"""
import argparse
import csv
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

MM2 = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/tools/minimap2-2.28_x64-linux/minimap2"
MMI = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/gene_window_eval/mmi/{line}.asm20.mmi"
IDX_LINES = ["B73", "B97", "CML103", "Il14H", "Oh43"]
OUT_LINES = ["A188", "CML459", "EP1", "Ia453", "Tx303"]
ALL_LINES = IDX_LINES + OUT_LINES
CS_OP = re.compile(r"(:\d+|\*[a-z]{2}|[+\-][a-z]+|~[a-z]{2}\d+[a-z]{2})")
SCORE_COLS = ["seq_hash", "line", "qlen", "query_N", "status", "contig", "tstart", "tend", "strand", "mapq",
              "n_blocks", "aligned_q", "matches", "mismatches", "ins", "dels", "unaligned", "truth_N", "errors", "rescued",
              "truth_uncalled"]


def split_pair(ind):
    cands = [(ind[:k], ind[k + 1:]) for k, ch in enumerate(ind) if ch == "x"]
    cands = [c for c in cands if c[0] in ALL_LINES and c[1] in ALL_LINES]
    if len(cands) != 1:
        raise ValueError(f"cannot split '{ind}'")
    return cands[0]


def truth_lines(kind, ind):
    return [ind] if kind == "inbred" else list(split_pair(ind))


def norm_chrom(c):
    m = re.fullmatch(r"(?:chr|Chr|CHR)?0*(\d+)", c)
    return f"chr{m.group(1)}" if m else c


def read_fasta_subset(path, keep):
    out, name, buf = {}, None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name in keep:
                    out[name] = "".join(buf)
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if name in keep:
        out[name] = "".join(buf)
    return out


class Claimed:
    def __init__(self):
        self.iv = []

    def free_parts(self, s, e):
        parts = [(s, e)]
        for a, b in self.iv:
            nxt = []
            for x, y in parts:
                if b <= x or a >= y:
                    nxt.append((x, y))
                else:
                    if x < a:
                        nxt.append((x, a))
                    if b < y:
                        nxt.append((b, y))
            parts = nxt
        return parts

    def add(self, parts):
        self.iv += parts

    def total(self):
        return sum(b - a for a, b in self.iv)


def inside(parts, q):
    return any(a <= q < b for a, b in parts)


def overlap(parts, s, e):
    return sum(max(0, min(b, e) - max(a, s)) for a, b in parts)


def score_query(qlen, nprefix, blocks, expect=None):
    """blocks: PAF records (dicts) for one query. nprefix: cumulative N counts of the query (len qlen+1).

    expect: the window's chromosome. If the best primary hit is elsewhere but some hit (primary or
    secondary) on `expect` scores >= RESCUE_FRAC of it, that hit is used instead (exact paralogs:
    minimap2 picks the primary among equal copies arbitrarily); flagged rescued=1."""
    r = dict(rescued=0, truth_uncalled=0, status="unaligned", contig=".", tstart=".", tend=".", strand=".", mapq=".", n_blocks=0,
             aligned_q=0, matches=0, mismatches=0, ins=0, dels=0, unaligned=0, truth_N=0)
    prim = [b for b in blocks if b["tp"] == "P"]
    if not prim:
        r["unaligned"] = int(qlen - nprefix[qlen])
        return r
    p = max(prim, key=lambda b: b["AS"])
    r["rescued"] = 0
    if expect is not None and norm_chrom(p["tname"]) != expect:
        on = [b for b in blocks if norm_chrom(b["tname"]) == expect]
        if on:
            cand = max(on, key=lambda b: b["AS"])
            if cand["AS"] >= RESCUE_FRAC * p["AS"]:
                p = cand
                r["rescued"] = 1
    r.update(status="aligned", contig=norm_chrom(p["tname"]), tstart=p["ts"], tend=p["te"], strand=p["strand"],
             mapq=p["mapq"])
    sup = [b for b in blocks if b is not p and b["tname"] == p["tname"] and b["strand"] == p["strand"]
           and b["ts"] < p["te"] + SLOP and b["te"] > p["ts"] - SLOP]
    chosen = []
    cl = Claimed()
    for b in [p] + sorted(sup, key=lambda b: -b["AS"]):
        parts = cl.free_parts(b["qs"], b["qe"])
        if not parts:
            continue
        cl.add(parts)
        chosen.append((b, parts))
        rev = b["strand"] == "-"
        q = b["qe"] if rev else b["qs"]
        prev_n = prev_del = 0      # adjacent '+N..' / '-..' ops: an N run aligned as a gap pair is not a deletion
        for op in CS_OP.findall(b["cs"]):
            c = op[0]
            if c != "+" and c != "-":
                prev_n = prev_del = 0
            if c == ":":
                n = int(op[1:])
                lo, hi = (q - n, q) if rev else (q, q + n)
                r["matches"] += overlap(parts, lo, hi)
                q = lo if rev else hi
            elif c == "*":
                qi = q - 1 if rev else q
                if inside(parts, qi):
                    if op[2] == "n":
                        pass
                    elif op[1] == "n":
                        r["truth_N"] += 1
                    else:
                        r["mismatches"] += 1
                q = qi if rev else q + 1
            elif c == "+":
                s = op[1:]
                nn = 0
                for k, ch in enumerate(s):
                    qi = q - 1 - k if rev else q + k
                    if inside(parts, qi):
                        if ch != "n":
                            r["ins"] += 1
                        else:
                            nn += 1
                q = q - len(s) if rev else q + len(s)
                comp = min(nn, prev_del)
                r["dels"] -= comp
                prev_n, prev_del = nn - comp, 0
            elif c == "-":
                d = 0
                if inside(parts, q - 1) and inside(parts, q):
                    d = len(op) - 1 - op.count("n")
                comp = min(d, prev_n)
                r["dels"] += d - comp
                prev_n, prev_del = prev_n - comp, d - comp
    chosen.sort(key=lambda x: x[0]["qs"])
    for (a, _), (b, _) in zip(chosen, chosen[1:]):
        tgap = (b["ts"] - a["te"]) if a["strand"] == "+" else (a["ts"] - b["te"])
        qgap = max(0, b["qs"] - a["qe"])
        if tgap > qgap:
            if qgap and nprefix[b["qs"]] > nprefix[a["qe"]]:
                r["truth_uncalled"] += tgap - qgap
            else:
                r["dels"] += tgap - qgap
    claimed_n = int(sum(nprefix[y] - nprefix[x] for x, y in cl.iv))
    r["aligned_q"] = cl.total()
    r["unaligned"] = int(qlen - nprefix[qlen]) - (cl.total() - claimed_n)
    r["n_blocks"] = len(chosen)
    return r


SLOP = 20000
RESCUE_FRAC = 0.99


def parse_paf(path):
    by = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            tags = {t[:2]: t[5:] for t in p[12:]}
            by[p[0]].append(dict(qlen=int(p[1]), qs=int(p[2]), qe=int(p[3]), strand=p[4], tname=p[5],
                                 ts=int(p[7]), te=int(p[8]), mapq=int(p[11]), tp=tags.get("tp", "P"),
                                 AS=int(tags.get("AS", 0)), cs=tags.get("cs", "")))
    return by


def nprefix_of(seq):
    out = np.zeros(len(seq) + 1, dtype=np.int64)
    np.cumsum(np.frombuffer(seq.upper().encode(), dtype=np.uint8) == ord("N"), out=out[1:])
    return out


def main():
    global SLOP
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitch", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lines", nargs="+", default=ALL_LINES)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--preset", default="asm20")
    ap.add_argument("--chain-slop", type=int, default=20000)
    ap.add_argument("--no-panel", action="store_true")
    ap.add_argument("--rescore-only", action="store_true", help="reuse existing PAFs")
    a = ap.parse_args()
    SLOP = a.chain_slop
    st, out = Path(a.stitch), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    need = defaultdict(set)
    qinfo, expect = {}, {}
    with open(st / "model_haps.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            qinfo[r["seq_hash"]] = (int(r["seq_len"]), int(r["n_N"]))
            expect.setdefault(r["seq_hash"], r["chrom"])
            for line in truth_lines(r["kind"], r["sample"]):
                need[line].add(r["seq_hash"])
    if not a.no_panel:
        with open(st / "panel.tsv") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["status"] == "ok":
                    qinfo[r["seq_hash"]] = (int(r["seq_len"]), int(r["n_N"]))
                    expect.setdefault(r["seq_hash"], r["chrom"])
                    for line in ALL_LINES:
                        need[line].add(r["seq_hash"])
    for line in a.lines:
        t0 = time.time()
        hs = {h for h in need[line] if qinfo[h][0] > qinfo[h][1]}
        empty = need[line] - hs
        seqs = read_fasta_subset(st / "seqs.fa", hs)
        qfa, paf = out / f"q_{line}.fa", out / f"{line}.paf"
        if not a.rescore_only:
            with open(qfa, "w") as f:
                for h in sorted(hs):
                    f.write(f">{h}\n{seqs[h]}\n")
            with open(paf, "w") as fo, open(out / f"{line}.mm2.log", "w") as fe:
                rc = subprocess.run([MM2, "-x", a.preset, "-c", "--cs", "--secondary=no", "-t", str(a.threads),
                                     MMI.format(line=line), str(qfa)], stdout=fo, stderr=fe).returncode
            if rc:
                sys.exit(f"minimap2 failed for {line}")
        t1 = time.time()
        by = parse_paf(paf)
        rows = {}
        for h in sorted(hs):
            sq = seqs[h]
            rows[h] = score_query(len(sq), nprefix_of(sq), by.get(h, []), expect.get(h))
        off = sorted(h for h, r in rows.items() if r["status"] == "aligned" and r["contig"] != expect.get(h))
        n_res = 0
        if off:
            rfa, rpaf = out / f"rescue_q_{line}.fa", out / f"{line}.rescue.paf"
        if off and not (a.rescore_only and rpaf.exists()):
            with open(rfa, "w") as f:
                for h in off:
                    f.write(f">{h}\n{seqs[h]}\n")
            with open(rpaf, "w") as fo, open(out / f"{line}.rescue.mm2.log", "w") as fe:
                rc = subprocess.run([MM2, "-x", a.preset, "-c", "--cs", "-N", "20", "-t", str(a.threads),
                                     MMI.format(line=line), str(rfa)], stdout=fo, stderr=fe).returncode
            if rc:
                sys.exit(f"minimap2 rescue failed for {line}")
            rfa.unlink()
        if off:
            rby = parse_paf(rpaf)
            for h in off:
                sq = seqs[h]
                r = score_query(len(sq), nprefix_of(sq), rby.get(h, []), expect.get(h))
                if r["rescued"]:
                    rows[h] = r
                    n_res += 1
        with open(out / f"scores_{line}.tsv", "w") as fo:
            fo.write("\t".join(SCORE_COLS) + "\n")
            for h in sorted(hs):
                r = rows[h]
                r.update(seq_hash=h, line=line, qlen=len(seqs[h]), query_N=qinfo[h][1])
                r["errors"] = r["mismatches"] + r["ins"] + r["dels"] + r["unaligned"]
                fo.write("\t".join(str(r[c]) for c in SCORE_COLS) + "\n")
            for h in sorted(empty):
                fo.write("\t".join(map(str, [h, line, qinfo[h][0], qinfo[h][1], "all_N", ".", ".", ".", ".", ".",
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) + "\n")
        print(f"{line}: {len(hs)} queries ({len(empty)} all-N), minimap2 {t1 - t0:.0f}s, "
              f"score+rescue {time.time() - t1:.0f}s; off-chromosome primaries {len(off)}, rescued {n_res}",
              file=sys.stderr, flush=True)
        if not a.rescore_only:
            qfa.unlink()


if __name__ == "__main__":
    main()
