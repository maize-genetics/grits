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
EVENT_COLS = ["ins_events", "del_events", "unaligned_segments", "event_errors"]
CS_OP = re.compile(r"(:\d+|\*[a-z]{2}|[+\-][a-z]+|~[a-z]{2}\d+[a-z]{2})")
SCORE_COLS = ["seq_hash", "line", "qlen", "query_N", "status", "contig", "tstart", "tend", "strand", "mapq",
              "n_blocks", "aligned_q", "matches", "mismatches", "ins", "dels", "unaligned", "truth_N", "errors", "rescued",
              "truth_uncalled", "n_noncollinear"] + [f"t{K}_{x}" for K in (50, 20)
                                   for x in ("anchored", "matches", "errors", "trimmed_bp", "called")] + \
             EVENT_COLS + [f"t{K}_{x}" for K in (50, 20) for x in EVENT_COLS]


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


TRIM_KS = (50, 20)
COLLINEAR = True
COLLINEAR_TOL = 50


def collinear_with(b, blocks):
    """Supplementary block b is kept only if, against every block already kept, query order equals
    target order (reversed on '-'): target gap >= min(query gap, 0) - COLLINEAR_TOL and
    <= max(query gap, 0) + SLOP."""
    for c in blocks:
        left, right = (b, c) if b["qs"] < c["qs"] else (c, b)
        qgap = right["qs"] - left["qe"]
        tgap = (right["ts"] - left["te"]) if b["strand"] == "+" else (left["ts"] - right["te"])
        if tgap < min(qgap, 0) - COLLINEAR_TOL or tgap > max(qgap, 0) + SLOP:
            return False
    return True


def score_query(qlen, nprefix, blocks, expect=None):
    """blocks: PAF records (dicts) for one query. nprefix: cumulative N counts of the query (len qlen+1).

    expect: the window's chromosome. If the best primary hit is elsewhere but some hit (primary or
    secondary) on `expect` scores >= RESCUE_FRAC of it, that hit is used instead (exact paralogs:
    minimap2 picks the primary among equal copies arbitrarily); flagged rescued=1.

    Trimmed agreement (t<K>_*, for K in TRIM_KS): the anchors are the first and last runs of >= K
    consecutive exactly matching bases (cs ':' ops, clipped to the claimed query range) in query order;
    errors are counted only inside [L, R) of the query (deletions strictly inside; gaps between blocks
    if both flanking block ends are inside), the non-N query bases outside are t<K>_trimmed_bp.
    No such run: t<K>_anchored = 0 (unanchored). Internal switch junctions are never trimmed.

    Event counts (full and t<K>_): ins_events = cs insertion ops with >= 1 non-N inserted base;
    del_events = cs deletion ops with >= 1 counted deleted base (an N-for-gap pair that cancels a deletion
    entirely removes it) plus each inter-block target gap; unaligned_segments = maximal runs of non-N
    query bases claimed by no block; event_errors = mismatches + ins_events + del_events +
    unaligned_segments (a 45 bp deletion counts 1, not 45)."""
    r = dict(rescued=0, truth_uncalled=0, n_noncollinear=0, status="unaligned", contig=".", tstart=".", tend=".", strand=".",
             mapq=".", n_blocks=0, aligned_q=0, matches=0, mismatches=0, ins=0, dels=0, unaligned=0, truth_N=0)
    called_all = int(qlen - nprefix[qlen])
    isN = np.diff(nprefix) > 0

    def runs_called(a, b):
        """Maximal runs of non-N bases in query [a, b)."""
        if b <= a:
            return 0
        x = ~isN[a:b]
        return int(x[0]) + int((x[1:] & ~x[:-1]).sum())
    for K in TRIM_KS:
        r.update({f"t{K}_anchored": 0, f"t{K}_matches": 0, f"t{K}_errors": 0, f"t{K}_trimmed_bp": called_all,
                  f"t{K}_called": 0})
        r.update({f"t{K}_{x}": 0 for x in EVENT_COLS})
    prim = [b for b in blocks if b["tp"] == "P"]
    if not prim:
        r["unaligned"] = called_all
        segs = runs_called(0, qlen)
        r.update(ins_events=0, del_events=0, unaligned_segments=segs, event_errors=segs)
        return r
    p = max(prim, key=lambda b: b["AS"])
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
    ev = []    # (kind, pos, n): M matches [pos, n=end); X mismatch; I inserted base; D deletion at boundary pos
    for b in [p] + sorted(sup, key=lambda b: -b["AS"]):
        if COLLINEAR and b is not p and not collinear_with(b, [c for c, _ in chosen]):
            r["n_noncollinear"] += 1
            continue
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
                for x, y in parts:
                    a0, b0 = max(x, lo), min(y, hi)
                    if b0 > a0:
                        ev.append(("M", a0, b0))
                q = lo if rev else hi
            elif c == "*":
                qi = q - 1 if rev else q
                if inside(parts, qi):
                    if op[2] == "n":
                        pass
                    elif op[1] == "n":
                        r["truth_N"] += 1
                    else:
                        ev.append(("X", qi, 1))
                q = qi if rev else q + 1
            elif c == "+":
                s = op[1:]
                nn = 0
                first = None
                for k, ch in enumerate(s):
                    qi = q - 1 - k if rev else q + k
                    if inside(parts, qi):
                        if ch != "n":
                            ev.append(("I", qi, 1))
                            first = qi if first is None else first
                        else:
                            nn += 1
                if first is not None:
                    ev.append(("IE", first, 1))
                q2 = q - len(s) if rev else q + len(s)
                comp = min(nn, prev_del)
                if comp:
                    ev.append(("D", q, -comp))
                    if comp == prev_del:
                        ev.append(("DE", q, -1))
                q = q2
                prev_n, prev_del = nn - comp, 0
            elif c == "-":
                d = 0
                if inside(parts, q - 1) and inside(parts, q):
                    d = len(op) - 1 - op.count("n")
                comp = min(d, prev_n)
                if d - comp:
                    ev.append(("D", q, d - comp))
                    ev.append(("DE", q, 1))
                prev_n, prev_del = prev_n - comp, d - comp
    chosen.sort(key=lambda x: x[0]["qs"])
    gaps = []
    for (a, _), (b, _) in zip(chosen, chosen[1:]):
        tgap = (b["ts"] - a["te"]) if a["strand"] == "+" else (a["ts"] - b["te"])
        qgap = max(0, b["qs"] - a["qe"])
        if tgap > qgap:
            if qgap and nprefix[b["qs"]] > nprefix[a["qe"]]:
                r["truth_uncalled"] += tgap - qgap
            else:
                gaps.append((a["qe"], b["qs"], tgap - qgap))

    def tally(L, R):
        m = x = i = d = ie = de = 0
        for k, a0, n in ev:
            if k == "M":
                m += max(0, min(n, R) - max(a0, L))
            elif k == "X":
                x += L <= a0 < R
            elif k == "I":
                i += L <= a0 < R
            elif k == "IE":
                ie += L <= a0 < R
            elif k == "DE":
                de += n if L < a0 < R else 0
            else:
                d += n if L < a0 < R else 0
        d += sum(n for a0, b0, n in gaps if a0 >= L and b0 <= R)
        de += sum(1 for a0, b0, n in gaps if a0 >= L and b0 <= R)
        segs, pos = 0, L                          # unclaimed stretches of [L, R), split at N runs
        for xx, y in sorted(cl.iv):
            if y <= L or xx >= R:
                continue
            segs += runs_called(pos, max(pos, xx))
            pos = max(pos, min(y, R))
        segs += runs_called(pos, R)
        tally.events = (ie, de, segs, x + ie + de + segs)
        clm = sum(max(0, min(y, R) - max(xx, L)) for xx, y in cl.iv)
        clm_n = sum(int(nprefix[min(y, R)] - nprefix[max(xx, L)]) for xx, y in cl.iv if min(y, R) > max(xx, L))
        span_called = (R - L) - int(nprefix[R] - nprefix[L])
        un = span_called - (clm - clm_n)
        return m, x, i, d, un, span_called

    m, x, i, d, un, _ = tally(0, qlen)
    r.update(matches=m, mismatches=x, ins=i, dels=d, unaligned=un, aligned_q=cl.total(), n_blocks=len(chosen))
    r.update(zip(EVENT_COLS, tally.events))
    for K in TRIM_KS:
        runs = [(a0, n) for k, a0, n in ev if k == "M" and n - a0 >= K]
        if not runs:
            continue
        L, R = min(a0 for a0, _ in runs), max(n for _, n in runs)
        m, x, i, d, un, sc = tally(L, R)
        r.update({f"t{K}_anchored": 1, f"t{K}_matches": m, f"t{K}_errors": x + i + d + un,
                  f"t{K}_trimmed_bp": called_all - sc, f"t{K}_called": sc})
        r.update({f"t{K}_{c}": v for c, v in zip(EVENT_COLS, tally.events)})
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


def all_n_row(h, line, qlen, n_n):
    """Score row of a query that is entirely N (never aligned)."""
    return dict(zip(SCORE_COLS, [h, line, qlen, n_n, "all_N", ".", ".", ".", ".", "."] + [0] * (len(SCORE_COLS) - 10)))


def align_line(line, seqs, expect, out, sfx="", threads=3, preset="asm20", rescore_only=False, keep_query=False):
    """Align the sequences `seqs` ({seq_hash: seq}, none all-N) to `line` with minimap2 and score each one
    (score_query, then the off-chromosome rescue pass). expect: {seq_hash: window chromosome}.
    Writes q_<line><sfx>.fa (deleted unless keep_query), <line><sfx>.paf and <line><sfx>.rescue.paf into
    `out`. Returns ({seq_hash: score row dict}, info dict)."""
    out = Path(out)
    t0 = time.time()
    hs = sorted(seqs)
    qfa, paf = out / f"q_{line}{sfx}.fa", out / f"{line}{sfx}.paf"
    if not rescore_only:
        with open(qfa, "w") as f:
            for h in hs:
                f.write(f">{h}\n{seqs[h]}\n")
        with open(paf, "w") as fo, open(out / f"{line}{sfx}.mm2.log", "w") as fe:
            rc = subprocess.run([MM2, "-x", preset, "-c", "--cs", "--secondary=no", "-t", str(threads),
                                 MMI.format(line=line), str(qfa)], stdout=fo, stderr=fe).returncode
        if rc:
            raise RuntimeError(f"minimap2 failed for {line}")
    t1 = time.time()
    by = parse_paf(paf)
    rows = {}
    for h in hs:
        sq = seqs[h]
        rows[h] = score_query(len(sq), nprefix_of(sq), by.get(h, []), expect.get(h))
    off = sorted(h for h, r in rows.items() if r["status"] == "aligned" and r["contig"] != expect.get(h))
    n_res = 0
    if off:
        rfa, rpaf = out / f"rescue_q_{line}{sfx}.fa", out / f"{line}{sfx}.rescue.paf"
    if off and not (rescore_only and rpaf.exists()):
        with open(rfa, "w") as f:
            for h in off:
                f.write(f">{h}\n{seqs[h]}\n")
        with open(rpaf, "w") as fo, open(out / f"{line}{sfx}.rescue.mm2.log", "w") as fe:
            rc = subprocess.run([MM2, "-x", preset, "-c", "--cs", "-N", "20", "-t", str(threads),
                                 MMI.format(line=line), str(rfa)], stdout=fo, stderr=fe).returncode
        if rc:
            raise RuntimeError(f"minimap2 rescue failed for {line}")
        rfa.unlink()
    if off:
        rby = parse_paf(rpaf)
        for h in off:
            sq = seqs[h]
            r = score_query(len(sq), nprefix_of(sq), rby.get(h, []), expect.get(h))
            if r["rescued"]:
                rows[h] = r
                n_res += 1
    for h in hs:
        r = rows[h]
        r.update(seq_hash=h, line=line, qlen=len(seqs[h]), query_N=int(seqs[h].upper().count("N")))
        r["errors"] = r["mismatches"] + r["ins"] + r["dels"] + r["unaligned"]
    if not rescore_only and not keep_query:
        qfa.unlink()
    return rows, dict(n=len(hs), mm2_s=round(t1 - t0, 1), score_s=round(time.time() - t1, 1), off=len(off),
                      rescued=n_res)


def main():
    global SLOP, COLLINEAR
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitch", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lines", nargs="+", default=ALL_LINES)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--preset", default="asm20")
    ap.add_argument("--chain-slop", type=int, default=20000)
    ap.add_argument("--no-collinear", action="store_true",
                    help="v1/v2 behaviour: combine supplementary blocks without the collinearity check")
    ap.add_argument("--set", choices=["all", "model", "panel"], default="all",
                    help="align the model haplotypes, the panel windows, or both; scores_<line>[.<set>].tsv")
    ap.add_argument("--rescore-only", action="store_true", help="reuse existing PAFs")
    a = ap.parse_args()
    SLOP = a.chain_slop
    COLLINEAR = not a.no_collinear
    st, out = Path(a.stitch), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    need = defaultdict(set)
    qinfo, expect = {}, {}
    with open(st / "model_haps.tsv") if a.set != "panel" else open("/dev/null") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            qinfo[r["seq_hash"]] = (int(r["seq_len"]), int(r["n_N"]))
            expect.setdefault(r["seq_hash"], r["chrom"])
            for line in truth_lines(r["kind"], r["sample"]):
                need[line].add(r["seq_hash"])
    if a.set != "model":
        with open(st / "panel.tsv") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["status"] == "ok":
                    qinfo[r["seq_hash"]] = (int(r["seq_len"]), int(r["n_N"]))
                    expect.setdefault(r["seq_hash"], r["chrom"])
                    for line in ALL_LINES:
                        need[line].add(r["seq_hash"])
    for line in a.lines:
        hs = {h for h in need[line] if qinfo[h][0] > qinfo[h][1]}
        empty = need[line] - hs
        seqs = {}
        for fa in sorted(st.glob("seqs*.fa")):
            seqs.update(read_fasta_subset(fa, hs - set(seqs)))
        sfx = "" if a.set == "all" else f".{a.set}"
        try:
            rows, info = align_line(line, seqs, expect, out, sfx, a.threads, a.preset, a.rescore_only)
        except RuntimeError as ex:
            sys.exit(str(ex))
        with open(out / f"scores_{line}{sfx}.tsv", "w") as fo:
            fo.write("\t".join(SCORE_COLS) + "\n")
            for h in sorted(hs):
                r = rows[h]
                r["query_N"] = qinfo[h][1]
                fo.write("\t".join(str(r[c]) for c in SCORE_COLS) + "\n")
            for h in sorted(empty):
                r = all_n_row(h, line, *qinfo[h])
                fo.write("\t".join(str(r[c]) for c in SCORE_COLS) + "\n")
        print(f"{line}: {len(hs)} queries ({len(empty)} all-N), minimap2 {info['mm2_s']:.0f}s, "
              f"score+rescue {info['score_s']:.0f}s; off-chromosome primaries {info['off']}, rescued "
              f"{info['rescued']}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
