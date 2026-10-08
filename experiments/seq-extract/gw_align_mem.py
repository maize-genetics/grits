"""Gene-window eval pilot: a MUMmer-style aligner from `ropebwt3 mem` (SMEMs) + collinear chaining.

Per truth line (its own FM-index, as for the sw pilot) and the same queries as gw_align.py:
  1. MEMs: `ropebwt3 mem -l L -p MAXOCC`; a MEM is kept only if its total occurrence count (col 4,
     both strands) is <= --max-occ, and every one of its positions becomes an anchor. Positions are the
     0-based forward start of the target interval on either strand.
  2. Chain: per (contig, strand), DP over anchors sorted by query start maximising
        covered query bases - C |qgap - tgap| - 0.01 min(qgap, tgap) - 1 (per gap), C = --gap-coef
     overlaps are trimmed (the later anchor starts after both the query and target end of the earlier
     one), a single target gap is capped at --max-tgap (50 kb), predecessors searched among the previous
     --window anchors. On '-' anchors use virtual target coordinates -(end) so the DP is the same.
     The chain on the window's chromosome is used unless a chain elsewhere scores > 1/0.99 x higher
     (then the query is mislocated, as in the minimap2 path's rescue rule).
  3. Gaps between consecutive chained anchors (q = query gap, t = target gap):
       q=t=0 nothing; q=t=1 SNP; q=0 deletion(t); t=0 insertion(q); both > 0 and max <= --small-gap:
       unit-cost global alignment of the two gap sequences (SNPs and indels from the traceback);
       larger: structural block, max(q, t) bases. A gap whose query part contains N is uncalled (its
       truth bases go to truth_uncalled, nothing is an error). The chain ends are extended ungapped
       (X-drop, +1/-3, drop --xdrop) so a SNP closer than L bp to a query end is a mismatch, not an
       unaligned tail; what is left beyond the extension is unaligned.
  4. errors = mismatches + ins + dels + unaligned + structural; trimmed K metrics as in gw_align
     (anchor runs = chained MEM pieces >= K bp). Writes scores_<line>[.<set>].tsv (gw_align columns +
     structural, n_struct, snps, n_small_ins, n_small_del) and variants_<line>[.<set>].tsv.gz.

Usage: python gw_align_mem.py --stitch DIR --out-dir DIR --idx-dir DIR --lines Oh43 Tx303 --set model|panel
       [--min-len 31] [--max-occ 4] [--threads 3]
"""
import argparse
import csv
import gzip
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import gw_align as G  # noqa: E402
from python.seq_extract import FastaFile, revcomp  # noqa: E402

RB3 = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/seq_extract_ropebwt/ropebwt3_v5"
TRUTH = {"Oh43": "/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/Oh43.fa",
         "Tx303": "/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_v2_heldout/asm/Tx303.fa"}
EXTRA_COLS = ["structural", "n_struct", "snps", "n_small_ins", "n_small_del"]


@njit(cache=True)
def chain_dp(qs, qe, ts, te, max_tgap, window, coef):
    n = len(qs)
    score = np.empty(n)
    pred = np.full(n, -1, dtype=np.int64)
    shift = np.zeros(n, dtype=np.int64)
    for j in range(n):
        lj = qe[j] - qs[j]
        best, bp, bs = float(lj), -1, 0
        for i in range(max(0, j - window), j):
            if qe[i] >= qe[j] or te[i] >= te[j] or qs[i] > qs[j] or ts[i] > ts[j]:
                continue
            s = max(0, qe[i] - qs[j], te[i] - ts[j])
            if s >= lj:
                continue
            qg = qs[j] + s - qe[i]
            tg = ts[j] + s - te[i]
            if tg > max_tgap:
                continue
            pen = coef * abs(qg - tg) + 0.01 * min(qg, tg) + (1.0 if qg + tg > 0 else 0.0)
            v = score[i] + (lj - s) - pen
            if v > best:
                best, bp, bs = v, i, s
        score[j] = best
        pred[j] = bp
        shift[j] = bs
    return score, pred, shift


@njit(cache=True)
def edit_align(a, b):
    """Unit-cost global alignment of byte arrays a (query) and b (truth); returns the op codes in order:
    0 match, 1 mismatch, 2 insertion (query base), 3 deletion (truth base)."""
    n, m = len(a), len(b)
    D = np.empty((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        D[i, 0] = i
    for j in range(m + 1):
        D[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = D[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else 1)
            x = D[i - 1, j] + 1
            y = D[i, j - 1] + 1
            D[i, j] = min(c, x, y)
    ops = np.empty(n + m, dtype=np.int8)
    k = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            ops[k] = 0 if a[i - 1] == b[j - 1] else 1
            i -= 1
            j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops[k] = 2
            i -= 1
        else:
            ops[k] = 3
            j -= 1
        k += 1
    return ops[:k][::-1]


def parse_mem(path, max_occ):
    by = defaultdict(list)
    st = defaultdict(int)
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            qs, qe, occ = int(p[1]), int(p[2]), int(p[3])
            st["mems"] += 1
            st["mem_bp"] += qe - qs
            if occ > max_occ:
                st["mems_dropped_occ"] += 1
                st["mem_bp_dropped_occ"] += qe - qs
                continue
            for pos in p[5:]:
                c, strand, t = pos.rsplit(":", 2)
                by[p[0]].append((qs, qe, c, strand, int(t)))
    return by, st


def score_mem_query(seq, nprefix, anchors, expect, truth, a, variants, h):
    qlen = len(seq)
    called_all = int(qlen - nprefix[qlen])
    r = dict(rescued=0, truth_uncalled=0, n_noncollinear=0, status="unaligned", contig=".", tstart=".",
             tend=".", strand=".", mapq=".", n_blocks=0, aligned_q=0, matches=0, mismatches=0, ins=0, dels=0,
             unaligned=called_all,
             truth_N=0, structural=0, n_struct=0, snps=0, n_small_ins=0, n_small_del=0)
    for K in G.TRIM_KS:
        r.update({f"t{K}_anchored": 0, f"t{K}_matches": 0, f"t{K}_errors": 0, f"t{K}_trimmed_bp": called_all,
                  f"t{K}_called": 0})
    if not anchors:
        return r
    groups = defaultdict(list)
    for qs, qe, c, strand, t in anchors:
        L = qe - qs
        if strand == "+":
            groups[(c, strand)].append((qs, qe, t, t + L))
        else:
            groups[(c, strand)].append((qs, qe, -(t + L), -t))
    best = {}
    for g, an in groups.items():
        an.sort()
        arr = np.array(an, dtype=np.int64)
        sc, pred, shift = chain_dp(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], a.max_tgap, a.window, a.gap_coef)
        j = int(np.argmax(sc))
        best[g] = (float(sc[j]), j, arr, pred, shift)
    glob = max(best, key=lambda g: best[g][0])
    on = [g for g in best if G.norm_chrom(g[0]) == expect]
    use = glob
    if on:
        g2 = max(on, key=lambda g: best[g][0])
        if best[g2][0] >= G.RESCUE_FRAC * best[glob][0]:
            use = g2
            r["rescued"] = int(g2 != glob)
    _, j, arr, pred, shift = best[use]
    ch = []
    while j >= 0:
        ch.append((int(arr[j, 0] + shift[j]), int(arr[j, 1]), int(arr[j, 2] + shift[j]), int(arr[j, 3])))
        j = int(pred[j])
    ch.reverse()
    contig, strand = use
    real = (lambda v: v) if strand == "+" else (lambda v: -v)
    tvals = [real(x) for c_ in ch for x in (c_[2], c_[3])]
    r.update(status="aligned", contig=G.norm_chrom(contig), tstart=min(tvals), tend=max(tvals), strand=strand,
             mapq=0, n_blocks=len(ch))
    ev = []   # (kind, qpos, n): M [qpos, n=end); X; I; D at boundary; S structural (qa, qb, size)
    for qs, qe, ts, te in ch:
        ev.append(("M", qs, qe))
    for (qs0, qe0, ts0, te0), (qs1, qe1, ts1, te1) in zip(ch, ch[1:]):
        q, t = qs1 - qe0, ts1 - te0
        if q == 0 and t == 0:
            continue
        qn = int(nprefix[qs1] - nprefix[qe0])
        tpos = real(te0) if strand == "+" else real(te0) - 1
        if qn:
            r["truth_uncalled"] += t
            continue
        if q == 1 and t == 1:
            ev.append(("X", qe0, 1))
            r["snps"] += 1
            variants.append((h, "SNP", qe0, tpos, 1, 1))
        elif q == 0:
            ev.append(("D", qe0, t))
            r["n_small_del" if t <= a.small_gap else "n_struct"] += 1
            if t > a.small_gap:
                ev[-1] = ("S", qe0, qe0, t)
            variants.append((h, "DEL" if t <= a.small_gap else "SV_DEL", qe0, tpos, 0, t))
        elif t == 0:
            if q <= a.small_gap:
                for k in range(q):
                    ev.append(("I", qe0 + k, 1))
                r["n_small_ins"] += 1
                variants.append((h, "INS", qe0, tpos, q, 0))
            else:
                ev.append(("S", qe0, qs1, q))
                r["n_struct"] += 1
                variants.append((h, "SV_INS", qe0, tpos, q, 0))
        elif max(q, t) <= a.small_gap:
            tseq = truth.fetch(contig, real(te0), real(ts1), upper=True) if strand == "+" else \
                revcomp(truth.fetch(contig, real(ts1), real(te0), upper=True))
            qa = np.frombuffer(seq[qe0:qs1].encode(), dtype=np.uint8)
            tb = np.frombuffer(tseq.encode(), dtype=np.uint8)
            ops = edit_align(qa, tb)
            qp, prev = qe0, 0
            for op in ops:
                if op == 0:
                    ev.append(("M", qp, qp + 1))
                    qp += 1
                elif op == 1:
                    ev.append(("X", qp, 1))
                    r["snps"] += 1
                    variants.append((h, "SNP", qp, ".", 1, 1, "gap_aln"))
                    qp += 1
                elif op == 2:
                    ev.append(("I", qp, 1))
                    if prev != 2:
                        r["n_small_ins"] += 1
                        variants.append((h, "INS", qp, ".", 1, 0, "gap_aln"))
                    qp += 1
                else:
                    ev.append(("D", qp, 1))
                    if prev != 3:
                        r["n_small_del"] += 1
                        variants.append((h, "DEL", qp, ".", 0, 1, "gap_aln"))
                prev = op
        else:
            ev.append(("S", qe0, qs1, max(q, t)))
            r["n_struct"] += 1
            variants.append((h, "SV_COMPLEX", qe0, tpos, q, t))
    clen = truth.length(contig)

    def tseg(vs, ve):
        if strand == "+":
            vs, ve = max(0, vs), min(clen, ve)
            return vs, truth.fetch(contig, vs, ve, upper=True) if ve > vs else ""
        lo, hi = max(0, -ve), min(clen, -vs)
        return -hi, revcomp(truth.fetch(contig, lo, hi, upper=True)) if hi > lo else ""

    def extend(qpos, tpos, direction):
        """Ungapped X-drop extension (+1 match, -3 mismatch, drop 20) from the chain end; returns the
        number of query bases taken and the mismatch query positions."""
        n = qpos if direction < 0 else qlen - qpos
        if n <= 0:
            return 0, []
        vs, tt = tseg(tpos - n, tpos) if direction < 0 else tseg(tpos, tpos + n)
        if direction < 0:
            qq, tt = seq[qpos - len(tt):qpos][::-1], tt[::-1]
        else:
            qq = seq[qpos:qpos + len(tt)]
        sc = best = 0
        bk, mis, bmis = 0, [], []
        for k, (x, y) in enumerate(zip(qq, tt)):
            if x == "N":
                continue
            if x == y:
                sc += 1
            else:
                sc -= 3
                mis.append(qpos - 1 - k if direction < 0 else qpos + k)
            if sc > best:
                best, bk, bmis = sc, k + 1, list(mis)
            elif sc < best - a.xdrop:
                break
        return bk, bmis

    lk, lmis = extend(ch[0][0], ch[0][2], -1) if a.xdrop > 0 else (0, [])
    rk, rmis = extend(ch[-1][1], ch[-1][3], 1) if a.xdrop > 0 else (0, [])
    ext = [(ch[0][0] - lk, ch[0][0]), (ch[-1][1], ch[-1][1] + rk)]
    for x0, x1 in ext:
        if x1 > x0:
            ev.append(("M", x0, x1))
    for x in lmis + rmis:
        ev.append(("X", x, 1))
        ev.append(("M", x, x + 1, -1))
        r["snps"] += 1
        variants.append((h, "SNP", x, ".", 1, 1, "end_ext"))
    span = (ch[0][0] - lk, ch[-1][1] + rk)

    def tally(L, R):
        m = x = i = d = s = 0
        for e in ev:
            k = e[0]
            if k == "M":
                lo, hi = max(e[1], L), min(e[2], R)
                if hi > lo:
                    m += ((hi - lo) - int(nprefix[hi] - nprefix[lo])) * (e[3] if len(e) > 3 else 1)
            elif k == "X":
                x += L <= e[1] < R
            elif k == "I":
                i += L <= e[1] < R
            elif k == "D":
                d += e[2] if L < e[1] < R else 0
            else:
                inside = (L < e[1] < R) if e[1] == e[2] else (e[1] >= L and e[2] <= R)
                s += e[3] if inside else 0
        cl = max(0, min(span[1], R) - max(span[0], L))
        cl_n = int(nprefix[min(span[1], R)] - nprefix[max(span[0], L)]) if cl else 0
        sc = (R - L) - int(nprefix[R] - nprefix[L])
        return m, x, i, d, s, sc - (cl - cl_n), sc

    m, x, i, d, s, un, _ = tally(0, qlen)
    r.update(matches=m, mismatches=x, ins=i, dels=d, structural=s, unaligned=un, aligned_q=span[1] - span[0])
    for K in G.TRIM_KS:
        runs = [(qs, qe) for qs, qe, _, _ in ch if qe - qs >= K]
        if not runs:
            continue
        L, R = runs[0][0], runs[-1][1]
        m, x, i, d, s, un, sc = tally(L, R)
        r.update({f"t{K}_anchored": 1, f"t{K}_matches": m, f"t{K}_errors": x + i + d + s + un,
                  f"t{K}_trimmed_bp": called_all - sc, f"t{K}_called": sc})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitch", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--idx-dir", required=True)
    ap.add_argument("--lines", nargs="+", required=True)
    ap.add_argument("--set", choices=["all", "model", "panel"], default="all")
    ap.add_argument("--min-len", type=int, default=31)
    ap.add_argument("--max-occ", type=int, default=4)
    ap.add_argument("--max-tgap", type=int, default=50000)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--gap-coef", type=float, default=0.2, help="chain penalty per bp of |qgap - tgap|")
    ap.add_argument("--small-gap", type=int, default=1000)
    ap.add_argument("--xdrop", type=int, default=20, help="ungapped end extension X-drop; 0 = off")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--rb3", default=RB3)
    ap.add_argument("--reuse-mem", action="store_true", help="reuse <out>/<line>.<set>.mem.tsv (rechain only)")
    a = ap.parse_args()
    st, out = Path(a.stitch), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    need, qinfo, expect = defaultdict(set), {}, {}
    if a.set != "panel":
        for r in csv.DictReader(open(st / "model_haps.tsv"), delimiter="\t"):
            qinfo[r["seq_hash"]] = (int(r["seq_len"]), int(r["n_N"]))
            expect.setdefault(r["seq_hash"], r["chrom"])
            for line in G.truth_lines(r["kind"], r["sample"]):
                need[line].add(r["seq_hash"])
    if a.set != "model":
        for r in csv.DictReader(open(st / "panel.tsv"), delimiter="\t"):
            if r["status"] == "ok":
                qinfo[r["seq_hash"]] = (int(r["seq_len"]), int(r["n_N"]))
                expect.setdefault(r["seq_hash"], r["chrom"])
                for line in G.ALL_LINES:
                    need[line].add(r["seq_hash"])
    sfx = "" if a.set == "all" else f".{a.set}"
    for line in a.lines:
        hs = {h for h in need[line] if qinfo[h][0] > qinfo[h][1]}
        empty = need[line] - hs
        seqs = {}
        for fa in sorted(st.glob("seqs*.fa")):
            seqs.update(G.read_fasta_subset(fa, hs - set(seqs)))
        qfa = out / f"q_{line}{sfx}.fa"
        with open(qfa, "w") as f:
            for h in sorted(hs):
                f.write(f">{h}\n{seqs[h]}\n")
        mem = out / f"{line}{sfx}.mem.tsv"
        tfile = out / f"{line}{sfx}.mem.time.txt"
        t0 = time.time()
        if not (a.reuse_mem and mem.exists()):
            with open(mem, "w") as fo, open(out / f"{line}{sfx}.mem.log", "w") as fe:
                rc = subprocess.run(["/usr/bin/time", "-v", "-o", str(tfile), a.rb3, "mem", "-t", str(a.threads),
                                     "-l", str(a.min_len), "-p", str(a.max_occ),
                                     str(Path(a.idx_dir) / f"{line}.fmd"), str(qfa)], stdout=fo, stderr=fe).returncode
            if rc:
                sys.exit(f"ropebwt3 mem failed for {line}")
        qfa.unlink()
        t_mem = time.time() - t0
        by, mst = parse_mem(mem, a.max_occ)
        truth = FastaFile(TRUTH[line], TRUTH[line] + ".fai")
        t1 = time.time()
        variants = []
        rows = {}
        for h in sorted(hs):
            rows[h] = score_mem_query(seqs[h], G.nprefix_of(seqs[h]), by.get(h, []), expect.get(h), truth, a,
                                      variants, h)
        t_chain = time.time() - t1
        cols = G.SCORE_COLS + EXTRA_COLS
        with open(out / f"scores_{line}{sfx}.tsv", "w") as fo:
            fo.write("\t".join(cols) + "\n")
            for h in sorted(hs):
                r = rows[h]
                r.update(seq_hash=h, line=line, qlen=len(seqs[h]), query_N=qinfo[h][1])
                r["errors"] = r["mismatches"] + r["ins"] + r["dels"] + r["unaligned"] + r["structural"]
                fo.write("\t".join(str(r.get(c, ".")) for c in cols) + "\n")
            for h in sorted(empty):
                fo.write("\t".join(map(str, [h, line, qinfo[h][0], qinfo[h][1], "all_N", ".", ".", ".", ".", "."]
                                       + [0] * (len(cols) - 10))) + "\n")
        with gzip.open(out / f"variants_{line}{sfx}.tsv.gz", "wt") as fo:
            fo.write("seq_hash\ttype\tqpos\ttpos\tqlen\ttlen\tsource\n")
            for v in variants:
                fo.write("\t".join(map(str, list(v) + ["gap"] * (7 - len(v)))) + "\n")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
        mem_rss = next((float(l.split(": ")[1]) / 1048576 for l in open(tfile) if "Maximum resident" in l), -1)
        with open(out / f"{line}{sfx}.stats.tsv", "w") as fo:
            fo.write("key\tvalue\n")
            for k, v in list(mst.items()) + [("queries", len(hs)), ("mem_wall_s", round(t_mem, 1)),
                                             ("mem_max_rss_GB", round(mem_rss, 3)),
                                             ("chain_gapdp_wall_s", round(t_chain, 1)),
                                             ("python_max_rss_GB", round(rss, 3))]:
                fo.write(f"{k}\t{v}\n")
        print(f"{line}{sfx}: {len(hs)} queries, mem {t_mem:.0f}s ({mem_rss:.2f} GB), chain+gap DP {t_chain:.0f}s; "
              f"{dict(mst)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
