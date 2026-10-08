"""Gene-window eval pilot: ropebwt3 `sw` as the aligner instead of minimap2.

Same queries as gw_align.py (the model haplotype windows whose truth involves the line, and the panel
windows), aligned with `ropebwt3 sw` against a per-truth-line FM-index (<line>.fmd + .ssa + .len.gz),
scored with gw_align.score_query (same cs parser, N handling, trimmed metric, chromosome check).

Differences from the minimap2 path (documented in the report):
  * `sw` reports ONE hit per query (no secondary/supplementary). With --split-rounds R > 0, the
    largest unaligned query piece (>= --split-min bp, after excluding N) left of / right of the
    current blocks is re-aligned on its own, up to R rounds; hits on the same contig and strand within
    gw_align.SLOP of the primary AND collinear with every accepted block (query order = target order,
    overlap <= --collinear-tol, target gap <= query gap + SLOP) are added as supplementary blocks ('tp'
    S) with query coordinates shifted back. This approximates minimap2's chained split alignments.
  * no paralog rescue: `sw` gives no secondary hits (the rh tag only counts BWT hits), so an
    off-chromosome primary stays mislocated.
  * the score fields mapq is 0 in `sw` output.

Usage: python gw_align_rb3.py --stitch DIR --out-dir DIR --idx-dir DIR --lines Oh43 Tx303 --set model|panel
       [--sw-args "-A1 -B4 -O6 -E1"] [--split-rounds 3] [--threads 3]
"""
import argparse
import csv
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gw_align as G  # noqa: E402

RB3 = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/seq_extract_ropebwt/ropebwt3_v3"


def run_sw(a, idx, fa, paf, log):
    with open(paf, "w") as fo, open(log, "a") as fe:
        rc = subprocess.run([a.rb3, "sw", "-t", str(a.threads)] + a.sw_args.split() + [str(idx), str(fa)],
                            stdout=fo, stderr=fe).returncode
    if rc:
        sys.exit(f"ropebwt3 sw failed ({fa})")
    return G.parse_paf(paf)


def free_pieces(qlen, blocks, nprefix, min_len):
    cl = sorted((b["qs"], b["qe"]) for b in blocks)
    out, pos = [], 0
    for s, e in cl + [(qlen, qlen)]:
        if s - pos >= min_len and (s - pos) - (nprefix[s] - nprefix[pos]) >= min_len:
            out.append((pos, s))
        pos = max(pos, e)
    return out


def collinear(b, blocks, tol):
    """A split block must keep query order = target order (reversed on '-') with every accepted block of
    the same strand, overlapping by at most `tol` and leaving a target gap of at most query gap + SLOP
    to its query neighbours (what a chaining aligner such as minimap2 enforces)."""
    for c in blocks:
        if c["tname"] != b["tname"] or c["strand"] != b["strand"]:
            continue
        left, right = (b, c) if b["qs"] < c["qs"] else (c, b)
        qgap = right["qs"] - left["qe"]
        tgap = (right["ts"] - left["te"]) if b["strand"] == "+" else (left["ts"] - right["te"])
        if tgap < -tol or tgap > max(qgap, 0) + G.SLOP:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitch", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--idx-dir", required=True)
    ap.add_argument("--lines", nargs="+", required=True)
    ap.add_argument("--set", choices=["all", "model", "panel"], default="all")
    ap.add_argument("--sw-args", default="")
    ap.add_argument("--split-rounds", type=int, default=3)
    ap.add_argument("--split-min", type=int, default=200)
    ap.add_argument("--collinear-tol", type=int, default=50)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--rb3", default=RB3)
    a = ap.parse_args()
    st, out = Path(a.stitch), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    need, qinfo, expect = defaultdict(set), {}, {}
    if a.set != "panel":
        with open(st / "model_haps.tsv") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                qinfo[r["seq_hash"]] = (int(r["seq_len"]), int(r["n_N"]))
                expect.setdefault(r["seq_hash"], r["chrom"])
                for line in G.truth_lines(r["kind"], r["sample"]):
                    need[line].add(r["seq_hash"])
    if a.set != "model":
        with open(st / "panel.tsv") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["status"] == "ok":
                    qinfo[r["seq_hash"]] = (int(r["seq_len"]), int(r["n_N"]))
                    expect.setdefault(r["seq_hash"], r["chrom"])
                    for line in G.ALL_LINES:
                        need[line].add(r["seq_hash"])
    sfx = "" if a.set == "all" else f".{a.set}"
    for line in a.lines:
        t0 = time.time()
        hs = {h for h in need[line] if qinfo[h][0] > qinfo[h][1]}
        empty = need[line] - hs
        seqs = {}
        for fa in sorted(st.glob("seqs*.fa")):
            seqs.update(G.read_fasta_subset(fa, hs - set(seqs)))
        idx = Path(a.idx_dir) / f"{line}.fmd"
        log = out / f"{line}{sfx}.sw.log"
        log.write_text("")
        qfa = out / f"q_{line}{sfx}.fa"
        with open(qfa, "w") as f:
            for h in sorted(hs):
                f.write(f">{h}\n{seqs[h]}\n")
        by = run_sw(a, idx, qfa, out / f"{line}{sfx}.paf", log)
        qfa.unlink()
        t_main = time.time() - t0
        nprefix = {h: G.nprefix_of(seqs[h]) for h in hs}
        n_split = n_noncol = 0
        for rnd in range(a.split_rounds):
            pieces = {}
            for h in hs:
                bl = by.get(h, [])
                if not bl:
                    continue
                for s, e in free_pieces(len(seqs[h]), bl, nprefix[h], a.split_min):
                    pieces[f"{h}:{s}:{e}"] = seqs[h][s:e]
            if not pieces:
                break
            pfa = out / f"split{rnd}_{line}{sfx}.fa"
            with open(pfa, "w") as f:
                for k, v in pieces.items():
                    f.write(f">{k}\n{v}\n")
            sby = run_sw(a, idx, pfa, out / f"{line}{sfx}.split{rnd}.paf", log)
            pfa.unlink()
            added = 0
            for k, recs in sby.items():
                h, s, _ = k.rsplit(":", 2)
                s = int(s)
                p = max(by[h], key=lambda b: b["AS"])
                for b in recs:
                    if b["tname"] != p["tname"] or b["strand"] != p["strand"] or \
                            not (b["ts"] < p["te"] + G.SLOP and b["te"] > p["ts"] - G.SLOP):
                        continue
                    b = dict(b, qs=b["qs"] + s, qe=b["qe"] + s, qlen=len(seqs[h]), tp="S")
                    if not collinear(b, by[h], a.collinear_tol):
                        n_noncol += 1
                        continue
                    by[h].append(b)
                    added += 1
            n_split += added
            if not added:
                break
        t1 = time.time()
        with open(out / f"scores_{line}{sfx}.tsv", "w") as fo:
            fo.write("\t".join(G.SCORE_COLS) + "\n")
            for h in sorted(hs):
                r = G.score_query(len(seqs[h]), nprefix[h], by.get(h, []), expect.get(h))
                r.update(seq_hash=h, line=line, qlen=len(seqs[h]), query_N=qinfo[h][1])
                r["errors"] = r["mismatches"] + r["ins"] + r["dels"] + r["unaligned"]
                fo.write("\t".join(str(r[c]) for c in G.SCORE_COLS) + "\n")
            for h in sorted(empty):
                r = G.all_n_row(h, line, *qinfo[h])
                fo.write("\t".join(str(r[c]) for c in G.SCORE_COLS) + "\n")
        print(f"{line}{sfx}: {len(hs)} queries, sw {t_main:.0f}s, split rounds added {n_split} blocks, rejected {n_noncol} non-collinear "
              f"({t1 - t0 - t_main:.0f}s), score {time.time() - t1:.0f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
