#!/usr/bin/env python
"""Independent check of ropebwt3 extract's projected / snapped ends on real maize, using minimap2.

For each B73 window (qid ref_contig ref_start ref_end assembly), the B73 sequence is aligned with
minimap2 to the WHOLE target founder assembly (no lift involved). The minimap2 window ends follow the
same convention as the snap: start = founder base aligned to the first B73 base of the window present
in the founder, end = one past the founder base aligned to the last one (deleted B73 bases skipped,
founder-only insertions excluded). An end is only judged when the primary alignment reaches it (the
query is not clipped on that side) and lies on the homologous chromosome.

Writes per-end rows comparing minimap2 with the snapped and the unsnapped ropebwt3 projections.
"""
import argparse
import csv
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from python.seq_extract import AssemblySet  # noqa: E402

CS = re.compile(r"(:\d+|\*[a-z][a-z]|[+\-][a-z]+|~[a-z]{2}\d+[a-z]{2})")


def chrom_of(name):
    m = re.search(r"chr(\d+)$", name, re.I)
    return f"chr{int(m.group(1))}" if m else None


def ends_from_cs(qlen, qs, qe, strand, ts, te, cs):
    """Founder coordinates (forward strand) of the first and last query bases present in the target.
    Returns (start, end, start_reached, end_reached) with end exclusive."""
    # walk target forward; query offset goes forward on '+', backward on '-' strand
    t = ts
    q = qs if strand == "+" else qe - 1
    step = 1 if strand == "+" else -1
    aligned = []                    # (query offset, target pos) for matched / substituted bases
    for tok in CS.findall(cs):
        c = tok[0]
        if c == ":":
            n = int(tok[1:])
            for k in range(n):
                aligned.append((q, t)); q += step; t += 1
        elif c == "*":
            aligned.append((q, t)); q += step; t += 1
        elif c == "+":                                  # bases in query (B73) absent from target
            q += step * (len(tok) - 1)
        elif c == "-":                                  # bases in target only (founder insertion)
            t += len(tok) - 1
    if not aligned:
        return None
    qa = sorted(aligned)
    first_q, first_t = qa[0]
    last_q, last_t = qa[-1]
    if strand == "+":
        return first_t, last_t + 1, first_q, last_q
    # '-': the first query base sits at the highest target position
    return last_t, first_t + 1, first_q, last_q


def run_mm2(mm2, asm_fa, queries_fa, out_paf, threads):
    with open(out_paf, "w") as f:
        subprocess.run([mm2, "-x", "asm10", "-c", "--cs", "-t", str(threads), "--secondary=no",
                        str(asm_fa), str(queries_fa)], stdout=f, stderr=subprocess.DEVNULL, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--snap", required=True)
    ap.add_argument("--nosnap", required=True)
    ap.add_argument("--fasta-dir", required=True)
    ap.add_argument("--fai-dir", required=True)
    ap.add_argument("--minimap2", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    qs = list(csv.DictReader(open(a.queries), delimiter="\t"))
    asms = sorted({q["assembly"] for q in qs} | {"B73"})
    AS = AssemblySet.from_dir(a.fasta_dir, pattern="{asm}_renamed.fa", assemblies=asms, fai_dir=a.fai_dir)
    by_asm = {}
    for q in qs:
        s = AS.fetch(q["ref_contig"], int(q["ref_start"]), int(q["ref_end"]), upper=True)
        by_asm.setdefault(q["assembly"], []).append((q["qid"], s))
    jobs = []
    for asm, items in by_asm.items():
        qfa = work / f"q_{asm}.fa"
        qfa.write_text("".join(f">{qid}\n{s}\n" for qid, s in items))
        paf = work / f"{asm}.paf"
        if not paf.exists() or paf.stat().st_size == 0:
            jobs.append((asm, qfa, paf))
    print(f"{len(qs)} windows over {len(by_asm)} assemblies; running minimap2 on {len(jobs)}", flush=True)

    def go(j):
        asm, qfa, paf = j
        run_mm2(a.minimap2, Path(a.fasta_dir) / f"{asm}_renamed.fa", qfa, paf, a.threads)
        print(f"  {asm} done", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(go, jobs))

    mm = {}
    for asm in by_asm:
        for line in open(work / f"{asm}.paf"):
            f = line.rstrip("\n").split("\t")
            tags = dict(t.split(":", 2)[0::2] for t in f[12:])
            if tags.get("tp") != "P":
                continue
            qid, qlen, qst, qen, strand, tname, ts, te = f[0], int(f[1]), int(f[2]), int(f[3]), f[4], f[5], int(f[7]), int(f[8])
            if qid in mm and mm[qid]["aln"] >= int(f[10]):
                continue
            mm[qid] = dict(qlen=qlen, qs=qst, qe=qen, strand=strand, tname=tname, ts=ts, te=te,
                           aln=int(f[10]), nm=int(f[9]), cs=tags.get("cs", ""))
    snap = {r["qid"]: r for r in csv.DictReader(open(a.snap), delimiter="\t")}
    nosnap = {r["qid"]: r for r in csv.DictReader(open(a.nosnap), delimiter="\t")}
    out = open(work / "per_end.tsv", "w")
    out.write("qid\tassembly\tend\tmm2_status\tmm2_pos\tmm2_identity\tsnap_label\tsnap_status\tsnap_pos\tsnap_diff\t"
              "nosnap_status\tnosnap_pos\tnosnap_diff\n")
    def bound(lo, hi, strand, e):
        """Founder interval bound that corresponds to B73 window end e ('start'/'end')."""
        return (lo if e == "start" else hi) if strand == "+" else (hi if e == "start" else lo)

    for q in qs:
        qid = q["qid"]
        m = mm.get(qid)
        ok_chr = m is not None and chrom_of(m["tname"]) == chrom_of(q["ref_contig"])
        r = ends_from_cs(m["qlen"], m["qs"], m["qe"], m["strand"], m["ts"], m["te"], m["cs"]) if ok_chr else None
        ident = f"{1 - m['nm'] / m['aln']:.4f}" if m and m["aln"] else ""
        for e, k in (("start", 0), ("end", 1)):
            if m is None:
                st, pos = "unaligned", ""
            elif not ok_chr:
                st, pos = "other_chrom", ""
            elif r is None:
                st, pos = "no_aligned_bases", ""
            else:
                reached = (r[2] == 0) if e == "start" else (r[3] == m["qlen"] - 1)
                st, pos = ("ok" if reached else "clipped"), bound(r[0], r[1], m["strand"], e)
            row = [qid, q["assembly"], e, st, pos, ident]
            for tab in (snap, nosnap):
                t = tab.get(qid)
                lab = []
                if tab is snap:
                    lab = [t["snap"].split(",")[k] if t and t.get("snap") else ""]
                if t is None or not t["status"].startswith("ok"):
                    row += lab + [t["status"] if t else "missing", "", ""]
                    continue
                tp = bound(int(t["start"]), int(t["end"]), t["strand"], e)
                if st != "ok":
                    diff = ""
                elif m["tname"] != t["contig"]:
                    diff = "other_contig"
                elif m["strand"] != t["strand"]:
                    diff = "other_strand"
                else:
                    diff = tp - pos
                row += lab + [t["status"], tp, diff]
            out.write("\t".join(map(str, row)) + "\n")
    out.close()
    print("wrote", work / "per_end.tsv")


if __name__ == "__main__":
    main()
