#!/usr/bin/env python
"""Cross-check python.seq_extract.snap against ropebwt3 `extract --lift` snapping.

Inputs: the C `--project-only --no-snap` table (raw projections) and the C `--project-only` table (snapped)
for the same queries. The Python SnapProvider is applied to every raw row and compared per end with C:
founder coordinate of the homologous base of the reference start/end, snap label, shift, and the row
status. With --truth (the fixture's ref_queries.tsv) both are also scored against the true intervals.
"""
import argparse
import collections
import time

from python.seq_extract.assembly import AssemblySet
from python.seq_extract.projection import is_usable, read_projection_table
from python.seq_extract.snap import SnapProvider


def ends(pr):
    if not is_usable(pr.status):
        return None
    return (pr.start, pr.end - 1) if pr.strand == "+" else (pr.end - 1, pr.start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nosnap", required=True)
    ap.add_argument("--snap", required=True)
    ap.add_argument("--fasta-dir", required=True)
    ap.add_argument("--pattern", default="{asm}.fa")
    ap.add_argument("--fai-dir", required=True)
    ap.add_argument("--truth")
    ap.add_argument("--out", help="per-end comparison TSV")
    a = ap.parse_args()

    asm = AssemblySet.from_dir(a.fasta_dir, a.pattern, fai_dir=a.fai_dir, build_missing=True)
    sp = SnapProvider(None, asm, asm)
    raw = read_projection_table(a.nosnap)
    csnap = {r.qid: r for r in read_projection_table(a.snap)}
    truth = {}
    if a.truth:
        with open(a.truth) as f:
            hdr = f.readline().rstrip("\n").split("\t")
            for line in f:
                d = dict(zip(hdr, line.rstrip("\n").split("\t")))
                truth[d["qid"]] = (d["true_contig"], int(d["true_start"]), int(d["true_end"]))

    mine, n_ends, t_tot = {}, 0, 0.0
    for r in raw:
        t0 = time.perf_counter()
        mine[r.qid] = sp.snap(r)
        dt = time.perf_counter() - t0
        if is_usable(r.status) and r.assembly != "B73":
            n_ends += 2
            t_tot += dt
    print(f"python snap: {n_ends} ends in {t_tot:.2f}s = {1e3 * t_tot / max(n_ends, 1):.2f} ms/end "
          f"(non-identity rows with a usable raw projection)")

    cnt = collections.Counter()
    labels_c, labels_p = collections.Counter(), collections.Counter()
    out = open(a.out, "w") if a.out else None
    if out:
        out.write("qid\tend\tassembly\tstrand\traw_cp\tc_cp\tpy_cp\tc_label\tpy_label\tc_shift\tpy_shift\tmatch\n")
    for r in raw:
        c, p = csnap[r.qid], mine[r.qid]
        if not is_usable(r.status):
            cnt["raw_failed_rows"] += 1
            cnt["raw_failed_rows_c_snap_" + ("usable" if is_usable(c.status) else "failed")] += 1
            continue
        cnt["rows"] += 1
        same_status = (c.status == p.status) or (c.status.split(":")[0] == p.status.split(":")[0]
                                                  and c.status == p.status)
        cnt["status_match" if same_status else "status_mismatch"] += 1
        if not same_status:
            print(f"STATUS {r.qid}: C={c.status!r} py={p.status!r}")
        ce, pe, re_ = ends(c), ends(p), ends(r)
        for i in (0, 1):
            cl, pl = (c.snap or ("?", "?"))[i], (p.snap or ("?", "?"))[i]
            cs, ps = (c.snap_shift or (None, None))[i], (p.snap_shift or (None, None))[i]
            cc, pc = (ce[i] if ce else None), (pe[i] if pe else None)
            ok = cc == pc and cl == pl and cs == ps
            cnt["ends"] += 1
            cnt["end_match" if ok else "end_mismatch"] += 1
            cnt["coord_match" if cc == pc else "coord_mismatch"] += 1
            cnt["label_match" if cl == pl else "label_mismatch"] += 1
            cnt["shift_match" if cs == ps else "shift_mismatch"] += 1
            labels_c[cl] += 1
            labels_p[pl] += 1
            if out:
                out.write(f"{r.qid}\t{'start' if i == 0 else 'end'}\t{r.assembly}\t{r.strand}\t{re_[i]}\t{cc}\t{pc}\t"
                          f"{cl}\t{pl}\t{cs}\t{ps}\t{int(ok)}\n")
            if not ok:
                print(f"END {r.qid} {'start' if i == 0 else 'end'} {r.assembly} {r.strand}: raw={re_[i]} "
                      f"C={cc} {cl} {cs} | py={pc} {pl} {ps}")
    if out:
        out.close()
    print("counts:", dict(cnt))
    print("C labels:", dict(labels_c))
    print("py labels:", dict(labels_p))
    if truth:
        sc = collections.Counter()
        for r in raw:
            tc, ts, te = truth[r.qid]
            for name, pr in (("raw", r), ("C", csnap[r.qid]), ("py", mine[r.qid])):
                okp = is_usable(pr.status) and pr.contig == tc
                sc[name + "_interval_exact"] += okp and (pr.start, pr.end) == (ts, te)
                sc[name + "_start_exact"] += okp and pr.start == ts
                sc[name + "_end_exact"] += okp and pr.end == te
        n = len(raw)
        for name in ("raw", "C", "py"):
            e = sc[name + "_start_exact"] + sc[name + "_end_exact"]
            print(f"truth {name}: intervals exact {sc[name + '_interval_exact']}/{n}, ends exact {e}/{2 * n}")


if __name__ == "__main__":
    main()
