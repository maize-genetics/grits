"""Gene-window eval: accuracy of ropebwt3 --project-only end points, per lift setting.

Reference: minimap2 (asm20) alignments of each B73 window to founder X (b73_to_<X>.paf). For a window end
covered by an alignment block (+ strand, block starts/ends within 200 bp of the window end), the
implied founder coordinate is extrapolated from that block. Each projection table p_<setting>.tsv (qid
<wid>_<X>) is compared end by end: |projected - implied| <= 10, <= 200, <= 2k, > 2k, or failed.

Usage (in the projtest dir): python gw_projection_accuracy.py ../windows.tsv p_*.tsv
"""
import collections
import sys

FOUNDERS = ["B97", "CML103", "Il14H", "Oh43"]


def main():
    win = {}
    for line in open(sys.argv[1]):
        p = line.split("\t")
        if p[0] != "wid":
            win[f"{p[1]}:{int(p[2]) + 1}-{p[3]}"] = (p[0], p[1])
    blocks = collections.defaultdict(list)
    for L in FOUNDERS:
        for line in open(f"b73_to_{L}.paf"):
            p = line.split("\t")
            w, ch = win[p[0]]
            if p[5] != ch or int(p[11]) < 5:
                continue
            blocks[(w, L)].append((int(p[1]), int(p[2]), int(p[3]), p[4], int(p[7]), int(p[8])))

    def ends(k):
        s = e = None
        for ql, qs, qe, st, ts, te in blocks.get(k, []):
            if st != "+":
                continue
            if qs <= 200 and s is None:
                s = ts - qs
            if ql - qe <= 200 and e is None:
                e = te + (ql - qe)
        return s, e

    for f in sys.argv[2:]:
        c = collections.Counter()
        for line in open(f):
            p = line.rstrip("\n").split("\t")
            if p[0] == "qid":
                continue
            s, e = ends(tuple(p[0].split("_")))
            for ref, val in ((s, p[6]), (e, p[7])):
                if ref is None:
                    continue
                c["n"] += 1
                if p[9] != "ok":
                    c["fail"] += 1
                    continue
                d = abs(int(val) - ref)
                c["<=10" if d <= 10 else "<=200" if d <= 200 else "<=2k" if d <= 2000 else ">2k"] += 1
        n = c.pop("n")
        print(f, n, {k: f"{100 * v / n:.1f}" for k, v in sorted(c.items())})


if __name__ == "__main__":
    main()
