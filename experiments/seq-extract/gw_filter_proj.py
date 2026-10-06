"""Gene-window eval: mark implausibly long projections as failures.

`ropebwt3 extract --project-only` projects the two ends of a B73 interval independently and accepts any
projected span; when one end falls back to the wide median fit the span can be 100s of kb to Mb for a
10 kb window. Rows with (projected length - reference length) > --max-extra are rewritten with status
`implausible_length` (contig/start/end kept in an extra column for inspection).

Usage: python gw_filter_proj.py IN.tsv OUT.tsv [--max-extra 50000]
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("out")
ap.add_argument("--max-extra", type=int, default=50000)
a = ap.parse_args()
n = bad = 0
with open(a.inp) as f, open(a.out, "w") as fo:
    hdr = f.readline().rstrip("\n").split("\t")
    fo.write("\t".join(hdr + ["original"]) + "\n")
    for line in f:
        p = line.rstrip("\n").split("\t")
        orig = "."
        if p[9] == "ok":
            n += 1
            if (int(p[7]) - int(p[6])) - (int(p[3]) - int(p[2])) > a.max_extra:
                bad += 1
                orig = f"{p[5]}:{p[6]}-{p[7]}:{p[8]}"
                p[5:10] = [".", ".", ".", ".", "implausible_length"]
        fo.write("\t".join(p + [orig]) + "\n")
print(f"{bad} of {n} ok projections rejected (extra length > {a.max_extra})")
