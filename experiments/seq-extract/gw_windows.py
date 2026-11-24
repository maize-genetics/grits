"""Gene-window eval, step 1: pick N protein-coding B73 v5 genes and write windows.tsv.

Window = gene centre +- half (default 5 kb), 0-based half-open B73 coordinates.
Selection: genes on chr1..chr10 with biotype=protein_coding whose window lies >= --end-margin from both
chromosome ends. N is allocated across chromosomes proportionally to their eligible gene counts
(largest remainder); within a chromosome, genes are drawn in a seeded random order and accepted
greedily if their window does not overlap an accepted window (windows are also >= --min-gap apart).

Usage: python gw_windows.py --out windows.tsv [--n 1000 --seed 20261006]
"""
import argparse
import random

GFF = "/workdir/shared_files/maize/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3"
B73_FAI = "/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/B73.fa.fai"
CHROMS = [f"chr{i}" for i in range(1, 11)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--half", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--end-margin", type=int, default=1_000_000)
    ap.add_argument("--min-gap", type=int, default=0)
    a = ap.parse_args()

    lens = {p[0]: int(p[1]) for p in (l.split("\t") for l in open(B73_FAI))}
    genes = {c: [] for c in CHROMS}
    for line in open(GFF):
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 9 or p[2] != "gene" or p[0] not in genes or "biotype=protein_coding" not in p[8]:
            continue
        gid = p[8].split("ID=")[1].split(";")[0]
        s0, e0 = int(p[3]) - 1, int(p[4])
        c = (s0 + e0) // 2
        ws, we = c - a.half, c + a.half
        if ws < a.end_margin or we > lens[p[0]] - a.end_margin:
            continue
        genes[p[0]].append((gid, ws, we, s0, e0, p[6]))

    tot = sum(len(v) for v in genes.values())
    quota = {c: a.n * len(genes[c]) / tot for c in CHROMS}
    alloc = {c: int(quota[c]) for c in CHROMS}
    for c in sorted(CHROMS, key=lambda c: quota[c] - alloc[c], reverse=True)[:a.n - sum(alloc.values())]:
        alloc[c] += 1

    rng = random.Random(a.seed)
    rows = []
    for c in CHROMS:
        cand = sorted(genes[c])
        rng.shuffle(cand)
        acc = []
        for g in cand:
            if len(acc) == alloc[c]:
                break
            if all(g[2] + a.min_gap <= x[1] or x[2] + a.min_gap <= g[1] for x in acc):
                acc.append(g)
        if len(acc) < alloc[c]:
            raise SystemExit(f"{c}: only {len(acc)} windows for quota {alloc[c]}")
        rows += [(c,) + g for g in sorted(acc, key=lambda x: x[1])]
    with open(a.out, "w") as f:
        f.write("wid\tchrom\tstart\tend\tgene_id\tgene_start\tgene_end\tstrand\n")
        for k, (c, gid, ws, we, s0, e0, st) in enumerate(rows):
            f.write(f"w{k:04d}\t{c}\t{ws}\t{we}\t{gid}\t{s0}\t{e0}\t{st}\n")
    print({c: alloc[c] for c in CHROMS}, "eligible genes:", tot)


if __name__ == "__main__":
    main()
