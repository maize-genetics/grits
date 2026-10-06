"""Gene-window eval: true parent of every window for the RIL2 samples.

A ril2 individual is ONE homozygous mosaic path (both haplotypes identical), drawn in B73 coordinates by
the corpus generator (mosaic.derive_ril2_mosaic(dataset_id, A, B, N_CROSSOVERS_RIL, replicate=0)); the
generator then projected each founder's B73 segments into that founder (AnchorWave anchorspro) and
simulated reads from the extracted regions (scratch/read_datasets/<CLASS>/<S>/{A,B}_<line>.segments.
regions.txt, 1-based inclusive, parent coordinates, plus kept unplaced contigs).

This replays the mosaic (read-only import of the generator; no bytecode written into the shared dir),
checks it against the regions files where that is exact (B73 parent: identity projection), and writes
for every RIL2 sample x window: truth parent line (or BP when a mosaic breakpoint falls inside the window).

Usage: python gw_ril_truth.py windows.tsv OUT.tsv
"""
import csv
import sys

sys.dont_write_bytecode = True
GEN = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/scripts"
sys.path.insert(0, GEN)
import config as cfg  # noqa: E402
import mosaic  # noqa: E402

READS = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/read_datasets"
CLASSES = ["IDX-RIL2", "MIX-RIL2", "OUT-RIL2"]


LINES = {"B73", "B97", "CML103", "Il14H", "Oh43", "A188", "CML459", "EP1", "Ia453", "Tx303"}


def split_pair(ind):
    """'B73xTx303' -> ('B73', 'Tx303'): the split point whose two sides are both known lines."""
    cands = [(ind[:k], ind[k + 1:]) for k, ch in enumerate(ind) if ch == "x"]
    cands = [c for c in cands if c[0] in LINES and c[1] in LINES]
    if len(cands) != 1:
        raise ValueError(f"cannot split '{ind}' into two known lines")
    return cands[0]


def regions(path):
    out = []
    for line in open(path):
        c, rng = line.strip().rsplit(":", 1)
        s, e = rng.split("-")
        out.append((c, int(s) - 1, int(e)))
    return out


def merged(segs):
    segs = sorted(segs)
    out = []
    for c, s, e in segs:
        if out and out[-1][0] == c and s <= out[-1][2]:
            out[-1] = (c, out[-1][1], max(out[-1][2], e))
        else:
            out.append((c, s, e))
    return out


def main():
    wins = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
    rows = []
    with open(cfg.MANIFEST_PATH) as f:
        inds = sorted({(r["dataset_id"], r["individual"]) for r in csv.DictReader(f, delimiter="\t")
                       if r["dataset_id"] in CLASSES})
    with open(sys.argv[2], "w") as fo:
        fo.write("class\tsample\twid\ttruth\tparent_A\tparent_B\tbreakpoints_in_window\n")
        for cls, ind in inds:
            a, b = split_pair(ind)
            m = mosaic.derive_ril2_mosaic(cls, a, b, cfg.N_CROSSOVERS_RIL, 0)
            nbp = sum(len(v) - 1 for v in m.values())
            for lab, line in (("A", a), ("B", b)):
                if line == "B73":
                    reg = [r for r in regions(f"{READS}/{cls}/{ind}/{lab}_{line}.segments.regions.txt")
                           if r[0] in m]
                    exp = merged(mosaic.segments_by_founder(m, lab))
                    print(f"{cls} {ind}: B73 parent {lab}: regions file == mosaic segments: {merged(reg) == exp} "
                          f"({len(exp)} segments)", file=sys.stderr)
            print(f"{cls} {ind}: {nbp} breakpoints", file=sys.stderr)
            for w in wins:
                ws, we = int(w["start"]), int(w["end"])
                hit = [(s, e, f) for s, e, f in m[w["chrom"]] if s < we and e > ws]
                bps = [s for s, e, f in hit if ws < s < we]
                truth = (a if hit[0][2] == "A" else b) if len(hit) == 1 else "BP"
                fo.write(f"{cls}\t{ind}\t{w['wid']}\t{truth}\t{a}\t{b}\t{','.join(map(str, bps)) or '.'}\n")


if __name__ == "__main__":
    main()
