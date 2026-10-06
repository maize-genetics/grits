"""Gene-window eval, step 0: verify the truth assemblies.

(a) in-panel truth FASTAs (index_asms/maize_v2/<Line>.fa, contigs chrN) carry the same sequence as the
    index's renamedFastas (<Line>_renamed.fa, contigs <Line>_chrN): md5 of every chr1..chr10 (uppercased).
(b) the simulated reads came from these FASTAs: for N reads per line from the read-simulation master
    FASTQs (wgsim names <contig>_<start1>_<end1>_...), the read (or its reverse complement) must match
    the assembly at the named fragment's start or end with <= 3 mismatches (wgsim -e 0.001).

Usage: python gw_verify_truth.py OUT.tsv
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from python.seq_extract import FastaFile, revcomp  # noqa: E402

IDX_ASM = Path("/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2")
OUT_ASM = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_v2_heldout/asm")
RENAMED = Path("/workdir/zrm22/HackathonJun2026/ropebwt_refMap/rope_bwt_index_v2/renamedFastas")
FAI = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/seq_extract_fai")
READS = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/read_datasets")
IDX_LINES = ["B73", "B97", "CML103", "Il14H", "Oh43"]
OUT_LINES = ["A188", "CML459", "EP1", "Ia453", "Tx303"]
CHROMS = [f"chr{i}" for i in range(1, 11)]


def md5_contig(ff, name):
    h = hashlib.md5()
    L = ff.length(name)
    step = 50_000_000
    for s in range(0, L, step):
        h.update(ff.fetch(name, s, min(L, s + step), upper=True).encode())
    return h.hexdigest(), L


def read_master(line):
    for cls in ("IDX-INBRED", "OUT-INBRED"):
        p = READS / cls / line / f"{line}.master.R1.fastq"
        if p.exists():
            return p
    return None


def check_reads(line, ff, n=2000):
    p = read_master(line)
    ok = bad = 0
    with open(p) as f:
        while ok + bad < n:
            h = f.readline()
            if not h:
                break
            seq = f.readline().strip()
            f.readline(); f.readline()
            name = h[1:].split("/")[0]
            parts = name.rsplit("_", 6)
            contig, s1, e1 = parts[0], int(parts[1]), int(parts[2])
            L = len(seq)
            best = L
            for a in (s1 - 1, e1 - L):
                try:
                    ref = ff.fetch(contig, a, a + L, upper=True)
                except Exception:
                    continue
                for q in (seq, revcomp(seq)):
                    best = min(best, sum(1 for x, y in zip(q, ref) if x != y))
            if best <= 3:
                ok += 1
            else:
                bad += 1
    return str(p), ok, bad


def main():
    out = open(sys.argv[1], "w")
    out.write("check\tline\tcontig\tvalue_truth\tvalue_index\tlength\tmatch\n")
    for line in IDX_LINES:
        t = FastaFile(IDX_ASM / f"{line}.fa", IDX_ASM / f"{line}.fa.fai")
        r = FastaFile(RENAMED / f"{line}_renamed.fa", FAI / f"{line}_renamed.fa.fai")
        for c in CHROMS:
            m1, l1 = md5_contig(t, c)
            m2, l2 = md5_contig(r, f"{line}_{c}")
            out.write(f"md5\t{line}\t{c}\t{m1}\t{m2}\t{l1}/{l2}\t{m1 == m2}\n")
            out.flush()
        p, ok, bad = check_reads(line, t)
        out.write(f"reads\t{line}\t{p}\t{ok}\t{bad}\t.\t{bad == 0}\n")
    for line in OUT_LINES:
        t = FastaFile(OUT_ASM / f"{line}.fa", OUT_ASM / f"{line}.fa.fai")
        p, ok, bad = check_reads(line, t)
        out.write(f"reads\t{line}\t{p}\t{ok}\t{bad}\t.\t{bad == 0}\n")
        out.flush()
    out.close()


if __name__ == "__main__":
    main()
