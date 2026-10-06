"""Gene-window eval: cross-check the two extractors on real projected founder segments.

Every status-ok row of a projection table (ropebwt3 extract --project-only) is extracted twice:
`ropebwt3 extract -f` (FM-index, direct coordinates) and python.seq_extract (faidx on renamedFastas);
both uppercased, compared byte for byte.

Usage: python gw_crosscheck.py PROJ.tsv OUTDIR [--threads 3] [--max N --seed S]
"""
import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from python.seq_extract import AssemblySet, read_projection_table  # noqa: E402

RB3 = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/seq_extract_ropebwt/ropebwt3"
FMD = "/workdir/zrm22/HackathonJun2026/ropebwt_refMap/rope_bwt_index_v2/maizeFastaIndex_SampleContig_v2.fmd"
RENAMED = "/workdir/zrm22/HackathonJun2026/ropebwt_refMap/rope_bwt_index_v2/renamedFastas"
FAI_DIR = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/seq_extract_fai"


def read_fasta(path):
    out, name, buf = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if name is not None:
                out[name] = "".join(buf)
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if name is not None:
        out[name] = "".join(buf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proj")
    ap.add_argument("outdir")
    ap.add_argument("--isa", required=True)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_projection_table(a.proj) if r.ok]
    if a.max and len(rows) > a.max:
        rows = random.Random(a.seed).sample(rows, a.max)
    with open(out / "direct_queries.tsv", "w") as f:
        f.write("qid\tcontig\tstart\tend\tstrand\n")
        for r in rows:
            f.write(f"{r.qid}\t{r.contig}\t{r.start}\t{r.end}\t{r.strand}\n")
    t0 = time.time()
    with open(out / "rb3.fa", "w") as fo, open(out / "rb3.log", "w") as fe:
        rc = subprocess.run([RB3, "extract", "-t", str(a.threads), "-I", a.isa, "-f",
                             str(out / "direct_queries.tsv"), FMD], stdout=fo, stderr=fe).returncode
    t_rb3 = time.time() - t0
    rb3 = read_fasta(out / "rb3.fa")
    asm = AssemblySet.from_dir(RENAMED, "{asm}_renamed.fa", fai_dir=FAI_DIR)
    t0 = time.time()
    n_ok = n_bad = n_missing = bp = 0
    by_strand = {"+": [0, 0], "-": [0, 0]}
    with open(out / "mismatches.tsv", "w") as fm:
        fm.write("qid\tcontig\tstart\tend\tstrand\tlen_rb3\tlen_faidx\tfirst_diff\n")
        for r in rows:
            s2 = asm.fetch(r.contig, r.start, r.end, r.strand, upper=True)
            s1 = rb3.get(r.qid)
            if s1 is None:
                n_missing += 1
                fm.write(f"{r.qid}\t{r.contig}\t{r.start}\t{r.end}\t{r.strand}\tNA\t{len(s2)}\tmissing\n")
                continue
            s1 = s1.upper()
            if s1 == s2:
                n_ok += 1
                bp += len(s2)
                by_strand[r.strand][0] += 1
            else:
                n_bad += 1
                by_strand[r.strand][1] += 1
                d = next((k for k, (x, y) in enumerate(zip(s1, s2)) if x != y), min(len(s1), len(s2)))
                fm.write(f"{r.qid}\t{r.contig}\t{r.start}\t{r.end}\t{r.strand}\t{len(s1)}\t{len(s2)}\t{d}\n")
    t_py = time.time() - t0
    summ = (f"segments\t{len(rows)}\nidentical\t{n_ok}\ndiffer\t{n_bad}\nmissing_rb3\t{n_missing}\n"
            f"identical_bp\t{bp}\nplus_identical/differ\t{by_strand['+']}\nminus_identical/differ\t{by_strand['-']}\n"
            f"rb3_exit\t{rc}\nrb3_seconds\t{t_rb3:.1f}\nfaidx_seconds\t{t_py:.1f}\n")
    (out / "summary.tsv").write_text(summ)
    print(summ)


if __name__ == "__main__":
    main()
