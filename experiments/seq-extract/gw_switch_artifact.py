"""Gene-window eval: how much error does a founder switch itself introduce (projection offsets)?

For an in-panel line X and every window [ws, we), split at the midpoint m and stitch
X[proj(ws, m)] + X[proj(m, we)] -- a "switch" from X to X, exactly as HaplotypeBuilder stitches two
founder pieces. Aligned back to X's own assembly, any error is a projection/stitching artifact at m
(duplicated or skipped bases where the two independent end projections disagree). The unsplit window
X[proj(ws, we)] is scored too (should be exact: it is a substring of X).

Usage: python gw_switch_artifact.py --windows windows.tsv --out-dir DIR [--lines B97 CML103 Il14H Oh43]
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from python.seq_extract import AssemblySet, TableProvider  # noqa: E402
import gw_align as G  # noqa: E402

RB3 = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/seq_extract_ropebwt/ropebwt3_v3"
IDX = "/workdir/zrm22/HackathonJun2026/ropebwt_refMap/rope_bwt_index_v2"
FMD = f"{IDX}/maizeFastaIndex_SampleContig_v2.fmd"
LIFT = f"{IDX}/maizeFastaIndex_SampleContig_v2_s200.lift"
ISA = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/gene_window_eval/maize_v2_b12.isa"
RENAMED = f"{IDX}/renamedFastas"
FAI_DIR = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/seq_extract_fai"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lines", nargs="+", default=["B97", "CML103", "Il14H", "Oh43"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--rb3", default=RB3)
    ap.add_argument("--rb3-args", default="", help="extra ropebwt3 extract options, e.g. --no-snap")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wins = list(csv.DictReader(open(a.windows), delimiter="\t"))
    q = out / "queries.tsv"
    with open(q, "w") as f:
        f.write("qid\tref_contig\tref_start\tref_end\tassembly\n")
        for L in a.lines:
            for w in wins:
                s, e = int(w["start"]), int(w["end"])
                m = (s + e) // 2
                for tag, x, y in (("full", s, e), ("left", s, m), ("right", m, e)):
                    f.write(f"{w['wid']}|{L}|{tag}\tB73_{w['chrom']}\t{x}\t{y}\t{L}\n")
    with open(out / "proj.tsv", "w") as fo:
        subprocess.run([a.rb3, "extract", f"--lift={LIFT}", "--ref-prefix=B73", "--project-only", "-I", ISA]
                       + a.rb3_args.split() + ["-f", str(q), FMD], stdout=fo, stderr=subprocess.DEVNULL, check=True)
    prov = TableProvider.from_tsv(out / "proj.tsv")
    asm = AssemblySet.from_dir(RENAMED, "{asm}_renamed.fa", fai_dir=FAI_DIR)
    rows = []
    for L in a.lines:
        seqs, meta = {}, {}
        for w in wins:
            s, e = int(w["start"]), int(w["end"])
            m = (s + e) // 2
            pf, pl, pr = (prov.project(w["chrom"], x, y, L) for x, y in ((s, e), (s, m), (m, e)))
            info = dict(wid=w["wid"], line=L, full_ok=pf.ok, split_ok=pl.ok and pr.ok)
            if pf.ok:
                seqs[f"{w['wid']}|full"] = asm.fetch(pf.contig, pf.start, pf.end, pf.strand, upper=True)
            if pl.ok and pr.ok:
                seqs[f"{w['wid']}|split"] = (asm.fetch(pl.contig, pl.start, pl.end, pl.strand, upper=True) +
                                             asm.fetch(pr.contig, pr.start, pr.end, pr.strand, upper=True))
                same = pl.contig == pr.contig and pl.strand == pr.strand
                info["junction_offset"] = (pr.start - pl.end) if same and pl.strand == "+" else \
                    ((pl.start - pr.end) if same else "NA")
                if pf.ok and same:
                    info["split_vs_full_len"] = len(seqs[f"{w['wid']}|split"]) - len(seqs[f"{w['wid']}|full"])
            meta[w["wid"]] = info
        fa = out / f"q_{L}.fa"
        with open(fa, "w") as f:
            for k, v in seqs.items():
                f.write(f">{k}\n{v}\n")
        paf = out / f"{L}.paf"
        with open(paf, "w") as fo:
            subprocess.run([G.MM2, "-x", "asm20", "-c", "--cs", "--secondary=no", "-t", str(a.threads),
                            G.MMI.format(line=L), str(fa)], stdout=fo, stderr=subprocess.DEVNULL, check=True)
        by = G.parse_paf(paf)
        for w in wins:
            info = meta[w["wid"]]
            for tag in ("full", "split"):
                k = f"{w['wid']}|{tag}"
                if k not in seqs:
                    continue
                r = G.score_query(len(seqs[k]), G.nprefix_of(seqs[k]), by.get(k, []))
                r["errors"] = r["mismatches"] + r["ins"] + r["dels"] + r["unaligned"]
                info[f"{tag}_errors"] = r["errors"]
                info[f"{tag}_ins"] = r["ins"]
                info[f"{tag}_dels"] = r["dels"]
                info[f"{tag}_mism"] = r["mismatches"]
                info[f"{tag}_unaln"] = r["unaligned"]
                info[f"{tag}_chrom_ok"] = r["contig"] == w["chrom"]
            rows.append(info)
    cols = ["wid", "line", "full_ok", "split_ok", "junction_offset", "split_vs_full_len", "full_errors",
            "full_chrom_ok", "split_errors", "split_ins", "split_dels", "split_mism", "split_unaln", "split_chrom_ok"]
    with open(out / "switch_artifact.tsv", "w") as fo:
        fo.write("\t".join(cols) + "\n")
        for r in rows:
            fo.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


if __name__ == "__main__":
    main()
