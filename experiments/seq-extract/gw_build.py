"""Gene-window eval, steps 2-3: projection queries and haplotype stitching.

  queries  windows.tsv + imputed BEDs -> queries.tsv (unique B73 intervals x founder the builders will
           request, plus the whole-window projection into every non-B73 founder for the panel ceiling)
  stitch   projection table (ropebwt3 extract --project-only output) -> per-window imputed haplotype
           sequences (HaplotypeBuilder, gap policy N, uppercase) and the 25 founder window sequences.
           Writes seqs_model.fa / seqs_panel.fa (unique sequences keyed by md5[:16]), model_haps.tsv,
           panel.tsv. The table is read through python.seq_extract's SpanCheckProvider (defaults); the
           number of rows it newly fails is printed (0 when the table came from ropebwt3 >= v3).

Samples: manifest rows of the 8 classes (old *-RIL rows ignored) at the requested depths; BEDs under
scratch/v6cfullfe_homo[_1x]_<scope>_fast/.../bed.
"""
import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from python.seq_extract import (AssemblySet, HaplotypeBuilder, SpanCheckProvider, TableProvider,  # noqa: E402
                                Window, read_bed_dir, write_projection_queries)
from python.seq_extract.haplotype import metadata_row, METADATA_COLUMNS  # noqa: E402
from python.seq_extract.projection import ProjectionQuery  # noqa: E402

MANIFEST = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv"
SCRATCH = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch")
RENAMED = "/workdir/zrm22/HackathonJun2026/ropebwt_refMap/rope_bwt_index_v2/renamedFastas"
FAI_DIR = str(SCRATCH / "seq_extract_fai")
CLASSES = ["IDX-INBRED", "IDX-HYB", "IDX-RIL2", "MIX-HYB", "MIX-RIL2", "OUT-INBRED", "OUT-HYB", "OUT-RIL2"]
DEPTH_TAG = {"0.1": ("v6cfullfe_homo", "0.1x"), "1.0": ("v6cfullfe_homo_1x", "1.0x")}


def samples(depths):
    out = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["dataset_id"] in CLASSES and r["coverage"] in depths:
                out.append((r["coverage"], r["dataset_id"], r["individual"], r["kind"]))
    return sorted(out, key=lambda x: (x[0], CLASSES.index(x[1]), x[2]))


def bed_dir(depth, cls, ind):
    tag, lab = DEPTH_TAG[depth]
    scope = cls.split("-")[0].lower()
    return SCRATCH / f"{tag}_{scope}_fast" / f"{tag}__{cls}__{ind}__{lab}" / "bed"


def read_windows(path):
    with open(path) as f:
        return [Window(r["wid"], r["chrom"], int(r["start"]), int(r["end"])) for r in csv.DictReader(f, delimiter="\t")]


def h16(s):
    return hashlib.md5(s.encode()).hexdigest()[:16]


def founders():
    return sorted(AssemblySet.from_dir(RENAMED, "{asm}_renamed.fa", fai_dir=FAI_DIR).assemblies)


def cmd_queries(a):
    wins = read_windows(a.windows)
    seen, qs = set(), []
    stats = []
    for depth, cls, ind, kind in samples(a.depths):
        bst = {}
        bed = read_bed_dir(bed_dir(depth, cls, ind), ind, "B73", "trim", bst)
        b = HaplotypeBuilder(bed, None, None, "N")
        n0 = len(qs)
        for q in b.projection_queries(wins):
            k = (q.ref_chrom, q.ref_start, q.ref_end, q.assembly)
            if k not in seen:
                seen.add(k)
                qs.append(ProjectionQuery(f"q{len(qs)}", *k))
        stats.append((depth, cls, ind, len(qs) - n0, bst))
    nm = len(qs)
    fs = founders()
    for w in wins:
        for f in fs:
            if f == "B73":
                continue
            k = (w.chrom, w.start, w.end, f)
            if k not in seen:
                seen.add(k)
                qs.append(ProjectionQuery(f"q{len(qs)}", *k))
    write_projection_queries(a.out, qs)
    for s in stats:
        print(*s, sep="\t", file=sys.stderr)
    print(f"{nm} model queries + {len(qs) - nm} panel queries -> {a.out}", file=sys.stderr)


class CountingSpanCheck(SpanCheckProvider):
    """python.seq_extract's span check at its defaults, counting the table rows it newly fails (the C
    tool already applies the same rule, so this should be 0)."""

    def __init__(self, base):
        super().__init__(base)
        self.newly_failed = set()

    def project(self, ref_chrom, ref_start, ref_end, assembly):
        pr = self.base.project(ref_chrom, ref_start, ref_end, assembly)
        out = self.check(pr)
        if pr.ok and not out.ok:
            self.newly_failed.add((ref_chrom, ref_start, ref_end, assembly))
        return out


def cmd_stitch(a):
    wins = read_windows(a.windows)
    t0 = time.time()
    prov = CountingSpanCheck(TableProvider.from_tsv(a.table))
    asm = AssemblySet.from_dir(RENAMED, "{asm}_renamed.fa", fai_dir=FAI_DIR)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seen = set()
    n_seq = 0

    def store(seq, fa):
        nonlocal n_seq
        h = h16(seq)
        if h not in seen:
            seen.add(h)
            fa.write(f">{h}\n{seq}\n")
            n_seq += 1
        return h

    if a.part in ("all", "model"):
        with open(out / "model_haps.tsv", "w") as md, open(out / "seqs_model.fa", "w") as fa:
            md.write("\t".join(["depth", "class", "sample", "kind"] + METADATA_COLUMNS + ["seq_hash", "n_N"]) + "\n")
            for depth, cls, ind, kind in samples(a.depths):
                bed = read_bed_dir(bed_dir(depth, cls, ind), ind, "B73", "trim")
                b = HaplotypeBuilder(bed, asm, prov, "N", upper=True)
                for r1, r2 in b.build_all(wins):
                    for r in (r1, r2):
                        md.write("\t".join(map(str, [depth, cls, ind, kind] + metadata_row(r) +
                                               [store(r.seq, fa), r.seq.count("N")])) + "\n")
                print(f"{depth} {cls} {ind} {time.time() - t0:.0f}s", file=sys.stderr)
    if a.part in ("all", "panel"):
        seen.clear()
        with open(out / "panel.tsv", "w") as pt, open(out / "seqs_panel.fa", "w") as fa:
            pt.write("wid\tchrom\tstart\tend\tfounder\tstatus\tcontig\tpstart\tpend\tstrand\tseq_len\tseq_hash\tn_N"
                     "\tclamped\n")
            for w in wins:
                for f in asm.assemblies:
                    if f == "B73":
                        ok, st, c, s, e, sd, cl = True, "ok", f"B73_{w.chrom}", w.start, w.end, "+", False
                    else:
                        p = prov.project(w.chrom, w.start, w.end, f)
                        ok, st, c, s, e, sd, cl = p.ok, p.status, p.contig, p.start, p.end, p.strand, p.clamped
                    if ok:
                        seq = asm.fetch(c, s, e, sd, upper=True)
                        pt.write(f"{w.wid}\t{w.chrom}\t{w.start}\t{w.end}\t{f}\tok\t{c}\t{s}\t{e}\t{sd}\t{len(seq)}\t"
                                 f"{store(seq, fa)}\t{seq.count('N')}\t{int(cl)}\n")
                    else:
                        pt.write(f"{w.wid}\t{w.chrom}\t{w.start}\t{w.end}\t{f}\t{st}\t.\t.\t.\t.\t0\t.\t0\t0\n")
    print(f"{n_seq} sequences written ({a.part}); span check newly failed {len(prov.newly_failed)} table rows; "
          f"{time.time() - t0:.0f}s", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("queries", "stitch"):
        p = sub.add_parser(name)
        p.add_argument("--windows", required=True)
        p.add_argument("--depths", nargs="+", default=["0.1"])
    sub.choices["queries"].add_argument("--out", required=True)
    sub.choices["stitch"].add_argument("--table", required=True)
    sub.choices["stitch"].add_argument("--out-dir", required=True)
    sub.choices["stitch"].add_argument("--part", choices=["all", "model", "panel"], default="all",
                                       help="model haplotypes, the 25-founder panel windows, or both")
    a = ap.parse_args()
    {"queries": cmd_queries, "stitch": cmd_stitch}[a.cmd](a)


if __name__ == "__main__":
    main()
