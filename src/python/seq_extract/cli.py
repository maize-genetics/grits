"""python -m python.seq_extract {faidx, extract, haplotypes}"""
from __future__ import annotations

import argparse
import contextlib
import csv
import sys
import time
from pathlib import Path

from .assembly import AssemblySet, fai_path_for
from .fai import FastaFile, FastaIndexError, RegionError, index_fasta
from .haplotype import METADATA_COLUMNS, HaplotypeBuilder, metadata_row, read_bed_dir, read_windows
from .projection import TableProvider, TruthMapProvider, identity_projections, write_projection_queries

PLACEHOLDER_TAG = "PLACEHOLDER_IDENTITY_PROJECTION_NOT_MEANINGFUL"


def write_fasta_record(fh, name, seq, width=0):
    fh.write(f">{name}\n")
    if width and width > 0:
        for k in range(0, len(seq), width):
            fh.write(seq[k:k + width] + "\n")
    elif seq:
        fh.write(seq + "\n")
    else:
        fh.write("\n")


def _open_out(path):
    return contextlib.nullcontext(sys.stdout) if path in (None, "-") else open(path, "w")


# --------------------------------------------------------------------------- faidx

def cmd_faidx(a):
    if a.out and len(a.fasta) != 1:
        sys.exit("--out takes exactly one FASTA; use --fai-dir for several")
    if not a.out and not a.fai_dir and not a.next_to_fasta:
        sys.exit("give --out, --fai-dir, or --next-to-fasta (the index is never written next to the input "
                 "unless asked)")
    for fa in a.fasta:
        dst = Path(a.out) if a.out else fai_path_for(fa, None if a.next_to_fasta else a.fai_dir)
        t0 = time.time()
        index_fasta(fa, dst)
        print(f"{fa} -> {dst} ({time.time() - t0:.1f}s)", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- extract

class _ContigFiles:
    """contig -> FastaFile over an explicit list of FASTAs (resolution by .fai membership)."""

    def __init__(self, fastas, fai_dir, build):
        self.by_contig = {}
        self.files = []
        for fa in fastas:
            fai = fai_path_for(fa, fai_dir)
            if not fai.exists():
                if not build:
                    raise FastaIndexError(f"missing index {fai} (use --build-fai with --fai-dir)")
                index_fasta(fa, fai)
            f = FastaFile(fa, fai)
            self.files.append(f)
            for n in f.names:
                if n in self.by_contig:
                    raise FastaIndexError(f"contig '{n}' appears in more than one FASTA")
                self.by_contig[n] = f

    def fetch(self, contig, start, end, strand="+", upper=False):
        f = self.by_contig.get(contig)
        if f is None:
            raise RegionError(f"unknown contig '{contig}'")
        return f.fetch(contig, start, end, strand, upper)


def _sources(a):
    if a.build_fai and not a.fai_dir:
        sys.exit("--build-fai requires --fai-dir")
    if a.fasta:
        return _ContigFiles(a.fasta, a.fai_dir, a.build_fai)
    if a.fasta_dir:
        return AssemblySet.from_dir(a.fasta_dir, a.pattern, fai_dir=a.fai_dir, build_missing=a.build_fai)
    sys.exit("give --fasta or --fasta-dir")


def read_direct_queries(path):
    out = []
    with open(path, newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        header = next(rd, None)
        if header is None or [h.strip() for h in header[:5]] != ["qid", "contig", "start", "end", "strand"]:
            raise ValueError(f"{path}: header must be 'qid contig start end strand'; got {header}")
        for ln, p in enumerate(rd, 2):
            if not p or not "".join(p).strip():
                continue
            try:
                out.append((p[0], p[1], int(p[2]), int(p[3]), p[4].strip()))
            except (ValueError, IndexError):
                raise ValueError(f"{path}: line {ln}: bad query row {p}") from None
    return out


def cmd_extract(a):
    src = _sources(a)
    qs = read_direct_queries(a.queries)
    n_err = 0
    with _open_out(a.out) as fh:
        for qid, contig, s, e, strand in qs:
            try:
                seq = src.fetch(contig, s, e, strand, upper=a.upper)
            except (RegionError, FastaIndexError) as ex:
                if a.on_error == "fail":
                    sys.exit(f"query {qid}: {ex}")
                print(f"query {qid}: {ex}", file=sys.stderr)
                n_err += 1
                continue
            write_fasta_record(fh, qid, seq, a.line_width)
    if n_err:
        print(f"{n_err} queries failed", file=sys.stderr)
    return 1 if n_err else 0


# --------------------------------------------------------------------------- haplotypes

def cmd_haplotypes(a):
    if a.build_fai and not a.fai_dir:
        sys.exit("--build-fai requires --fai-dir")
    bst = {}
    bed = read_bed_dir(a.bed, a.sample, a.ref_assembly, a.bed_overlap, bst)
    if bst["n_trimmed_intervals"] or bst["n_dropped_intervals"]:
        print(f"BED overlaps resolved ({a.bed_overlap}): {bst}", file=sys.stderr)
    windows = read_windows(a.windows, a.ref_assembly)
    asm = AssemblySet.from_dir(a.fasta_dir, a.pattern, fai_dir=a.fai_dir, build_missing=a.build_fai)
    placeholder = a.provider == "identity-placeholder"
    builder = HaplotypeBuilder(bed, asm, None, a.gap_policy, a.ref_assembly, not a.no_reference_identity,
                               a.upper)
    if a.emit_queries:
        qs = builder.projection_queries(windows)
        write_projection_queries(a.emit_queries, qs, a.ref_assembly)
        print(f"{len(qs)} projection queries -> {a.emit_queries}", file=sys.stderr)
        return 0
    if a.provider == "truth":
        if not a.maps_dir:
            sys.exit("--provider truth needs --maps-dir")
        builder.provider = TruthMapProvider(a.maps_dir, a.ref_assembly)
    elif a.provider == "table":
        if not a.table:
            sys.exit("--provider table needs --table")
        builder.provider = TableProvider.from_tsv(a.table, a.ref_assembly)
    else:
        builder.provider = TableProvider(identity_projections(builder.projection_queries(windows)),
                                         a.ref_assembly)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f" {PLACEHOLDER_TAG}" if placeholder else ""
    t0 = time.time()
    n = 0
    with open(out / f"{a.prefix}hap1.fa", "w") as h1, open(out / f"{a.prefix}hap2.fa", "w") as h2, \
            open(out / f"{a.prefix}metadata.tsv", "w") as md:
        if placeholder:
            md.write(f"# {PLACEHOLDER_TAG}: identity projection, sequences are not real haplotypes\n")
        md.write("\t".join(METADATA_COLUMNS + ["provider", "gap_policy"]) + "\n")
        for r1, r2 in builder.build_all(windows):
            for r, fh in ((r1, h1), (r2, h2)):
                w = r.window
                write_fasta_record(fh, f"{w.wid} hap={r.hap} ref={w.chrom}:{w.start}-{w.end} "
                                       f"switches={r.n_switches} gap_bases={r.gap_bases} "
                                       f"failures={len(r.failures)}{tag}", r.seq, a.line_width)
                md.write("\t".join(map(str, metadata_row(r) + [a.provider, a.gap_policy])) + "\n")
            n += 1
    print(f"{n} windows x 2 haplotypes in {time.time() - t0:.2f}s -> {out}", file=sys.stderr)
    asm.close()
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m python.seq_extract", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("faidx", help="build samtools-compatible .fai index(es)")
    p.add_argument("fasta", nargs="+")
    p.add_argument("--out", help="index path (single FASTA)")
    p.add_argument("--fai-dir", help="write <basename>.fai here")
    p.add_argument("--next-to-fasta", action="store_true", help="write FASTA.fai next to the input")
    p.set_defaults(func=cmd_faidx)

    def add_src(p, dir_only=False):
        if not dir_only:
            p.add_argument("--fasta", nargs="+", help="FASTA files; contigs resolved by their .fai")
        p.add_argument("--fasta-dir", required=dir_only, help="directory of per-assembly FASTAs")
        p.add_argument("--pattern", default="{asm}.fa", help="file name pattern with {asm} (default {asm}.fa)")
        p.add_argument("--fai-dir", help="where .fai files live (default: next to each FASTA)")
        p.add_argument("--build-fai", action="store_true", help="build missing .fai into --fai-dir")
        p.add_argument("--upper", action="store_true", help="uppercase output (default: preserve case)")
        p.add_argument("--line-width", type=int, default=0, help="wrap output FASTA (0 = one line)")

    p = sub.add_parser("extract", help="extract regions listed in a TSV: qid contig start end strand")
    add_src(p)
    p.add_argument("--queries", required=True)
    p.add_argument("--out", default="-")
    p.add_argument("--on-error", choices=["fail", "skip"], default="fail")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("haplotypes", help="stitch imputed haplotype sequences over reference windows")
    add_src(p, dir_only=True)
    p.add_argument("--bed", required=True, help="dir of <sample>_<chr>_imputed.bed files, or one BED file")
    p.add_argument("--sample", help="sample name if --bed holds several")
    p.add_argument("--bed-overlap", choices=["trim", "error"], default="trim",
                   help="overlapping BED intervals: trim the later one (default) or fail")
    p.add_argument("--windows", required=True, help="TSV: wid chrom start end (reference coordinates)")
    p.add_argument("--provider", choices=["truth", "table", "identity-placeholder"], default="table")
    p.add_argument("--maps-dir", help="truth maps (<Asm>_<chr>.npy) for --provider truth")
    p.add_argument("--table", help="projection table TSV for --provider table")
    p.add_argument("--gap-policy", choices=["N", "skip", "extend"], default="N")
    p.add_argument("--ref-assembly", default="B73")
    p.add_argument("--no-reference-identity", action="store_true",
                   help="project reference-assembly segments through the provider too")
    p.add_argument("--emit-queries", help="write the projection queries this run needs and exit")
    p.add_argument("--out-dir", help="output directory")
    p.add_argument("--prefix", default="", help="output file prefix")
    p.set_defaults(func=cmd_haplotypes)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "haplotypes" and not a.emit_queries and not a.out_dir:
        sys.exit("--out-dir is required")
    return a.func(a)
