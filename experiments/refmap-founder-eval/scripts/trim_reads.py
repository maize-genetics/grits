#!/usr/bin/env python
"""
Quality-trim FASTQ reads before alignment: a standalone stand-in for what
could eventually be a ropebwt3 refmap preprocessing option (see the
grits_workdir NAM-founder trim sweep, scripts/nam_trim_sweep.py and
results/nam_trim_sweep.md).

Per read: trim leading bases with Phred quality < --min-qual, then trim
trailing bases with quality < --min-qual (contiguous runs from each end only
-- internal low-quality bases are left alone, matching Trimmomatic's
LEADING/TRAILING behavior, not a sliding-window or whole-read filter). If
what remains is shorter than --min-len, the read is dropped entirely.

Usage:
    trim_reads.py --in reads.fastq[.gz] --out trimmed.fastq \
        --min-qual 35 --min-len 75 [--stats stats.tsv] [--phred-offset 33]
"""
import argparse
import gzip
import sys
from pathlib import Path


def _open_read(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def _open_write(path):
    path = str(path)
    return gzip.open(path, "wt") if path.endswith(".gz") else open(path, "w")


def iter_fastq(fh):
    """Yield (header, seq, plus, qual) 4-line records, stripped of newlines."""
    while True:
        header = fh.readline()
        if not header:
            return
        seq = fh.readline()
        plus = fh.readline()
        qual = fh.readline()
        if not qual:
            raise ValueError("truncated FASTQ record (EOF mid-record)")
        yield header.rstrip("\n"), seq.rstrip("\n"), plus.rstrip("\n"), qual.rstrip("\n")


def trim_read(seq, qual, min_qual, offset):
    """Trim contiguous below-threshold bases off each end only; leave
    internal bases untouched regardless of quality."""
    n = len(qual)
    start = 0
    while start < n and (ord(qual[start]) - offset) < min_qual:
        start += 1
    end = n
    while end > start and (ord(qual[end - 1]) - offset) < min_qual:
        end -= 1
    return seq[start:end], qual[start:end]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="input FASTQ, plain or .gz")
    ap.add_argument("--out", dest="out", required=True, help="output FASTQ (plain, or .gz if named *.gz)")
    ap.add_argument("--min-qual", type=int, required=True, help="Phred quality threshold for end-trimming")
    ap.add_argument("--min-len", type=int, required=True, help="drop the read if trimmed length < this")
    ap.add_argument("--phred-offset", type=int, default=33)
    ap.add_argument("--stats", default=None, help="optional TSV sidecar with summary counts")
    args = ap.parse_args()

    reads_in = reads_kept = reads_dropped = 0
    len_sum_out = 0

    with _open_read(args.inp) as fin, _open_write(args.out) as fout:
        for header, seq, plus, qual in iter_fastq(fin):
            reads_in += 1
            if len(seq) != len(qual):
                raise ValueError(f"seq/qual length mismatch at record {reads_in} ({header!r})")
            t_seq, t_qual = trim_read(seq, qual, args.min_qual, args.phred_offset)
            if len(t_seq) < args.min_len:
                reads_dropped += 1
                continue
            reads_kept += 1
            len_sum_out += len(t_seq)
            fout.write(f"{header}\n{t_seq}\n{plus}\n{t_qual}\n")

    mean_len_out = (len_sum_out / reads_kept) if reads_kept else 0.0
    pct_kept = (reads_kept / reads_in * 100) if reads_in else 0.0
    print(f"reads_in={reads_in} reads_kept={reads_kept} reads_dropped={reads_dropped} "
          f"pct_kept={pct_kept:.2f} mean_len_out={mean_len_out:.2f}", file=sys.stderr)

    if args.stats:
        stats_path = Path(args.stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            "reads_in\treads_kept\treads_dropped\tpct_kept\tmean_len_out\n"
            f"{reads_in}\t{reads_kept}\t{reads_dropped}\t{pct_kept:.4f}\t{mean_len_out:.4f}\n"
        )

    if reads_kept == 0:
        print("WARNING: 0 reads survived trimming -- output FASTQ is empty", file=sys.stderr)


if __name__ == "__main__":
    main()
