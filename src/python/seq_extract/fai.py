"""samtools-compatible .fai index building/reading and random-access FASTA extraction.

Conventions (shared with `ropebwt3 extract`):
  * coordinates are 0-based, half-open [start, end) on the forward strand of the named sequence;
  * strand '-' returns the reverse complement of that forward interval;
  * case is preserved by default (soft-masking survives, like `samtools faidx`); pass upper=True
    to uppercase (what an FM-index based extractor returns);
  * reverse complement follows IUPAC: A<->T, C<->G, R<->Y, K<->M, B<->V, D<->H, S, W, N map to
    themselves; case is preserved; any other byte (e.g. '-', '*', '.') is left unchanged.

Index format: one line per sequence, `name length offset linebases linewidth` (tab-separated),
byte-identical to `samtools faidx` for valid FASTA, including CRLF files (linewidth = linebases+2).
Like samtools, zero-length sequences are not indexed. Unlike samtools, duplicate names, malformed
line lengths, internal whitespace and blank lines inside a sequence are hard errors.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_COMP_SRC = b"ACGTRYKMBVDHSWNUacgtrykmbvdhswnu"
_COMP_DST = b"TGCAYRMKVBHDSWNAtgcayrmkvbhdswna"
_COMP = bytes.maketrans(_COMP_SRC, _COMP_DST)
_UPPER = bytes.maketrans(b"abcdefghijklmnopqrstuvwxyz", b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class FastaIndexError(ValueError):
    """Malformed FASTA / .fai, or a mismatch between them."""


class RegionError(ValueError):
    """Unknown sequence name, or an invalid / out-of-range interval."""


def revcomp(seq):
    """Reverse complement of a str or bytes (IUPAC aware, case preserving)."""
    if isinstance(seq, str):
        return seq.encode("ascii").translate(_COMP)[::-1].decode("ascii")
    return bytes(seq).translate(_COMP)[::-1]


@dataclass(frozen=True)
class FaiEntry:
    name: str
    length: int
    offset: int
    linebases: int
    linewidth: int

    def line(self) -> str:
        return f"{self.name}\t{self.length}\t{self.offset}\t{self.linebases}\t{self.linewidth}\n"

    def byte_span(self, start: int, end: int) -> tuple[int, int]:
        """File byte range [b0, b1) holding bases [start, end) (end > start)."""
        lb, lw = self.linebases, self.linewidth
        b0 = self.offset + (start // lb) * lw + start % lb
        last = end - 1
        b1 = self.offset + (last // lb) * lw + last % lb + 1
        return b0, b1


# --------------------------------------------------------------------------- building

class _SeqState:
    __slots__ = ("name", "hdr_line", "offset", "lb", "tw", "length", "phase")

    def __init__(self, name, hdr_line):
        self.name = name
        self.hdr_line = hdr_line
        self.offset = None
        self.lb = None
        self.tw = None
        self.length = 0
        self.phase = 0          # 0: full lines; 1: after the short last line; 2: after a blank line


def _err(path, line_no, msg):
    return FastaIndexError(f"{path}: line {line_no}: {msg}")


def _process_run(path, st: _SeqState, starts, clen, term, line_no0):
    """Sequence lines of one record inside one chunk. clen = bases per line, term = terminator bytes."""
    n = len(clen)
    if n == 0:
        return
    i = 0
    if st.phase == 0 and st.lb is None:
        if clen[0] == 0:
            st.phase = 2
        else:
            st.lb = int(clen[0])
            st.offset = int(starts[0])
            st.tw = int(term[0])
    if st.phase == 0:
        neq = np.flatnonzero(clen != st.lb)
        j = int(neq[0]) if len(neq) else n
        if j > 0:
            bad_t = np.flatnonzero(term[:j] != st.tw)
            if len(bad_t):
                k = int(bad_t[0])
                if not (term[k] == 0 and k == n - 1):     # unterminated last line of the file
                    raise _err(path, line_no0 + k, f"inconsistent line terminator in sequence '{st.name}'")
        st.length += j * st.lb
        if j < n:
            L = int(clen[j])
            if L > st.lb:
                raise _err(path, line_no0 + j, f"different line length in sequence '{st.name}'")
            if L == 0:
                st.phase = 2
            else:
                if term[j] != st.tw and not (term[j] == 0 and j == n - 1):
                    raise _err(path, line_no0 + j, f"inconsistent line terminator in sequence '{st.name}'")
                st.length += L
                st.phase = 1
            i = j + 1
        else:
            i = n
    if i < n:
        nz = np.flatnonzero(clen[i:] != 0)
        if len(nz):
            k = i + int(nz[0])
            what = "sequence line after a blank line" if st.phase == 2 else "different line length"
            raise _err(path, line_no0 + k, f"{what} in sequence '{st.name}'")
        st.phase = 2


def build_fai(fasta_path, chunk_bytes: int = 1 << 26) -> list[FaiEntry]:
    """Scan a plain (uncompressed) FASTA and return its index entries."""
    path = str(fasta_path)
    entries: list[FaiEntry] = []
    names: set[str] = set()
    fh = open(path, "rb")
    try:
        if fh.read(2) == b"\x1f\x8b":
            raise FastaIndexError(f"{path}: gzip-compressed FASTA is not supported")
        fh.seek(0)
        st: _SeqState | None = None
        line_no = 1
        pos = 0
        carry = b""

        def finish(s):
            if s is None or s.length == 0:
                return
            if s.name in names:
                raise _err(path, s.hdr_line, f"duplicate sequence name '{s.name}'")
            names.add(s.name)
            entries.append(FaiEntry(s.name, s.length, s.offset, s.lb, s.lb + (s.tw or 1)))

        eof = False
        while not eof:
            block = fh.read(chunk_bytes)
            eof = len(block) == 0
            data = carry + block
            if not eof:
                k = data.rfind(b"\n")
                if k < 0:
                    carry = data
                    continue
                carry = data[k + 1:]
                data = data[:k + 1]
            else:
                carry = b""
                if not data:
                    break
            ce = pos + len(data)
            arr = np.frombuffer(data, dtype=np.uint8)
            nl = np.flatnonzero(arr == 10)
            ends = nl
            if len(nl) == 0 or nl[-1] != len(arr) - 1:
                ends = np.r_[nl, len(arr)]
            starts = np.r_[0, nl[:len(ends) - 1] + 1] if len(ends) else np.zeros(0, np.int64)
            raw = ends - starts
            has_nl = np.ones(len(ends), dtype=bool)
            if len(ends) and (len(nl) == 0 or nl[-1] != len(arr) - 1):
                has_nl[-1] = False
            cr = np.zeros(len(ends), dtype=bool)
            nonempty = raw > 0
            cr[nonempty] = arr[ends[nonempty] - 1] == 13
            clen = raw - cr
            term = has_nl.astype(np.int64) + cr
            first = np.zeros(len(ends), dtype=np.uint8)
            first[clen > 0] = arr[starts[clen > 0]]
            is_hdr = first == ord(">")
            hdr_idx = np.flatnonzero(is_hdr)

            n_cr = int(np.count_nonzero(arr == 13))
            if n_cr != int(cr.sum()):
                raise FastaIndexError(f"{path}: stray carriage return inside a line (chunk at byte {pos})")
            n_ws = int(np.count_nonzero((arr == 32) | (arr == 9) | (arr == 11) | (arr == 12)))
            if n_ws:
                hdr_ws = sum(int(np.count_nonzero((arr[starts[h]:ends[h]] == 32) | (arr[starts[h]:ends[h]] == 9)
                                                  | (arr[starts[h]:ends[h]] == 11) | (arr[starts[h]:ends[h]] == 12)))
                             for h in hdr_idx)
                if hdr_ws != n_ws:
                    raise FastaIndexError(f"{path}: whitespace inside a sequence line (chunk at byte {pos})")

            bounds = list(hdr_idx) + [len(ends)]
            prev = 0
            for b in bounds:
                if b > prev:
                    if st is None:
                        if np.any(clen[prev:b] > 0):
                            k = prev + int(np.flatnonzero(clen[prev:b] > 0)[0])
                            raise _err(path, line_no + k, "sequence data before the first '>' header")
                    else:
                        _process_run(path, st, starts[prev:b] + pos, clen[prev:b], term[prev:b], line_no + prev)
                if b < len(ends):
                    finish(st)
                    hdr = bytes(arr[starts[b] + 1:starts[b] + clen[b]])
                    name = hdr.split(None, 1)[0].decode("ascii", "replace") if hdr.strip() else ""
                    if not name:
                        raise _err(path, line_no + b, "empty sequence name")
                    st = _SeqState(name, line_no + b)
                    st.offset = pos + int(starts[b]) + int(raw[b]) + 1
                prev = b + 1
            line_no += len(ends)
            pos = ce
        finish(st)
    finally:
        fh.close()
    return entries


def write_fai(entries: Iterable[FaiEntry], fai_path) -> Path:
    fai_path = Path(fai_path)
    fai_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = fai_path.with_name(fai_path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        for e in entries:
            f.write(e.line())
    os.replace(tmp, fai_path)
    return fai_path


def index_fasta(fasta_path, fai_path) -> Path:
    """Build an index for fasta_path and write it to fai_path (an explicit location)."""
    return write_fai(build_fai(fasta_path), fai_path)


def read_fai(fai_path) -> list[FaiEntry]:
    out = []
    seen = set()
    with open(fai_path) as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            p = line.rstrip("\r\n").split("\t")
            if len(p) < 5:
                raise FastaIndexError(f"{fai_path}: line {ln}: expected 5 columns, got {len(p)}")
            try:
                e = FaiEntry(p[0], int(p[1]), int(p[2]), int(p[3]), int(p[4]))
            except ValueError as ex:
                raise FastaIndexError(f"{fai_path}: line {ln}: {ex}") from None
            if e.length < 0 or e.offset < 0 or e.linebases <= 0 or e.linewidth < e.linebases:
                raise FastaIndexError(f"{fai_path}: line {ln}: invalid entry {line.strip()!r}")
            if e.name in seen:
                raise FastaIndexError(f"{fai_path}: duplicate name '{e.name}'")
            seen.add(e.name)
            out.append(e)
    return out


# --------------------------------------------------------------------------- extraction

@dataclass(frozen=True)
class Region:
    contig: str
    start: int
    end: int
    strand: str = "+"
    qid: str | None = None


class FastaFile:
    """Random access into one plain FASTA through its .fai. Opens the file once (os.pread)."""

    def __init__(self, fasta_path, fai_path=None):
        self.path = Path(fasta_path)
        if fai_path is None:
            fai_path = Path(str(self.path) + ".fai")
            if not fai_path.exists():
                raise FastaIndexError(f"no index for {self.path}: build one with index_fasta(fasta, fai_path) "
                                      f"and pass fai_path")
        self.fai_path = Path(fai_path)
        self.entries = {e.name: e for e in read_fai(self.fai_path)}
        self._fd = os.open(self.path, os.O_RDONLY)
        fsize = os.fstat(self._fd).st_size
        for e in self.entries.values():
            if e.length and e.byte_span(e.length - 1, e.length)[1] > fsize:
                self.close()
                raise FastaIndexError(f"{self.fai_path} does not match {self.path}: '{e.name}' ends past EOF")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        if getattr(self, "_fd", None) is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __contains__(self, name):
        return name in self.entries

    @property
    def names(self) -> list[str]:
        return list(self.entries)

    def length(self, name: str) -> int:
        return self._entry(name).length

    def _entry(self, name):
        try:
            return self.entries[name]
        except KeyError:
            raise RegionError(f"unknown sequence '{name}' in {self.path}") from None

    def fetch_bytes(self, name: str, start: int, end: int, strand: str = "+", upper: bool = False) -> bytes:
        e = self._entry(name)
        if not (isinstance(start, (int, np.integer)) and isinstance(end, (int, np.integer))):
            raise RegionError(f"non-integer coordinates {start!r}, {end!r}")
        start, end = int(start), int(end)
        if start < 0 or end > e.length:
            raise RegionError(f"{name}:[{start},{end}) out of range (length {e.length})")
        if end <= start:
            raise RegionError(f"{name}:[{start},{end}) is empty or reversed")
        if strand not in ("+", "-"):
            raise RegionError(f"strand must be '+' or '-', got {strand!r}")
        b0, b1 = e.byte_span(start, end)
        if self._fd is None:
            raise ValueError("FastaFile is closed")
        buf = os.pread(self._fd, b1 - b0, b0)
        seq = buf.translate(None, b"\r\n")
        if len(seq) != end - start:
            raise FastaIndexError(f"{self.path}: index/file mismatch at {name}:[{start},{end}) "
                                  f"(got {len(seq)} bases)")
        if upper:
            seq = seq.translate(_UPPER)
        if strand == "-":
            seq = seq.translate(_COMP)[::-1]
        return seq

    def fetch(self, name: str, start: int, end: int, strand: str = "+", upper: bool = False) -> str:
        return self.fetch_bytes(name, start, end, strand, upper).decode("ascii")

    def fetch_many(self, regions: Sequence, upper: bool = False) -> list[str]:
        """regions: Region objects or (contig, start, end[, strand]) tuples."""
        out = []
        for r in regions:
            if isinstance(r, Region):
                out.append(self.fetch(r.contig, r.start, r.end, r.strand, upper))
            else:
                out.append(self.fetch(*r, upper=upper))
        return out
