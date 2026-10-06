"""Reference interval -> assembly interval projection providers.

All providers answer `project(ref_chrom, ref_start, ref_end, assembly) -> Projection` and never raise
for an unprojectable interval: they return a Projection whose `status` is not "ok".

Reference chromosomes are handled in bare form (`chr1`); `B73_chr1`-style names are accepted and the
reference-assembly prefix stripped.

Projection table TSV (shared with `ropebwt3 extract --project-only`), header required:
    qid  ref_contig  ref_start  ref_end  assembly  contig  start  end  strand  status
optionally followed by `method  snap  snap_shift` (ropebwt3 a4df980+: start/end projector, start,end snap
labels, start,end shifts in bp); further columns are ignored.
status "ok" means [start, end) on `contig`/`strand` is the projection. "ok_clamped" (ropebwt3: the projected
end ran past the contig end and was clamped to it) is usable as well, but flagged (Projection.clamped):
the interval is real founder sequence, it just stops at the contig end, so the stitched haplotype may lack
the bases beyond it. Anything else (deleted, unmapped, "projection_failed: ...") is a failure, and
contig/start/end/strand may then be '.'.
A projection *query* TSV (what a projecting tool needs) is the first five columns.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

OK = "ok"
OK_CLAMPED = "ok_clamped"
USABLE_STATUSES = frozenset((OK, OK_CLAMPED))
EXTRA_COLUMNS = ["method", "snap", "snap_shift"]
TABLE_COLUMNS = ["qid", "ref_contig", "ref_start", "ref_end", "assembly", "contig", "start", "end", "strand",
                 "status"]
QUERY_COLUMNS = TABLE_COLUMNS[:5]


@dataclass(frozen=True)
class Projection:
    ref_chrom: str
    ref_start: int
    ref_end: int
    assembly: str
    contig: str | None = None
    start: int | None = None
    end: int | None = None
    strand: str = "+"
    status: str = OK
    qid: str | None = None
    method: str | None = None
    snap: tuple[str, str] | None = None
    snap_shift: tuple[int, int] | None = None

    @property
    def ok(self) -> bool:
        """Usable: status ok or ok_clamped."""
        return self.status in USABLE_STATUSES

    @property
    def clamped(self) -> bool:
        return self.status == OK_CLAMPED


def is_usable(status: str) -> bool:
    return status in USABLE_STATUSES


@dataclass(frozen=True)
class ProjectionQuery:
    qid: str
    ref_chrom: str
    ref_start: int
    ref_end: int
    assembly: str


class ProjectionProvider(Protocol):
    def project(self, ref_chrom: str, ref_start: int, ref_end: int, assembly: str) -> Projection: ...


def bare_chrom(chrom: str, ref_assembly: str = "B73") -> str:
    p = ref_assembly + "_"
    return chrom[len(p):] if chrom.startswith(p) and len(chrom) > len(p) else chrom


class TruthMapProvider:
    """Simulator maps: maps_dir/<Asm>_<chr>.npy, int64 [ref_len + 1], -1 = reference base deleted in Asm.

    Projection of [s, e) = [map[first present], map[last present] + 1), on '+'.
    """

    def __init__(self, maps_dir, ref_assembly: str = "B73"):
        self.maps_dir = Path(maps_dir)
        self.ref_assembly = ref_assembly
        self._maps: dict[str, np.ndarray | None] = {}

    def _map(self, contig):
        if contig not in self._maps:
            p = self.maps_dir / f"{contig}.npy"
            self._maps[contig] = np.load(p, mmap_mode="r") if p.exists() else None
        return self._maps[contig]

    def project(self, ref_chrom, ref_start, ref_end, assembly) -> Projection:
        chrom = bare_chrom(ref_chrom, self.ref_assembly)
        base = dict(ref_chrom=chrom, ref_start=int(ref_start), ref_end=int(ref_end), assembly=assembly)
        contig = f"{assembly}_{chrom}"
        mp = self._map(contig)
        if mp is None:
            return Projection(**base, status="no_map")
        ref_len = len(mp) - 1
        if ref_start < 0 or ref_end > ref_len or ref_end <= ref_start:
            return Projection(**base, status="out_of_range")
        present = np.flatnonzero(np.asarray(mp[ref_start:ref_end]) >= 0)
        if len(present) == 0:
            return Projection(**base, contig=contig, status="deleted")
        return Projection(**base, contig=contig, start=int(mp[ref_start + present[0]]),
                          end=int(mp[ref_start + present[-1]]) + 1, strand="+", status=OK)


class TableProvider:
    """Exact-match lookup of (ref_chrom, ref_start, ref_end, assembly) in a projection table."""

    def __init__(self, rows: Iterable[Projection], ref_assembly: str = "B73"):
        self.ref_assembly = ref_assembly
        self._rows: dict[tuple, Projection] = {}
        for r in rows:
            key = (bare_chrom(r.ref_chrom, ref_assembly), r.ref_start, r.ref_end, r.assembly)
            old = self._rows.get(key)
            if old is not None and (old.contig, old.start, old.end, old.strand, old.status) != \
                    (r.contig, r.start, r.end, r.strand, r.status):
                raise ValueError(f"conflicting projection rows for {key}: {old} vs {r}")
            self._rows[key] = r

    @classmethod
    def from_tsv(cls, path, ref_assembly: str = "B73") -> "TableProvider":
        return cls(read_projection_table(path), ref_assembly)

    def __len__(self):
        return len(self._rows)

    def project(self, ref_chrom, ref_start, ref_end, assembly) -> Projection:
        chrom = bare_chrom(ref_chrom, self.ref_assembly)
        r = self._rows.get((chrom, int(ref_start), int(ref_end), assembly))
        if r is None:
            return Projection(chrom, int(ref_start), int(ref_end), assembly, status="not_in_table")
        return r


def _int_or_none(x):
    return None if x in (".", "", "NA") else int(x)


def _pair(x, conv):
    if x in (None, "", "."):
        return None
    p = x.split(",")
    if len(p) != 2:
        raise ValueError(f"expected 'a,b', got {x!r}")
    return conv(p[0]), conv(p[1])


def read_projection_table(path) -> list[Projection]:
    rows = []
    with open(path, newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        header = next(rd, None)
        if header is None or [h.strip() for h in header[:len(TABLE_COLUMNS)]] != TABLE_COLUMNS:
            raise ValueError(f"{path}: header must start with {' '.join(TABLE_COLUMNS)}; got {header}")
        extra = {h.strip(): k for k, h in enumerate(header) if h.strip() in EXTRA_COLUMNS}
        for ln, p in enumerate(rd, 2):
            if not p or not "".join(p).strip():
                continue
            if len(p) < len(TABLE_COLUMNS):
                raise ValueError(f"{path}: line {ln}: expected {len(TABLE_COLUMNS)} columns")
            qid, rc, rs, re_, asm, contig, s, e, strand, status = p[:10]
            status = status.strip()

            def col(name):
                k = extra.get(name)
                return p[k].strip() if k is not None and k < len(p) else None
            try:
                pr = Projection(rc, int(rs), int(re_), asm, None if contig in (".", "") else contig,
                                _int_or_none(s), _int_or_none(e), strand if strand in ("+", "-") else "+",
                                status, qid, None if col("method") in (None, "", ".") else col("method"), _pair(col("snap"), str),
                                _pair(col("snap_shift"), int))
            except ValueError as ex:
                raise ValueError(f"{path}: line {ln}: {ex}") from None
            if pr.ok:
                if pr.contig is None or pr.start is None or pr.end is None or strand not in ("+", "-"):
                    raise ValueError(f"{path}: line {ln}: status {status} but contig/start/end/strand missing")
                if pr.end <= pr.start or pr.start < 0:
                    raise ValueError(f"{path}: line {ln}: invalid interval [{pr.start},{pr.end})")
            rows.append(pr)
    return rows


def write_projection_table(path, rows: Iterable[Projection], ref_assembly: str = "B73"):
    with open(path, "w") as f:
        f.write("\t".join(TABLE_COLUMNS + EXTRA_COLUMNS) + "\n")
        for k, r in enumerate(rows):
            f.write("\t".join(map(str, (
                r.qid or f"p{k}", f"{ref_assembly}_{bare_chrom(r.ref_chrom, ref_assembly)}", r.ref_start,
                r.ref_end, r.assembly, r.contig or ".", "." if r.start is None else r.start,
                "." if r.end is None else r.end, r.strand if r.ok else ".", r.status, r.method or ".",
                ",".join(r.snap) if r.snap else ".",
                ",".join(map(str, r.snap_shift)) if r.snap_shift else "."))) + "\n")


def write_projection_queries(path, queries: Iterable[ProjectionQuery], ref_assembly: str = "B73"):
    with open(path, "w") as f:
        f.write("\t".join(QUERY_COLUMNS) + "\n")
        for q in queries:
            f.write(f"{q.qid}\t{ref_assembly}_{bare_chrom(q.ref_chrom, ref_assembly)}\t{q.ref_start}\t"
                    f"{q.ref_end}\t{q.assembly}\n")


def identity_projections(queries: Iterable[ProjectionQuery]) -> list[Projection]:
    """PLACEHOLDER rows: reference coordinates reused verbatim on <Asm>_<chr>. Not a real projection;
    only for exercising the code path when no projecting tool is available."""
    return [Projection(q.ref_chrom, q.ref_start, q.ref_end, q.assembly, f"{q.assembly}_{q.ref_chrom}",
                       q.ref_start, q.ref_end, "+", OK, q.qid) for q in queries]
