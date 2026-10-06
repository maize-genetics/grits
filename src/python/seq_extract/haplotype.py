"""Stitch an imputed haplotype's sequence over reference windows from founder assemblies.

Haplotype 1 follows the BED `parent1` column, haplotype 2 `parent2`. For a window [ws, we) on a
reference chromosome, the BED is first resolved chromosome-wide into a gap-free tiling (founder or
"uncalled"), with consecutive pieces of the same founder merged, then clipped to the window. Each founder
piece [s, e) is projected into that founder (provider.project) and extracted; pieces are concatenated
in reference order. Insertions that sit exactly at a founder switch point belong to neither piece and are
not emitted (the projection of [s, e) runs from the first to the last reference base present).

Uncalled stretches (between BED intervals, before the first, after the last) follow `gap_policy`:
  "N"       N x reference length (default);
  "skip"    nothing is emitted;
  "extend"  the neighbouring founders are extended: an interior gap is split at its midpoint (left
            founder gets floor(len/2)), as make_bed_contiguous does; a leading/trailing gap takes the
            first/last founder of the chromosome. A chromosome without intervals stays uncalled (N).
Gap bases are counted in the metadata under every policy.

Projection or extraction failures are recorded and emitted as N x reference length; nothing is guessed.
Segments whose founder is the reference assembly use the identity projection when reference_identity
is set (the reference coordinate system *is* that assembly).
"""
from __future__ import annotations

import bisect
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from .fai import FastaIndexError, RegionError
from .projection import OK, Projection, ProjectionQuery, bare_chrom, is_usable

GAP_POLICIES = ("N", "skip", "extend")
_INF = 1 << 62
BED_COLUMNS = ["chrom", "start", "end", "parent1", "parent2"]
_BED_NAME = re.compile(r"^(?P<sample>.+)_(?P<chrom>[^_]+)_imputed\.bed$")


@dataclass(frozen=True)
class BedInterval:
    chrom: str
    start: int
    end: int
    parent1: str
    parent2: str

    def founder(self, hap: int) -> str:
        return self.parent1 if hap == 1 else self.parent2


@dataclass(frozen=True)
class Window:
    wid: str
    chrom: str
    start: int
    end: int


@dataclass
class Segment:
    founder: str | None          # None = uncalled reference stretch
    ref_start: int
    ref_end: int
    contig: str | None = None
    start: int | None = None
    end: int | None = None
    strand: str = "+"
    status: str = "gap"          # ok | ok_clamped | gap | <failure status>
    emitted: int = 0             # bases contributed to the output
    snap: tuple | None = None    # (start label, end label) when the provider snapped the ends
    snap_shift: tuple | None = None

    @property
    def ref_len(self):
        return self.ref_end - self.ref_start

    def describe(self) -> str:
        r = f"{self.ref_start}-{self.ref_end}"
        if self.founder is None:
            return f"GAP:{r}"
        if is_usable(self.status):
            d = f"{self.founder}:{r}>{self.contig}:{self.start}-{self.end}:{self.strand}"
            if self.status != OK:
                d += f":{self.status}"
            if self.snap:
                d += f"[{','.join(self.snap)}]"
            return d
        return f"{self.founder}:{r}!{self.status}"


@dataclass
class HaplotypeResult:
    window: Window
    hap: int
    seq: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def founder_segments(self):
        return [s for s in self.segments if s.founder is not None]

    @property
    def n_switches(self) -> int:
        f = [s.founder for s in self.founder_segments]
        return sum(1 for a, b in zip(f, f[1:]) if a != b)

    @property
    def gap_bases(self) -> int:
        return sum(s.ref_len for s in self.segments if s.founder is None)

    @property
    def failures(self) -> list[Segment]:
        return [s for s in self.founder_segments if not is_usable(s.status)]

    @property
    def n_clamped(self) -> int:
        return sum(1 for s in self.founder_segments if s.status == "ok_clamped")

    @property
    def failed_bases(self) -> int:
        return sum(s.ref_len for s in self.failures)

    @property
    def founders(self) -> list[str]:
        out = []
        for s in self.founder_segments:
            if s.founder not in out:
                out.append(s.founder)
        return out


# --------------------------------------------------------------------------- inputs

def read_imputed_bed(path) -> list[BedInterval]:
    out = []
    with open(path, newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        for ln, p in enumerate(rd, 1):
            if not p or not "".join(p).strip() or p[0].startswith("#"):
                continue
            if ln == 1 and [x.strip() for x in p[:5]] == BED_COLUMNS:
                continue
            if len(p) < 5:
                raise ValueError(f"{path}: line {ln}: expected >= 5 columns ({' '.join(BED_COLUMNS)})")
            try:
                s, e = int(p[1]), int(p[2])
            except ValueError:
                raise ValueError(f"{path}: line {ln}: non-integer start/end") from None
            if s < 0 or e <= s:
                raise ValueError(f"{path}: line {ln}: invalid interval [{s},{e})")
            out.append(BedInterval(p[0], s, e, p[3].strip(), p[4].strip()))
    return out


def group_bed(intervals: Iterable[BedInterval], ref_assembly: str = "B73", overlap: str = "trim",
              stats: dict | None = None) -> dict[str, list[BedInterval]]:
    """By bare chromosome, sorted, overlaps resolved.

    overlap="trim": an interval starting before the previous one's end is trimmed to start there (the
    earlier interval keeps the overlapped bases; a fully covered interval is dropped). Real GRITS BEDs
    contain 1 bp overlaps where two decoded bins share a position. overlap="error" raises instead.
    stats, if given, receives n_trimmed_intervals / trimmed_bases / n_dropped_intervals.
    """
    if overlap not in ("trim", "error"):
        raise ValueError("overlap must be 'trim' or 'error'")
    st = {"n_trimmed_intervals": 0, "trimmed_bases": 0, "n_dropped_intervals": 0}
    by: dict[str, list[BedInterval]] = {}
    for iv in intervals:
        c = bare_chrom(iv.chrom, ref_assembly)
        by.setdefault(c, []).append(BedInterval(c, iv.start, iv.end, iv.parent1, iv.parent2))
    for c, ivs in by.items():
        ivs.sort(key=lambda x: (x.start, x.end))
        out = []
        for b in ivs:
            if out and b.start < out[-1].end:
                a = out[-1]
                if overlap == "error":
                    raise ValueError(f"overlapping BED intervals on {c}: [{a.start},{a.end}) and [{b.start},{b.end})")
                if b.end <= a.end:
                    st["n_dropped_intervals"] += 1
                    st["trimmed_bases"] += b.end - b.start
                    continue
                st["n_trimmed_intervals"] += 1
                st["trimmed_bases"] += a.end - b.start
                b = BedInterval(c, a.end, b.end, b.parent1, b.parent2)
            out.append(b)
        by[c] = out
    if stats is not None:
        stats.update(st)
    return by


def read_bed_dir(bed_dir, sample: str | None = None, ref_assembly: str = "B73", overlap: str = "trim",
                 stats: dict | None = None) -> dict[str, list[BedInterval]]:
    """All `<sample>_<chr>_imputed.bed` files of one sample (a single .bed file path also works)."""
    p = Path(bed_dir)
    if p.is_file():
        return group_bed(read_imputed_bed(p), ref_assembly, overlap, stats)
    files = sorted(p.glob("*_imputed.bed"))
    by_sample: dict[str, list[Path]] = {}
    for f in files:
        m = _BED_NAME.match(f.name)
        if m:
            by_sample.setdefault(m["sample"], []).append(f)
    if sample is None:
        if len(by_sample) != 1:
            raise ValueError(f"{p}: expected exactly one sample's *_imputed.bed files, found "
                             f"{sorted(by_sample) or 'none'}; pass sample=")
        sample = next(iter(by_sample))
    if sample not in by_sample:
        raise ValueError(f"{p}: no *_imputed.bed files for sample '{sample}'")
    ivs = []
    for f in by_sample[sample]:
        ivs += read_imputed_bed(f)
    return group_bed(ivs, ref_assembly, overlap, stats)


def read_windows(path, ref_assembly: str = "B73") -> list[Window]:
    """TSV with header `wid chrom start end` (0-based half-open, reference coordinates)."""
    out, seen = [], set()
    with open(path, newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        header = next(rd, None)
        if header is None or [h.strip() for h in header[:4]] != ["wid", "chrom", "start", "end"]:
            raise ValueError(f"{path}: header must be 'wid chrom start end'; got {header}")
        for ln, p in enumerate(rd, 2):
            if not p or not "".join(p).strip():
                continue
            try:
                w = Window(p[0], bare_chrom(p[1], ref_assembly), int(p[2]), int(p[3]))
            except (ValueError, IndexError):
                raise ValueError(f"{path}: line {ln}: bad window row {p}") from None
            if w.start < 0 or w.end <= w.start:
                raise ValueError(f"{path}: line {ln}: invalid window [{w.start},{w.end})")
            if w.wid in seen:
                raise ValueError(f"{path}: line {ln}: duplicate wid '{w.wid}'")
            seen.add(w.wid)
            out.append(w)
    return out


# --------------------------------------------------------------------------- builder

def resolve_tiling(intervals: list[BedInterval], hap: int, gap_policy: str) -> list[tuple[int, int, str | None]]:
    """Chromosome-wide gap-free tiling [0, inf) for one haplotype, same-founder neighbours merged."""
    if gap_policy not in GAP_POLICIES:
        raise ValueError(f"gap_policy must be one of {GAP_POLICIES}, got {gap_policy!r}")
    called = [(iv.start, iv.end, iv.founder(hap)) for iv in intervals]
    if not called:
        return [(0, _INF, None)]
    tiles: list[list] = []
    pos = 0
    for k, (s, e, f) in enumerate(called):
        if s > pos:
            if gap_policy == "extend":
                if k == 0:
                    tiles.append([pos, s, f])
                else:
                    mid = pos + (s - pos) // 2
                    tiles[-1][1] = mid
                    tiles.append([mid, s, f])
            else:
                tiles.append([pos, s, None])
        tiles.append([s, e, f])
        pos = e
    tiles.append([pos, _INF, called[-1][2] if gap_policy == "extend" else None])
    merged: list[list] = []
    for t in tiles:
        if t[1] <= t[0]:
            continue
        if merged and merged[-1][2] == t[2] and merged[-1][1] == t[0]:
            merged[-1][1] = t[1]
        else:
            merged.append(t)
    return [tuple(t) for t in merged]


class HaplotypeBuilder:
    def __init__(self, bed: dict[str, list[BedInterval]], assemblies, provider, gap_policy: str = "N",
                 ref_assembly: str = "B73", reference_identity: bool = True, upper: bool = False):
        if gap_policy not in GAP_POLICIES:
            raise ValueError(f"gap_policy must be one of {GAP_POLICIES}, got {gap_policy!r}")
        self.bed = {bare_chrom(c, ref_assembly): v for c, v in bed.items()}
        self.assemblies = assemblies
        self.provider = provider
        self.gap_policy = gap_policy
        self.ref_assembly = ref_assembly
        self.reference_identity = reference_identity
        self.upper = upper
        self._tilings: dict[tuple[str, int], tuple[list, list]] = {}

    def _tiling(self, chrom, hap):
        key = (chrom, hap)
        if key not in self._tilings:
            t = resolve_tiling(self.bed.get(chrom, []), hap, self.gap_policy)
            self._tilings[key] = ([x[0] for x in t], t)
        return self._tilings[key]

    def layout(self, window: Window, hap: int) -> list[tuple[int, int, str | None]]:
        if hap not in (1, 2):
            raise ValueError("hap must be 1 or 2")
        chrom = bare_chrom(window.chrom, self.ref_assembly)
        starts, tiles = self._tiling(chrom, hap)
        k = max(0, bisect.bisect_right(starts, window.start) - 1)
        out = []
        while k < len(tiles) and tiles[k][0] < window.end:
            s, e, f = tiles[k]
            s, e = max(s, window.start), min(e, window.end)
            if e > s:
                out.append((s, e, f))
            k += 1
        return out

    def _uses_identity(self, founder):
        return self.reference_identity and founder == self.ref_assembly

    def projection_queries(self, windows: Iterable[Window]) -> list[ProjectionQuery]:
        """Unique (chrom, start, end, founder) projections the build will request, for a projecting tool."""
        seen, out = set(), []
        for w in windows:
            chrom = bare_chrom(w.chrom, self.ref_assembly)
            for hap in (1, 2):
                for s, e, f in self.layout(w, hap):
                    if f is None or self._uses_identity(f):
                        continue
                    key = (chrom, s, e, f)
                    if key not in seen:
                        seen.add(key)
                        out.append(ProjectionQuery(f"q{len(out)}", chrom, s, e, f))
        return out

    def _project(self, chrom, s, e, founder) -> Projection:
        if self._uses_identity(founder):
            return Projection(chrom, s, e, founder, f"{founder}_{chrom}", s, e, "+", OK)
        return self.provider.project(chrom, s, e, founder)

    def build(self, window: Window, hap: int) -> HaplotypeResult:
        chrom = bare_chrom(window.chrom, self.ref_assembly)
        parts, segs = [], []
        for s, e, f in self.layout(window, hap):
            if f is None:
                seg = Segment(None, s, e)
                if self.gap_policy != "skip":
                    parts.append("N" * (e - s))
                    seg.emitted = e - s
                segs.append(seg)
                continue
            seg = Segment(f, s, e)
            pr = self._project(chrom, s, e, f)
            if pr.ok:
                seg.contig, seg.start, seg.end, seg.strand = pr.contig, pr.start, pr.end, pr.strand
                seg.snap, seg.snap_shift = pr.snap, pr.snap_shift
                try:
                    seq = self.assemblies.fetch(pr.contig, pr.start, pr.end, pr.strand, upper=self.upper)
                    seg.status = pr.status
                except (RegionError, FastaIndexError, FileNotFoundError) as ex:
                    seq = None
                    seg.status = f"extract_error:{type(ex).__name__}"
            else:
                seq = None
                seg.status = pr.status or "failed"
            if seq is None:
                seq = "N" * (e - s)
            parts.append(seq)
            seg.emitted = len(seq)
            segs.append(seg)
        return HaplotypeResult(window, hap, "".join(parts), segs)

    def build_all(self, windows: Iterable[Window]) -> Iterator[tuple[HaplotypeResult, HaplotypeResult]]:
        for w in windows:
            yield self.build(w, 1), self.build(w, 2)


METADATA_COLUMNS = ["wid", "hap", "chrom", "start", "end", "seq_len", "n_segments", "n_switches", "gap_bases",
                    "n_failures", "failed_bases", "founders", "segments", "failures", "n_clamped"]


def metadata_row(r: HaplotypeResult) -> list:
    w = r.window
    return [w.wid, r.hap, w.chrom, w.start, w.end, len(r.seq), len(r.founder_segments), r.n_switches,
            r.gap_bases, len(r.failures), r.failed_bases, ",".join(r.founders) or ".",
            ";".join(s.describe() for s in r.segments) or ".",
            ";".join(s.describe() for s in r.failures) or ".", r.n_clamped]
