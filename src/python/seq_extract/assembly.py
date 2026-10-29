"""Assembly name -> FASTA (+ .fai) mapping, and contig `<Asm>_<chr>` -> assembly resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from .fai import FastaFile, FastaIndexError, RegionError, index_fasta


def fai_path_for(fasta_path, fai_dir=None) -> Path:
    fasta_path = Path(fasta_path)
    if fai_dir is None:
        return Path(str(fasta_path) + ".fai")
    return Path(fai_dir) / (fasta_path.name + ".fai")


class AssemblySet:
    """Lazily opened FASTAs, one per assembly. Each file is opened once and reused.

    fai_dir: where the .fai files live (named <fasta basename>.fai). None means next to the FASTA.
    build_missing: build absent indexes into fai_dir (never next to the FASTA: fai_dir is required).
    """

    def __init__(self, fastas: Mapping[str, str | Path], fai_dir=None, build_missing: bool = False):
        if build_missing and fai_dir is None:
            raise ValueError("build_missing requires an explicit fai_dir")
        self.fastas = {a: Path(p) for a, p in fastas.items()}
        self.fai_dir = Path(fai_dir) if fai_dir is not None else None
        self.build_missing = build_missing
        self._open: dict[str, FastaFile] = {}
        self._by_len = sorted(self.fastas, key=len, reverse=True)

    @classmethod
    def from_dir(cls, fasta_dir, pattern: str = "{asm}.fa", assemblies: Iterable[str] | None = None,
                 fai_dir=None, build_missing: bool = False) -> "AssemblySet":
        """pattern contains '{asm}', e.g. '{asm}_renamed.fa'. Without `assemblies`, globs the directory."""
        fasta_dir = Path(fasta_dir)
        if "{asm}" not in pattern:
            raise ValueError("pattern must contain '{asm}'")
        pre, post = pattern.split("{asm}", 1)
        if assemblies is None:
            found = {}
            for p in sorted(fasta_dir.glob(pattern.replace("{asm}", "*"))):
                n = p.name
                if n.startswith(pre) and n.endswith(post) and len(n) > len(pre) + len(post):
                    found[n[len(pre):len(n) - len(post)]] = p
            if not found:
                raise FileNotFoundError(f"no FASTA matching {pattern!r} in {fasta_dir}")
            return cls(found, fai_dir, build_missing)
        return cls({a: fasta_dir / pattern.format(asm=a) for a in assemblies}, fai_dir, build_missing)

    @property
    def assemblies(self) -> list[str]:
        return list(self.fastas)

    def __contains__(self, asm):
        return asm in self.fastas

    def fai_path(self, asm: str) -> Path:
        return fai_path_for(self._fasta(asm), self.fai_dir)

    def _fasta(self, asm):
        try:
            return self.fastas[asm]
        except KeyError:
            raise RegionError(f"unknown assembly '{asm}' (known: {', '.join(sorted(self.fastas))})") from None

    def get(self, asm: str) -> FastaFile:
        f = self._open.get(asm)
        if f is None:
            fa = self._fasta(asm)
            if not fa.exists():
                raise FileNotFoundError(f"FASTA for assembly '{asm}' not found: {fa}")
            fai = self.fai_path(asm)
            if not fai.exists():
                if not self.build_missing:
                    raise FastaIndexError(f"missing index {fai} for {fa}")
                index_fasta(fa, fai)
            f = self._open[asm] = FastaFile(fa, fai)
        return f

    def assembly_of(self, contig: str) -> str:
        """`Oh43_chr1` -> `Oh43` (longest known assembly name followed by '_')."""
        for a in self._by_len:
            if contig.startswith(a + "_") and len(contig) > len(a) + 1:
                return a
        raise RegionError(f"contig '{contig}' does not start with a known assembly name + '_'")

    def file_for(self, contig: str) -> FastaFile:
        return self.get(self.assembly_of(contig))

    def length(self, contig: str) -> int:
        return self.file_for(contig).length(contig)

    def fetch(self, contig: str, start: int, end: int, strand: str = "+", upper: bool = False) -> str:
        return self.file_for(contig).fetch(contig, start, end, strand, upper)

    def fetch_many(self, regions, upper: bool = False) -> list[str]:
        """regions: Region objects or (contig, start, end[, strand]) tuples, in any assembly mix."""
        out = []
        for r in regions:
            if hasattr(r, "contig"):
                out.append(self.fetch(r.contig, r.start, r.end, r.strand, upper))
            else:
                out.append(self.fetch(*r, upper=upper))
        return out

    def close(self):
        for f in self._open.values():
            f.close()
        self._open.clear()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
