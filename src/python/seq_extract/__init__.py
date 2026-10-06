"""faidx-based sequence extraction and imputed-haplotype sequence stitching.

Conventions: 0-based half-open [start, end) on the forward strand; '-' = reverse complement;
assembly contigs are named <Asm>_<chr> (e.g. Oh43_chr1); reference chromosomes are bare (chr1).
See fai.py (extraction, case, IUPAC), projection.py (providers, table format) and haplotype.py
(stitching, gap policies, failures).
"""
from .fai import (FaiEntry, FastaFile, FastaIndexError, Region, RegionError, build_fai, index_fasta, read_fai,
                  revcomp, write_fai)
from .assembly import AssemblySet, fai_path_for
from .projection import (OK, OK_CLAMPED, Projection, is_usable, ProjectionProvider, ProjectionQuery, TableProvider, TruthMapProvider,
                         identity_projections, read_projection_table, write_projection_queries,
                         write_projection_table)
from .haplotype import (BedInterval, HaplotypeBuilder, HaplotypeResult, Segment, Window, read_bed_dir,
                        read_imputed_bed, read_windows, resolve_tiling)
from .snap import SnapOpt, SnapProvider, SnapRes, nt6, revcomp6, snap_align
