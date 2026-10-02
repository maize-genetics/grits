#!/usr/bin/env python
"""
Founder-path bp-weighted error -- productized version of the ad hoc scoring
recipe used throughout this session (originally re-derived from scratch
every time as e.g. /home/zrm22/.claude/jobs/0e12e67f/tmp/ril2_two_tables.py).
See /home/zrm22/.claude/plans/wondrous-discovering-octopus.md.

Metric: bp where EITHER decoded haplotype disagrees with the true RIL2
founder ("pair_error" -- the convention used in every published table so
far), over total decoded bp. Decoded BED intervals are split at any true
mosaic breakpoint they straddle before comparison, so a decoded interval
spanning a true crossover is scored correctly on each side of it.

Ground truth comes from simval_oracle_bed.build_ril_mosaics, which exactly
replays the corpus's own breakpoint/founder-assignment RNG (crc32-based
simlib.seed_for) -- no truth gVCF or comparator needed, this reads only the
decoded BED files.
"""
import argparse
import bisect
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from simval_oracle_bed import build_ril_mosaics  # noqa: E402

SCRATCH_ROOT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/simval_eval")


def _founder_at(segs, starts, pos):
    i = bisect.bisect_right(starts, pos) - 1
    return segs[i][2]


def score_founder_path_error(bed_dir, sample, mosaic_h1, label_to_name):
    """bed_dir: dir containing {sample}_{chrom}_imputed.bed files (one per
    contig, per write_imputed_bed's own convention). mosaic_h1: {chrom:
    [(start, end, 'A'|'B'), ...]} -- for ril2, mosaic_h1 IS mosaic_h2 (both
    haplotypes share one crossover path), so only one mosaic is needed here.
    Returns (wrong_bp, total_bp, error_pct); error_pct is None if no decoded
    BED files were found (row not yet run)."""
    bed_dir = Path(bed_dir)
    n_beds = len(list(bed_dir.glob(f"{sample}_*_imputed.bed"))) if bed_dir.exists() else 0
    if n_beds < len(mosaic_h1):
        return None, None, None

    total_bp = 0
    wrong_bp = 0
    for chrom, segs in mosaic_h1.items():
        bed_path = bed_dir / f"{sample}_{chrom}_imputed.bed"
        decoded = pd.read_csv(bed_path, sep="\t")
        starts = [s[0] for s in segs]
        for _, row in decoded.iterrows():
            s, e = int(row["start"]), int(row["end"])
            cuts = [s] + [x for x in starts if s < x < e] + [e]
            for i in range(len(cuts) - 1):
                p, q = cuts[i], cuts[i + 1]
                width = q - p
                true_founder = label_to_name[_founder_at(segs, starts, p)]
                wrong = (row["parent1"] != true_founder) or (row["parent2"] != true_founder)
                total_bp += width
                if wrong:
                    wrong_bp += width
    error_pct = (wrong_bp / total_bp * 100) if total_bp else None
    return wrong_bp, total_bp, error_pct


def score_row(individual, coverage, tag, parent_a, parent_b, replicate=0):
    """Convenience wrapper for the IDX-RIL2 scratch layout convention:
    outdir = IDX-RIL2__{individual}__{coverage}x__{tag}, sample =
    {individual}_ril2_{tag}."""
    mosaic_h1, _, label_to_name = build_ril_mosaics("IDX-RIL2", parent_a, parent_b, replicate)
    sample = f"{individual}_ril2_{tag}"
    bed_dir = SCRATCH_ROOT / f"IDX-RIL2__{individual}__{coverage}x__{tag}" / "bed"
    return score_founder_path_error(bed_dir, sample, mosaic_h1, label_to_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-a", required=True)
    ap.add_argument("--parent-b", required=True)
    ap.add_argument("--coverages", nargs="*", default=["0.01", "0.1", "0.5", "1.0", "2.0"])
    ap.add_argument("--tags", nargs="*", required=True)
    args = ap.parse_args()

    individual = f"{args.parent_a}x{args.parent_b}"
    print(f"pair".ljust(20) + "".join(c.rjust(10) for c in args.coverages))
    for tag in args.tags:
        row_str = tag.ljust(20)
        for coverage in args.coverages:
            _, _, pct = score_row(individual, coverage, tag, args.parent_a, args.parent_b)
            row_str += (f"{pct:.4f}".rjust(10) if pct is not None else "MISSING".rjust(10))
        print(row_str)


if __name__ == "__main__":
    main()
