#!/usr/bin/env python
"""
Extract the actual list of WRONG founder-path intervals (not just raw
decoded segment count) for a given IDX-RIL2 row, for downstream annotation
(N-gap distance, Il14H SV overlap, local SSR content -- see
/home/zrm22/.claude/plans/wondrous-discovering-octopus.md follow-up).
Reuses founder_path_error.py's exact split-at-true-breakpoint bp-weighted
logic, but instead of summing bp, records each individual wrong
sub-interval.
"""
import argparse
import bisect
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from simval_oracle_bed import build_ril_mosaics  # noqa: E402
import simval_paths as P  # noqa: E402

INDIVIDUAL = "Oh43xIl14H"


def _founder_at(segs, starts, pos):
    i = bisect.bisect_right(starts, pos) - 1
    return segs[i][2]


def extract_wrong_intervals(bed_dir, sample, mosaic_h1, label_to_name):
    wrong_intervals = []
    total_bp = wrong_bp = 0
    for chrom, segs in mosaic_h1.items():
        bed_path = Path(bed_dir) / f"{sample}_{chrom}_imputed.bed"
        if not bed_path.exists():
            continue
        decoded = pd.read_csv(bed_path, sep="\t")
        starts = [s[0] for s in segs]
        for _, row in decoded.iterrows():
            s, e = int(row["start"]), int(row["end"])
            cuts = [s] + [x for x in starts if s < x < e] + [e]
            for i in range(len(cuts) - 1):
                p, q = cuts[i], cuts[i + 1]
                width = q - p
                true_founder = label_to_name[_founder_at(segs, starts, p)]
                total_bp += width
                wrong = (row["parent1"] != true_founder) or (row["parent2"] != true_founder)
                if wrong:
                    wrong_bp += width
                    wrong_intervals.append({
                        "chrom": chrom, "start": p, "end": q, "width": width,
                        "decoded_parent1": row["parent1"], "decoded_parent2": row["parent2"],
                        "true_founder": true_founder,
                    })
    return wrong_intervals, wrong_bp, total_bp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--json-out", required=True)
    cli = ap.parse_args()

    mosaic_h1, _, label_to_name = build_ril_mosaics("IDX-RIL2", "Oh43", "Il14H")
    sample = f"{INDIVIDUAL}_ril2_{cli.tag}"
    bed_dir = P.SCRATCH_ROOT / f"IDX-RIL2__{INDIVIDUAL}__{cli.coverage}x__{cli.tag}" / "bed"

    wrong_intervals, wrong_bp, total_bp = extract_wrong_intervals(bed_dir, sample, mosaic_h1, label_to_name)

    print(f"{cli.coverage}x/{cli.tag}: {len(wrong_intervals)} wrong intervals, "
          f"wrong_bp={wrong_bp:,} total_bp={total_bp:,} error_pct={wrong_bp/total_bp*100:.4f}%")

    by_chrom = {}
    for iv in wrong_intervals:
        by_chrom[iv["chrom"]] = by_chrom.get(iv["chrom"], 0) + 1
    print("wrong intervals per chromosome:", by_chrom)

    Path(cli.json_out).write_text(json.dumps(wrong_intervals, indent=1))
    print(f"wrote {cli.json_out}")


if __name__ == "__main__":
    main()
