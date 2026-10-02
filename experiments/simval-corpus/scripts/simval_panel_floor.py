#!/usr/bin/env python
"""
Panel-representability floor for held-out (OUT-INBRED) individuals
(Diagnostic 2 of /home/zrm22/.claude/plans/swift-chasing-melody.md's
chain-error investigation).

Answers: at each panel site, does ANY of the 24 founders actually used by
the model (panel_25founders_v2.vcf minus the fixed-dropped P39, index 23 --
same drop every real batch row used) carry the held-out individual's true
allele? The non-representable fraction is a hard error floor no model,
however good, could beat. Reported alongside two baselines computed in the
SAME pass (each panel record's own founder GT columns already give
everything needed): "always predict B73" and "best single founder"
(whichever of the 24 founders' own genotype the truth agrees with most
often, picked in hindsight -- an upper bound no model actually achieves,
but a useful ceiling).

Scope: chr1, SYSTEMATICALLY subsampled (every SAMPLE_STRIDE-th panel
record, position order preserved -- required, since TruthCursor is
forward-only) rather than every one of chr1's ~26M records, purely for
Python-loop tractability -- a head-based subsample would only cover chr1's
first ~15-40Mb (this file is position-sorted) and badly misrepresent
distal/centromeric structure, so this uses a systematic stride across the
WHOLE chromosome instead, same discipline as the earlier deletion/insertion
measurement. Every number below states this scope explicitly.

Reuses UNMODIFIED: compare_gvcf_truth.TruthCursor, .iter_truth_records,
.partition_truth_by_contig; vcf_eval.accuracy.gt_to_allele_multiset.

Usage:
    simval_panel_floor.py [--stride 10] [--limit-panel-records N]
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import simval_paths as P  # noqa: E402

CRF_REPO_ROOT = Path("/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness")
sys.path.insert(0, str(CRF_REPO_ROOT / "src"))
from python.vcf_eval import compare_gvcf_truth as cgt  # noqa: E402
from python.vcf_eval.accuracy import gt_to_allele_multiset  # noqa: E402

HELDOUT_GVCF_DIR = P.GRITS_WORKDIR / "data/maize_v2_heldout/gvcf_sorted"
CONTROL_GVCF_DIR = P.GRITS_WORKDIR / "data/maize_v2_rebuild/gvcf_sorted"

HELDOUT_SAMPLES = ["A188", "CML459", "EP1", "Ia453", "Tx303"]
CONTROL_SAMPLES = ["B73", "B97", "CML103", "Il14H", "Oh43"]  # in-panel sanity check
DROP_IDX = P.FIXED_DROP_IDX  # 23 == P39, same fixed drop every real batch row used


def build_cursor(gvcf_path, chrom, tmp_dir):
    contig_files = cgt.partition_truth_by_contig(str(gvcf_path), tmp_dir)
    path = contig_files.get(chrom)
    records = cgt.iter_truth_records(str(path)) if path is not None else iter(())
    return cgt.TruthCursor(records)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10,
                     help="process every Nth chr1 panel record, position order preserved")
    ap.add_argument("--limit-panel-records", type=int, default=None,
                     help="stop after this many chr1 panel records seen (pre-stride) -- for smoke tests")
    ap.add_argument("--chrom", default="chr1")
    args = ap.parse_args()
    chrom = args.chrom

    names = HELDOUT_SAMPLES + CONTROL_SAMPLES
    print(f"Building truth cursors for {names} on {chrom} ...", flush=True)
    tmpdirs = []
    cursors = {}
    for name in names:
        gvcf = (HELDOUT_GVCF_DIR if name in HELDOUT_SAMPLES else CONTROL_GVCF_DIR) / f"{name}.g.vcf.gz"
        td = tempfile.TemporaryDirectory()
        tmpdirs.append(td)
        cursors[name] = build_cursor(gvcf, chrom, Path(td.name))
    print("Cursors ready.", flush=True)

    stats = {name: {"compared": 0, "non_representable": 0} for name in names}
    founder_names = None
    founder_correct = {name: None for name in names}  # np arrays, lazily sized

    import numpy as np

    seen_chr1 = 0
    used = 0
    with open(P.PANEL_VCF_V2) as f:
        for line in f:
            if line[0] == "#":
                if line.startswith("#CHROM"):
                    header = line.rstrip("\n").split("\t")
                    founder_names = header[9:]
                    assert founder_names[DROP_IDX] == "P39", (
                        f"expected DROP_IDX={DROP_IDX} to be P39, got {founder_names[DROP_IDX]!r} "
                        f"-- panel column order changed, floor computation would silently use the "
                        f"wrong founder set")
                    keep_founder_idx = [i for i in range(len(founder_names)) if i != DROP_IDX]
                    keep_founder_names = [founder_names[i] for i in keep_founder_idx]
                    for name in names:
                        founder_correct[name] = np.zeros(len(keep_founder_idx), dtype=np.int64)
                continue

            c = line[:line.index("\t")]
            if c != chrom:
                if seen_chr1 > 0:
                    break  # chr1 block ended (file is position-sorted per contig)
                continue
            seen_chr1 += 1
            if args.limit_panel_records and seen_chr1 > args.limit_panel_records:
                break
            if (seen_chr1 - 1) % args.stride != 0:
                continue
            used += 1

            parts = line.rstrip("\n").split("\t")
            pos = int(parts[1])
            ref = parts[3]
            alt_field = parts[4]
            founder_gts = parts[9:]

            alts = [] if alt_field in (".", "") else alt_field.split(",")

            def gt_allele(g):
                g = g.strip()
                if g in (".", ""):
                    return None
                try:
                    idx = int(g)
                except ValueError:
                    return None
                if idx == 0:
                    return ref
                j = idx - 1
                return alts[j] if 0 <= j < len(alts) else None

            kept_alleles = [gt_allele(founder_gts[i]) for i in keep_founder_idx]
            representable_set = {a for a in kept_alleles if a is not None}

            for name in names:
                r = cursors[name].resolve(chrom, pos, 1)
                if r is None:
                    continue
                t_alt, t_gt = r
                truth_tuple = gt_to_allele_multiset(ref, t_alt, t_gt, phase_sensitive=False)
                if truth_tuple is None:
                    continue
                truth_allele = truth_tuple[0]

                st = stats[name]
                st["compared"] += 1
                if truth_allele not in representable_set:
                    st["non_representable"] += 1

                fc = founder_correct[name]
                for k, a in enumerate(kept_alleles):
                    if a == truth_allele:
                        fc[k] += 1

            if used % 200000 == 0:
                print(f"  ... {used:,} sampled records processed ({seen_chr1:,} chr1 records seen)",
                      flush=True)

    print(f"\nDone: {seen_chr1:,} chr1 panel records seen, {used:,} sampled "
          f"(stride={args.stride}) on {chrom}.\n")

    b73_k = keep_founder_names.index("B73")
    rows = []
    for name in names:
        st = stats[name]
        fc = founder_correct[name]
        compared = st["compared"]
        floor = st["non_representable"] / compared if compared else None
        b73_err = 1 - fc[b73_k] / compared if compared else None
        best_k = int(np.argmax(fc))
        best_err = 1 - fc[best_k] / compared if compared else None
        rows.append({
            "sample": name, "role": "heldout" if name in HELDOUT_SAMPLES else "control",
            "compared_sites": compared,
            "representability_floor": floor,
            "always_B73_error": b73_err,
            "best_single_founder": keep_founder_names[best_k],
            "best_single_founder_error": best_err,
        })

    import csv
    out_path = P.RESULTS_DIR / "simval_panel_floor.tsv"
    P.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"{'sample':10s} {'role':8s} {'compared':>10s} {'floor%':>8s} "
          f"{'B73base%':>9s} {'best_founder':>13s} {'best%':>8s}")
    for r in rows:
        print(f"{r['sample']:10s} {r['role']:8s} {r['compared_sites']:>10d} "
              f"{100*r['representability_floor']:>7.3f}% "
              f"{100*r['always_B73_error']:>8.3f}% {r['best_single_founder']:>13s} "
              f"{100*r['best_single_founder_error']:>7.3f}%")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
