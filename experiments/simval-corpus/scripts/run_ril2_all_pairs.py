#!/usr/bin/env python
"""
Run every IDX-RIL2 pair at a given coverage through the align stage, with
the new --max-hit-frac / binarize windowing features (see
/home/zrm22/.claude/plans/wondrous-discovering-octopus.md and
test_crf_relatedness/src/python/crf/eval/{PLAN,RESULTS,HANDOFF}.md on
branch windowing-quality-filters).

Uses the corpus's OWN official reads (manifest.tsv r1/r2/truth paths), not
an ad hoc resample -- confirmed the earlier one-off Oh43xIl14H subsample
this session built (scratch/read_datasets/...) does NOT byte-match the
now-published official 0.1x row (different md5), so refmap is re-run fresh
here against the official fastqs for every pair, Oh43xIl14H included, for
consistency with "the rest of the RIL2 datasets are done".

The edited ropebwt_npy_to_matrix.py lives on branch windowing-quality-filters
(worktree grits-windowfilter-worktree), not merged yet, so this monkey-patches
nam_baseline.WINDOW_SCRIPT to point there for this run only -- same pattern
run_ril2_windowfilter_test.py already uses.

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src \
      /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/run_ril2_all_pairs.py \
        --coverage 0.1 --max-hit-frac 0.5
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import heldout_assembly_eval as hae  # noqa: E402
import simval_eval_one as seo  # noqa: E402

WORKTREE_WINDOW_SCRIPT = Path(
    "/local/workdir/zrm22/HackathonJun2026/grits-windowfilter-worktree/"
    "src/python/crf/ropebwt_npy_to_matrix.py")
assert WORKTREE_WINDOW_SCRIPT.exists(), WORKTREE_WINDOW_SCRIPT
hae.nb.WINDOW_SCRIPT = WORKTREE_WINDOW_SCRIPT
print(f"[patched] hae.nb.WINDOW_SCRIPT -> {WORKTREE_WINDOW_SCRIPT}")

MANIFEST = Path(
    "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv")
SCRATCH_ROOT = Path(
    "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/simval_eval")
PAIRS = ["B73xOh43", "B73xCML103", "Oh43xIl14H", "B97xCML103", "Il14HxB97"]


def load_manifest_row(individual, coverage):
    with open(MANIFEST) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["dataset_id"] == "IDX-RIL2" and row["individual"] == individual \
                    and row["coverage"] == coverage:
                return row
    raise KeyError(f"no manifest row for IDX-RIL2/{individual}/{coverage}x")


COVERAGES = ["0.01", "0.1", "0.5", "1.0", "2.0"]


RAW_FILES = ("raw.npy", "raw.npy.bins.tsv", "raw.npy.gametes.tsv", "raw.ps4g", "raw.tsv", "raw.log")


def stage_raw_from_sibling(individual, coverage, outdir):
    """refmap's raw.npy is produced BEFORE any --max-hit-frac/binarize
    filtering (that happens downstream, at the windowing step) -- so it's
    identical across every filter setting for the same (individual,
    coverage). If any sibling IDX-RIL2__{individual}__{coverage}x__* dir
    (any OTHER tag) already has a complete raw.npy/raw.tsv, symlink it into
    this run's outdir so hae.run_refmap's own resumability check
    (npy_path.exists() and tsv_path.exists()) skips refmap entirely instead
    of re-aligning reads it's already aligned once."""
    if (outdir / "raw.npy").exists():
        return
    for sib in sorted(SCRATCH_ROOT.glob(f"IDX-RIL2__{individual}__{coverage}x__*")):
        if sib == outdir or not (sib / "raw.npy").exists() or not (sib / "raw.tsv").exists():
            continue
        for f in RAW_FILES:
            src, dst = sib / f, outdir / f
            if src.exists() and not dst.exists():
                dst.symlink_to(src.resolve())
        print(f"  staged raw.npy for {individual}/{coverage}x from {sib.name}")
        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverages", nargs="*", default=COVERAGES)
    ap.add_argument("--max-hit-frac", type=float, default=None,
                     help="default None = unfiltered (off); pass a float to enable the fan-out filter")
    ap.add_argument("--pairs", nargs="*", default=PAIRS)
    cli = ap.parse_args()
    tag = f"hitfrac{cli.max_hit_frac}-bin" if cli.max_hit_frac is not None else "unfiltered-bin"

    hae.THREADS = "20"
    for coverage in cli.coverages:
        for individual in cli.pairs:
            mrow = load_manifest_row(individual, coverage)
            outdir = SCRATCH_ROOT / f"IDX-RIL2__{individual}__{coverage}x__{tag}"
            outdir.mkdir(parents=True, exist_ok=True)
            stage_raw_from_sibling(individual, coverage, outdir)
            print(f"\n=== IDX-RIL2 / {individual} / {coverage}x / {tag} ===")

            args = argparse.Namespace(
                stage="align", sample=f"{individual}_ril2_{tag}",
                r1=mrow["r1_path"], r2=mrow["r2_path"], truth_h1=None, truth_h2=None,
                outdir=str(outdir), out_json=str(outdir / "align_result.json"),
                drop_idx=23, max_hit_frac=cli.max_hit_frac, retain_counts=False,
                region=None, threads=20, no_cleanup=False,
                panel_vcf=str(seo.P.PANEL_VCF_V2), ckpt=str(seo.P.CKPT_DIPLOID),
                dataset_id="IDX-RIL2", coverage=coverage, dataset_class="indexed",
                kind="ril2", arm="refmap",
            )
            result = seo.do_align(args)
            print(f"  -> {result}")


if __name__ == "__main__":
    main()
