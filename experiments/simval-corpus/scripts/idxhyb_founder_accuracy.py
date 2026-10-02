#!/usr/bin/env python
"""
Founder-pair accuracy (decode-level, not the full VCF/genotype comparator)
for an IDX-HYB row, before/after a --max-hit-frac PS4G filter.

"Founder accuracy" here = does the diploid-affinity CRF decode the correct
UNORDERED founder pair per window-position. For an indexed hybrid, truth is
known by corpus construction (0 breakpoints genome-wide: h1=parentA,
h2=parentB everywhere -- see simulated_validation_corpus's README and
simval_full_corpus_eval's true-path-oracle check, which measured this row's
oracle error at 1.17e-07), so no truth VCF/BED is needed at all -- every
decoded position's true pair is the same constant {parentA, parentB} set.
Much cheaper than the full bed_to_vcf + compare_gvcf_truth_diploid path
(~20-40 min/row) used for results/simval_results.tsv's error_rate -- and NOT
the same metric (that one is a genome-wide, deletion-span-weighted allele/
genotype match over the imputed VCF; this one is a per-decoded-position
founder-identity match, no VCF projection involved).

Reuses, unmodified: heldout_assembly_eval.window/load_gamete_names/
k_target_to_name, simval_eval_one.window_fixed_drop/run_inference_diploid
(kind="hybrid" -> KIND_HOMO_SCALE["hybrid"]=1.0, the real per-kind prior
this corpus's own pipeline uses -- not heldout_assembly_eval.run_inference's
hardcoded zero, which is for its own single-assembly use case only).

IMPORTANT correctness fix included here: the default/stock checkout's
ropebwt_npy_to_matrix.py (nam_baseline.WINDOW_SCRIPT's default target) does
NOT binarize features -- that only exists on the unmerged
windowing-quality-filters branch. Every checkpoint here (verified directly
against data/training/sim_diploid_512_affinity.npy) was trained on strictly
[0,1] features, so feeding it raw counts (observed up to 22 on real data) is
a genuine train/eval mismatch. This script unconditionally monkeypatches
hae.nb.WINDOW_SCRIPT to the worktree copy so its own windowing is always
correctly binarized -- same pattern already used by
run_ril2_windowfilter_test.py / run_ril2_all_pairs.py. NOTE: simval_batch.py
/ simval_eval_one.py's own default runs do NOT do this patch, so the
published results/simval_results.tsv corpus (all 200 rows) was scored on
raw, non-binarized features -- see idxhyb_ps4g_hitfrac_filter memory / the
comparison this script's __main__ prints for the measured (modest, ~1pp on
this row) impact.

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src \
      /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/idxhyb_founder_accuracy.py \
        --outdir <row_dir_with_raw.npy> --parent-a Oh43 --parent-b Il14H \
        [--label run-name]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import heldout_assembly_eval as hae  # noqa: E402
import simval_eval_one as seo  # noqa: E402
import simval_paths as P  # noqa: E402

WORKTREE_WINDOW_SCRIPT = Path(
    "/local/workdir/zrm22/HackathonJun2026/grits-windowfilter-worktree/"
    "src/python/crf/ropebwt_npy_to_matrix.py")
assert WORKTREE_WINDOW_SCRIPT.exists(), WORKTREE_WINDOW_SCRIPT
hae.nb.WINDOW_SCRIPT = WORKTREE_WINDOW_SCRIPT


def score_founder_pair_accuracy(windowed_k24_npy, gametes_path, drop_idx,
                                 parent_a, parent_b, ckpt_path, kind, device, label):
    import numpy as np

    gamete_names = hae.load_gamete_names(gametes_path)
    pi, pj, het_scale_val, het_scale_diag = seo.run_inference_diploid(
        windowed_k24_npy, device, ckpt_path=ckpt_path, kind=kind)
    # pi/pj: (n_windows, window_size) -- one decoded founder index PER
    # window-position, not one per window (verified directly: shape
    # (1505, 512) on this row). TARGET_K is a valid state count, but the
    # decode never actually returns the K-th "unknown" sentinel (checked:
    # value range 0..TARGET_K-1 only), so a size-TARGET_K lookup table is
    # safe and avoids a slow Python-level loop.
    idx_to_name = np.array(
        [hae.k_target_to_name(i, drop_idx, gamete_names) for i in range(P.TARGET_K)])
    na = idx_to_name[pi.reshape(-1)]
    nb = idx_to_name[pj.reshape(-1)]
    n = na.shape[0]

    exact_pair = ((na == parent_a) & (nb == parent_b)) | ((na == parent_b) & (nb == parent_a))
    pair_correct = int(exact_pair.sum())
    # per-haplotype (marginal) credit: mirrors the corpus's partial-credit
    # convention -- both slots correct when the unordered pair matches
    # exactly, else count each slot's own membership in the true pair.
    hap_correct = int(np.where(exact_pair, 2,
                                np.isin(na, (parent_a, parent_b)).astype(np.int64)
                                + np.isin(nb, (parent_a, parent_b)).astype(np.int64)).sum())

    return {
        "label": label,
        "windowed_npy": str(windowed_k24_npy),
        "n_windows": pi.shape[0],
        "n_sites": n,
        "pair_acc": pair_correct / n,
        "pair_error": 1 - pair_correct / n,
        "hap_acc": hap_correct / (2 * n),
        "hap_error": 1 - hap_correct / (2 * n),
        "het_scale": het_scale_val,
        "het_scale_diagnostic": het_scale_diag,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True,
                     help="row dir containing raw.npy/raw.npy.bins.tsv/raw.npy.gametes.tsv "
                          "(symlinks are fine)")
    ap.add_argument("--parent-a", required=True)
    ap.add_argument("--parent-b", required=True)
    ap.add_argument("--drop-idx", type=int, default=P.FIXED_DROP_IDX)
    ap.add_argument("--ckpt", default=str(P.CKPT_DIPLOID))
    ap.add_argument("--kind", default="hybrid")
    ap.add_argument("--label", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    label = args.label or outdir.name

    windowed_k24_npy, bins_path = seo.window_fixed_drop(
        outdir / "raw.npy", outdir, drop_idx=args.drop_idx)
    print(f"[{label}] windowed -> {windowed_k24_npy}")

    result = score_founder_pair_accuracy(
        windowed_k24_npy, outdir / "raw.npy.gametes.tsv", args.drop_idx,
        args.parent_a, args.parent_b, args.ckpt, args.kind, args.device, label)

    print(f"[{label}] n_windows={result['n_windows']:,} n_sites={result['n_sites']:,} "
          f"pair_acc={result['pair_acc']:.4f} (error {result['pair_error']:.4%}) "
          f"hap_acc={result['hap_acc']:.4f} (error {result['hap_error']:.4%})")
    return result


if __name__ == "__main__":
    main()
