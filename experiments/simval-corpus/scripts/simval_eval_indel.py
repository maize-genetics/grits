#!/usr/bin/env python
"""
Per-row driver for genome-wide genotype-level scoring of GRITSCRFDiploidIndel
checkpoints (ternary+distance+count, K25 no-drop) -- Verification item 5 for
the depth-confidence-fix / read-count investigation, extended to IDX/OUT/MIX
panel classes (not just in-panel IDX).

Why this exists instead of reusing simval_eval_one.py/heldout_assembly_eval.py
unmodified: both hardcode the OLD binary (non-indel) GRITSCRFDiploid model
class + checkpoint, and an obsolete K25->K24 fixed-drop windowing convention
this project has moved away from. This script reuses everything
model-agnostic from those two (refmap alignment paths/binary constants via
`nb`, hae.write_imputed_bed/bed_to_vcf/bgzip_and_index_vcf/
filter_to_autosomes-equivalent, compare_gvcf_truth_diploid.py) and swaps only
the alignment export mode (refmap --anchor-dist-npy, the
lift-ridx-ternary-dist-map branch binary -- NOT nb.BIN, which is the older
plain-count-npy binary) and the inference call
(GRITSCRFDiploidIndel.load_from_checkpoint + infer_real_founder_pairs, K=25,
no drop -- the same canonical real-data inference path eval_common.py uses).

Usage:
    simval_eval_indel.py --stage {align,score,all} --sample NAME
        --r1 R1.fastq.gz --r2 R2.fastq.gz
        --truth-h1 H1.g.vcf.gz --truth-h2 H2.g.vcf.gz
        --outdir SCRATCH_DIR --out-json RESULT.json
        --ckpt PATH [--panel-vcf PATH]
        [--dataset-id ID] [--coverage C] [--dataset-class CLASS] [--kind KIND]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import heldout_assembly_eval as hae  # noqa: E402  (bed-write/bed-to-vcf/bgzip, model-agnostic)
import simval_paths as P  # noqa: E402

hae.nb.FMD = P.FMD_V2
hae.nb.LIFT = P.LIFT_V2

# refmap binary with --anchor-dist-npy (ternary+distance+count export) --
# NOT hae.nb.BIN (older plain-count-npy binary). Verified working command
# for this exact flag set: experiments/simulator-indels/results/
# real_anchor_dist_npy_oh43xil14h.md (2026-09-16, IDX-HYB Oh43xIl14H 0.1x).
ANCHORDIST_BIN = Path(
    "/workdir/zrm22/HackathonJun2026/ropebwt_refMap/ropebwt3-phg/.claude/worktrees/"
    "lift-ridx-ternary-dist-map/ropebwt3")
ANCHOR_DIST_THRESH = 2000  # matches the real --lift stride (-s 2000), not the
                            # simulator's indel_anchor_thresh=0 (different unit space)
K = 25
CRF_SRC = str(Path(__file__).parent.parent.parent.parent / "src")


def prep_fastq(r1_gz, r2_gz, out_fastq):
    out_fastq = Path(out_fastq)
    if out_fastq.exists():
        return out_fastq
    tmp = out_fastq.with_suffix(out_fastq.suffix + ".tmp")
    with open(tmp, "wb") as out_f:
        for gz in (r1_gz, r2_gz):
            proc = subprocess.run(["zcat", str(gz)], stdout=out_f, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                raise RuntimeError(f"zcat failed for {gz}: {proc.stderr[-2000:]}")
    tmp.rename(out_fastq)
    return out_fastq


def run_refmap_anchordist(sample, reads_fastq, outdir, threads="20"):
    """refmap --anchor-dist-npy: writes raw.npy as the FLAT [n_bins, 3K+2]
    (count|ternary|distance|gA,gB) export directly -- the windowed
    [N,512,3K+2] training/eval contract is a SEPARATE second stage
    (ropebwt_npy_to_matrix.py --anchor-dist-npy --emit-read-counts, below)."""
    outdir = Path(outdir)
    npy_path = outdir / "raw.npy"
    ps4g_path = outdir / "raw.ps4g"
    tsv_path = outdir / "raw.tsv"
    log_path = outdir / "raw.log"
    labels_path = hae.nb.make_labels_bed("B73", outdir)

    if npy_path.exists() and tsv_path.exists():
        print(f"  [{sample}] anchor-dist refmap output already exists, skipping")
        return npy_path

    cmd = [str(ANCHORDIST_BIN), "refmap", "--ref-prefix=B73", "--max-occ=-1", "-l", "19",
           f"--lift={hae.nb.LIFT}", "-t", threads,
           f"--label-bed={labels_path}", f"--ps4g={ps4g_path}", f"--npy={npy_path}",
           "--anchor-dist-npy", f"--anchor-dist-thresh={ANCHOR_DIST_THRESH}",
           str(hae.nb.FMD), str(reads_fastq)]
    print(f"  [{sample}] running: {' '.join(cmd)}")
    t0 = time.time()
    with open(tsv_path, "w") as out_f:
        proc = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True)
    log_path.write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"anchor-dist refmap failed for {sample}: {proc.stderr[-2000:]}")
    print(f"  [{sample}] anchor-dist refmap done in {time.time() - t0:.1f}s")
    return npy_path


def window_k25_wcount(raw_npy, outdir):
    """ropebwt_npy_to_matrix.py --anchor-dist-npy --emit-read-counts, K25
    native (no --target-num-parents / no drop) -- the exact conversion
    eval_common.py's cached *_wcount.npy real-data files already use for
    IDX samples."""
    outdir = Path(outdir)
    bins_path = outdir / "raw.npy.bins.tsv"
    gametes_path = outdir / "raw.npy.gametes.tsv"
    out_path = outdir / "windowed_k25native_wcount.npy"
    if out_path.exists():
        return out_path, bins_path, gametes_path

    cmd = [sys.executable, str(Path(CRF_SRC) / "python/crf/ropebwt_npy_to_matrix.py"),
           f"--npy={raw_npy}", f"--bins={bins_path}", f"--gametes={gametes_path}",
           "--num-parents", str(K), "--window-size", "512",
           "--anchor-dist-npy", "--emit-read-counts", f"--out={out_path}"]
    print(f"  running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"windowing (K25 wcount) failed: {proc.stderr[-2000:]}")
    return out_path, bins_path, gametes_path


def run_inference_indel(data_path, device, ckpt_path):
    """Canonical real-data inference path (matches eval_common.py exactly) --
    infer_real_founder_pairs already handles homo_scale adaptively/
    per-window internally (the "unified homo_scale fix" this project already
    landed), no manual kind-based override needed unlike the old model."""
    sys.path.insert(0, CRF_SRC)
    import torch
    from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, infer_real_founder_pairs

    model = GRITSCRFDiploidIndel.load_from_checkpoint(
        str(ckpt_path), map_location=device, strict=False).eval().to(device)
    data = np.load(data_path)
    pred_lo, pred_hi = infer_real_founder_pairs(model, data, K, device=device)
    print(f"  windows={pred_lo.shape[0]:,}")

    # Release GPU memory NOW, not at process exit. This process stays alive
    # for the long (~25-45min) CPU-only comparator phase afterward, and
    # PyTorch's CUDA caching allocator does not return memory to the driver
    # on its own -- pred_lo/pred_hi are already plain numpy (.cpu().numpy()
    # inside infer_real_founder_pairs), so nothing downstream references the
    # model/GPU tensors after this point. Without this, N concurrent workers
    # each hold their model's GPU memory for their ENTIRE row lifetime (not
    # just their ~5-25s inference window), oversubscribing the GPU under any
    # real parallelism (confirmed: CUDA OOM under 10-way parallelism here).
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return pred_lo, pred_hi  # each [N_windows, T], full K25 index space, no drop


def filter_to_autosomes(vcf_path, out_path):
    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    chroms = ",".join(f"chr{i}" for i in range(1, 11))
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    proc = subprocess.run(["bcftools", "view", "-t", chroms, "-o", str(tmp), str(vcf_path)],
                           capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"bcftools view (organelle filter) failed: {proc.stderr[-2000:]}")
    tmp.rename(out_path)
    return out_path


def do_align(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # NOTE: a cache-reuse attempt here (pointing at scratch/simval_eval/'s
    # raw.npy) was tried and REVERTED -- that directory is simval_eval_one.py's
    # own scratch space, populated by hae.run_refmap WITHOUT --anchor-dist-npy
    # (plain [n_rows, K+2] format, confirmed directly: OUT-INBRED CML459's
    # cached raw.npy has 27 cols = K+2 for K=25, not 77 = 3K+2). It does NOT
    # carry the ternary/distance/count blocks this pipeline needs -- always
    # run fresh anchor-dist alignment here.
    fastq = prep_fastq(args.r1, args.r2, outdir / "reads.fastq")
    t_prep = time.time()

    raw_npy = run_refmap_anchordist(args.sample, fastq, outdir, threads=str(args.threads))
    t_refmap = time.time()
    if not args.no_cleanup:
        fastq.unlink(missing_ok=True)

    windowed_npy, bins_path, gametes_path = window_k25_wcount(raw_npy, outdir)
    t_window = time.time()

    gamete_names = hae.load_gamete_names(gametes_path)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_lo, pred_hi = run_inference_indel(windowed_npy, device, ckpt_path=args.ckpt)
    t_infer = time.time()

    bed_dir = outdir / "bed"
    hae.write_imputed_bed(args.sample, pred_lo, pred_hi, None, gamete_names,
                           bins_path, bed_dir, bin_size=256)
    t_bed = time.time()

    timings = {
        "prep_fastq_s": round(t_prep - t0, 2), "refmap_s": round(t_refmap - t_prep, 2),
        "window_s": round(t_window - t_refmap, 2), "inference_s": round(t_infer - t_window, 2),
        "write_bed_s": round(t_bed - t_infer, 2), "align_total_s": round(t_bed - t0, 2),
    }
    (outdir / "align_timings.json").write_text(json.dumps(timings, indent=1))
    print(f"[{args.sample}] align stage done in {t_bed - t0:.1f}s")
    return timings


def do_score(args):
    outdir = Path(args.outdir)
    bed_dir = outdir / "bed"

    cached_path = outdir / "score_result.json"
    if cached_path.exists():
        cached = json.loads(cached_path.read_text())
        print(f"[{args.sample}] score stage already done, reusing {cached_path} "
              f"(error_rate={cached.get('error_rate')})")
        return cached

    t0 = time.time()
    imputed_vcf = outdir / f"{args.sample}_imputed.vcf"
    filtered_vcf = outdir / f"{args.sample}_imputed.autosomes.vcf"
    imputed_vcf_gz = filtered_vcf.with_suffix(filtered_vcf.suffix + ".gz")

    if not imputed_vcf_gz.exists():
        if not filtered_vcf.exists():
            if not imputed_vcf.exists():
                hae.bed_to_vcf(bed_dir, args.panel_vcf, imputed_vcf)
            filter_to_autosomes(imputed_vcf, filtered_vcf)
            if not args.no_cleanup:
                imputed_vcf.unlink(missing_ok=True)
        imputed_vcf_gz = hae.bgzip_and_index_vcf(filtered_vcf)
        if not args.no_cleanup:
            filtered_vcf.unlink(missing_ok=True)
    t_bedvcf = time.time()

    sys.path.insert(0, CRF_SRC)
    import compare_gvcf_truth_diploid as cgtd
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw = str(td / "imputed.raw.tsv")
        indexed = (Path(str(imputed_vcf_gz) + ".tbi").exists())
        if indexed:
            tsv_for_compare = raw
            cgtd.write_query_tsv(str(imputed_vcf_gz), raw, args.sample, None)
        else:
            sorted_tsv = str(td / "imputed.sorted.tsv")
            cgtd.write_query_tsv(str(imputed_vcf_gz), raw, args.sample, None)
            cgtd.sort_tsv(raw, sorted_tsv)
            tsv_for_compare = sorted_tsv
        counts = cgtd.compare_diploid(tsv_for_compare, args.truth_h1, args.truth_h2,
                                       phase_sensitive=False, partial_credit=True,
                                       class_breakdown=True, snp_refcall_metrics=True)
    t_compare = time.time()

    compared = counts.get("compared_sites", 0)
    error_rate = (1.0 - counts["gt_allele_matches"] / compared) if compared else None
    partial_error_rate = (1.0 - counts["partial_credit_sum"] / compared) if compared else None
    snprc_compared = counts.get("snprc_compared_sites", 0)
    snprc_error_rate = (1.0 - counts["snprc_gt_allele_matches"] / snprc_compared
                         ) if snprc_compared else None

    result = {
        "error_rate": error_rate, "partial_error_rate": partial_error_rate,
        "compared_sites": compared, "snprc_compared_sites": snprc_compared,
        "snprc_error_rate": snprc_error_rate,
        "snprc_partial_credit_sum": counts.get("snprc_partial_credit_sum"),
        "excluded_no_info": counts.get("excluded_no_info", 0),
        "imputed_records": counts.get("imputed_records", 0),
        "gt_allele_matches": counts.get("gt_allele_matches", 0),
        "gt_allele_mismatches": counts.get("gt_allele_mismatches", 0),
        "bed_to_vcf_s": round(t_bedvcf - t0, 2), "compare_s": round(t_compare - t_bedvcf, 2),
        "score_total_s": round(t_compare - t0, 2),
    }
    (outdir / "score_result.json").write_text(json.dumps(result, indent=1))
    print(f"[{args.sample}] score stage done in {t_compare - t0:.1f}s, error_rate={error_rate}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["align", "score", "all"], required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--r1")
    ap.add_argument("--r2")
    ap.add_argument("--truth-h1")
    ap.add_argument("--truth-h2")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--no-cleanup", action="store_true")
    ap.add_argument("--panel-vcf", default=str(P.PANEL_VCF_V2))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset-id", default=None)
    ap.add_argument("--coverage", default=None)
    ap.add_argument("--dataset-class", default=None)
    ap.add_argument("--kind", default=None)
    args = ap.parse_args()

    t_wall0 = time.time()
    out = {"sample": args.sample, "dataset_id": args.dataset_id, "coverage": args.coverage,
           "dataset_class": args.dataset_class, "kind": args.kind}

    if args.stage in ("align", "all"):
        if not (args.r1 and args.r2):
            raise SystemExit("--r1/--r2 required for align stage")
        out["align"] = do_align(args)

    if args.stage in ("score", "all"):
        if not (args.truth_h1 and args.truth_h2):
            raise SystemExit("--truth-h1/--truth-h2 required for score stage")
        out["score"] = do_score(args)

    out["wall_seconds"] = round(time.time() - t_wall0, 2)
    Path(args.out_json).write_text(json.dumps(out, indent=1))
    print(f"[{args.sample}] wrote {args.out_json}")


if __name__ == "__main__":
    main()
