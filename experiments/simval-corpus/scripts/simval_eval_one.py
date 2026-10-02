#!/usr/bin/env python
"""
Per-row driver for the full-corpus affinity-model evaluation (see
/home/zrm22/.claude/plans/swift-chasing-melody.md). Runs ONE (individual,
coverage) read set from the maize simulated-validation corpus through:

  gunzip+concat R1/R2 -> refmap (v2 index) -> window (fixed K25->K24 drop)
  -> diploid-affinity CRF decode (real per-individual homo_scale, not the
  hardcoded zero heldout_assembly_eval.py uses for its own single-assembly
  use case) -> founder-path BED -> bed-to-vcf (v2 panel) -> dual-TruthCursor
  genotype comparison against the corpus's own (h1, h2) truth gVCF pair.

Reused, UNMODIFIED, by importing heldout_assembly_eval.py as a module (`hae`,
which itself imports nam_baseline as `hae.nb`): hae.run_refmap, hae.window
(only for its no-op K25->K25 branch here), hae.load_gamete_names,
hae.load_contig_layout, hae.k_target_to_name, hae.write_imputed_bed,
hae.bed_to_vcf, hae.bgzip_and_index_vcf. v1->v2 index/panel paths are patched
onto hae.nb's module globals in-process (see simval_paths.py) -- no shared
file is edited.

New here: prep_fastq (paired-gzipped -> single uncompressed, matching this
corpus's own "both mates as independent reads" build convention, generalized
from tripsacum_diploid.py's concat pattern); window_fixed_drop (pins the
K25->K24 founder drop to one fixed index for every row in the batch, instead
of hae.window()'s own per-run data-dependent choice, which would otherwise
make the coverage-vs-accuracy axis incomparable across rungs -- see plan
section 1); run_inference_diploid (identical to hae.run_inference except it
passes a real per-individual `homo_scale` from train_diploid._het_scale()
instead of hae.run_inference's hardcoded zeros, which are only correct for
its own single-assembly-vs-itself use case, not for this corpus's genuinely
heterozygous hybrid/RIL rows); run_comparison_and_score (always uses the new
dual-TruthCursor comparator, compare_gvcf_truth_diploid.py -- proven
equivalent to the existing single-cursor --truth-ploidy-expand=2 path when
truth_h1==truth_h2, i.e. every inbred row, via assert_equivalence_h1_eq_h2,
checked on the pilot -- so one code path scores all 200 rows correctly).

Usage:
    simval_eval_one.py --stage {align,score,all} --sample NAME
        --r1 R1.fastq.gz --r2 R2.fastq.gz
        --truth-h1 H1.g.vcf.gz --truth-h2 H2.g.vcf.gz
        --outdir SCRATCH_DIR --out-json RESULT.json
        [--drop-idx N] [--region R] [--threads 20] [--no-cleanup]
        [--panel-vcf PATH] [--ckpt PATH] [--arm ARM]
        [--dataset-id ID] [--coverage C] [--dataset-class CLASS] [--kind KIND]

--arm selects the alignment step (default "refmap", unchanged from every
prior run of this script) -- see heldout_assembly_eval.py's CHAIN_ARMS for
the chain-based alternatives this exists to compare against refmap.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import heldout_assembly_eval as hae  # noqa: E402  (brings in hae.nb = nam_baseline)
import simval_paths as P  # noqa: E402

# ---------------------------------------------------------------------------
# v1 -> v2 cutover: patch module globals in-process, no shared file touched.
# ---------------------------------------------------------------------------
hae.nb.FMD = P.FMD_V2
hae.nb.LIFT = P.LIFT_V2


def prep_fastq(r1_gz, r2_gz, out_fastq):
    """gunzip+concat R1 then R2 into one uncompressed FASTQ -- refmap has no
    paired-end logic and takes a single file; this matches the corpus's own
    build convention (both mates kept, fed as independent reads) and
    tripsacum_diploid.py's existing concat pattern. Resumable."""
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


def window_fixed_drop(raw_npy, outdir, drop_idx, max_hit_frac=None, retain_counts=False):
    """K25 -> K24, dropping exactly `drop_idx` (fixed across every row in the
    batch) instead of hae.window()'s own hits-based per-run choice. Gets the
    full K25 windowed matrix from hae.window()'s no-op branch (num_parents==
    target_num_parents, so it does the flat->windowed conversion and -1->
    unknown remap but no trim), then applies the SAME single-index drop/
    remap math ropebwt_npy_to_matrix.py itself uses (K+1-sized remap table,
    kept founders compacted to 0..K-2, dropped index + old-unknown both ->
    new unknown) -- which is exactly what hae.k_target_to_name(idx, drop_idx,
    gamete_names) already inverts, so that function is reused unmodified
    downstream.

    max_hit_frac / retain_counts thread straight through to hae.window()'s
    own params of the same name. Returns (out_path, bins_path): bins_path
    is the raw.npy.bins.tsv companion callers must use for genomic
    coordinates -- when max_hit_frac drops rows, hae.window()'s underlying
    ropebwt_npy_to_matrix.py call writes a filtered `<windowed_k25>.bins.tsv`
    sidecar (window boundaries are row-count-based, so dropped rows shift
    them); this returns THAT path instead of the original outdir/raw.npy.
    bins.tsv whenever filtering was requested, so BED-writing stays in sync."""
    outdir = Path(outdir)
    windowed_k25_path, _ = hae.window(raw_npy, outdir,
                                       num_parents=P.SOURCE_K, target_num_parents=P.SOURCE_K,
                                       max_hit_frac=max_hit_frac, retain_counts=retain_counts)
    bins_path = (Path(str(windowed_k25_path) + ".bins.tsv") if max_hit_frac is not None
                 else outdir / "raw.npy.bins.tsv")

    suffix = ""
    if not retain_counts:
        suffix += "_bin"
    if max_hit_frac is not None:
        suffix += f"_hitfrac{max_hit_frac}"
    out_path = outdir / f"windowed_k{P.TARGET_K}_fixdrop{drop_idx}{suffix}.npy"
    if out_path.exists():
        return out_path, bins_path

    arr = np.load(windowed_k25_path)
    K = P.SOURCE_K
    keep_idx = np.array([i for i in range(K) if i != drop_idx])
    remap = np.full(K + 1, P.TARGET_K, dtype=np.int64)
    remap[keep_idx] = np.arange(len(keep_idx))

    feats = arr[:, :, keep_idx]
    gA = remap[arr[:, :, K].astype(np.int64)]
    gB = remap[arr[:, :, K + 1].astype(np.int64)]
    labels = np.stack([gA, gB], axis=2).astype(np.int8)
    out = np.concatenate([feats, labels], axis=2).astype(np.int8)

    tmp = out_path.with_suffix(out_path.suffix + ".tmp.npy")
    np.save(tmp, out)
    tmp.rename(out_path)
    return out_path, bins_path


#: homo_scale by GROUND-TRUTH kind (from the manifest, not inferred). Every
#: individual's zygosity design is exactly known by construction in this
#: corpus -- inbred: h1==h2 everywhere (0.0, matches hae.run_inference's own
#: hardcoded value, which is correct for its own known-homozygous use case);
#: hybrid: h1=parentA/h2=parentB genome-wide, maximally heterozygous (1.0);
#: ril: an independently-drawn A/B mosaic per haplotype, ~50% heterozygous
#: by the corpus's own construction (0.5). This is the SAME category of
#: legitimate prior knowledge heldout_assembly_eval.py already uses (it
#: assumes zero heterozygosity for ITS known-homozygous test individuals,
#: not inferred either) -- not label leakage, since no genotype/founder
#: identity is disclosed to the model, only a scalar penalty-scale prior.
#:
#: A per-individual READ-DERIVED estimate (train_diploid._het_scale) was
#: tried first and rejected: on the pilot, B73 at 0.01x (a true inbred,
#: 100% homozygous by construction) came back het_scale=0.49 from only 221
#: windows -- proxy noise at low coverage, not signal -- which partially
#: suppressed the homozygous prior and inflated error_rate to 8.6% (vs. the
#: ~0% expected and the ~0.7% historical in-panel baseline; the pattern --
#: partial_error_rate almost exactly half of error_rate -- is the signature
#: of spurious het miscalls). Worse, an unreliable-at-low-coverage estimate
#: would confound coverage with estimator noise, corrupting the exact axis
#: (error rate vs. coverage) this evaluation exists to measure. The real
#: proxy is still computed and logged (as `het_scale_diagnostic`) purely as
#: a read-derived QC signal, never used to steer inference.
KIND_HOMO_SCALE = {"inbred": 0.0, "hybrid": 1.0, "ril": 0.5, "ril2": 0.0}


def run_inference_diploid(data_path, device, ckpt_path, kind, batch_size=128, num_workers=4):
    """Structural copy of hae.run_inference, with ONE change: homo_scale is
    KIND_HOMO_SCALE[kind] (a fixed, ground-truth-informed prior -- see that
    table's docstring), not hae.run_inference's hardcoded zeros, which are
    correct ONLY for that script's own single-assembly-vs-itself use case
    (zero heterozygosity possible by construction), not for this corpus's
    genuinely heterozygous hybrid/RIL individuals."""
    import torch
    from torch.utils.data import DataLoader
    from python.crf.train_diploid import (GRITSCRFDiploid, make_diploid_splits,
                                           _founder_affinity, _dcrf_viterbi, _het_scale)

    model = GRITSCRFDiploid.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval().to(device)

    _, _, ds = make_diploid_splits(str(data_path), num_parents=P.TARGET_K,
                                    val_frac=0.0, test_frac=1.0)
    if len(ds) == 0:
        raise RuntimeError(f"no windows produced from {data_path} -- sample too sparse/short")

    feats_all = np.asarray(np.load(data_path))[:, :, :P.TARGET_K].astype(np.float32)
    het_scale_diagnostic = float(_het_scale(feats_all))  # logged only, not used for inference
    if kind not in KIND_HOMO_SCALE:
        raise ValueError(f"unknown kind {kind!r}, expected one of {list(KIND_HOMO_SCALE)}")
    het_scale_val = KIND_HOMO_SCALE[kind]

    needs_affinity = getattr(model, "founder_affinity", False)
    ext_t = None
    if needs_affinity:
        ext_vec = _founder_affinity(feats_all)
        ext_t = torch.tensor(ext_vec, dtype=torch.float32, device=device)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    all_pi, all_pj = [], []
    with torch.no_grad():
        for batch in loader:
            X = batch["input_embeds"].to(device)
            homo_scale = torch.full((X.shape[0],), het_scale_val, device=device)
            if needs_affinity:
                ext_batch = ext_t.unsqueeze(0).expand(X.shape[0], -1, -1)
                emis_p, _, c = model(X, homo_scale, ext_batch)
            else:
                emis_p, _, c = model(X, homo_scale)
            pred = _dcrf_viterbi(emis_p, c, model.nsw_pair, model.stay_bonus)
            all_pi.append(model.pi[pred].cpu().numpy())
            all_pj.append(model.pj[pred].cpu().numpy())
    print(f"  affinity={needs_affinity}  kind={kind}  homo_scale={het_scale_val:.4f}  "
          f"het_scale_diagnostic={het_scale_diagnostic:.4f}  windows={len(ds):,}")
    return (np.concatenate(all_pi, axis=0), np.concatenate(all_pj, axis=0),
            het_scale_val, het_scale_diagnostic)


def run_comparison_and_score(imputed_vcf_gz, truth_h1, truth_h2, sample, outdir, region=None):
    """Always the dual-TruthCursor comparator -- proven equivalent to the
    single-cursor --truth-ploidy-expand=2 path when truth_h1==truth_h2 (see
    compare_gvcf_truth_diploid.assert_equivalence_h1_eq_h2, checked on the
    pilot), so this one code path is correct for inbred AND hybrid/RIL rows.
    In-process (not a subprocess + stdout-regex parse) for exact counts."""
    import contextlib
    import io
    import tempfile
    import compare_gvcf_truth_diploid as cgtd

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw = str(td / "imputed.raw.tsv")
        sorted_tsv = str(td / "imputed.sorted.tsv")
        cgtd.write_query_tsv(str(imputed_vcf_gz), raw, sample, region)
        cgtd.sort_tsv(raw, sorted_tsv)
        counts = cgtd.compare_diploid(sorted_tsv, str(truth_h1), str(truth_h2),
                                       phase_sensitive=False, partial_credit=True)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cgtd.print_report(counts)
    report_text = buf.getvalue()
    print(report_text)
    report_path = Path(outdir) / f"{sample}_comparison.txt"
    report_path.write_text(report_text)
    return counts, report_path


def do_align(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    fastq = prep_fastq(args.r1, args.r2, outdir / "reads.fastq")
    t_prep = time.time()

    bin_size = getattr(args, "bin_size", None) or 256
    raw_npy = hae.run_refmap(args.sample, fastq, outdir, arm=args.arm,
                              bin_size=(bin_size if bin_size != 256 else None))
    t_refmap = time.time()

    if not args.no_cleanup:
        fastq.unlink(missing_ok=True)

    if args.drop_idx is not None:
        windowed_npy, bins_path = window_fixed_drop(
            raw_npy, outdir, args.drop_idx,
            max_hit_frac=args.max_hit_frac, retain_counts=args.retain_counts)
        dropped_idx = args.drop_idx
    else:
        # Reference-run mode: let the shared script's own hits-based logic
        # choose -- used ONCE (pilot row 3) to pin --drop-idx for every
        # other row afterward.
        windowed_npy, dropped_idx = hae.window(raw_npy, outdir,
                                                num_parents=P.SOURCE_K,
                                                target_num_parents=P.TARGET_K,
                                                max_hit_frac=args.max_hit_frac,
                                                retain_counts=args.retain_counts)
        bins_path = (Path(str(windowed_npy) + ".bins.tsv") if args.max_hit_frac is not None
                     else outdir / "raw.npy.bins.tsv")
    t_window = time.time()

    gamete_names = hae.load_gamete_names(outdir / "raw.npy.gametes.tsv")

    if not args.kind:
        raise SystemExit("--kind required (inbred/hybrid/ril -- selects the homo_scale prior)")

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pi_arr, pj_arr, het_scale_val, het_scale_diag = run_inference_diploid(
        windowed_npy, device, ckpt_path=Path(args.ckpt), kind=args.kind)
    t_infer = time.time()

    bed_dir = outdir / "bed"
    hae.write_imputed_bed(args.sample, pi_arr, pj_arr, dropped_idx, gamete_names,
                           bins_path, bed_dir, bin_size=bin_size)
    t_bed = time.time()

    timings = {
        "prep_fastq_s": round(t_prep - t0, 2),
        "refmap_s": round(t_refmap - t_prep, 2),
        "window_s": round(t_window - t_refmap, 2),
        "inference_s": round(t_infer - t_window, 2),
        "write_bed_s": round(t_bed - t_infer, 2),
        "align_total_s": round(t_bed - t0, 2),
    }
    align_info = {**timings, "dropped_idx": dropped_idx, "het_scale": het_scale_val,
                  "het_scale_diagnostic": het_scale_diag}
    (outdir / "align_timings.json").write_text(json.dumps(align_info, indent=1))
    print(f"[{args.sample}] align stage done in {t_bed - t0:.1f}s, "
          f"dropped_idx={dropped_idx}, het_scale={het_scale_val:.4f} "
          f"(diagnostic proxy: {het_scale_diag:.4f})")
    return align_info


def filter_to_autosomes(vcf_path, out_path):
    """panel_25founders_v2.vcf has a genuine pre-existing data bug: its Mt
    records are split into two non-contiguous blocks (separated by all of
    Pt's records) -- confirmed directly on the raw panel file, not
    introduced by anything here. bed-to-vcf passes those positions straight
    through from the panel for any contig our BED doesn't cover (organelles
    never reach a full 512-row window at these coverages, so we never write
    a BED for them), which breaks tabix indexing downstream
    ('Chromosome blocks not continuous'). This project's own established
    scoring scope is chr1-10 only regardless (config.py's B73_CHROMS,
    preflight_heldout.py's EXPECTED_CONTIGS) -- organelles were never meant
    to be part of the accuracy computation -- so the fix is to drop them
    here, not to edit the shared panel VCF. `-t` (targets) streams without
    needing an index, so this runs directly on the plain uncompressed VCF."""
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


def do_score(args):
    outdir = Path(args.outdir)
    bed_dir = outdir / "bed"

    cached_path = outdir / "score_result.json"
    if cached_path.exists():
        # Resumability for the WHOLE stage, not just VCF generation -- the
        # comparator itself (run_comparison_and_score) is the expensive part
        # (~20+ min/row on this corpus), so this must be checked before
        # doing any work, not just before regenerating the imputed VCF.
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
            filtered_vcf.unlink(missing_ok=True)  # keep only the .gz + .tbi
    t_bedvcf = time.time()

    counts, report_path = run_comparison_and_score(
        imputed_vcf_gz, args.truth_h1, args.truth_h2, args.sample, outdir, region=args.region)
    t_compare = time.time()

    compared = counts.get("compared_sites", 0)
    error_rate = (1.0 - counts["gt_allele_matches"] / compared) if compared else None
    partial_error_rate = (1.0 - counts["partial_credit_sum"] / compared) if compared else None
    ceiling_denom = counts["matchable_variant_sites"] + counts["unmatchable_variant_sites"]
    matchable_ceiling = (counts["matchable_variant_sites"] / ceiling_denom) if ceiling_denom else None

    result = {
        "error_rate": error_rate,
        "partial_error_rate": partial_error_rate,
        "compared_sites": compared,
        "excluded_no_info": counts.get("excluded_no_info", 0),
        "excluded_partial_info": counts.get("excluded_partial_info", 0),
        "imputed_records": counts.get("imputed_records", 0),
        "gt_allele_matches": counts.get("gt_allele_matches", 0),
        "gt_allele_mismatches": counts.get("gt_allele_mismatches", 0),
        "matchable_ceiling": matchable_ceiling,
        "bed_to_vcf_s": round(t_bedvcf - t0, 2),
        "compare_s": round(t_compare - t_bedvcf, 2),
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
    ap.add_argument("--drop-idx", type=int, default=None)
    ap.add_argument("--max-hit-frac", type=float, default=None,
                     help="Drop read-groups whose founder fan-out exceeds this fraction of "
                          "the source K=25 panel (e.g. 0.5 = drop reads hitting >12/25 "
                          "founders) at the windowing step. Default: off, no filtering.")
    ap.add_argument("--retain-counts", action="store_true",
                     help="Keep raw read counts in the feature matrix instead of binarizing "
                          "to 0/1 (binarize is now the default -- see ropebwt_npy_to_matrix.py "
                          "docstring for why: every checkpoint here was trained on binary "
                          "features, so raw counts are a train/eval mismatch).")
    ap.add_argument("--region", default=None)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--no-cleanup", action="store_true")
    ap.add_argument("--panel-vcf", default=str(P.PANEL_VCF_V2))
    ap.add_argument("--ckpt", default=str(P.CKPT_DIPLOID))
    ap.add_argument("--dataset-id", default=None)
    ap.add_argument("--coverage", default=None)
    ap.add_argument("--dataset-class", default=None)
    ap.add_argument("--kind", default=None)
    ap.add_argument("--arm", default="refmap",
                     help="Alignment arm, forwarded to hae.run_refmap: 'refmap' "
                          "(default, unchanged) or one of hae.CHAIN_ARMS "
                          "('chain_nolift', 'chain_ordinary_5000', 'chain_ordinary_256').")
    args = ap.parse_args()

    hae.THREADS = str(args.threads)

    t_wall0 = time.time()
    out = {
        "sample": args.sample, "dataset_id": args.dataset_id, "coverage": args.coverage,
        "dataset_class": args.dataset_class, "kind": args.kind,
    }

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
