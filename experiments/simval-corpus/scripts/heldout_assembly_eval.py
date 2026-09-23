#!/usr/bin/env python
"""
Held-out-assembly evaluation: manufacture ground truth for a maize assembly
NOT in our ropebwt3 index by simulating reads from it, running it through the
same refmap -> window -> CRF+affinity pipeline as nam_baseline.py/
nam_diploid.py, converting the decoded founder path into an imputed VCF
(`Sample bed-to-vcf`), and scoring it against the assembly's own truth gVCF
(`build_truth_gvcf.sh`, biokotlin-tools) with the gVCF-aware comparator
(`vcf_eval/compare_gvcf_truth.py`) -- a real per-site error rate for a
genuinely out-of-index sample, not just in-panel accuracy.

CAUTION -- this only measures true generalization if ASSEMBLY_FASTA is
genuinely excluded from the ropebwt3 index this script points at (nb.FMD /
nb.LIFT, our own rebuilt 25-founder index -- NOT smm477's 24-founder one, see
the plan doc's "Founder-side data" correction). Confirm the index's keyfile
(ropebwt_refMap/keyfile_fastas.txt) does not include your sample before
trusting the result; this script does not verify that for you.

Reused, unmodified: nam_baseline.py's BIN/LIFT/FMD/WINDOW_SCRIPT/THREADS,
make_labels_bed(); crf/ropebwt_npy_to_matrix.py (same window()-wrapper pattern
as nam_diploid.py, including its stdout-regex dropped_idx capture for the
K25->K24 trim); crf/train_diploid.py's GRITSCRFDiploid/make_diploid_splits/
_founder_affinity/_dcrf_viterbi; bed_io/bed.py's output_collapse_bed; the
`sample bed-to-vcf` Kotlin CLI; scripts/build_truth_gvcf.sh;
vcf_eval/compare_gvcf_truth.py.

New here (the genuinely new wiring this plan called for): simulate_reads()
for an arbitrary assembly FASTA (tripsacum_diploid.py's wgsim convention,
generalized off its species-specific assembly_path() lookup); pure decode-only
inference (no label-based accuracy -- there IS no true-founder label for a
held-out sample, that's the whole point); and the windowed-model-output ->
per-contig BED bridge (load_contig_layout/write_imputed_bed), which nothing
in this codebase already does -- nam_diploid.py never converts its windowed
predictions to BED at all, it stays in windowed-matrix space for its
label-based accuracy computation.

Usage:
    python heldout_assembly_eval.py <assembly.fa> <sample_name> [--depth N]
        [--panel-vcf PATH] [--ckpt PATH] [--truth-gvcf PATH]

--truth-gvcf lets you skip the AnchorWave alignment here and reuse an
already-built truth gVCF for this exact assembly (e.g. a real founder that
already has one from elsewhere in the pipeline) -- see
scripts/nam_inpanel_smoketest.py for the main use case.
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../refmap-founder-eval/scripts"))  # nam_baseline.py lives in the refmap-founder-eval experiment
import nam_baseline as nb  # noqa: E402  (reuse BIN/LIFT/FMD/WINDOW_SCRIPT/THREADS/make_labels_bed)

# This codebase has two inconsistent import-root conventions across its own
# files: crf/*.py imports as `python.crf...` (needs .../src on sys.path),
# while bed_io/*.py imports as `src.python.bed_io...` (needs .../src's PARENT,
# i.e. the repo root, on sys.path). Both are needed here since this script
# uses both. See CRF_SRC below.

WGSIM = Path("/programs/samtools-1.20/bin/wgsim")  # matches tripsacum_diploid.py's WGSIM
AFFINITY_CKPT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/checkpoints/"
                      "diploid-affinity-sim512-h3/d-epoch=04-val_pair_acc=0.6179.ckpt")
PANEL_VCF_DEFAULT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/"
                          "maize_panel_vcf/panel_25founders.vcf")
SAMPLE_BIN = Path("/local/workdir/zrm22/HackathonJun2026/DebugSim/sample-1.0-SNAPSHOT/bin/sample")
BUILD_TRUTH_GVCF_SH = Path(__file__).parent / "build_truth_gvcf.sh"
COMPARE_SCRIPT = Path("/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/"
                       "src/python/vcf_eval/compare_gvcf_truth.py")
CRF_REPO_ROOT = Path("/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness")
CRF_SRC = CRF_REPO_ROOT / "src"
sys.path.insert(0, str(CRF_SRC))         # for `python.crf...` (train_diploid, etc.)
sys.path.insert(0, str(CRF_REPO_ROOT))   # for `src.python.bed_io...` (bed.py)

SCRATCH = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/heldout_assembly_eval")
RESULTS_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results")

PY = sys.executable
THREADS = nb.THREADS
DEFAULT_DEPTH = 1_000_000
WINDOW_SIZE = 512
SOURCE_K = 25   # our rebuilt index's founder count (includes B73)
TARGET_K = 24   # the diploid-affinity checkpoint's trained founder count


def simulate_reads(assembly_fasta, sample, depth, outdir):
    """wgsim-simulate `depth` read pairs from the assembly FASTA; R1 only
    (matches tripsacum_diploid.py's convention -- refmap has no paired-end
    logic). -r0/-R0 = no injected mutations, so truth = the assembly itself.
    Deterministic seed for cache/resumability."""
    outdir.mkdir(parents=True, exist_ok=True)
    r1 = outdir / f"{sample}_{depth}.R1.fastq"
    if r1.exists():
        return r1
    r1_tmp = outdir / f"{sample}_{depth}.R1.fastq.tmp"
    r2_tmp = outdir / f"{sample}_{depth}.R2.fastq.tmp"
    seed = abs(hash(sample)) % (2 ** 31)
    cmd = [str(WGSIM), "-e", "0.001", "-r", "0", "-R", "0",
           "-1", "100", "-2", "100", "-N", str(depth), "-S", str(seed),
           str(assembly_fasta), str(r1_tmp), str(r2_tmp)]
    print(f"  [{sample}] simulating {depth:,} read pairs: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    r1_tmp.rename(r1)
    r2_tmp.unlink(missing_ok=True)
    return r1


#: ropebwt3 built from the `pav-ps4g-insertion-rows` branch (chain --ps4g/--npy
#: wiring + --insertion-rows), in its own worktree off the same real checkout
#: nb.BIN lives under -- see /home/zrm22/.claude/plans/the-next-task-i-humming-
#: lobster.md. Built, not the ephemeral job-tmp clone, so it survives past any
#: one session; the checkout's pre-existing dirty tree and other worktrees are
#: untouched (git worktree add is additive).
CHAIN_BIN = nb.REFMAP_ROOT / "ropebwt3-phg/.claude/worktrees/pav-ps4g-insertion-rows/ropebwt3"

#: arm -> extra ropebwt3 `chain` CLI args, appended after `--ref-prefix=B73`.
#: "refmap" isn't here -- it's the separate, unchanged default code path below.
#: `--max-occ` (refmap's occurrence cap) is deliberately absent for every chain
#: arm: confirmed by reading search.c that it's consumed only by
#: refmap_anchor_flank/refmap_query, never by chain's own SMEM path (which
#: auto-scales --chain-max-occ instead) -- passing it would parse but do nothing.
CHAIN_ARMS = {
    "chain_nolift": [],
    "chain_ordinary_5000": ["--lift={lift}", "--insertion-rows=ordinary", "--pav-agree=5000"],
    "chain_ordinary_256": ["--lift={lift}", "--insertion-rows=ordinary", "--pav-agree=256"],
    # --max-intron=1 variants: requested to have a real, documented full-pipeline
    # accuracy result on record. Row-count/gamete-set-cardinality checks on a 2M-
    # read sample already showed --max-intron=1 (and --gap-intron=1 stacked on
    # top) change chain's output by <0.2% either way (648,208 -> 649,152 rows,
    # gameteSet-cardinality mean 14.33 -> 14.38) -- expected, since these gap/
    # intron knobs only affect multi-SMEM chaining, and a 100bp WGS read with no
    # splicing almost always forms a single maximal SMEM with nothing to gap
    # over. Kept as separate named arms (not a CLI toggle) so both the default-
    # and max-intron=1 results exist side by side for comparison.
    "chain_nolift_maxintron1": ["--max-intron=1"],
    "chain_ordinary_5000_maxintron1": ["--lift={lift}", "--insertion-rows=ordinary",
                                        "--pav-agree=5000", "--max-intron=1"],
    "chain_ordinary_256_maxintron1": ["--lift={lift}", "--insertion-rows=ordinary",
                                       "--pav-agree=256", "--max-intron=1"],
}


def run_refmap(sample, reads_fastq, outdir, arm="refmap", bin_size=None):
    """Same ropebwt3 refmap invocation as nam_baseline.py/nam_diploid.py, on a
    single held-out sample's simulated reads. --label-bed's founder name is
    irrelevant -- this sample is not a founder, so there is no true-founder
    label to compare against; any valid panel name satisfies the flag.

    arm="refmap" (default) is byte-identical to every call site that predates
    this parameter -- no other caller passes arm=, so behavior for
    nam_baseline.py-style callers is unchanged. arm="chain_*" instead runs
    `ropebwt3 chain` from CHAIN_BIN with CHAIN_ARMS[arm]'s extra flags; see
    that dict's comment for why --max-occ is dropped. Filenames stay fixed
    (raw.npy/raw.ps4g/raw.tsv/raw.log) regardless of arm -- callers running
    more than one arm on the same sample must pass a distinct `outdir` per
    arm, the same way they already must for distinct samples.

    bin_size=None (default) omits --bin-size entirely, so refmap uses its
    own built-in default (256) -- byte-identical to every call site that
    predates this parameter. Pass an explicit int (e.g. 1) to change the
    PS4G/npy position bin width; callers doing so MUST also pass the same
    bin_size through to load_contig_layout/write_imputed_bed below, or
    decoded BED coordinates will silently be wrong by (256/bin_size)x."""
    npy_path = outdir / "raw.npy"
    ps4g_path = outdir / "raw.ps4g"
    tsv_path = outdir / "raw.tsv"
    log_path = outdir / "raw.log"
    labels_path = nb.make_labels_bed("B73", outdir)

    if npy_path.exists() and tsv_path.exists():
        print(f"  [{sample}] {arm} output already exists, skipping")
        return npy_path

    if arm == "refmap":
        cmd = [str(nb.BIN), "refmap", "--ref-prefix=B73", "--max-occ=-1",
               f"--lift={nb.LIFT}", "-t", THREADS,
               f"--label-bed={labels_path}", f"--ps4g={ps4g_path}", f"--npy={npy_path}",
               str(nb.FMD), str(reads_fastq)]
        if bin_size is not None:
            cmd.append(f"--bin-size={bin_size}")
    elif arm in CHAIN_ARMS:
        extra = [a.format(lift=nb.LIFT) for a in CHAIN_ARMS[arm]]
        cmd = [str(CHAIN_BIN), "chain", "--ref-prefix=B73", *extra, "-t", THREADS,
               f"--label-bed={labels_path}", f"--ps4g={ps4g_path}", f"--npy={npy_path}",
               str(nb.FMD), str(reads_fastq)]
    else:
        raise ValueError(f"unknown arm {arm!r}, expected 'refmap' or one of {list(CHAIN_ARMS)}")

    print(f"  [{sample}] running ({arm}): {' '.join(cmd)}")
    t0 = time.time()
    with open(tsv_path, "w") as out_f:
        proc = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True)
    log_path.write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{arm} failed for {sample}: {proc.stderr[-2000:]}")
    print(f"  [{sample}] {arm} done in {time.time() - t0:.1f}s")
    return npy_path


def window(raw_npy, outdir, num_parents=SOURCE_K, target_num_parents=TARGET_K,
           max_hit_frac=None, retain_counts=False):
    """crf/ropebwt_npy_to_matrix.py wrapper -- identical pattern to
    nam_diploid.py's window() (stdout-regex dropped_idx capture included;
    only correct for dropping exactly one founder, which SOURCE_K->TARGET_K
    (25->24) is). Returns (out_path, dropped_idx).

    max_hit_frac / retain_counts are threaded straight through to
    ropebwt_npy_to_matrix.py's own flags of the same name (see that
    module's docstring). retain_counts=False (the default here, matching
    that script's own default) means features are binarized -- this
    changes the output filename (`_bin` suffix) so the existing cache
    check below can never silently hand back a stale count-valued array
    under the old (pre-default-flip) filename; `retain_counts=True`
    reproduces the exact old filename, so anything already on disk from
    before this flag existed stays a valid, reusable cache hit."""
    bins_path = outdir / "raw.npy.bins.tsv"
    gametes_path = outdir / "raw.npy.gametes.tsv"
    suffix = f"_k{target_num_parents}" if target_num_parents != num_parents else f"_k{num_parents}"
    if not retain_counts:
        suffix += "_bin"
    if max_hit_frac is not None:
        suffix += f"_hitfrac{max_hit_frac}"
    out_path = outdir / f"windowed{suffix}.npy"
    sidecar = outdir / f"windowed{suffix}.dropped_idx.txt"

    if out_path.exists():
        dropped_idx = int(sidecar.read_text()) if sidecar.exists() else None
        return out_path, dropped_idx

    cmd = [PY, str(nb.WINDOW_SCRIPT), f"--npy={raw_npy}", f"--bins={bins_path}",
           f"--gametes={gametes_path}", f"--num-parents={num_parents}",
           f"--window-size={WINDOW_SIZE}", f"--out={out_path}"]
    if target_num_parents and target_num_parents != num_parents:
        cmd.append(f"--target-num-parents={target_num_parents}")
    if max_hit_frac is not None:
        cmd.append(f"--max-hit-frac={max_hit_frac}")
    if retain_counts:
        cmd.append("--retain-counts")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"windowing failed: {proc.stderr[-2000:]}")

    dropped_idx = None
    m = re.search(r"^\s*(\S+)\s+idx=(\d+)\s+hits=", proc.stdout, re.MULTILINE)
    if m:
        dropped_idx = int(m.group(2))
        sidecar.write_text(str(dropped_idx))
    return out_path, dropped_idx


def load_gamete_names(gametes_path):
    """Ordered list of founder names indexed by original (pre-trim, SOURCE_K)
    gamete index, from ropebwt_npy_to_matrix.py's --gametes sidecar."""
    df = pd.read_csv(gametes_path, sep="\t")
    return df.sort_values("gameteIndex")["sampleName"].tolist()


def k_target_to_name(idx_target, dropped_idx, gamete_names):
    """Map a predicted TARGET_K-space founder index back to its name in the
    original SOURCE_K space, inverting window()'s single-founder drop."""
    idx_source = idx_target
    if dropped_idx is not None and idx_target >= dropped_idx:
        idx_source += 1
    return gamete_names[idx_source]


def run_inference(data_path, device, ckpt_path=AFFINITY_CKPT, batch_size=128, num_workers=4):
    """Pure decode -- no label-based accuracy (a held-out sample has no true
    founder label to score against; that's the entire premise of this
    script). Mirrors nam_diploid.py's evaluate_diploid_with_affinity for the
    model-forward + Viterbi-decode mechanics, but returns the decoded path
    instead of comparing it to anything.

    homo_scale=0.0 is passed explicitly (not left as forward()'s default
    None). GRITSCRFDiploid.forward(X, homo_scale, ext_emb) scales a FIXED,
    checkpoint-level heterozygous penalty (confirmed on the
    diploid-affinity-sim512-h3 checkpoint: learned_het=False,
    homo_penalty=3.0) by homo_scale; when homo_scale is None the *full*
    penalty applies regardless of the sample's true zygosity. Every sample
    this driver evaluates is a single assembly compared to itself -- zero
    heterozygosity is possible by construction -- so the penalty must be
    fully suppressed here, not left at its outbred-individual default.
    (Found via real-data check: Oh43's imputed BED called heterozygous in
    70-84% of regions despite Oh43 being 100% homozygous truth, with a
    striking single-slot asymmetry -- one haplotype landing on Oh43
    correctly far more than the other, the signature of a het prior
    fighting genuinely homozygous read evidence, not IBD noise.)"""
    import torch
    from torch.utils.data import DataLoader
    from python.crf.train_diploid import (GRITSCRFDiploid, make_diploid_splits,
                                           _founder_affinity, _dcrf_viterbi)

    model = GRITSCRFDiploid.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval().to(device)

    # val_frac=0/test_frac=1 + shuffle=False below -> the "test" split is the
    # whole file in original (window) order, same pattern nam_diploid.py's
    # run_diploid_eval relies on for a single real individual.
    _, _, ds = make_diploid_splits(str(data_path), num_parents=TARGET_K,
                                    val_frac=0.0, test_frac=1.0)
    if len(ds) == 0:
        raise RuntimeError(f"no windows produced from {data_path} -- sample too sparse/short")

    needs_affinity = getattr(model, "founder_affinity", False)
    ext_t = None
    if needs_affinity:
        feats = np.asarray(np.load(data_path))[:, :, :TARGET_K]
        ext_vec = _founder_affinity(feats)  # [K,2], label-free -> safe at inference
        ext_t = torch.tensor(ext_vec, dtype=torch.float32, device=device)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    all_pi, all_pj = [], []
    with torch.no_grad():
        for batch in loader:
            X = batch["input_embeds"].to(device)
            homo_scale = torch.zeros(X.shape[0], device=device)
            if needs_affinity:
                ext_batch = ext_t.unsqueeze(0).expand(X.shape[0], -1, -1)
                emis_p, _, c = model(X, homo_scale, ext_batch)
            else:
                emis_p, _, c = model(X, homo_scale)
            pred = _dcrf_viterbi(emis_p, c, model.nsw_pair, model.stay_bonus)
            all_pi.append(model.pi[pred].cpu().numpy())
            all_pj.append(model.pj[pred].cpu().numpy())
    print(f"  affinity={needs_affinity}  windows={len(ds):,}")
    return np.concatenate(all_pi, axis=0), np.concatenate(all_pj, axis=0)  # each [N_windows, T]


def load_contig_layout(bins_path, window_size=WINDOW_SIZE, bin_size=256):
    """Independently recompute the same per-contig window layout
    ropebwt_npy_to_matrix.py's windowing loop produces (contiguous
    window_size row-chunks per contig, in on-disk order, dropping a trailing
    partial window), so predictions can be mapped back to real (contig, pos)
    without needing anything ropebwt_npy_to_matrix.py doesn't already save.
    Returns [(contig, positions[n_windows*window_size], n_windows), ...] in
    the same contig order the windowed .npy concatenates them.

    bin_size must match whatever --bin-size the refmap run that produced
    bins_path actually used (default 256, refmap's own default) -- this is
    the only place `bin * bin_size` (refPosBinned -> real bp) happens on
    this code path."""
    bins_df = pd.read_csv(bins_path, sep="\t")
    layout = []
    for contig, idx in bins_df.groupby("contig", sort=False).indices.items():
        idx = np.sort(idx)
        n_windows = len(idx) // window_size
        if n_windows == 0:
            continue
        used = idx[: n_windows * window_size]
        positions = bins_df.loc[used, "bin"].to_numpy() * bin_size  # refPosBinned -> real bp
        layout.append((contig, positions, n_windows))
    return layout


def write_imputed_bed(sample, pi_arr, pj_arr, dropped_idx, gamete_names, bins_path, bed_dir,
                       bin_size=256):
    """Bridge from windowed model output back to founder-path BEDs, using
    load_contig_layout for positions instead of re-parsing the PS4G file (we
    already have everything from bins.tsv).

    bin_size must match the refmap run that produced bins_path (see
    load_contig_layout) -- default 256 is refmap's own default, so every
    pre-existing caller is unaffected.

    IMPORTANT: writes ONE FILE PER CONTIG (`{sample}_{contig}_imputed.bed`),
    not one combined file -- BedToVcf.kt's own filename regex
    (`^(.+?)(?:_chr([A-Za-z0-9]+)_imputed)?$`, buildRangeMaps) requires this
    per-contig split to correctly parse sample name vs. contig; confirmed
    against real production output (smm477/phg-maize/bed-files/.../
    B97_chr1_imputed.bed, one file per chromosome). A single combined file
    would silently mis-parse."""
    from src.python.bed_io.bed import output_collapse_bed

    T = pi_arr.shape[1]
    pi_flat = pi_arr.reshape(-1)
    pj_flat = pj_arr.reshape(-1)

    layout = load_contig_layout(bins_path, T, bin_size=bin_size)
    total_layout_windows = sum(n for _, _, n in layout)
    if total_layout_windows != pi_arr.shape[0]:
        raise RuntimeError(
            f"window-count mismatch: model produced {pi_arr.shape[0]} windows, "
            f"bins.tsv layout implies {total_layout_windows} -- inference/windowing "
            f"desynced, do not trust the resulting BED")

    bed_dir.mkdir(parents=True, exist_ok=True)
    cursor = 0
    written = []
    for contig, positions, n_windows in layout:
        n_rows = n_windows * T
        pi_c = pi_flat[cursor: cursor + n_rows]
        pj_c = pj_flat[cursor: cursor + n_rows]
        cursor += n_rows
        # A checkpoint trained with a genuine null/no-visible-lineage-mate
        # target (experiments/depth-confidence-fix/, synthetic leave-one-out
        # held-out augmentation) can legally decode a predicted index >=
        # len(gamete_names) -- the null state -- which k_target_to_name has
        # no real founder name for (unlike every prior checkpoint this
        # pipeline has scored, whose training never incentivized predicting
        # "no call"). Skip those sites rather than crash or fabricate a
        # founder guess; downstream scoring already tolerates BED gaps (the
        # project's existing "no-fill" gVCF convention).
        n_gamete = len(gamete_names)
        rows = []
        n_skipped = 0
        for pos, p1, p2 in zip(positions, pi_c, pj_c):
            p1, p2 = int(p1), int(p2)
            src1 = p1 + 1 if dropped_idx is not None and p1 >= dropped_idx else p1
            src2 = p2 + 1 if dropped_idx is not None and p2 >= dropped_idx else p2
            if src1 >= n_gamete or src2 >= n_gamete:
                n_skipped += 1
                continue
            rows.append({
                "chrom": contig, "start": int(pos), "end": int(pos) + 1,
                "parent1": gamete_names[src1],
                "parent2": gamete_names[src2],
            })
        out_bed = bed_dir / f"{sample}_{contig}_imputed.bed"
        output_collapse_bed(pd.DataFrame(rows), str(out_bed))
        written.append(out_bed)
        if n_skipped:
            print(f"  {contig}: wrote {len(rows):,} sites -> {out_bed} "
                  f"({n_skipped:,} null-state sites skipped)")
        else:
            print(f"  {contig}: wrote {len(rows):,} sites -> {out_bed}")
    return written


def bed_to_vcf(bed_dir, panel_vcf, out_vcf):
    """`sample bed-to-vcf` (BedToVcf.kt) -- composes the founder-path BED with
    the reference panel VCF into the imputed diploid VCF."""
    import os
    env = os.environ.copy()
    env["PATH"] = "/programs/jdk-21.0.1/bin:" + env.get("PATH", "")
    cmd = [str(SAMPLE_BIN), "bed-to-vcf", f"--bed-dir={bed_dir}",
           f"--reference-panel-vcf={panel_vcf}", f"--out-file={out_vcf}"]
    print(f"  running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"bed-to-vcf failed: {proc.stderr[-2000:]}")
    if not out_vcf.exists():
        raise RuntimeError(f"bed-to-vcf reported success but {out_vcf} was not written")
    return out_vcf


def bgzip_and_index_vcf(vcf_path):
    """bgzip-compress + tabix-index the imputed VCF. compare_gvcf_truth.py's
    write_query_tsv() calls `bcftools query -r <region>` whenever --region is
    given, and bcftools requires BGZF compression + an index for random-
    access region queries -- a plain .vcf works fine WITHOUT --region (no
    random access needed), which is exactly why this was never needed before
    --region existed. Always run this (not just when --region is passed) so
    behavior doesn't silently depend on whether --region happens to be set."""
    vcf_path = Path(vcf_path)
    gz_path = vcf_path.with_suffix(vcf_path.suffix + ".gz")
    subprocess.run(["bgzip", "-f", "-k", str(vcf_path)], check=True)
    subprocess.run(["tabix", "-f", "-p", "vcf", str(gz_path)], check=True)
    return gz_path


def build_truth_gvcf(assembly_fasta, outdir):
    cmd = ["bash", str(BUILD_TRUTH_GVCF_SH), str(assembly_fasta), str(outdir)]
    print(f"  running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"build_truth_gvcf.sh failed: {proc.stderr[-3000:]}")
    sample_name = Path(assembly_fasta).stem
    gvcf = outdir / "gvcf" / f"{sample_name}.g.vcf.gz"
    if not gvcf.exists():
        raise RuntimeError(f"expected truth gVCF not found: {gvcf}")
    return gvcf


def run_comparison(imputed_vcf, truth_gvcf, sample, outdir, region=None):
    cmd = [PY, str(COMPARE_SCRIPT), f"--imputed-vcf={imputed_vcf}",
           f"--truth-gvcf={truth_gvcf}", f"--sample={sample}",
           "--truth-ploidy-expand=2", "--partial-credit"]
    if region:
        cmd.append(f"--region={region}")
    env = {"PYTHONPATH": str(CRF_SRC)}
    import os
    env = {**os.environ, **env}
    print(f"  running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    report = outdir / f"{sample}_comparison.txt"
    report.write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"compare_gvcf_truth.py failed: {proc.stderr[-3000:]}")
    print(proc.stdout)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("assembly_fasta")
    ap.add_argument("sample_name")
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--panel-vcf", default=str(PANEL_VCF_DEFAULT))
    ap.add_argument("--ckpt", default=str(AFFINITY_CKPT))
    ap.add_argument("--truth-gvcf", default=None,
                     help="Use an already-built truth gVCF instead of running "
                          "build_truth_gvcf.sh (AnchorWave) here -- e.g. for a "
                          "founder that already has one, so its own alignment "
                          "cost is skipped. Must be the assembly's own truth, "
                          "not derived from a different assembly.")
    ap.add_argument("--region", default=None,
                     help="Restrict the comparison step to one region, e.g. "
                          "chr1 or chr1:1000000-2000000 (passed straight through "
                          "to compare_gvcf_truth.py's own --region). Use this for "
                          "a chromosome-scale dry run against a chr-subset truth "
                          "gVCF -- without it, every non-covered site reads as "
                          "'no info, excluded' instead of being left out of scope.")
    args = ap.parse_args()

    assembly_fasta = Path(args.assembly_fasta).resolve()
    sample = args.sample_name
    outdir = SCRATCH / f"{sample}_{args.depth // 1000}k"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {sample}: simulate reads ===")
    reads = simulate_reads(assembly_fasta, sample, args.depth, outdir)

    print(f"\n=== {sample}: refmap ===")
    raw_npy = run_refmap(sample, reads, outdir)

    print(f"\n=== {sample}: window (K={SOURCE_K}->{TARGET_K}) ===")
    windowed_npy, dropped_idx = window(raw_npy, outdir)
    gamete_names = load_gamete_names(outdir / "raw.npy.gametes.tsv")

    print(f"\n=== {sample}: CRF+affinity inference (decode only) ===")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pi_arr, pj_arr = run_inference(windowed_npy, device, ckpt_path=Path(args.ckpt))

    print(f"\n=== {sample}: path -> BED ===")
    bed_dir = outdir / "bed"
    write_imputed_bed(sample, pi_arr, pj_arr, dropped_idx, gamete_names,
                       outdir / "raw.npy.bins.tsv", bed_dir)

    print(f"\n=== {sample}: BED -> imputed VCF ===")
    imputed_vcf = outdir / f"{sample}_imputed.vcf"
    bed_to_vcf(bed_dir, args.panel_vcf, imputed_vcf)
    imputed_vcf = bgzip_and_index_vcf(imputed_vcf)

    if args.truth_gvcf:
        truth_gvcf = Path(args.truth_gvcf)
        if not truth_gvcf.exists():
            raise RuntimeError(f"--truth-gvcf given but not found: {truth_gvcf}")
        print(f"\n=== {sample}: using provided truth gVCF (skipping AnchorWave): "
              f"{truth_gvcf} ===")
    else:
        print(f"\n=== {sample}: build truth gVCF ===")
        truth_gvcf_dir = outdir / "truth"
        truth_gvcf = build_truth_gvcf(assembly_fasta, truth_gvcf_dir)

    print(f"\n=== {sample}: compare imputed vs. truth ===")
    report = run_comparison(imputed_vcf, truth_gvcf, sample, outdir, region=args.region)

    print(f"\nDone. Report: {report}")


if __name__ == "__main__":
    main()
