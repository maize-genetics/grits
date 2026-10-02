#!/usr/bin/env python
"""
True-path oracle for the full-corpus affinity-model evaluation (Diagnostic 1
of /home/zrm22/.claude/plans/swift-chasing-melody.md's chain-error
investigation). Writes GROUND-TRUTH founder-path BEDs -- no alignment, no
model inference -- and runs them through the exact same bed_to_vcf ->
dual-cursor-comparator chain the real model output goes through
(simval_eval_one.py --stage score, completely unmodified). Any error that
appears is chain error (merge-gvcfs/BedToVcf.kt/comparator) by construction,
not model error, since the "prediction" fed in IS the truth.

Three dataset kinds, chr1 only (--region chr1 on the comparator; the BEDs
themselves cover the whole genome so bed-to-vcf sees complete input):

  IDX-INBRED  -- identity control: parent1=parent2=sample, genome-wide.
                 Must reproduce ~0% (same path the real batch already
                 validated near-perfectly at whole-genome scope).
  IDX-HYB     -- parent1=parentA, parent2=parentB, genome-wide (0
                 breakpoints by this corpus's own design -- truth.py's
                 hybrid_truth is literally the two parent gVCFs unmodified).
                 THE key row: exercises BedToVcf.kt's heterozygous
                 composition and the dual-cursor comparator on genuinely
                 different h1/h2, which IDX-INBRED's h1==h2 path never
                 touches.
  IDX-RIL     -- two variants:
                 "exact"   -- the true mosaic breakpoints, continuous.
                 "wsnap"   -- the true founder at each position the model
                              ACTUALLY predicted at (real bin positions from
                              a completed row's raw.npy.bins.tsv, 2.0x --
                              the largest/most complete window layout
                              available), i.e. the best ANY model could ever
                              score at this window resolution. The gap
                              between "exact" and "wsnap" is pure
                              window-grid coarseness cost, not chain error
                              and not model error.

RIL mosaic reconstruction is EXACT, not approximate: build_read_datasets.py
derives its breakpoint/founder-assignment RNGs from
simlib.seed_for(dataset_id, parent_a_name, parent_b_name, <label>, replicate)
-- crc32-based, so re-deriving the same seeds and calling
mosaic.draw_breakpoints / mosaic.build_haplotype_mosaic here reproduces the
IDENTICAL mosaic used to build that row's own truth gVCF (replicate=0 always
-- confirmed, build_ril_master's only caller never passes a non-default
value). All positions/segments here are B73 REFERENCE coordinates directly
(chrom_lengths from mosaic.load_b73_chrom_lengths(), the same B73-coordinate
space the panel VCF, the real windowed .npy, and the founders' own gVCFs
all already share -- no anchorspro projection needed; that projection is
only for read-simulation FASTA extraction in build_read_datasets.py, which
this oracle skips entirely since we never simulate reads).

Usage:
    simval_oracle_bed.py [--limit N] [--no-cleanup]
"""
import bisect
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import simval_paths as P  # noqa: E402

CORPUS_SCRIPTS = Path(
    "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/scripts"
)
sys.path.insert(0, str(CORPUS_SCRIPTS))
import config as cfg  # noqa: E402
import mosaic  # noqa: E402
import simlib  # noqa: E402

CRF_REPO_ROOT = Path("/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness")
CRF_SRC = CRF_REPO_ROOT / "src"
sys.path.insert(0, str(CRF_SRC))
sys.path.insert(0, str(CRF_REPO_ROOT))  # bed_io/bed.py itself imports "src.python...."
from python.bed_io.bed import output_collapse_bed  # noqa: E402

PY = "/home/zrm22/mambaforge/envs/phg-ml/bin/python"
EVAL_ONE = Path(__file__).parent / "simval_eval_one.py"
WINDOW_SIZE = 512  # heldout_assembly_eval.WINDOW_SIZE -- see write_imputed_bed

ORACLE_ROOT = P.GRITS_WORKDIR / "scratch" / "simval_oracle"
RESULTS_TSV = P.RESULTS_DIR / "simval_oracle_results.tsv"

MANIFEST_ROWS = None  # lazy-loaded


def load_manifest_row(dataset_id, individual, coverage="2.0"):
    """Truth paths don't depend on coverage -- reuse the manifest's own
    truth_h1/truth_h2 rather than re-deriving via config/truth.py, so this
    exercises the SAME truth files the real batch scored against."""
    global MANIFEST_ROWS
    if MANIFEST_ROWS is None:
        import csv
        with open(P.MANIFEST) as f:
            MANIFEST_ROWS = list(csv.DictReader(f, delimiter="\t"))
    for r in MANIFEST_ROWS:
        if r["dataset_id"] == dataset_id and r["individual"] == individual and r["coverage"] == coverage:
            return r
    raise KeyError(f"no manifest row for {dataset_id}/{individual}/{coverage}x")


# ---------------------------------------------------------------------------
# BED construction
# ---------------------------------------------------------------------------

def write_bed_full_genome(sample_or_names, bed_dir):
    """Identity (inbred) or trivial hybrid (0 breakpoints) BED: one full-span
    row per chromosome. `sample_or_names` is either a single name (inbred,
    parent1=parent2=name) or a (name1, name2) pair (hybrid)."""
    chrom_lengths = mosaic.load_b73_chrom_lengths()
    if isinstance(sample_or_names, tuple):
        name1, name2 = sample_or_names
        tag = f"{name1}x{name2}"
    else:
        name1 = name2 = sample_or_names
        tag = name1
    bed_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for chrom, length in chrom_lengths.items():
        df = pd.DataFrame([{"chrom": chrom, "start": 0, "end": length,
                             "parent1": name1, "parent2": name2}])
        out = bed_dir / f"{tag}_{chrom}_imputed.bed"
        output_collapse_bed(df, str(out))
        written.append(out)
    return written


def build_ril_mosaics(dataset_id, parent_a_name, parent_b_name, replicate=0):
    """Exact reproduction of build_read_datasets.build_ril_master's (legacy
    "ril" kind) or build_ril2_master's (true-inbred "ril2" kind) breakpoint/
    founder-assignment RNG derivation (read-simulation parts skipped -- not
    needed here). Dispatches on cfg.DATASETS[dataset_id]["kind"]. Returns
    (mosaic_h1, mosaic_h2, label_to_name):
      - "ril" (legacy, ~50% het): two INDEPENDENT per-haplotype mosaics.
      - "ril2" (true RIL, homozygous): mosaic_h1 IS mosaic_h2 (same object,
        one shared crossover path -- see mosaic.derive_ril2_mosaic).
    Each mosaic is {chrom: [(start, end, 'A'|'B'), ...]} in B73 coordinates.
    label_to_name = {'A': parent_a_name, 'B': parent_b_name} (matches the
    generator's own `label_to_parent = {"A": parent_a, "B": parent_b}`, i.e.
    parent_a is always founder 'A')."""
    import random

    kind = cfg.DATASETS[dataset_id]["kind"]
    label_to_name = {"A": parent_a_name, "B": parent_b_name}

    if kind == "ril2":
        mosaic_h = mosaic.derive_ril2_mosaic(dataset_id, parent_a_name, parent_b_name,
                                              cfg.N_CROSSOVERS_RIL, replicate)
        return mosaic_h, mosaic_h, label_to_name

    chrom_lengths = mosaic.load_b73_chrom_lengths()

    seed_h1 = simlib.seed_for(dataset_id, parent_a_name, parent_b_name, "bp_h1", replicate)
    seed_h2 = simlib.seed_for(dataset_id, parent_a_name, parent_b_name, "bp_h2", replicate)
    bp_h1 = mosaic.draw_breakpoints(cfg.N_BREAKPOINTS, chrom_lengths, random.Random(seed_h1))
    bp_h2 = mosaic.draw_breakpoints(cfg.N_BREAKPOINTS, chrom_lengths, random.Random(seed_h2))

    seed_founder_h1 = simlib.seed_for(dataset_id, parent_a_name, parent_b_name, "founder_h1", replicate)
    seed_founder_h2 = simlib.seed_for(dataset_id, parent_a_name, parent_b_name, "founder_h2", replicate)
    mosaic_h1 = mosaic.build_haplotype_mosaic(bp_h1, chrom_lengths, random.Random(seed_founder_h1))
    mosaic_h2 = mosaic.build_haplotype_mosaic(bp_h2, chrom_lengths, random.Random(seed_founder_h2))

    return mosaic_h1, mosaic_h2, label_to_name


def _founder_at(sorted_segs, starts, pos):
    """sorted_segs: [(start, end, label), ...] sorted by start, covering the
    whole chromosome contiguously. starts: parallel list of segment starts
    (precomputed once per chrom for bisect). Returns the label whose
    [start, end) contains pos."""
    i = bisect.bisect_right(starts, pos) - 1
    return sorted_segs[i][2]


def write_bed_ril_exact(mosaic_h1, mosaic_h2, label_to_name, bed_dir, tag):
    """Merge two independent per-haplotype mosaics into combined
    (parent1, parent2) segments by walking the union of both haplotypes'
    breakpoints per chromosome -- exact, continuous, no window-grid
    resolution loss."""
    bed_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for chrom in mosaic_h1:
        segs1 = mosaic_h1[chrom]
        segs2 = mosaic_h2[chrom]
        starts1 = [s[0] for s in segs1]
        starts2 = [s[0] for s in segs2]
        boundaries = sorted(set([s[0] for s in segs1] + [s[0] for s in segs2] + [segs1[-1][1]]))
        rows = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            lab1 = _founder_at(segs1, starts1, start)
            lab2 = _founder_at(segs2, starts2, start)
            rows.append({"chrom": chrom, "start": start, "end": end,
                         "parent1": label_to_name[lab1], "parent2": label_to_name[lab2]})
        out = bed_dir / f"{tag}_{chrom}_imputed.bed"
        output_collapse_bed(pd.DataFrame(rows), str(out))
        written.append(out)
    return written


def load_contig_layout_positions(bins_path, window_size=WINDOW_SIZE):
    """Same layout heldout_assembly_eval.load_contig_layout computes (real
    per-bin bp positions actually covered by a completed row's own
    alignment), reimplemented locally to avoid importing heldout_assembly_eval
    (which patches nam_baseline.FMD/LIFT as an import side effect -- fine
    to import too, but this keeps the oracle independent of that module
    entirely, matching the plan's framing of an independent check)."""
    import numpy as np
    bins_df = pd.read_csv(bins_path, sep="\t")
    layout = {}
    for contig, idx in bins_df.groupby("contig", sort=False).indices.items():
        idx = np.sort(idx)
        n_windows = len(idx) // window_size
        if n_windows == 0:
            continue
        used = idx[: n_windows * window_size]
        positions = bins_df.loc[used, "bin"].to_numpy() * 256
        layout[contig] = positions
    return layout


def write_bed_ril_windowsnapped(mosaic_h1, mosaic_h2, label_to_name, bins_path, bed_dir, tag,
                                 chrom_scope=("chr1",)):
    """True founder at each REAL bin position the model actually predicted
    at (from a completed row's own raw.npy.bins.tsv), restricted to
    chrom_scope (only chr1 is ever scored via --region, so only chr1 needs
    this treatment) -- chromosomes outside chrom_scope fall back to the
    exact continuous mosaic (unscored filler, just so bed-to-vcf sees
    complete genome coverage)."""
    bed_dir.mkdir(parents=True, exist_ok=True)
    layout = load_contig_layout_positions(bins_path)
    written = []
    for chrom in mosaic_h1:
        segs1 = mosaic_h1[chrom]
        segs2 = mosaic_h2[chrom]
        starts1 = [s[0] for s in segs1]
        starts2 = [s[0] for s in segs2]
        if chrom in chrom_scope and chrom in layout:
            positions = layout[chrom]
            rows = []
            for pos in positions:
                pos = int(pos)
                lab1 = _founder_at(segs1, starts1, pos)
                lab2 = _founder_at(segs2, starts2, pos)
                rows.append({"chrom": chrom, "start": pos, "end": pos + 1,
                             "parent1": label_to_name[lab1], "parent2": label_to_name[lab2]})
            df = pd.DataFrame(rows)
        else:
            boundaries = sorted(set(starts1 + starts2 + [segs1[-1][1]]))
            rows = []
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                lab1 = _founder_at(segs1, starts1, start)
                lab2 = _founder_at(segs2, starts2, start)
                rows.append({"chrom": chrom, "start": start, "end": end,
                             "parent1": label_to_name[lab1], "parent2": label_to_name[lab2]})
            df = pd.DataFrame(rows)
        out = bed_dir / f"{tag}_{chrom}_imputed.bed"
        output_collapse_bed(df, str(out))
        written.append(out)
    return written


# ---------------------------------------------------------------------------
# Scoring (reuses simval_eval_one.py --stage score UNMODIFIED)
# ---------------------------------------------------------------------------

def run_score(sample, outdir, truth_h1, truth_h2, region="chr1"):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(EVAL_ONE), "--stage", "score", "--sample", sample,
           "--outdir", str(outdir), "--out-json", str(outdir / "stage_out.json"),
           "--truth-h1", str(truth_h1), "--truth-h2", str(truth_h2),
           "--panel-vcf", str(P.PANEL_VCF_V2), "--region", region]
    log_path = outdir / "score.log"
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if proc.returncode != 0:
        return {"status": "failed", "returncode": proc.returncode, "log": str(log_path)}
    score = json.loads((outdir / "score_result.json").read_text())
    return {"status": "ok", "wall_seconds": round(dt, 2), **score}


# ---------------------------------------------------------------------------
# Rows to run
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--region", default="chr1")
    ap.add_argument("--parallel-score", type=int, default=6,
                     help="concurrent scoring subprocesses (each streams the 13GB panel "
                          "VCF through bed-to-vcf + runs the comparator) -- BED writing "
                          "itself stays serial, it's cheap")
    ap.add_argument("--skip-existing", action="store_true",
                     help="skip a job if its score_result.json already exists")
    args = ap.parse_args()

    rows_out = []
    rows_lock = threading.Lock()

    def emit(dataset_id, individual, label, sample, outdir, truth_h1, truth_h2):
        print(f"=== {dataset_id}/{individual}/{label} (sample={sample}) ===", flush=True)
        if args.skip_existing and (outdir / "score_result.json").exists():
            score = json.loads((outdir / "score_result.json").read_text())
            row = {"dataset_id": dataset_id, "individual": individual, "variant": label,
                   "row_key": f"{dataset_id}__{individual}__{label}", "status": "cached", **score}
        else:
            res = run_score(sample, outdir, truth_h1, truth_h2, region=args.region)
            row = {"dataset_id": dataset_id, "individual": individual, "variant": label,
                   "row_key": f"{dataset_id}__{individual}__{label}", **res}
        with rows_lock:
            rows_out.append(row)
        print(f"  -> {row}", flush=True)
        return row

    jobs = []

    # --- IDX-INBRED: identity control ---
    for sample in cfg.IDX_SAMPLES:
        jobs.append(("IDX-INBRED", sample, "identity"))

    # --- IDX-HYB: trivial (0 breakpoints, genome-wide) ---
    for a, b in cfg.IDX_PAIRS:
        jobs.append(("IDX-HYB", f"{a}x{b}", "hybrid"))

    # --- IDX-RIL: exact + window-snapped ---
    for a, b in cfg.IDX_PAIRS:
        jobs.append(("IDX-RIL", f"{a}x{b}", "exact"))
        jobs.append(("IDX-RIL", f"{a}x{b}", "wsnap"))

    if args.limit:
        jobs = jobs[: args.limit]

    # --- Phase 1: write all BEDs serially (cheap, no subprocess) ---
    score_jobs = []
    for dataset_id, individual, label in jobs:
        mrow = load_manifest_row(dataset_id, individual)
        truth_h1, truth_h2 = mrow["truth_h1"], mrow["truth_h2"]
        outdir = ORACLE_ROOT / f"{dataset_id}__{individual}__{label}"
        bed_dir = outdir / "bed"

        if args.skip_existing and (outdir / "score_result.json").exists():
            score_jobs.append((dataset_id, individual, label, individual, outdir, truth_h1, truth_h2))
            continue

        if dataset_id == "IDX-INBRED":
            write_bed_full_genome(individual, bed_dir)

        elif dataset_id == "IDX-HYB":
            a, b = individual.split("x")
            write_bed_full_genome((a, b), bed_dir)

        elif dataset_id == "IDX-RIL":
            a, b = individual.split("x")
            mosaic_h1, mosaic_h2, label_to_name = build_ril_mosaics("IDX-RIL", a, b)
            if label == "exact":
                write_bed_ril_exact(mosaic_h1, mosaic_h2, label_to_name, bed_dir, individual)
            else:  # wsnap
                bins_path = (P.SCRATCH_ROOT / f"IDX-RIL__{individual}__2.0x" / "raw.npy.bins.tsv")
                if not bins_path.exists():
                    print(f"  SKIP wsnap for {individual}: no bins.tsv at {bins_path}")
                    rows_out.append({"dataset_id": dataset_id, "individual": individual,
                                      "variant": label, "status": "skipped_no_bins"})
                    continue
                write_bed_ril_windowsnapped(mosaic_h1, mosaic_h2, label_to_name, bins_path,
                                             bed_dir, individual, chrom_scope=(args.region,))
        print(f"  wrote BED for {dataset_id}/{individual}/{label}", flush=True)
        score_jobs.append((dataset_id, individual, label, individual, outdir, truth_h1, truth_h2))

    # --- Phase 2: score in parallel (each is its own subprocess: JVM bed-to-vcf
    # + Python comparator, streaming the shared 13GB panel VCF -- OS page cache
    # makes repeated reads across concurrent jobs cheap after the first) ---
    print(f"\nScoring {len(score_jobs)} jobs with --parallel-score={args.parallel_score} ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel_score) as pool:
        futs = [pool.submit(emit, *job) for job in score_jobs]
        for fut in as_completed(futs):
            fut.result()  # surface any exception immediately

    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows_out)
    df.to_csv(RESULTS_TSV, sep="\t", index=False)
    print(f"\nWrote {len(df)} rows -> {RESULTS_TSV}")
    if "error_rate" in df.columns:
        print(df[["dataset_id", "individual", "variant", "error_rate", "compared_sites"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
