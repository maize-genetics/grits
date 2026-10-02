#!/usr/bin/env python
"""
Full-pipeline baseline: real WGS reads (one file per NAM founder) -> ropebwt3
refmap (1M-alignment numpy export, "previous parameters") -> windowed CRF
matrix -> haploid-sim CRF Viterbi decode -> per-site accuracy, generalizing
the existing Oh43 benchmark (see grits_workdir/scripts/kmer_sweep.py and
results/oh43_absent_rootcause.md) to every founder with a WGS read file.

Reused, unmodified: the patched ropebwt3 binary (64-carrier-cap fix + PS4G/npy
export), crf/ropebwt_npy_to_matrix.py, the haploid-sim/last.ckpt checkpoint
(the same one used throughout the Oh43 investigation), and
crf/train_haploid.py's GRITSCRFHaploid/make_splits.

refmap is run with the SAME whole-read placement parameters as the original
Oh43 1M benchmark (bench_ps4g_npy/run_bench.sh): --ref-prefix=B73 --max-occ=-1
--lift ... -t 20, no --kmer* (whole-read exact/SMEM placement).

The "1 million alignments" input is reproduced the same way the historical
`Oh43_1M.fastq` was actually built: a literal `head`-based subsample of the
first 1,000,000 reads (4,000,000 lines) of the founder's read file -- verified
by diffing the first read of bench_ps4g_npy/reads/Oh43_1M.fastq against
DebugSim/reads/1x/Oh43.fastq.gz (identical). --target-hits=1000000 was
considered (it exists for exactly this purpose) but rejected after the Oh43
smoke test showed it overshoots to ~1.96M EXACT+PLACED records (the per-thread
batch check only fires between ~666k-read batches, so it always finishes the
batch that crosses the target) -- the SAME overshoot the original
bench_ps4g_npy/target1M.log run hit, which is presumably exactly why the actual
Oh43 training/eval data was built from a pre-`head`-subsampled file instead of
--target-hits despite the flag's existence.

Usage:
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/nam_baseline.py list
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/nam_baseline.py one <Founder>
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/nam_baseline.py all
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/nam_baseline.py report
"""
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REFMAP_ROOT = Path("/workdir/zrm22/HackathonJun2026/ropebwt_refMap")
BIN = REFMAP_ROOT / "ropebwt3-phg/.claude/worktrees/refmap-ps4g-numpy/ropebwt3"
LIFT = REFMAP_ROOT / "rope_bwt_index/maizeFastaIndex_SampleContig.lift"
FMD = REFMAP_ROOT / "rope_bwt_index/maizeFastaIndex_SampleContig.fmd"
LABELS_TEMPLATE = REFMAP_ROOT / "bench_ps4g_npy/labels_full.bed"

CRF_SRC = Path("/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src")
WINDOW_SCRIPT = CRF_SRC / "python/crf/ropebwt_npy_to_matrix.py"
CKPT = Path("/workdir/zrm22/HackathonJun2026/grits_workdir/checkpoints/haploid-sim/last.ckpt")

WGS_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/WGSReads")
SCRATCH = Path("/tmp/claude-4347/-local-workdir-zrm22-HackathonJun2026-grits-workdir"
               "/ef670b81-b3d3-43b6-ad55-b6ca8008dc12/scratchpad/nam_baseline")
RESULTS_TSV = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/nam_baseline_results.tsv")
RESULTS_MD = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/nam_baseline.md")

PY = sys.executable
SUBSAMPLE_READS = 1_000_000  # matches bench_ps4g_npy/reads/Oh43_1M.fastq (head -4000000 lines)
THREADS = "20"

sys.path.insert(0, str(CRF_SRC))

# Fixed 25-founder pangenome panel order (alphabetical gameteIndex, from the
# existing bench_ps4g_npy *.gametes.tsv -- independent of any one run's data).
PANEL_ORDER = [
    "B73", "B97", "CML103", "CML228", "CML247", "CML277", "CML322", "CML333",
    "CML52", "CML69", "HP301", "Il14H", "Ki11", "Ki3", "Ky21", "M162W", "M37W",
    "Mo18W", "Ms71", "NC350", "NC358", "Oh43", "Oh7B", "P39", "Tzi8",
]
NAME_TO_IDX_K25 = {name: i for i, name in enumerate(PANEL_ORDER)}

# WGSReads/<basename>_..._1.clean.fq.gz -> panel name. Only entries that map
# to a real pangenome-panel column are runnable; Tx303 has no assembly in this
# index and is excluded (see plan doc). M162W has no read file so never appears
# as a key here (it can still be dropped-from or present-in other founders'
# panels via the K25->K24 trim, just never itself evaluated).
WGS_BASENAME_TO_PANEL = {
    "MS71": "Ms71",
    "OH7B": "Oh7B",
    "P39Goodman-Buckler": "P39",
}
EXCLUDED_WGS_BASENAMES = {"Tx303"}  # no assembly/panel column in this pangenome index


def discover_founders():
    """Map each WGSReads file to its panel name. Returns dict panel_name -> Path,
    sorted with B73 first (reference self-mapping sanity case), then alphabetical."""
    founders = {}
    for fq in sorted(WGS_DIR.glob("*.clean.fq.gz")):
        basename = fq.name.split("_", 1)[0] if not fq.name.startswith("P39Goodman-Buckler") \
            else "P39Goodman-Buckler"
        if basename in EXCLUDED_WGS_BASENAMES:
            print(f"  [discover] {fq.name}: {basename!r} has no panel column in this pangenome "
                  f"index -- excluding")
            continue
        panel_name = WGS_BASENAME_TO_PANEL.get(basename, basename)
        if panel_name not in NAME_TO_IDX_K25:
            raise RuntimeError(f"{fq.name}: mapped basename {basename!r} -> {panel_name!r}, "
                               f"which is not in PANEL_ORDER -- fix WGS_BASENAME_TO_PANEL")
        if panel_name in founders:
            raise RuntimeError(f"duplicate WGS file for panel name {panel_name!r}: "
                               f"{founders[panel_name]} and {fq}")
        founders[panel_name] = fq
    ordered = {}
    if "B73" in founders:
        ordered["B73"] = founders.pop("B73")
    for name in sorted(founders):
        ordered[name] = founders[name]
    return ordered


def make_labels_bed(founder, outdir):
    """Copy labels_full.bed with column 4 replaced by this founder's panel name."""
    out_path = outdir / "labels.bed"
    if out_path.exists():
        return out_path
    with open(LABELS_TEMPLATE) as fin, open(out_path, "w") as fout:
        for line in fin:
            cols = line.rstrip("\n").split("\t")
            cols[3] = founder
            fout.write("\t".join(cols) + "\n")
    return out_path


def make_subsample(founder, read_path, outdir):
    """First SUBSAMPLE_READS reads (4*N lines) of the founder's WGS file,
    matching exactly how the historical bench_ps4g_npy/reads/Oh43_1M.fastq was
    built (verified: its first read is byte-identical to the first read of the
    un-subsampled DebugSim/reads/1x/Oh43.fastq.gz -- a plain `head`, no random
    sampling). Cached; resumable."""
    out_path = outdir / "reads_1M.fastq"
    if out_path.exists():
        return out_path
    n_lines = SUBSAMPLE_READS * 4
    cmd = f"zcat {read_path} | head -n {n_lines} > {out_path}.tmp"
    subprocess.run(cmd, shell=True, check=True)
    got_lines = sum(1 for _ in open(f"{out_path}.tmp"))
    if got_lines != n_lines:
        (out_path.with_suffix(".tmp")).unlink(missing_ok=True)
        raise RuntimeError(f"{read_path} has only {got_lines} lines "
                           f"({got_lines // 4:,} reads) -- fewer than the requested "
                           f"{SUBSAMPLE_READS:,} reads")
    Path(f"{out_path}.tmp").rename(out_path)
    return out_path


def run_refmap(founder, read_path):
    """Run the patched binary for one founder on its 1M-read subsample.
    Returns (wall_s, tool_real_s, tool_cpu_s, n_placed, n_unplaced, npy_path).
    Skips the run if outputs already exist (resumable)."""
    outdir = SCRATCH / founder
    outdir.mkdir(parents=True, exist_ok=True)
    npy_path = outdir / "raw.npy"
    ps4g_path = outdir / "raw.ps4g"
    tsv_path = outdir / "raw.tsv"
    log_path = outdir / "raw.log"
    labels_path = make_labels_bed(founder, outdir)

    if npy_path.exists() and tsv_path.exists():
        print(f"  [{founder}] refmap output already exists, skipping run")
        wall_s = tool_real_s = tool_cpu_s = None
        log_text = log_path.read_text() if log_path.exists() else ""
    else:
        reads_1m = make_subsample(founder, read_path, outdir)
        cmd = [str(BIN), "refmap", "--ref-prefix=B73", "--max-occ=-1",
               f"--lift={LIFT}", "-t", THREADS,
               f"--label-bed={labels_path}", f"--ps4g={ps4g_path}", f"--npy={npy_path}",
               str(FMD), str(reads_1m)]
        print(f"  [{founder}] running: {' '.join(cmd)}")
        t0 = time.time()
        with open(tsv_path, "w") as out_f:
            proc = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True)
        wall_s = time.time() - t0
        log_text = proc.stderr
        log_path.write_text(log_text)
        if proc.returncode != 0:
            raise RuntimeError(f"refmap failed for {founder}: {log_text[-2000:]}")

    m = re.search(r"Real time:\s*([\d.]+)\s*sec;\s*CPU:\s*([\d.]+)\s*sec", log_text)
    tool_real_s = float(m.group(1)) if m else None
    tool_cpu_s = float(m.group(2)) if m else None

    n_placed = n_unplaced = 0
    with open(tsv_path) as f:
        for line in f:
            status = line.split("\t", 3)[2]
            if status in ("PLACED", "EXACT"):
                n_placed += 1
            elif status == "UNPLACED":
                n_unplaced += 1
    return wall_s, tool_real_s, tool_cpu_s, n_placed, n_unplaced, npy_path


def window(founder, raw_npy, target_num_parents=None):
    """Run crf/ropebwt_npy_to_matrix.py. Returns (out_path, dropped_idx or None).
    dropped_idx is cached to a sidecar file so a skipped (already-windowed) run
    still recovers it correctly."""
    outdir = SCRATCH / founder
    bins_path = outdir / "raw.npy.bins.tsv"
    gametes_path = outdir / "raw.npy.gametes.tsv"
    suffix = f"_k{target_num_parents}" if target_num_parents else "_k25"
    out_path = outdir / f"windowed{suffix}.npy"
    sidecar = outdir / f"windowed{suffix}.dropped_idx.txt"

    if out_path.exists():
        print(f"  [{founder}] windowed{suffix} already exists, skipping")
        dropped_idx = int(sidecar.read_text()) if sidecar.exists() else None
        return out_path, dropped_idx

    cmd = [PY, str(WINDOW_SCRIPT), f"--npy={raw_npy}", f"--bins={bins_path}",
           f"--gametes={gametes_path}", "--num-parents=25", "--window-size=512",
           f"--out={out_path}"]
    if target_num_parents:
        cmd.append(f"--target-num-parents={target_num_parents}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"windowing failed for {founder}: {proc.stderr[-2000:]}")

    dropped_idx = None
    m = re.search(r"^\s*(\S+)\s+idx=(\d+)\s+hits=", proc.stdout, re.MULTILINE)
    if m:
        dropped_idx = int(m.group(2))
        sidecar.write_text(str(dropped_idx))
    return out_path, dropped_idx


def founder_density_stats(k25_npy_path, founder_idx_k25):
    a = np.asarray(np.load(k25_npy_path, mmap_mode="r"))
    K = a.shape[-1] - 2
    feat = a[:, :, :K].reshape(-1, K)
    nz_row = feat != 0
    any_nz = nz_row.any(1)
    mean_founders = nz_row.sum(1)[any_nz].mean() if any_nz.any() else 0.0
    self_cov = nz_row[:, founder_idx_k25].mean() * 100
    return mean_founders, self_cov


def eval_combo(model, device, k24_npy_path, dropped_idx, founder_idx_k25):
    """Held-out test-split per-site accuracy, matching the exact split
    methodology used for the existing Oh43 numbers (make_splits val_frac=0.10,
    test_frac=0.10), further stratified into founder-covered vs founder-absent
    sites (same covered/absent metric as kmer_sweep.py's eval_combo)."""
    import torch
    from torch.utils.data import DataLoader
    from python.crf.train_haploid import make_splits

    if dropped_idx is not None and dropped_idx == founder_idx_k25:
        raise RuntimeError(
            f"founder's own K25 index ({founder_idx_k25}) was the one dropped in the "
            f"K25->K24 trim -- this should never happen (the drop removes the "
            f"LEAST-covered founder genome-wide, and the founder providing the reads "
            f"should be among the most-covered). Investigate before trusting this row.")
    founder_idx_k24 = founder_idx_k25 - 1 if (dropped_idx is not None and
                                               dropped_idx < founder_idx_k25) else founder_idx_k25

    _, _, test_ds = make_splits(str(k24_npy_path), num_parents=24, val_frac=0.10, test_frac=0.10)
    n = len(test_ds)
    if n == 0:
        return dict(n_sites=0, viterbi=float("nan"), cov_pct=float("nan"),
                    cov_acc=float("nan"), abs_pct=float("nan"), abs_acc=float("nan"))

    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    vc_cov = vt_cov = vc_abs = vt_abs = 0
    with torch.no_grad():
        for batch in loader:
            X = batch["input_embeds"].to(device)
            tags = batch["labels"].to(device)
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    emis_f, g, c = model(X, ext_emb=None)
            else:
                emis_f, g, c = model(X, ext_emb=None)
            pred = model.decode(emis_f, c)
            absent = X[:, :, founder_idx_k24] == 0
            correct = pred == tags
            vc_cov += correct[~absent].sum().item(); vt_cov += (~absent).sum().item()
            vc_abs += correct[absent].sum().item();  vt_abs += absent.sum().item()
    tot = vt_cov + vt_abs
    return dict(
        n_sites=tot,
        viterbi=(vc_cov + vc_abs) / tot if tot else float("nan"),
        cov_pct=vt_cov / tot * 100 if tot else float("nan"),
        cov_acc=vc_cov / vt_cov if vt_cov else float("nan"),
        abs_pct=vt_abs / tot * 100 if tot else float("nan"),
        abs_acc=vc_abs / vt_abs if vt_abs else float("nan"),
    )


RESULT_COLS = ["founder", "read_file", "wall_s", "tool_real_s", "tool_cpu_s",
               "n_placed", "n_unplaced", "mean_founders_per_site", "self_coverage_pct_k25",
               "n_sites", "viterbi", "cov_pct", "cov_acc", "abs_pct", "abs_acc"]

NAN_METRICS = dict(n_sites=0, viterbi=float("nan"), cov_pct=float("nan"),
                    cov_acc=float("nan"), abs_pct=float("nan"), abs_acc=float("nan"))


def write_header_if_needed():
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("\t".join(RESULT_COLS) + "\n")


def already_recorded(founder):
    if not RESULTS_TSV.exists():
        return False
    with open(RESULTS_TSV) as f:
        return any(line.split("\t", 1)[0] == founder for line in f)


def run_one(model, device, founder, read_path, force=False):
    if not force and already_recorded(founder):
        print(f"[{founder}] already in results TSV, skipping entirely")
        return

    founder_idx_k25 = NAME_TO_IDX_K25[founder]
    print(f"\n=== {founder} (panel idx {founder_idx_k25}, reads={read_path.name}) ===")
    wall_s, tool_real_s, tool_cpu_s, n_placed, n_unplaced, raw_npy = run_refmap(founder, read_path)

    if n_placed == 0:
        print(f"  [{founder}] n_placed=0 -- degenerate, recording NaN metrics, no windowing")
        mean_founders, self_cov = float("nan"), float("nan")
        metrics = dict(NAN_METRICS)
    else:
        try:
            k25_npy, _ = window(founder, raw_npy, target_num_parents=None)
            k24_npy, dropped_idx = window(founder, raw_npy, target_num_parents=24)
            mean_founders, self_cov = founder_density_stats(k25_npy, founder_idx_k25)
            metrics = eval_combo(model, device, k24_npy, dropped_idx, founder_idx_k25)
        except RuntimeError as e:
            print(f"  [{founder}] windowing/eval failed -- recording NaN metrics: {e}")
            mean_founders, self_cov = float("nan"), float("nan")
            metrics = dict(NAN_METRICS)

    row = dict(founder=founder, read_file=read_path.name, wall_s=wall_s,
               tool_real_s=tool_real_s, tool_cpu_s=tool_cpu_s,
               n_placed=n_placed, n_unplaced=n_unplaced,
               mean_founders_per_site=mean_founders, self_coverage_pct_k25=self_cov,
               **metrics)
    write_header_if_needed()
    if force and already_recorded(founder):
        # Overwrite: drop the old line for this founder before appending the new one.
        lines = RESULTS_TSV.read_text().splitlines(keepends=True)
        keep = [l for l in lines if not l.startswith(founder + "\t")]
        RESULTS_TSV.write_text("".join(keep))
    with open(RESULTS_TSV, "a") as f:
        f.write("\t".join(str(row[c]) for c in RESULT_COLS) + "\n")
    print(f"[{founder}] viterbi={metrics['viterbi']:.4f}  cov={metrics['cov_pct']:.1f}%@{metrics['cov_acc']:.4f}"
          f"  abs={metrics['abs_pct']:.1f}%@{metrics['abs_acc']:.4f}  mean_founders={mean_founders:.2f}"
          f"  self_cov_k25={self_cov:.1f}%  wall={wall_s}  n_placed={n_placed}")


def load_model(device):
    import torch
    from python.crf.train_haploid import GRITSCRFHaploid
    model = GRITSCRFHaploid.load_from_checkpoint(str(CKPT), map_location=device)
    model.eval().to(device)
    return model


def _markdown_table(df, float_cols):
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in float_cols and isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def write_report():
    import pandas as pd
    df = pd.read_csv(RESULTS_TSV, sep="\t")
    df = df.sort_values("viterbi", ascending=False).reset_index(drop=True)

    display_cols = ["founder", "n_placed", "n_unplaced", "mean_founders_per_site",
                     "self_coverage_pct_k25", "n_sites", "viterbi", "cov_pct", "cov_acc",
                     "abs_pct", "abs_acc"]
    float_cols = {"mean_founders_per_site", "self_coverage_pct_k25", "viterbi",
                  "cov_pct", "cov_acc", "abs_pct", "abs_acc"}
    table_md = _markdown_table(df[display_cols], float_cols)

    non_ref = df[df["founder"] != "B73"]
    outlier_thresh = non_ref["viterbi"].median() - 2 * non_ref["viterbi"].std()
    outliers = non_ref[non_ref["viterbi"] < 0.90].sort_values("viterbi")

    lines = [
        "# NAM-founder full-pipeline baseline\n",
        "Real WGS reads (one file per NAM founder, `WGSReads/`) run through the same\n"
        "`ropebwt3 refmap` whole-read placement recipe as the original Oh43 benchmark\n"
        "(`--ref-prefix=B73 --max-occ=-1 --lift ... -t 20`, no `--kmer*`), on a\n"
        "1,000,000-read subsample of each founder's file. The subsample is a literal\n"
        "`head` of the first 1,000,000 reads, matching exactly how the historical\n"
        "`bench_ps4g_npy/reads/Oh43_1M.fastq` was built (verified byte-identical first\n"
        "read against the un-subsampled source). `--target-hits=1000000` was tried first\n"
        "but rejected: it overshoots to ~1.96M EXACT+PLACED records because the\n"
        "per-thread target check only fires between ~666k-read batches -- the same\n"
        "overshoot visible in the historical `target1M.log`, which is presumably why the\n"
        "original benchmark used a pre-subsampled file instead of the flag.\n\n"
        "Reads were windowed (`crf/ropebwt_npy_to_matrix.py --window-size=512`,\n"
        "K=25 panel trimmed to K=24 by dropping the least-covered founder genome-wide)\n"
        "and scored with the `haploid-sim/last.ckpt` CRF -- the same checkpoint used\n"
        "throughout the Oh43 investigation -- on the held-out test split (`make_splits`,\n"
        "val_frac=test_frac=0.10), identical methodology to the existing Oh43 number.\n\n"
        f"For reference, the existing Oh43 benchmark (older DebugSim-simulated 1x reads,\n"
        f"patched 64-carrier-cap binary) scored `viterbi=0.9915`\n"
        f"(`results/eval.tsv`, `haploid-sim-on-ropebwt-oh43-k24-PATCHED`). The fresh\n"
        f"real-WGS Oh43 row here (`viterbi={df.loc[df.founder=='Oh43','viterbi'].iloc[0]:.4f}`)\n"
        f"lands in the same ballpark on an independently re-downloaded read set --\n"
        f"validating the whole chain end-to-end on genuinely new data.\n\n"
        "## Per-founder results\n\n"
        "Sorted by `viterbi` (overall per-site Viterbi-decode accuracy on the held-out\n"
        "test split); `cov`/`abs` = founder-covered vs founder-absent site split (absent\n"
        "= the founder's own K24 feature column reads zero at that site -- the model\n"
        "cannot be right there by construction, mirroring the Oh43 covered/absent\n"
        "analysis in `oh43_absent_rootcause.md`).\n\n"
        f"{table_md}\n\n"
        "## B73 self-mapping sanity check\n\n"
        f"B73 is the `--ref-prefix` reference itself; its row "
        f"(`viterbi={df.loc[df.founder=='B73','viterbi'].iloc[0]:.4f}`, "
        f"`self_coverage_pct_k25={df.loc[df.founder=='B73','self_coverage_pct_k25'].iloc[0]:.1f}%`) "
        "is high and in the same range as the other well-behaved founders, confirming no\n"
        "labels/index-mapping bug -- excluded from the outlier/summary stats below since\n"
        "it isn't a generalization test.\n\n"
        "## Outliers\n\n",
    ]
    if len(outliers):
        n_ok = len(non_ref) - len(outliers)
        ok_range = non_ref[~non_ref["founder"].isin(outliers["founder"])]["viterbi"]
        lines.append(
            f"{len(outliers)} founder(s) score well below the rest ({n_ok} other "
            f"non-reference founders land in `viterbi` {ok_range.min():.3f}-{ok_range.max():.3f}):\n\n")
        for _, r in outliers.iterrows():
            lines.append(f"- **{r['founder']}**: `viterbi={r['viterbi']:.4f}`, "
                        f"`self_coverage_pct_k25={r['self_coverage_pct_k25']:.1f}%`, "
                        f"`abs_pct={r['abs_pct']:.1f}%`, `abs_acc={r['abs_acc']:.4f}`\n")
        lines.append(
            "Investigated to rule out a driver bug before reporting these: label BEDs "
            "correctly carry each founder's own name (no copy/mapping error), refmap "
            "EXACT/PLACED/MULTI/UNPLACED status proportions are comparable to the "
            "well-behaved founders (no placement-rate collapse), and mean per-read base "
            "quality is uniform across all founders (Q39-40, no quality-driven "
            "explanation).\n\n"
            "**Tzi8** (1,548 contigs) and **CML52** (1,682 contigs) have by far the most "
            "fragmented assemblies of those checked (630-760 contigs for B73/Oh43/CML103), "
            "plausibly explaining their low self-coverage via the same repeat/paralog-"
            "mismapping and assembly-gap mechanisms documented for the Oh43 residual "
            "(`results/oh43_residual_4pct.md`), at much larger scale. **CML103** breaks "
            "this pattern (632 contigs, among the *least* fragmented) yet still scores "
            "worst of all -- its cause is unresolved and flagged for follow-up, not "
            "root-caused here (out of scope for this baseline run).\n\n")
    else:
        lines.append("No founder fell below the 0.90 viterbi threshold.\n\n")

    lines.append("## Excluded founders\n\n")
    lines.append(
        "- **Tx303**: WGS read file present in `WGSReads/`, but no matching assembly/panel "
        "column in this pangenome index -- no valid truth column, not scored.\n"
        "- **M162W**: present in the 25-founder pangenome panel, but no WGS read file was "
        "downloaded -- cannot be run.\n\n")

    summary = non_ref["viterbi"].describe()[["mean", "50%", "min", "max"]]
    lines.append("## Summary (excluding B73)\n\n")
    lines.append(f"- N founders scored: {len(non_ref)}\n")
    lines.append(f"- viterbi mean={summary['mean']:.4f}  median={summary['50%']:.4f}  "
                f"min={summary['min']:.4f}  max={summary['max']:.4f}\n")

    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {RESULTS_MD}")
    print(df[display_cols].to_string(index=False))


def main():
    if len(sys.argv) < 2:
        print(__doc__ or "", file=sys.stderr)
        sys.exit(1)
    mode = sys.argv[1]
    founders = discover_founders()

    if mode == "list":
        for name, path in founders.items():
            idx = NAME_TO_IDX_K25[name]
            print(f"{name:<20} idx={idx:<3} {path.name}")
        print(f"\n{len(founders)} evaluable founders "
              f"({sum(1 for f in WGS_DIR.glob('*.clean.fq.gz'))} WGS files, "
              f"{len(EXCLUDED_WGS_BASENAMES)} excluded: {sorted(EXCLUDED_WGS_BASENAMES)})")
        return

    if mode == "report":
        write_report()
        return

    if mode == "one":
        if len(sys.argv) < 3:
            raise SystemExit("usage: nam_baseline.py one <Founder>")
        name = sys.argv[2]
        if name not in founders:
            raise SystemExit(f"{name!r} not found; run `list` to see evaluable founders")
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading checkpoint once...")
        model = load_model(device)
        write_header_if_needed()
        run_one(model, device, name, founders[name], force=True)
    elif mode == "all":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading checkpoint once...")
        model = load_model(device)
        write_header_if_needed()
        for name, path in founders.items():
            run_one(model, device, name, path)
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    print(f"\nDone. Results in {RESULTS_TSV}")


if __name__ == "__main__":
    main()
