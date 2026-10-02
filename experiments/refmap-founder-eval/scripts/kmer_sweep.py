#!/usr/bin/env python
"""
Sweep ropebwt3-phg's --kmer placement parameters (length, step, min-agree,
cluster) on the Oh43 1M-read benchmark, and score each combination two ways:
  1. founder-support density (mean founders/site, Oh43 coverage %) from the
     raw K=25 windowed matrix -- the fixed-index diagnostic.
  2. CRF Viterbi accuracy from the K=24 windowed matrix (matching the trained
     checkpoint), same haploid-sim checkpoint used throughout this
     investigation, same Oh43-covered/absent split methodology.

Single process: the checkpoint (and pytorch_lightning) is loaded ONCE, not
once per combo, since that import alone costs ~40s cold.

Reused, unmodified: the patched ropebwt3 binary (carrier-cap fix + kmer
gameteSet export, both already applied), crf/ropebwt_npy_to_matrix.py,
crf/train_haploid.py's GRITSCRFHaploid/make_splits.

Usage:
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/kmer_sweep.py
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
LABEL_BED = REFMAP_ROOT / "bench_ps4g_npy/labels_full.bed"
READS = REFMAP_ROOT / "bench_ps4g_npy/reads/Oh43_1M.fastq"

CRF_SRC = Path("/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src")
WINDOW_SCRIPT = CRF_SRC / "python/crf/ropebwt_npy_to_matrix.py"
CKPT = Path("/workdir/zrm22/HackathonJun2026/grits_workdir/checkpoints/haploid-sim/last.ckpt")

SCRATCH = Path("/tmp/claude-4347/-local-workdir-zrm22-HackathonJun2026-grits-workdir"
               "/be856a00-7aeb-43da-bfcc-56639fe63142/scratchpad/kmer_sweep")
RESULTS_TSV = Path("/workdir/zrm22/HackathonJun2026/grits_workdir/results/kmer_sweep_results.tsv")

OH43_NAME = "Oh43"
OH43_IDX_K25 = 21  # fixed: alphabetical panel position, independent of data
PY = sys.executable

sys.path.insert(0, str(CRF_SRC))


def combo_tag(k, step, agree, cluster):
    return f"k{k}_s{step}_a{agree}_c{cluster}"


def run_refmap(tag, k, step, agree, cluster):
    """Run the patched binary for one combo. Returns (wall_s, tool_real_s, tool_cpu_s,
    n_placed, n_unplaced), skipping the run if outputs already exist."""
    outdir = SCRATCH / tag
    outdir.mkdir(parents=True, exist_ok=True)
    npy_path = outdir / "raw.npy"
    ps4g_path = outdir / "raw.ps4g"
    tsv_path = outdir / "raw.tsv"
    log_path = outdir / "raw.log"

    if npy_path.exists() and tsv_path.exists():
        print(f"  [{tag}] refmap output already exists, skipping run")
        wall_s = None
        tool_real_s = tool_cpu_s = None
        log_text = log_path.read_text() if log_path.exists() else ""
    else:
        cmd = [str(BIN), "refmap", "--ref-prefix=B73", "--max-occ=-1",
               f"--lift={LIFT}", f"--kmer={k}", f"--kmer-step={step}",
               f"--min-agree={agree}", f"--kmer-cluster={cluster}", "-t", "20",
               f"--label-bed={LABEL_BED}", f"--ps4g={ps4g_path}", f"--npy={npy_path}",
               str(FMD), str(READS)]
        print(f"  [{tag}] running: {' '.join(cmd)}")
        t0 = time.time()
        with open(tsv_path, "w") as out_f:
            proc = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True)
        wall_s = time.time() - t0
        log_text = proc.stderr
        log_path.write_text(log_text)
        if proc.returncode != 0:
            raise RuntimeError(f"refmap failed for {tag}: {log_text[-2000:]}")

    m = re.search(r"Real time:\s*([\d.]+)\s*sec;\s*CPU:\s*([\d.]+)\s*sec", log_text)
    tool_real_s = float(m.group(1)) if m else None
    tool_cpu_s = float(m.group(2)) if m else None

    n_placed = n_unplaced = 0
    with open(tsv_path) as f:
        for line in f:
            status = line.split("\t", 3)[2]
            if status == "PLACED":
                n_placed += 1
            elif status == "UNPLACED":
                n_unplaced += 1
    return wall_s, tool_real_s, tool_cpu_s, n_placed, n_unplaced, npy_path


def window(tag, raw_npy, target_num_parents=None):
    """Run crf/ropebwt_npy_to_matrix.py. Returns (out_path, dropped_idx or None).
    dropped_idx is cached to a sidecar file so a skipped (already-windowed) run
    still recovers it correctly -- it is NOT recomputable from the windowed
    array alone (windowing drops trailing partial-window rows per contig, so
    the founder hit-counts wouldn't exactly match what the tool itself saw)."""
    outdir = SCRATCH / tag
    bins_path = outdir / "raw.npy.bins.tsv"
    gametes_path = outdir / "raw.npy.gametes.tsv"
    suffix = f"_k{target_num_parents}" if target_num_parents else "_k25"
    out_path = outdir / f"windowed{suffix}.npy"
    sidecar = outdir / f"windowed{suffix}.dropped_idx.txt"

    if out_path.exists():
        print(f"  [{tag}] windowed{suffix} already exists, skipping")
        dropped_idx = int(sidecar.read_text()) if sidecar.exists() else None
        return out_path, dropped_idx

    cmd = [PY, str(WINDOW_SCRIPT), f"--npy={raw_npy}", f"--bins={bins_path}",
           f"--gametes={gametes_path}", "--num-parents=25", "--window-size=512",
           f"--out={out_path}"]
    if target_num_parents:
        cmd.append(f"--target-num-parents={target_num_parents}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"windowing failed for {tag}: {proc.stderr[-2000:]}")

    dropped_idx = None
    m = re.search(r"^\s*(\S+)\s+idx=(\d+)\s+hits=", proc.stdout, re.MULTILINE)
    if m:
        dropped_idx = int(m.group(2))
        sidecar.write_text(str(dropped_idx))
    return out_path, dropped_idx


def founder_density_stats(k25_npy_path):
    a = np.asarray(np.load(k25_npy_path, mmap_mode="r"))
    K = a.shape[-1] - 2
    feat = a[:, :, :K].reshape(-1, K)
    nz_row = feat != 0
    any_nz = nz_row.any(1)
    mean_founders = nz_row.sum(1)[any_nz].mean() if any_nz.any() else 0.0
    oh43_cov = nz_row[:, OH43_IDX_K25].mean() * 100
    n_windows = a.shape[0]
    return mean_founders, oh43_cov, n_windows


def eval_combo(model, k24_npy_path, dropped_idx):
    import torch
    from python.crf.train_haploid import make_splits

    oh43_idx_k24 = OH43_IDX_K25 - 1 if (dropped_idx is not None and dropped_idx < OH43_IDX_K25) else OH43_IDX_K25
    _, _, test_ds = make_splits(str(k24_npy_path), num_parents=24, val_frac=0.10, test_frac=0.10)
    n = len(test_ds)
    if n == 0:
        return dict(n_sites=0, viterbi=float("nan"), cov_pct=float("nan"),
                    cov_acc=float("nan"), abs_pct=float("nan"), abs_acc=float("nan"))
    vc_cov = vt_cov = vc_abs = vt_abs = 0
    with torch.no_grad():
        for i in range(n):
            item = test_ds[i]
            X = item["input_embeds"].unsqueeze(0)
            tags = item["labels"]
            emis_f, g, c = model(X, ext_emb=None)
            pred = model.decode(emis_f, c)[0]
            absent = X[0, :, oh43_idx_k24] == 0
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


RESULT_COLS = ["tag", "k", "step", "agree", "cluster", "wall_s", "tool_real_s", "tool_cpu_s",
               "n_placed", "n_unplaced", "mean_founders_per_site", "oh43_coverage_pct_k25",
               "n_sites", "viterbi", "cov_pct", "cov_acc", "abs_pct", "abs_acc"]


def write_header_if_needed():
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("\t".join(RESULT_COLS) + "\n")


def already_recorded(tag):
    if not RESULTS_TSV.exists():
        return False
    with open(RESULTS_TSV) as f:
        return any(line.split("\t", 1)[0] == tag for line in f)


NAN_METRICS = dict(n_sites=0, viterbi=float("nan"), cov_pct=float("nan"),
                    cov_acc=float("nan"), abs_pct=float("nan"), abs_acc=float("nan"))


def run_one(model, k, step, agree, cluster):
    tag = combo_tag(k, step, agree, cluster)
    if already_recorded(tag):
        print(f"[{tag}] already in results TSV, skipping entirely")
        return

    print(f"\n=== {tag} ===")
    wall_s, tool_real_s, tool_cpu_s, n_placed, n_unplaced, raw_npy = run_refmap(
        tag, k, step, agree, cluster)

    if n_placed == 0:
        # Structural degenerate case: e.g. min-agree exceeds the number of
        # distinct k-mer tiles a read can produce at this k/step, so nothing
        # is ever PLACED and there is no PS4G/npy data to window at all.
        print(f"  [{tag}] n_placed=0 -- degenerate combo (min-agree likely exceeds tiles/read), "
              f"recording NaN metrics, no windowing attempted")
        mean_founders, oh43_cov = float("nan"), float("nan")
        metrics = dict(NAN_METRICS)
    else:
        try:
            k25_npy, _ = window(tag, raw_npy, target_num_parents=None)
            k24_npy, dropped_idx = window(tag, raw_npy, target_num_parents=24)
            mean_founders, oh43_cov, n_windows = founder_density_stats(k25_npy)
            metrics = eval_combo(model, k24_npy, dropped_idx)
        except RuntimeError as e:
            print(f"  [{tag}] windowing failed despite n_placed={n_placed} "
                  f"(too few/sparse for a full 512-row window anywhere) -- recording NaN metrics: {e}")
            mean_founders, oh43_cov = float("nan"), float("nan")
            metrics = dict(NAN_METRICS)

    row = dict(tag=tag, k=k, step=step, agree=agree, cluster=cluster,
               wall_s=wall_s, tool_real_s=tool_real_s, tool_cpu_s=tool_cpu_s,
               n_placed=n_placed, n_unplaced=n_unplaced,
               mean_founders_per_site=mean_founders, oh43_coverage_pct_k25=oh43_cov,
               **metrics)
    write_header_if_needed()
    with open(RESULTS_TSV, "a") as f:
        f.write("\t".join(str(row[c]) for c in RESULT_COLS) + "\n")
    print(f"[{tag}] viterbi={metrics['viterbi']:.4f}  cov={metrics['cov_pct']:.1f}%@{metrics['cov_acc']:.4f}"
          f"  abs={metrics['abs_pct']:.1f}%@{metrics['abs_acc']:.4f}  mean_founders={mean_founders:.2f}"
          f"  oh43_cov_k25={oh43_cov:.1f}%  wall={wall_s}  n_placed={n_placed}")


def load_model():
    import torch
    from python.crf.train_haploid import GRITSCRFHaploid
    model = GRITSCRFHaploid.load_from_checkpoint(str(CKPT), map_location="cpu")
    model.eval()
    return model


def stage_a(model):
    """Sweep kmer length, step=15/agree=2/cluster=2000 fixed."""
    for k in (25, 50, 75, 100, 125):
        run_one(model, k, 15, 2, 2000)


def stage_b(model, best_k):
    """Sweep kmer-step at the given (Stage-A-winning) k, agree=2/cluster=2000 fixed."""
    for step in (5, 10, 15, 25, 40):
        run_one(model, best_k, step, 2, 2000)


def stage_c(model, best_k, best_step):
    """Sweep min-agree at the given (Stage-A/B-winning) k/step, cluster=2000 fixed."""
    for agree in (1, 2, 3, 4, 5):
        run_one(model, best_k, best_step, agree, 2000)


def stage_d(model, best_k, best_step, best_agree):
    """Optional: sweep kmer-cluster at the winning k/step/agree."""
    for cluster in (500, 1000, 2000, 4000):
        run_one(model, best_k, best_step, best_agree, cluster)


def main():
    """Usage: kmer_sweep.py <stage> [args...]
      kmer_sweep.py A
      kmer_sweep.py B <best_k>
      kmer_sweep.py C <best_k> <best_step>
      kmer_sweep.py D <best_k> <best_step> <best_agree>
      kmer_sweep.py one <k> <step> <agree> <cluster>   (single ad-hoc combo)
    """
    if len(sys.argv) < 2:
        print(__doc__ or "", file=sys.stderr)
        print(main.__doc__, file=sys.stderr)
        sys.exit(1)
    stage = sys.argv[1]
    args = [int(a) for a in sys.argv[2:]]

    print("Loading checkpoint once...")
    model = load_model()
    write_header_if_needed()

    if stage == "A":
        stage_a(model)
    elif stage == "B":
        stage_b(model, *args)
    elif stage == "C":
        stage_c(model, *args)
    elif stage == "D":
        stage_d(model, *args)
    elif stage == "one":
        run_one(model, *args)
    else:
        raise SystemExit(f"unknown stage {stage!r}")

    print(f"\nDone. Results in {RESULTS_TSV}")


if __name__ == "__main__":
    main()
