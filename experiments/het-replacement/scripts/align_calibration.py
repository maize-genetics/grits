#!/usr/bin/env python
"""Align calibration-corpus read sets (simval_eval_indel.py --stage align: refmap,
windowing, a ternary-baseline inference + BED) so measure_real_stats.py has
raw.tsv / windowed inputs. Calibration corpus only (see PLAN.md §2.1/§2.2)."""
import argparse
import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

S = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/regionfix-out-snprc-wt/experiments/simval-corpus/scripts"
PY = "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default/bin/python"
CK = ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
      "diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt")
EVAL_LINES = {"B73", "Oh43", "Il14H", "B97", "CML103", "Tx303", "A188", "EP1", "CML459", "Ia453"}

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_calibration/manifest.tsv")
ap.add_argument("--out-root", required=True)
ap.add_argument("--coverages", nargs="+", default=["0.1"])
ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--gpu", default="0")
args = ap.parse_args()
rows = [r for r in csv.DictReader(open(args.manifest), delimiter="\t") if r["coverage"] in args.coverages]
assert not any(set(r["individual"].split("x")) & EVAL_LINES for r in rows), "evaluation line in calibration manifest"


def run(r):
    d = f"{args.out_root}/calib__{r['dataset_id']}__{r['individual']}__{r['coverage']}x"
    if os.path.exists(f"{d}/windowed_k25native_wcount.npy") and os.path.exists(f"{d}/raw.tsv"):
        return r["dataset_id"], r["individual"], r["coverage"], "exists"
    os.makedirs(d, exist_ok=True)
    cmd = [PY, f"{S}/simval_eval_indel.py", "--stage", "align", "--sample", r["individual"],
           "--r1", r["r1_path"], "--r2", r["r2_path"], "--truth-h1", r["truth_h1"], "--truth-h2", r["truth_h2"],
           "--outdir", d, "--out-json", f"{d}/result.json", "--threads", "16", "--no-cleanup",
           "--dataset-id", r["dataset_id"], "--coverage", r["coverage"], "--dataset-class", r["class"],
           "--kind", r["kind"], "--ckpt", CK]
    p = subprocess.run(cmd, stdout=open(f"{d}/driver.log", "w"), stderr=subprocess.STDOUT,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu))
    return r["dataset_id"], r["individual"], r["coverage"], p.returncode


with ThreadPoolExecutor(args.workers) as ex:
    for res in ex.map(run, rows):
        print(*res, flush=True)
