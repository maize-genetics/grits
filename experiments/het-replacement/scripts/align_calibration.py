#!/usr/bin/env python
"""Align corpus read sets (simval_eval_indel.py --stage align: refmap, windowing,
a ternary-baseline inference + BED) with the standard real-data alignment.

--corpus calibration: the calibration corpus, feeds measure_real_stats.py
  (PLAN.md §2.1/§2.2); refuses any evaluation line.
--corpus evaluation: re-aligns the evaluation corpus (simulated_validation) so
  every checkpoint is scored on the same alignment (fast_eval_ckpt.py --input-glob).

Standard alignment (adopted 2026-09-25): lift built with `ropebwt3 lift -s 200`
plus refmap --lift-local-win=20000 (adaptive projection, ropebwt3-phg branch
lift-adaptive-projection); --anchor-dist-thresh stays 2000 as in the tested setting.
--lift-preset s2000 reproduces the old alignment."""
import argparse
import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

S = str(Path(__file__).resolve().parents[2] / "simval-corpus/scripts")
PY = "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default/bin/python"
CK = ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
      "diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt")
EVAL_LINES = {"B73", "Oh43", "Il14H", "B97", "CML103", "Tx303", "A188", "EP1", "CML459", "Ia453"}
RB3 = "/workdir/zrm22/HackathonJun2026/ropebwt_refMap"
PRESETS = {
    "s200local": dict(bin=f"{RB3}/ropebwt3-phg/.claude/worktrees/lift-adaptive-projection/ropebwt3",
                      lift=f"{RB3}/rope_bwt_index_v2/maizeFastaIndex_SampleContig_v2_s200.lift", local=20000),
    "s2000": dict(bin=None, lift=None, local=None),
}
MANIFESTS = {"calibration": "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_calibration/manifest.tsv",
             "evaluation": "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv"}

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", choices=sorted(MANIFESTS), default="calibration")
ap.add_argument("--out-root", required=True)
ap.add_argument("--coverages", nargs="+", default=["0.1"])
ap.add_argument("--datasets", nargs="*", default=None)
ap.add_argument("--lift-preset", choices=sorted(PRESETS), default="s200local")
ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--gpu", default="0")
args = ap.parse_args()
rows = [r for r in csv.DictReader(open(MANIFESTS[args.corpus]), delimiter="\t")
        if r["coverage"] in args.coverages and (not args.datasets or r["dataset_id"] in args.datasets)]
if args.corpus == "calibration":
    assert not any(set(r["individual"].split("x")) & EVAL_LINES for r in rows), "evaluation line in calibration manifest"
pre = PRESETS[args.lift_preset]
prefix = "calib" if args.corpus == "calibration" else "eval"


def run(r):
    d = f"{args.out_root}/{prefix}__{r['dataset_id']}__{r['individual']}__{r['coverage']}x"
    if os.path.exists(f"{d}/windowed_k25native_wcount.npy") and os.path.exists(f"{d}/raw.tsv"):
        return r["dataset_id"], r["individual"], r["coverage"], "exists"
    os.makedirs(d, exist_ok=True)
    cmd = [PY, f"{S}/simval_eval_indel.py", "--stage", "align", "--sample", r["individual"],
           "--r1", r["r1_path"], "--r2", r["r2_path"], "--truth-h1", r["truth_h1"], "--truth-h2", r["truth_h2"],
           "--outdir", d, "--out-json", f"{d}/result.json", "--threads", "16", "--no-cleanup",
           "--dataset-id", r["dataset_id"], "--coverage", r["coverage"], "--dataset-class", r["class"],
           "--kind", r["kind"], "--ckpt", CK]
    if pre["bin"]:
        cmd += ["--refmap-bin", pre["bin"], "--lift", pre["lift"], "--lift-local-win", str(pre["local"])]
    p = subprocess.run(cmd, stdout=open(f"{d}/driver.log", "w"), stderr=subprocess.STDOUT,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu))
    return r["dataset_id"], r["individual"], r["coverage"], p.returncode


with ThreadPoolExecutor(args.workers) as ex:
    for res in ex.map(run, rows):
        print(*res, flush=True)
