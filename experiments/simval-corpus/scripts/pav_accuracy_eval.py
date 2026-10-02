#!/usr/bin/env python
"""
Compares imputation accuracy across alignment arms on one simval-corpus row:

  A: refmap             -- current production baseline
  B: chain_nolift        -- chain, no PAV (isolates "did switching engines
                             alone shift accuracy" from "did PAV help")
  C: chain_ordinary_5000 -- chain --lift --insertion-rows=ordinary --pav-agree=5000
  D: chain_ordinary_256  -- chain --lift --insertion-rows=ordinary --pav-agree=256

Answers the question the pav-ps4g-insertion-rows branch (see
/home/zrm22/.claude/plans/the-next-task-i-humming-lobster.md) was built for:
does dropping the `pav:` prefix (--insertion-rows=ordinary) help accuracy, and
does tightening --pav-agree toward the pipeline's 256bp bin size help further
or just cost recall for nothing.

Each arm is `simval_eval_one.py --stage all` (unmodified, only new here: the
--arm flag it now forwards to heldout_assembly_eval.run_refmap). Arm A reuses
the SAME outdir simval_batch.py already used for this row (row_key =
f"{dataset_id}__{individual}__{coverage}x", the established convention) so an
already-scored row returns its cached result instantly instead of re-running
~45 minutes of alignment+comparison; arms B-D get fresh, arm-suffixed outdirs.

--drop-idx is pinned to simval_paths.FIXED_DROP_IDX (23) for every arm, not
left to each arm's own hits-based auto-selection: refmap and chain produce
genuinely different per-founder hit distributions, so auto-selecting per arm
could drop a DIFFERENT founder in each one, confounding the comparison via a
different K25->K24 panel rather than via the alignment method itself.

Usage:
    pav_accuracy_eval.py [--dataset-id IDX-INBRED] [--individual Oh43]
        [--coverage 1.0] [--threads 20] [--arms refmap,chain_nolift,...]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import simval_paths as P  # noqa: E402
import heldout_assembly_eval as hae  # noqa: E402  (for CHAIN_ARMS, so this list can't drift from run_refmap's own)

PY = sys.executable
EVAL_ONE = Path(__file__).parent / "simval_eval_one.py"

ALL_ARMS = ["refmap", *hae.CHAIN_ARMS]


def load_manifest_row(dataset_id, individual, coverage):
    with open(P.MANIFEST) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if (row["dataset_id"] == dataset_id and row["individual"] == individual
                    and row["coverage"] == coverage):
                return row
    raise SystemExit(f"no manifest row for dataset_id={dataset_id!r} "
                      f"individual={individual!r} coverage={coverage!r}")


def outdir_for_arm(row_key, arm):
    # arm="refmap" reuses the pre-existing simval_batch.py outdir for this row
    # (no suffix), so the deep per-stage caches inside simval_eval_one.py
    # (raw.npy, windowed_*.npy, score_result.json) short-circuit almost all of
    # it -- but NOT that outdir's own "result.json": simval_batch.py writes a
    # different, flattened schema there (align_*-prefixed keys, no "score"/
    # "align" nesting), so this script uses its own out-json filename below
    # rather than risk silently misreading that file's shape.
    if arm == "refmap":
        return P.SCRATCH_ROOT / row_key
    return P.SCRATCH_ROOT / f"{row_key}__{arm}"


def run_arm(row, arm, threads, drop_idx):
    dataset_id, klass, kind = row["dataset_id"], row["class"], row["kind"]
    individual, coverage = row["individual"], row["coverage"]
    row_key = f"{dataset_id}__{individual}__{coverage}x"
    outdir = outdir_for_arm(row_key, arm)
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / "pav_eval_result.json"

    if out_json.exists():
        cached = json.loads(out_json.read_text())
        print(f"[{row_key}/{arm}] cached, reusing {out_json}")
        return cached

    cmd = [PY, str(EVAL_ONE), "--stage", "all", "--sample", individual,
           "--r1", row["r1_path"], "--r2", row["r2_path"],
           "--truth-h1", row["truth_h1"], "--truth-h2", row["truth_h2"],
           "--outdir", str(outdir), "--out-json", str(out_json),
           "--threads", str(threads), "--drop-idx", str(drop_idx),
           "--panel-vcf", str(P.PANEL_VCF_V2), "--ckpt", str(P.CKPT_DIPLOID),
           "--dataset-id", dataset_id, "--coverage", coverage,
           "--dataset-class", klass, "--kind", kind, "--arm", arm]
    print(f"[{row_key}/{arm}] running: {' '.join(cmd)}")
    t0 = time.time()
    log_path = outdir / "run.log"
    env = {**os.environ, "TMPDIR": str(P.TMPDIR)}  # keep bcftools/comparator intermediates off shared /tmp
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        tail = log_path.read_text()[-3000:]
        raise RuntimeError(f"[{row_key}/{arm}] failed (see {log_path}):\n{tail}")
    print(f"[{row_key}/{arm}] done in {time.time() - t0:.1f}s")
    return json.loads(out_json.read_text())


def write_report(row, results, out_md, out_tsv):
    dataset_id, individual, coverage = row["dataset_id"], row["individual"], row["coverage"]
    cols = ["arm", "error_rate", "partial_error_rate", "compared_sites",
            "gt_allele_mismatches", "matchable_ceiling", "align_dropped_idx",
            "wall_seconds"]

    with open(out_tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for arm, r in results.items():
            score = r.get("score", {})
            align = r.get("align", {})
            vals = [arm, score.get("error_rate"), score.get("partial_error_rate"),
                     score.get("compared_sites"), score.get("gt_allele_mismatches"),
                     score.get("matchable_ceiling"), align.get("dropped_idx"),
                     r.get("wall_seconds")]
            f.write("\t".join(str(v) for v in vals) + "\n")

    lines = [f"# PAV accuracy eval: {dataset_id} {individual} {coverage}x", ""]
    lines.append("| arm | error_rate | partial_error_rate | compared_sites | mismatches |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm, r in results.items():
        s = r.get("score", {})
        er, per = s.get("error_rate"), s.get("partial_error_rate")
        if er is None:
            lines.append(f"| {arm} | -- | -- | -- | -- |")
        else:
            lines.append(f"| {arm} | {er:.6f} | {per:.6f} | "
                          f"{s.get('compared_sites'):,} | {s.get('gt_allele_mismatches'):,} |")
    Path(out_md).write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md}\nWrote {out_tsv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-id", default="IDX-INBRED")
    ap.add_argument("--individual", default="Oh43")
    ap.add_argument("--coverage", default="1.0")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--drop-idx", type=int, default=P.FIXED_DROP_IDX)
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    args = ap.parse_args()

    arms = args.arms.split(",")
    for arm in arms:
        if arm not in ALL_ARMS:
            raise SystemExit(f"unknown arm {arm!r}, expected one of {ALL_ARMS}")

    row = load_manifest_row(args.dataset_id, args.individual, args.coverage)
    print(f"row: {row['dataset_id']} {row['individual']} {row['coverage']}x "
          f"(kind={row['kind']}, drop_idx={args.drop_idx})")

    results = {}
    for arm in arms:
        results[arm] = run_arm(row, arm, args.threads, args.drop_idx)

    P.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.individual}_{args.coverage}x"
    out_md = P.RESULTS_DIR / f"pav_accuracy_eval_{tag}.md"
    out_tsv = P.RESULTS_DIR / f"pav_accuracy_eval_{tag}.tsv"
    write_report(row, results, out_md, out_tsv)


if __name__ == "__main__":
    main()
