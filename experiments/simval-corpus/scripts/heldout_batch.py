#!/usr/bin/env python
"""
Generalized batch orchestrator for heldout_assembly_eval.py: run N samples
(each a (sample_name, assembly_fasta, optional truth_gvcf) triple) through
the full held-out-assembly pipeline, in parallel, with per-sample subprocess
isolation, per-sample logs, and a summary JSON.

This is nam_inpanel_smoketest.py's own proven orchestration shape (subprocess
isolation, ThreadPoolExecutor concurrency, per-sample logs, summary JSON),
generalized off its hardcoded 25-founder keyfile work list so the SAME
orchestrator also drives:
  - a genuine held-out external assembly (Type A -- the real use case)
  - a leave-one-out founder rebuild (Type B)
  - diverged material, e.g. Tripsacum (Type C)

nam_inpanel_smoketest.py is now a thin caller over this module: it builds the
25-founder work list from the keyfile and calls main() with its own label so
its summary JSON / log directory names (and therefore
build_inpanel_report.py, which parses that summary JSON) are unchanged.

Work list format -- a TSV, one sample per line:
    sample_name<TAB>assembly_fasta<TAB>[truth_gvcf]
The third column is optional; when blank or absent, heldout_assembly_eval.py
builds a fresh truth gVCF via build_truth_gvcf.sh (AnchorWave) for that
sample instead of reusing an existing one. Blank lines and lines starting
with '#' are skipped.

Usage:
    python heldout_batch.py <work_list.tsv> [--depth N] [--parallel N]
        [--region R] [--label NAME] [--samples A,B,C]
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DRIVER = Path(__file__).parent / "heldout_assembly_eval.py"
RESULTS_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results")
DEFAULT_DEPTH = 250_000

PY = sys.executable


def load_work_list(path):
    items = []
    for lineno, raw_line in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise SystemExit(f"{path}:{lineno}: expected at least 2 tab-separated "
                              f"fields (sample_name, assembly_fasta), got: {raw_line!r}")
        name, fasta = parts[0], parts[1]
        truth_gvcf = parts[2] if len(parts) > 2 and parts[2] else None
        items.append((name, fasta, truth_gvcf))
    return items


def run_one(name, assembly_fasta, truth_gvcf, depth, region, log_dir, extra_args):
    cmd = [PY, str(DRIVER), str(assembly_fasta), name, f"--depth={depth}"]
    if truth_gvcf:
        cmd.append(f"--truth-gvcf={truth_gvcf}")
        print(f"[{name}] reusing existing truth gVCF (skipping AnchorWave): {truth_gvcf}")
    else:
        print(f"[{name}] no truth gVCF given -- self-building via AnchorWave")
    if region:
        cmd.append(f"--region={region}")
    cmd.extend(extra_args)

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    log_path.write_text(proc.stdout + "\n" + proc.stderr)
    ok = proc.returncode == 0
    print(f"[{name}] {'OK' if ok else 'FAILED'} in {dt:.1f}s (log: {log_path})")
    # NOTE: "founder" is a legacy key name from nam_inpanel_smoketest.py, kept
    # (rather than renamed to e.g. "sample") so build_inpanel_report.py's
    # existing parser keeps working unchanged for that caller.
    return {"founder": name, "ok": ok, "seconds": round(dt, 1), "log": str(log_path),
            "reused_truth_gvcf": bool(truth_gvcf)}


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work_list", help="TSV: sample_name<TAB>assembly_fasta<TAB>[truth_gvcf]")
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--region", default=None,
                     help="passed through to heldout_assembly_eval.py's --region "
                          "(e.g. chr1 for a chromosome-scale dry-run batch)")
    ap.add_argument("--parallel", type=int, default=3,
                     help="concurrent samples (default 3). Each run_one() is fully "
                          "isolated (own subprocess, own outdir keyed by sample "
                          "name); refmap uses 20 threads per run, so 3 concurrent "
                          "stays near a 62-core budget.")
    ap.add_argument("--label", default=None,
                     help="summary JSON / log dir name stem (default: work list's "
                          "own filename stem). Written to "
                          "results/<label>_summary.json and results/<label>_logs/.")
    ap.add_argument("--samples", default=None,
                     help="comma-separated subset of sample names to run "
                          "(default: all rows in the work list)")
    return ap


def main(argv=None):
    ap = build_arg_parser()
    args, extra_args = ap.parse_known_args(argv)

    label = args.label or Path(args.work_list).stem
    log_dir = RESULTS_DIR / f"{label}_logs"
    summary_json = RESULTS_DIR / f"{label}_summary.json"

    work = load_work_list(args.work_list)
    if args.samples:
        wanted = set(args.samples.split(","))
        work = [w for w in work if w[0] in wanted]
        missing = wanted - {w[0] for w in work}
        if missing:
            raise SystemExit(f"unknown sample(s): {missing}")

    print(f"Running heldout batch for {len(work)} sample(s), depth={args.depth:,}, "
          f"parallel={args.parallel}, label={label!r}")

    todo = []
    results = []
    for name, fasta, truth_gvcf in work:
        if not Path(fasta).exists():
            print(f"[{name}] SKIP -- assembly not found: {fasta}")
            results.append({"founder": name, "ok": False, "seconds": 0.0, "log": None,
                             "reused_truth_gvcf": False, "skip_reason": "assembly not found"})
            continue
        todo.append((name, fasta, truth_gvcf))

    if args.parallel <= 1:
        for name, fasta, truth_gvcf in todo:
            results.append(run_one(name, fasta, truth_gvcf, args.depth, args.region,
                                    log_dir, extra_args))
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(run_one, name, fasta, truth_gvcf, args.depth,
                                    args.region, log_dir, extra_args): name
                       for name, fasta, truth_gvcf in todo}
            for fut in as_completed(futures):
                results.append(fut.result())

    ok = sum(r["ok"] for r in results)
    print(f"\nDone: {ok}/{len(results)} succeeded")
    failed = [r["founder"] for r in results if not r["ok"]]
    if failed:
        print(f"Failed: {failed}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(
        {"depth": args.depth, "region": args.region, "results": results}, indent=2))
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
