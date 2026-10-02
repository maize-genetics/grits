#!/usr/bin/env python
"""
Q x min-length read-trimming sweep for the NAM founder-recovery baseline.

Reuses nam_baseline.py's pipeline (refmap -> windowed CRF -> Viterbi accuracy)
completely unmodified, loaded as a module and monkeypatched per condition:
SCRATCH/RESULTS_TSV/RESULTS_MD point at a per-condition location, and
make_subsample/run_refmap are wrapped so refmap runs on a quality-trimmed
copy of the ALREADY-cached 1,000,000-read baseline subsample (scripts/
trim_reads.py) instead of a fresh, untrimmed one -- no re-subsampling from
the source .fq.gz. The trimmed FASTQ is deleted right after refmap finishes
with it (raw.npy/raw.tsv are what downstream steps need), so per-founder disk
stays close to the baseline's own ~200MB (npy+tsv+ps4g), not +360MB/condition
for the trimmed reads.

Grid: min-qual in {30, 35, 40} x min-len in {50, 75, 100} (9 conditions),
each compared against the existing no-trim results/nam_baseline_results.tsv.
Every condition writes its own results/nam_trim_q{Q}_l{L}.tsv, resumable
exactly like nam_baseline.py itself (skips founders already recorded, skips
refmap when its outputs already exist).

Usage:
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/nam_trim_sweep.py sweep
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/nam_trim_sweep.py sweep --quals 35 --lens 75 --founders B73
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/nam_trim_sweep.py report
"""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NAM_BASELINE_PATH = SCRIPT_DIR / "nam_baseline.py"
TRIM_SCRIPT = SCRIPT_DIR / "trim_reads.py"

# Sweep scratch lives on /local (104T free), not /tmp -- 9 conditions x 24
# founders x ~200MB/founder of npy/tsv/ps4g outputs adds up.
SWEEP_SCRATCH_ROOT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/nam_trim_sweep")
RESULTS_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results")
BASELINE_RESULTS_TSV = RESULTS_DIR / "nam_baseline_results.tsv"

PY = sys.executable

QUALS = [30, 35, 40]
LENS = [50, 75, 100]
WEAK_FOUNDERS = ["CML103", "CML52", "Tzi8"]  # known low-accuracy outliers, see nam_baseline.md


def tag(q, l):
    return f"q{q}_l{l}"


def load_nam_baseline():
    """Import nam_baseline.py as a module (not __main__), so its functions are
    reusable and monkeypatchable without touching the committed file."""
    spec = importlib.util.spec_from_file_location("nam_baseline", NAM_BASELINE_PATH)
    nb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb)
    return nb


def make_trim_hook(baseline_scratch, orig_make_subsample, min_qual, min_len):
    """Returns a make_subsample(founder, read_path, outdir) replacement that
    trims the cached baseline reads_1M.fastq (falling back to building a
    fresh untrimmed subsample via the original function if no cache exists)."""
    def _trim_subsample(founder, read_path, outdir):
        out_path = outdir / "reads_1M_trimmed.fastq"
        if out_path.exists():
            return out_path
        cached = baseline_scratch / founder / "reads_1M.fastq"
        made_fresh = False
        if cached.exists():
            subsample_path = cached
        else:
            print(f"  [{founder}] no cached baseline subsample at {cached}, building fresh")
            subsample_path = orig_make_subsample(founder, read_path, outdir)
            made_fresh = True
        stats_path = outdir / "trim_stats.tsv"
        cmd = [PY, str(TRIM_SCRIPT), f"--in={subsample_path}", f"--out={out_path}",
               f"--min-qual={min_qual}", f"--min-len={min_len}", f"--stats={stats_path}"]
        print(f"  [{founder}] trimming: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        if made_fresh:
            subsample_path.unlink(missing_ok=True)
        return out_path
    return _trim_subsample


def run_condition(nb, orig_run_refmap, orig_make_subsample, baseline_scratch,
                   founders, model, device, q, l):
    t = tag(q, l)
    print(f"\n########## condition {t}: min-qual={q} min-len={l} ##########")
    nb.SCRATCH = SWEEP_SCRATCH_ROOT / t
    nb.SCRATCH.mkdir(parents=True, exist_ok=True)
    nb.RESULTS_TSV = RESULTS_DIR / f"nam_trim_{t}.tsv"
    nb.RESULTS_MD = RESULTS_DIR / f"nam_trim_{t}.md"

    nb.make_subsample = make_trim_hook(baseline_scratch, orig_make_subsample, q, l)

    def _run_refmap_and_cleanup(founder, read_path):
        result = orig_run_refmap(founder, read_path)
        trimmed = nb.SCRATCH / founder / "reads_1M_trimmed.fastq"
        trimmed.unlink(missing_ok=True)
        return result
    nb.run_refmap = _run_refmap_and_cleanup

    nb.write_header_if_needed()
    for name, path in founders.items():
        nb.run_one(model, device, name, path)


def run_all(quals, lens, only_founders=None):
    import torch

    nb = load_nam_baseline()
    baseline_scratch = nb.SCRATCH  # cached-baseline location, captured before any override
    orig_make_subsample = nb.make_subsample
    orig_run_refmap = nb.run_refmap

    founders = nb.discover_founders()
    if only_founders:
        missing = set(only_founders) - set(founders)
        if missing:
            raise SystemExit(f"unknown founder(s): {sorted(missing)}; run nam_baseline.py list")
        founders = {k: v for k, v in founders.items() if k in only_founders}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint once (device={device})...")
    model = nb.load_model(device)

    for q in quals:
        for l in lens:
            run_condition(nb, orig_run_refmap, orig_make_subsample, baseline_scratch,
                          founders, model, device, q, l)
    print("\nAll conditions complete.")


def _trim_agg_stats(founders, tag_):
    import pandas as pd
    rows = []
    outdir = SWEEP_SCRATCH_ROOT / tag_
    for founder in founders:
        p = outdir / founder / "trim_stats.tsv"
        if p.exists():
            rows.append(pd.read_csv(p, sep="\t").iloc[0])
    if not rows:
        return None
    return pd.DataFrame(rows)[["pct_kept", "mean_len_out"]].mean()


def write_report(quals, lens):
    import pandas as pd

    nb = load_nam_baseline()  # reuse its _markdown_table helper

    if not BASELINE_RESULTS_TSV.exists():
        raise SystemExit(f"missing baseline results {BASELINE_RESULTS_TSV} -- run nam_baseline.py all first")
    baseline_df = pd.read_csv(BASELINE_RESULTS_TSV, sep="\t")
    base_non_ref = baseline_df[baseline_df["founder"] != "B73"]
    base_mean, base_median = base_non_ref["viterbi"].mean(), base_non_ref["viterbi"].median()
    founders = list(baseline_df["founder"])

    matrix_rows, trim_rows, condition_dfs = [], [], {}
    for q in quals:
        for l in lens:
            t = tag(q, l)
            path = RESULTS_DIR / f"nam_trim_{t}.tsv"
            if not path.exists():
                print(f"WARNING: missing {path}, skipping condition {t} in report")
                continue
            df = pd.read_csv(path, sep="\t")
            condition_dfs[t] = df
            non_ref = df[df["founder"] != "B73"]
            mean_v, median_v = non_ref["viterbi"].mean(), non_ref["viterbi"].median()
            matrix_rows.append(dict(tag=t, min_qual=q, min_len=l,
                                     mean_viterbi=mean_v, median_viterbi=median_v,
                                     delta_vs_baseline=mean_v - base_mean))
            agg = _trim_agg_stats(founders, t)
            trim_rows.append(dict(tag=t, min_qual=q, min_len=l,
                                   mean_pct_kept=agg["pct_kept"] if agg is not None else float("nan"),
                                   mean_len_out=agg["mean_len_out"] if agg is not None else float("nan")))

    if not matrix_rows:
        raise SystemExit("no condition results found -- run `sweep` first")

    matrix_df = pd.DataFrame(matrix_rows)
    matrix_df.to_csv(RESULTS_DIR / "nam_trim_sweep_matrix.tsv", sep="\t", index=False)
    pivot = matrix_df.pivot(index="min_qual", columns="min_len", values="mean_viterbi")
    trim_df = pd.DataFrame(trim_rows).sort_values(["min_qual", "min_len"])

    weak_rows = []
    for f in WEAK_FOUNDERS:
        if f not in founders:
            continue
        base_v = baseline_df.loc[baseline_df.founder == f, "viterbi"]
        row = dict(founder=f, baseline=base_v.iloc[0] if len(base_v) else float("nan"))
        for t, df in condition_dfs.items():
            v = df.loc[df.founder == f, "viterbi"]
            row[t] = v.iloc[0] if len(v) else float("nan")
        weak_rows.append(row)
    weak_df = pd.DataFrame(weak_rows)

    lines = []
    lines.append("# NAM founder-recovery: read-trimming sweep\n\n")
    lines.append(
        "Quality-trimmed the same 1,000,000-read subsample used in the no-trim baseline "
        "(`results/nam_baseline.md`) with `scripts/trim_reads.py` -- end-trim leading/trailing "
        "bases below a Phred threshold, drop the read if what remains is shorter than a minimum "
        "length (internal low-quality runs are left untouched) -- then re-ran the identical "
        "refmap -> windowed-CRF -> Viterbi pipeline (`scripts/nam_trim_sweep.py`, driving "
        "`nam_baseline.py`'s functions unmodified) across a "
        f"{len(quals)}x{len(lens)} grid: min-qual in {quals}, min-len in {lens}.\n\n"
        f"No-trim baseline (excluding B73): mean viterbi={base_mean:.4f}, median={base_median:.4f} "
        "(`results/nam_baseline_results.tsv`).\n\n"
    )

    lines.append("## Mean viterbi (excl. B73) by condition\n\n")
    lines.append(f"Rows = min-qual, columns = min-len. Baseline (no trim) = {base_mean:.4f}.\n\n")
    cols = list(pivot.columns)
    header = "| Q \\ L | " + " | ".join(str(c) for c in cols) + " |"
    sep = "| --- | " + " | ".join("---" for _ in cols) + " |"
    rows_md = [header, sep]
    for q in pivot.index:
        vals = " | ".join(f"{pivot.loc[q, c]:.4f}" if pd.notna(pivot.loc[q, c]) else "n/a" for c in cols)
        rows_md.append(f"| {q} | {vals} |")
    lines.append("\n".join(rows_md) + "\n\n")

    lines.append("## Delta vs baseline, sorted best-to-worst\n\n")
    delta_cols = ["tag", "min_qual", "min_len", "mean_viterbi", "delta_vs_baseline"]
    lines.append(nb._markdown_table(
        matrix_df[delta_cols].sort_values("delta_vs_baseline", ascending=False),
        {"mean_viterbi", "delta_vs_baseline"}) + "\n\n")

    lines.append("## Weak-founder detail (CML103, CML52, Tzi8)\n\n")
    if len(weak_df):
        float_cols = set(weak_df.columns) - {"founder"}
        lines.append(nb._markdown_table(weak_df, float_cols) + "\n\n")
    else:
        lines.append("(no weak-founder rows available)\n\n")

    lines.append("## Trim aggressiveness by condition\n\n")
    lines.append(
        "Mean over founders of `pct_kept` (reads surviving trim+min-len) and `mean_len_out` "
        "(mean trimmed length, kept reads only) -- the cost side of the accuracy/length "
        "tradeoff.\n\n")
    lines.append(nb._markdown_table(trim_df, {"mean_pct_kept", "mean_len_out"}) + "\n\n")

    out_path = RESULTS_DIR / "nam_trim_sweep.md"
    out_path.write_text("".join(lines))
    print(f"Wrote {out_path}")
    print(matrix_df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["sweep", "report"])
    ap.add_argument("--quals", type=int, nargs="+", default=QUALS)
    ap.add_argument("--lens", type=int, nargs="+", default=LENS)
    ap.add_argument("--founders", nargs="+", default=None,
                     help="restrict to these founder names (smoke testing)")
    args = ap.parse_args()

    if args.mode == "sweep":
        run_all(args.quals, args.lens, only_founders=args.founders)
    else:
        write_report(args.quals, args.lens)


if __name__ == "__main__":
    main()
