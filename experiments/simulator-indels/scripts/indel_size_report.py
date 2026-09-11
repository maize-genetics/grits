"""
Measure real indel structure from maize founder gVCFs, to calibrate
`simulate_alleles.py`'s planned indel model (see `../PLAN.md`).

Reads PHGv2-style founder gVCFs (`ASM_Start`/`ASM_End` INFO fields record
each block's span in the founder's own assembly coordinates; comparing
that to the block's reference span gives the indel delta directly --
no alignment or variant-calling assumptions beyond what's already in the
gVCF). This is the same method used ad hoc during the design-review
session that produced the 39-42.6% indel-affected-bp figures recorded in
`../results/indel_biology_notes.md`; this script makes that measurement
reproducible and extends it across chromosomes/founders with a figure.

Two numbers per (founder, chromosome), both load-bearing for the
simulator design:
  1. Indel-affected reference bp, as a fraction of the chromosome's
     reference span -- the calibration target for `--indel-frac`-style
     tuning (~40% expected, confirmed here).
  2. The indel-length distribution, reported BOTH event-weighted (%
     of indel events per size class) and bp-weighted (% of indel-
     affected bp per size class) -- these are starkly different
     (bimodal-event, large-dominated-bp), and calibrating the simulator
     to the wrong one would silently produce ~0% indel-affected genome.
     See "Why bp-weighted, not event-weighted" in indel_biology_notes.md.

Usage:
    pixi run -- python experiments/simulator-indels/scripts/indel_size_report.py \\
        --gvcf-dir /workdir/zrm22/HackathonJun2026/grits_workdir/data/founder_truth_gvcfs_nofill \\
        --founders B97 CML103 Tzi8 Oh43 Ky21 P39 \\
        --chroms chr1 chr3 chr5 chr10 \\
        --out-tsv experiments/simulator-indels/results/indel_size_report.tsv \\
        --out-fig experiments/simulator-indels/results/indel_size_distribution.png

Run under an environment with matplotlib (the project's pixi env does
not currently include it; `/home/zrm22/mambaforge/envs/phg-ml` does) if
`--out-fig` is requested; `--out-tsv` alone needs only numpy/pandas.
"""
import argparse
import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# A188 is a confirmed assembly-quality outlier (see indel_biology_notes.md):
# ~19-20% indel fraction vs ~11-13% event fraction for every other founder,
# 60%+ of its indels at exactly 1bp vs ~41% for everyone else. Reported if
# present in --founders, but never used for fitting simulator parameters.
OUTLIER_FOUNDERS = {"A188"}

END_RE = re.compile(r"(?:^|;)END=(\d+)")
ASM_START_RE = re.compile(r"ASM_Start=(\d+)")
ASM_END_RE = re.compile(r"ASM_End=(\d+)")


def iter_gvcf_blocks(gvcf_path, chrom):
    """Yield (pos, refspan, asmspan) for every record on `chrom`.

    pos is 1-based VCF POS. refspan/asmspan are >=1. Records with no
    ASM_Start/ASM_End (shouldn't happen in this gVCF flavor, but don't
    assume) are skipped rather than silently treated as zero-length.
    """
    opener = gzip.open if str(gvcf_path).endswith(".gz") else open
    with opener(gvcf_path, "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            fields = line.rstrip("\n").split("\t", 8)
            if fields[0] != chrom:
                continue
            pos = int(fields[1])
            ref = fields[3]
            info = fields[7]
            m_end = END_RE.search(info)
            end = int(m_end.group(1)) if m_end else pos + len(ref) - 1
            m_as = ASM_START_RE.search(info)
            m_ae = ASM_END_RE.search(info)
            if not (m_as and m_ae):
                continue
            asm_start, asm_end = int(m_as.group(1)), int(m_ae.group(1))
            refspan = end - pos + 1
            asmspan = abs(asm_end - asm_start) + 1
            yield pos, end, refspan, asmspan


def size_bucket(delta):
    """floor(log2(delta)) + 1, i.e. the same power-of-two bucketing used
    during the ad hoc design-review measurement (bucket b covers
    [2^(b-1), 2^b - 1])."""
    return int(delta).bit_length()


def measure_founder_chrom(gvcf_path, founder, chrom):
    """Returns (summary_row, events_df) for one (founder, chromosome)."""
    prev_end = None
    max_end = 0
    gap_bp = 0
    n_gaps = 0
    covered_bp = 0
    events = []  # (bucket, delta, is_insertion)
    for pos, end, refspan, asmspan in iter_gvcf_blocks(gvcf_path, chrom):
        covered_bp += refspan
        if prev_end is not None and pos > prev_end + 1:
            gap_bp += pos - prev_end - 1
            n_gaps += 1
        if refspan != asmspan:
            delta = abs(refspan - asmspan)
            events.append((size_bucket(delta), delta, asmspan > refspan))
        prev_end = end
        if end > max_end:
            max_end = end

    ev = pd.DataFrame(events, columns=["bucket", "delta", "is_insertion"])
    # Reference-genome-homology fraction: deletions (and true gaps) remove
    # reference bp from founder homology; insertions do NOT -- an inserted
    # run sits at a single reference anchor point and adds zero reference
    # span (its bp exist only in the founder's own assembly coordinates,
    # where they drive read-stacking density, not reference-genome loss).
    # Summing insertion delta into this fraction was an early bug in this
    # measurement (caught during script verification, see PLAN.md/HANDOFF.md)
    # that nearly doubled the reported fraction -- deletion-only is the
    # correct, unambiguous "founder has no homologous sequence here" statistic.
    del_bp = int(ev.loc[~ev["is_insertion"], "delta"].sum()) if len(ev) else 0
    ins_bp = int(ev.loc[ev["is_insertion"], "delta"].sum()) if len(ev) else 0
    indel_bp = gap_bp + del_bp
    summary = {
        "founder": founder,
        "chrom": chrom,
        "ref_span_bp": max_end,
        "n_indel_events": len(ev),
        "deletion_bp": del_bp,
        "insertion_bp": ins_bp,
        "indel_affected_bp": indel_bp,
        "indel_affected_frac": indel_bp / max_end if max_end else float("nan"),
        "n_reference_gaps": n_gaps,
        "gap_bp": gap_bp,
    }
    ev["founder"] = founder
    ev["chrom"] = chrom
    return summary, ev


def build_size_table(all_events):
    """Event- and bp-weighted size-class table, pooled across every
    (founder, chrom) passed in (excluding outlier founders by default is
    the caller's job -- this just tabulates what it's given)."""
    ev = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame(
        columns=["bucket", "delta", "is_insertion"])
    if ev.empty:
        return pd.DataFrame()
    tot_events = len(ev)
    tot_bp = ev["delta"].sum()
    rows = []
    for b, g in ev.groupby("bucket"):
        lo, hi = 2 ** (b - 1), 2 ** b - 1
        n_ins = int(g["is_insertion"].sum())
        n_del = len(g) - n_ins
        rows.append({
            "size_range": f"{lo}-{hi}",
            "bucket": b,
            "n_events": len(g),
            "event_pct": 100 * len(g) / tot_events,
            "indel_bp": int(g["delta"].sum()),
            "bp_pct": 100 * g["delta"].sum() / tot_bp,
            "ins_del_ratio": (n_ins / n_del) if n_del else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


def make_figure(size_table, summary_df, out_path, organism="maize"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit_df = summary_df[~summary_df["founder"].isin(OUTLIER_FOUNDERS)]
    outlier_df = summary_df[summary_df["founder"].isin(OUTLIER_FOUNDERS)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    x = np.arange(len(size_table))
    w = 0.38
    ax.bar(x - w / 2, size_table["event_pct"], width=w, label="% of events", color="#4C72B0")
    ax.bar(x + w / 2, size_table["bp_pct"], width=w, label="% of indel bp", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(size_table["size_range"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("% of total")
    ax.set_title("A. Event count vs. bp: the size distributions invert")
    ax.legend()

    ax = axes[0, 1]
    st = size_table.sort_values("bucket")
    cum_bp = np.cumsum(st["indel_bp"].to_numpy())
    cum_bp_pct = 100 * cum_bp / cum_bp[-1]
    lo_bp = 2.0 ** (st["bucket"].to_numpy() - 1)
    ax.plot(lo_bp, cum_bp_pct, marker="o", color="#55A868")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("size threshold (bp, lower bound of bucket)")
    ax.set_ylabel("cumulative % of indel bp, sizes < threshold")
    ax.axvline(4096, color="gray", linestyle="--", linewidth=1)
    idx4k = np.searchsorted(st["bucket"].to_numpy(), size_bucket(4096))
    below4k = cum_bp_pct[idx4k - 1] if idx4k > 0 else 0.0
    ax.annotate(f"<4kb: {below4k:.1f}%\n>=4kb: {100 - below4k:.1f}%",
                xy=(4096, below4k), xytext=(8, 40), textcoords="offset points",
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_title("B. Cumulative indel bp by size threshold")

    ax = axes[1, 0]
    ax.plot(lo_bp, st["ins_del_ratio"], marker="o", color="#C44E52")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 2)
    ax.set_xlabel("size class lower bound (bp)")
    ax.set_ylabel("insertion : deletion ratio")
    ax.set_title("C. ins:del symmetry holds size-by-size")

    ax = axes[1, 1]
    piv = fit_df.pivot_table(index="chrom", columns="founder", values="indel_affected_frac")
    # Sort chromosomes by their trailing digits, independent of naming scheme
    # (maize's "chr1".."chr10", cassava's "Chromosome01".."Chromosome18", ...).
    piv = piv.reindex(sorted(piv.index, key=lambda c: int(re.search(r"(\d+)$", c).group(1))))
    for founder in piv.columns:
        ax.plot(range(len(piv)), 100 * piv[founder], marker="o", alpha=0.7, label=founder)
    ax.axhline(40, color="gray", linestyle="--", linewidth=1, label="40% target")
    ax.set_xticks(range(len(piv)))
    ax.set_xticklabels(piv.index, rotation=45, ha="right")
    ax.set_ylabel("% of reference bp indel-affected")
    title = "D. Indel-affected fraction by chromosome/founder"
    if len(outlier_df):
        title += f"\n(excludes outlier: {', '.join(sorted(outlier_df['founder'].unique()))})"
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=2, loc="lower right")

    fig.suptitle(f"Real {organism} founder indel structure (founder gVCFs, ASM_Start/ASM_End spans)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gvcf-dir", required=True, help="Directory of <founder>.g.vcf.gz files.")
    p.add_argument("--founders", nargs="+", required=True)
    p.add_argument("--chroms", nargs="+", required=True)
    p.add_argument("--out-tsv", required=True, help="Prefix for output TSVs "
                    "(writes <prefix>_summary.tsv and <prefix>_sizes.tsv).")
    p.add_argument("--out-fig", default=None, help="If set, write the 4-panel PNG here "
                    "(requires matplotlib).")
    p.add_argument("--organism", default="maize", help="Used only in the figure's title "
                    "(e.g. 'cassava'); does not affect the measurement itself.")
    return p.parse_args()


def main():
    args = parse_args()
    gvcf_dir = Path(args.gvcf_dir)

    summaries = []
    all_events = []
    fit_events = []  # excludes outlier founders, for the size table
    for founder in args.founders:
        gvcf_path = gvcf_dir / f"{founder}.g.vcf.gz"
        if not gvcf_path.exists():
            print(f"skip {founder}: {gvcf_path} not found", file=sys.stderr)
            continue
        for chrom in args.chroms:
            summary, ev = measure_founder_chrom(gvcf_path, founder, chrom)
            summaries.append(summary)
            all_events.append(ev)
            if founder not in OUTLIER_FOUNDERS:
                fit_events.append(ev)
            print(f"{founder}\t{chrom}\tindel_frac={summary['indel_affected_frac']:.4f}\t"
                  f"n_events={summary['n_indel_events']}", file=sys.stderr)

    summary_df = pd.DataFrame(summaries)
    size_table = build_size_table(fit_events)

    summary_path = f"{args.out_tsv}_summary.tsv"
    sizes_path = f"{args.out_tsv}_sizes.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    size_table.to_csv(sizes_path, sep="\t", index=False)
    print(f"wrote {summary_path} ({len(summary_df)} rows)")
    print(f"wrote {sizes_path} ({len(size_table)} rows)")

    fit_df = summary_df[~summary_df["founder"].isin(OUTLIER_FOUNDERS)]
    if len(fit_df):
        print(f"\nmean indel-affected frac (excl. outliers): "
              f"{fit_df['indel_affected_frac'].mean():.4f}  "
              f"(min={fit_df['indel_affected_frac'].min():.4f}, "
              f"max={fit_df['indel_affected_frac'].max():.4f})")
    if len(size_table):
        ge4k = size_table[size_table["bucket"] >= size_bucket(4096)]["bp_pct"].sum()
        print(f"indel bp from events >=4kb: {ge4k:.1f}%")

    if args.out_fig:
        make_figure(size_table, summary_df, args.out_fig, organism=args.organism)


if __name__ == "__main__":
    main()
