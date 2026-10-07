"""Gene-window eval: aggregate per_hapwin.tsv into per-class / per-sample tables.

Definitions (bases): called = seq length - N; errors = mismatches + ins + dels + unaligned.
  err_per_kb   pooled: 1000 * sum(errors) / sum(called) over scored haplotype-windows
  identity     pooled: sum(matches) / sum(matches + errors); mean_identity = per-window mean of the same
  exact        scored windows with errors == 0 and no N; exact_called = errors == 0 (N allowed)
  uncalled     % of all window bases that are N: BED gap / projection failure / assembly N
  mislocated   % of windows with an alignment whose primary hit is off the window's chromosome
  ceiling      the best single founder for the same (window, truth line), over the same scored windows

Usage: python gw_report.py RESULTS_DIR
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CLASSES = ["IDX-INBRED", "IDX-HYB", "IDX-RIL2", "MIX-HYB", "MIX-RIL2", "OUT-INBRED", "OUT-HYB", "OUT-RIL2"]
SNPRC = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/eval_suite/v6cfullfe_homo/per_sample.tsv"


def summarize(d):
    sc = d[d.state == "scored"]
    called = (sc.qlen - sc.query_N).sum()
    ident = sc.matches / (sc.matches + sc.errors).replace(0, np.nan)
    ce = sc[sc.ceil_state == "scored"]
    ce_called = ce.ceil_called.astype(float).sum()
    ce_err = ce.ceil_errors.astype(float)
    ce_match = ce.ceil_matches.astype(float)
    tot = d.qlen.sum()
    has_aln = d[d.state.isin(["scored", "mislocated"])]
    out = {}
    for K in (50, 20):
        an = sc[sc[f"t{K}_anchored"] == 1]
        tc = an[f"t{K}_called"].sum()
        sw = an.n_switches > 0
        diff = an.errors - an[f"t{K}_errors"]
        ce_t = an[an[f"ceil_t{K}_founder"].astype(str) != "."]
        out.update({
            f"t{K}_err_per_kb": 1000 * an[f"t{K}_errors"].sum() / tc,
            f"t{K}_identity": an[f"t{K}_matches"].sum() / (an[f"t{K}_matches"].sum() + an[f"t{K}_errors"].sum()),
            f"t{K}_pct_exact": 100 * (an[f"t{K}_errors"] == 0).mean(),
            f"t{K}_mean_trimmed_bp": an[f"t{K}_trimmed_bp"].mean(),
            f"t{K}_pct_unanchored": 100 * (sc[f"t{K}_anchored"] == 0).mean(),
            f"t{K}_full_err_per_kb_anchored": 1000 * an.errors.sum() / (an.qlen - an.query_N).sum(),
            f"t{K}_pct_diff_from_switch_win": 100 * diff[sw].sum() / diff.sum() if diff.sum() else np.nan,
            f"t{K}_model_err_per_kb_ceilrows": 1000 * ce_t[f"t{K}_errors"].sum() / ce_t[f"t{K}_called"].sum(),
            f"t{K}_ceil_err_per_kb": 1000 * ce_t[f"ceil_t{K}_errors"].astype(float).sum() /
                                     ce_t[f"ceil_t{K}_called"].astype(float).sum(),
            f"t{K}_ceil_pct_exact": 100 * (ce_t[f"ceil_t{K}_errors"].astype(float) == 0).mean(),
        })
    return pd.Series({
        **{"n_hapwin": len(d)},
        "scored": len(sc),
        "pct_all_N": 100 * (d.state == "all_N").mean(),
        "pct_mislocated": 100 * (has_aln.state == "mislocated").mean() if len(has_aln) else np.nan,
        "pct_uncalled_bp": 100 * d.query_N.sum() / tot,
        "pct_gap_bp": 100 * d.gap_bases.sum() / tot,
        "pct_projfail_bp": 100 * d.failed_bases.sum() / tot,
        "identity": sc.matches.sum() / (sc.matches.sum() + sc.errors.sum()),
        "mean_identity": ident.mean(),
        "err_per_kb": 1000 * sc.errors.sum() / called,
        "mism_per_kb": 1000 * sc.mismatches.sum() / called,
        "indel_per_kb": 1000 * (sc.ins.sum() + sc.dels.sum()) / called,
        "unaln_per_kb": 1000 * sc.unaligned.sum() / called,
        "pct_exact": 100 * ((sc.errors == 0) & (sc.query_N == 0)).mean(),
        "pct_exact_called": 100 * (sc.errors == 0).mean(),
        "model_err_per_kb_ceilrows": 1000 * ce.errors.sum() / (ce.qlen - ce.query_N).sum(),
        "ceil_err_per_kb": 1000 * ce_err.sum() / ce_called,
        "ceil_identity": ce_match.sum() / (ce_match.sum() + ce_err.sum()),
        "ceil_pct_exact_called": 100 * (ce_err == 0).mean(),
        "pct_win_switch": 100 * (sc.n_switches > 0).mean(),
        "err_per_kb_noswitch": 1000 * sc[sc.n_switches == 0].errors.sum() /
                               (sc[sc.n_switches == 0].qlen - sc[sc.n_switches == 0].query_N).sum(),
        "err_per_kb_switch": (1000 * sc[sc.n_switches > 0].errors.sum() /
                              (sc[sc.n_switches > 0].qlen - sc[sc.n_switches > 0].query_N).sum())
                             if (sc.n_switches > 0).any() else np.nan,
        "pct_exact_called_noswitch": 100 * (sc[sc.n_switches == 0].errors == 0).mean(),
        "pct_exact_called_switch": 100 * (sc[sc.n_switches > 0].errors == 0).mean() if (sc.n_switches > 0).any()
        else np.nan,
        "pct_errors_in_switch_win": 100 * sc[sc.n_switches > 0].errors.sum() / max(1, sc.errors.sum()),
        "ceil_err_per_kb_noswitch": 1000 * ce[ce.n_switches == 0].ceil_errors.astype(float).sum() /
                                    ce[ce.n_switches == 0].ceil_called.astype(float).sum(),
        "ceil_err_per_kb_switch": (1000 * ce[ce.n_switches > 0].ceil_errors.astype(float).sum() /
                                   ce[ce.n_switches > 0].ceil_called.astype(float).sum())
                                  if (ce.n_switches > 0).any() else np.nan,
        "pct_at_ceiling": 100 * (ce.errors <= ce_err).mean(),
        "pct_mislocated_exact": 100 * ((has_aln.state == "mislocated") & (has_aln.errors == 0)).mean()
        if len(has_aln) else np.nan,
        "pct_mislocated_to_scaffold": 100 * ((has_aln.state == "mislocated") &
                                             ~has_aln.contig.astype(str).str.fullmatch(r"chr\d+")).mean()
        if len(has_aln) else np.nan,
        **out,
    })


def main():
    warnings.filterwarnings("ignore")
    rd = Path(sys.argv[1])
    d = pd.read_csv(rd / "per_hapwin.tsv", sep="\t", dtype={"depth": str})
    d["class"] = pd.Categorical(d["class"], CLASSES, ordered=True)
    pc = d.groupby(["depth", "class"], observed=True).apply(summarize, include_groups=False).reset_index()
    pc.to_csv(rd / "per_class.tsv", sep="\t", index=False, float_format="%.4g")
    ps = d.groupby(["depth", "class", "sample"], observed=True).apply(summarize, include_groups=False).reset_index()
    sn = pd.read_csv(SNPRC, sep="\t", dtype={"depth": str})
    sn = sn.rename(columns={"dataset": "class", "individual": "sample"})[["class", "sample", "depth", "snprc_err",
                                                                          "all_err"]]
    sn["class"] = pd.Categorical(sn["class"], CLASSES, ordered=True)
    ps = ps.merge(sn, on=["class", "sample", "depth"], how="left")
    ps.to_csv(rd / "per_sample.tsv", sep="\t", index=False, float_format="%.4g")
    lines = []
    for depth, g in ps.groupby("depth"):
        for m in ("err_per_kb", "t50_err_per_kb", "t20_err_per_kb"):
            r, p = spearmanr(g[m], g.snprc_err)
            r2, p2 = spearmanr(g[m], g.all_err)
            lines.append(f"depth {depth} {m}: n={len(g)} spearman vs snprc_err={r:.3f} (p={p:.2g}); "
                         f"vs all_err={r2:.3f} (p={p2:.2g})")
            lines.append("    per class vs snprc_err: " + ", ".join(
                f"{cls} {spearmanr(gg[m], gg.snprc_err)[0]:.2f}" for cls, gg in g.groupby("class", observed=True)))
    (rd / "snprc_agreement.txt").write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print(pc.to_string(float_format=lambda x: f"{x:.3g}"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
