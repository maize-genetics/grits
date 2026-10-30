#!/usr/bin/env python
"""Build the imputation report-card page from every model registered with eval_suite.py.

Reads results/eval_suite/<model>/class_summary.tsv for each model in results/eval_suite/registry.json
and writes results/eval_suite/report.html: a headline table (SNP+RC error with % genome covered, plus
the panel-ceiling reference), then one section per area of the problem, each a models x classes table
per depth. Lower-is-better metrics shade the best full-coverage model per class.
"""
import os
import html
import json
from pathlib import Path

import pandas as pd

OUT = Path(os.environ.get("GRITS_WORKDIR", "/nonexistent-GRITS_WORKDIR-unset")) / "results/eval_suite"
CLASSES = ["IDX-INBRED", "IDX-HYB", "IDX-RIL2", "MIX-HYB", "MIX-RIL2", "OUT-INBRED", "OUT-HYB", "OUT-RIL2"]
DEPTHS = [(0.1, "0.1x"), (1.0, "1x")]
FULL_COVERAGE = 0.9          # models below this genome coverage in a class are not eligible for "best"

# (section id, title, intro, [(metric, label, kind, best)]); kind: pct (x100, 2 dp) | pct0 | num | num2 | sec
SECTIONS = [
    ("accuracy", "Accuracy", "SNP+RefCall genotype error, then split by whether the true genotype is heterozygous.",
     [("snprc_err", "SNP+RC error %", "pct", "min"), ("all_err", "All-sites error %", "pct", "min"),
      ("snprc_err_truth_het", "SNP+RC error at truly het sites %", "pct", "min"),
      ("snprc_err_truth_hom", "SNP+RC error at truly homozygous sites %", "pct", "min")]),
    ("coverage", "Coverage", "How much of the genome gets a call. Error rates only count called sites.",
     [("genome_covered", "Genome inside BED intervals %", "pct0", "max"),
      ("snprc_called_frac", "Reference SNP+RC sites called %", "pct0", "max"),
      ("gaps_100kb", "Uncalled stretches ≥ 100 kb", "num", "min"),
      ("longest_gap_mb", "Longest uncalled stretch (Mb)", "num2", "min")]),
    ("zygosity", "Zygosity", "Allele level on compared sites: calls with two different alleles vs the truth.",
     [("pred_het_frac", "Predicted het sites %", "pct", None), ("truth_het_frac", "True het sites %", "pct", None),
      ("het_precision", "Het precision %", "pct", "max"), ("het_recall", "Het recall %", "pct", "max"),
      ("router_correct", "Router inbred/hybrid call correct %", "pct0", "max")]),
    ("recombination", "Recombination", "Founder-pair changes along the genome vs the reference path (the true path "
     "in panel; the best possible panel path out of panel). Breakpoint precision: share of predicted changes "
     "within the distance of a reference change; recall the reverse.",
     [("pair_changes", "Predicted founder-pair changes", "num", None),
      ("ref_pair_changes", "Reference founder-pair changes", "num", None),
      ("bp_precision_10kb", "Breakpoint precision, 10 kb %", "pct0", "max"),
      ("bp_recall_10kb", "Breakpoint recall, 10 kb %", "pct0", "max"),
      ("bp_precision_100kb", "Breakpoint precision, 100 kb %", "pct0", "max"),
      ("bp_recall_100kb", "Breakpoint recall, 100 kb %", "pct0", "max"),
      ("median_segment_mb", "Median segment (Mb)", "num2", None),
      ("ref_median_segment_mb", "Reference median segment (Mb)", "num2", None)]),
    ("frequency", "Frequency", "Founder share across the genome vs the reference path (total variation distance, "
     "0 = identical), and error by how many of the 25 panel founders carry the non-reference allele.",
     [("founder_share_tvd", "Founder-share distance (TVD)", "num2", "min"),
      ("snprc_err_af_singleton", "Error %, allele in 1 founder", "pct", "min"),
      ("snprc_err_af_rare_2-3", "Error %, allele in 2–3 founders", "pct", "min"),
      ("snprc_err_af_low_4-6", "Error %, allele in 4–6 founders", "pct", "min"),
      ("snprc_err_af_common_7+", "Error %, allele in 7+ founders", "pct", "min")]),
    ("affinity", "Affinity", "The read-based founder affinity every model uses as its prior, vs the reference "
     "founder share (Spearman rank correlation). Parents in top 2: in-panel parents only.",
     [("affinity_spearman", "Affinity vs reference share (Spearman)", "num2", "max"),
      ("parents_in_aff_top2", "True parents in affinity top 2 %", "pct0", "max"),
      ("pred_parents_in_share_top2", "True parents in predicted founder share top 2 %", "pct0", "max")]),
    ("headroom", "Headroom", "SNP+RC error minus the panel ceiling's: how far the model is from the best any "
     "panel-founder path could do.",
     [("excess_over_ceiling", "Excess over ceiling, points", "pct", "min")]),
    ("distance", "Distance to breakpoint", "SNP+RC error by distance to the nearest reference founder change.",
     [("snprc_err_dist_0-10kb", "Error %, < 10 kb", "pct", "min"),
      ("snprc_err_dist_10-100kb", "Error %, 10–100 kb", "pct", "min"),
      ("snprc_err_dist_100-1000kb", "Error %, 100 kb – 1 Mb", "pct", "min"),
      ("snprc_err_dist_1000-infkb", "Error %, > 1 Mb", "pct", "min")]),
    ("cost", "Cost", "Decode wall-clock seconds per sample (fast_eval_ckpt, includes scoring).",
     [("decode_seconds", "Seconds per sample", "num", "min")]),
]


def fmt(v, kind):
    if v is None or pd.isna(v):
        return "–"
    return {"pct": f"{100 * v:.2f}", "pct0": f"{100 * v:.0f}", "num": f"{v:,.0f}", "num2": f"{v:.2f}",
            "sec": f"{v:.0f}"}[kind]


def load():
    reg = json.loads((OUT / "registry.json").read_text())
    models = {}
    for m, info in reg.items():
        f = OUT / m / "class_summary.tsv"
        if f.exists():
            df = pd.read_csv(f, sep="\t").set_index(["depth", "dataset"])
            models[m] = (info, df)
    return models


def table(models, metric, label, kind, best, depth):
    head = "".join(f"<th scope='col'>{c}</th>" for c in CLASSES)
    rows = []
    vals = {m: {c: (df.loc[(depth, c), metric] if (depth, c) in df.index and metric in df.columns else None)
                for c in CLASSES} for m, (_, df) in models.items()}
    cover = {m: {c: (df.loc[(depth, c), "genome_covered"] if (depth, c) in df.index else 0) for c in CLASSES}
             for m, (_, df) in models.items()}
    bests = {}
    if best:
        for c in CLASSES:
            cand = [(round(vals[m][c], 6), m) for m in models if vals[m][c] is not None and not pd.isna(vals[m][c])
                    and cover[m][c] >= FULL_COVERAGE]
            if cand:
                tgt = min(cand)[0] if best == "min" else max(cand)[0]
                bests[c] = {m for v, m in cand if v == tgt}
    for m, (info, _) in models.items():
        tds = "".join(f"<td class='{'best' if m in bests.get(c, ()) else ''}'>{fmt(vals[m][c], kind)}</td>"
                      for c in CLASSES)
        rows.append(f"<tr><th scope='row'><span class='m'>{html.escape(m)}</span></th>{tds}</tr>")
    return (f"<div class='scroll'><table><caption>{html.escape(label)}</caption><thead><tr><th scope='col'>Model"
            f"</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def headline(models, depth):
    head = "".join(f"<th scope='col'>{c}</th>" for c in CLASSES) + "<th scope='col'>Mean</th>"
    rows = []
    best = {}
    for c in CLASSES:
        cand = [(round(df.loc[(depth, c), "snprc_err"], 6), m) for m, (_, df) in models.items()
                if (depth, c) in df.index and df.loc[(depth, c), "genome_covered"] >= FULL_COVERAGE]
        if cand:
            best[c] = {m for v, m in cand if v == min(cand)[0]}
    any_df = next(iter(models.values()))[1]
    ceil = [any_df.loc[(depth, c), "ceiling_snprc_err"] if (depth, c) in any_df.index else None for c in CLASSES]
    tds = "".join(f"<td class='ref'><span class='e'>{fmt(v, 'pct')}</span></td>" for v in ceil)
    rows.append(f"<tr class='ceil'><th scope='row'><span class='m'>panel ceiling</span><span class='sub'>best possible "
                f"panel path</span></th>{tds}<td class='ref'><span class='e'>{fmt(pd.Series(ceil).mean(), 'pct')}"
                f"</span></td></tr>")
    for m, (info, df) in models.items():
        cells, errs = [], []
        for c in CLASSES:
            if (depth, c) not in df.index:
                cells.append("<td>–</td>"); continue
            e, cov = df.loc[(depth, c), "snprc_err"], df.loc[(depth, c), "genome_covered"]
            errs.append(e)
            gap = cov < FULL_COVERAGE
            cells.append(f"<td class='{'best' if m in best.get(c, ()) else ''}{' partial' if gap else ''}'>"
                         f"<span class='e'>{fmt(e, 'pct')}</span><span class='cov{' gap' if gap else ''}'>"
                         f"{100 * cov:.0f}% covered</span></td>")
        partial = any(df.loc[(depth, c), "genome_covered"] < FULL_COVERAGE for c in CLASSES if (depth, c) in df.index)
        mean = f"<td><span class='e'>{fmt(pd.Series(errs).mean(), 'pct')}</span>" + \
               ("<span class='cov gap'>partial coverage</span>" if partial else "") + "</td>"
        rows.append(f"<tr><th scope='row'><span class='m'>{html.escape(m)}</span><span class='sub'>"
                    f"{html.escape(info.get('note', ''))}</span></th>{''.join(cells)}{mean}</tr>")
    return (f"<div class='scroll'><table class='head'><caption>SNP+RefCall error %, {dict(DEPTHS)[depth]} · lower is "
            f"better</caption><thead><tr><th scope='col'>Model</th>{head}</tr></thead><tbody>{''.join(rows)}"
            f"</tbody></table></div>")


def main():
    models = load()
    nav = "".join(f"<a href='#{sid}'>{t}</a>" for sid, t, _, _ in SECTIONS)
    body = [f"<section id='headline'><h2>Headline</h2><p class='intro'>Shaded: best model in each class among those "
            f"covering at least {int(100 * FULL_COVERAGE)}% of the genome. Amber: partial coverage — error counts only "
            f"called sites, so it is not comparable.</p>"
            + "".join(headline(models, d) for d, _ in DEPTHS) + "</section>"]
    for sid, title, intro, metrics in SECTIONS:
        parts = [f"<section id='{sid}'><h2>{title}</h2><p class='intro'>{intro}</p>"]
        for d, dl in DEPTHS:
            parts.append(f"<h3>{dl}</h3>")
            parts += [table(models, *mt, d) for mt in metrics]
        body.append("".join(parts) + "</section>")
    tpl = (Path(__file__).parent / "eval_suite_report_template.html").read_text()
    page = tpl.replace("{{NAV}}", nav).replace("{{BODY}}", "".join(body)).replace(
        "{{MODELS}}", str(len(models)))
    (OUT / "report.html").write_text(page)
    print(OUT / "report.html")


if __name__ == "__main__":
    main()
