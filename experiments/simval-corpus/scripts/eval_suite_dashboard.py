#!/usr/bin/env python
"""At-a-glance model dashboard (results/eval_suite/dashboard.html), built from the eval-suite registry.

One self-contained page; the data is inlined as JSON and every comparison (baseline, depth, status,
verdict) is computed in the page, so the baseline and depth can be switched there. Inputs per model:
<model>/class_summary.tsv (core sections) and <model>/genewindow_v2/summary.tsv (optional; gene-window
scorer v2 = tile-on-failure projection, GW_DIR).

Scope numbers (IDX / MIX / OUT) per depth:
  snprc  mean of the class SNP+RC error rates (every class has 5 samples)
  cov    mean % genome inside BED intervals
  gwx    gene-window error / panel-ceiling error, pooled over the scope's haplotype windows; per event
         (event_x_ceiling, the default and primary measure) or per bp (err_x_ceiling), switched on the page.
         Gene-window rates are shown as % (err_pct_capped or event_err_pct); the per-kb columns stay in the
         JSON but are not displayed.
  r2c    % of the way from the random-founder baseline to the panel ceiling (r2c_event_pct / r2c_bp_pct):
         0 = random founder per window, 100 = best panel founder per window.
  swr    switch ratio = sum(predicted founder-pair changes) / sum(reference-path changes)
The status rules and the verdict rules are documented on the page (Rules) and in the template's JS.
"""
import json
import math
import time
from pathlib import Path

import pandas as pd

OUT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/eval_suite")
CLASSES = ["IDX-INBRED", "IDX-HYB", "IDX-RIL2", "MIX-HYB", "MIX-RIL2", "OUT-INBRED", "OUT-HYB", "OUT-RIL2"]
SCOPES = ["IDX", "MIX", "OUT"]
DEPTHS = {0.1: "0.1", 1.0: "1"}

CORE = ["snprc_err", "all_err", "snprc_err_truth_het", "snprc_err_truth_hom", "excess_over_ceiling",
        "ceiling_snprc_err", "genome_covered", "snprc_called_frac", "gaps_100kb", "longest_gap_mb",
        "pred_het_frac", "truth_het_frac", "het_precision", "het_recall", "router_correct",
        "pair_changes", "ref_pair_changes", "bp_precision_10kb", "bp_recall_10kb", "bp_precision_100kb",
        "bp_recall_100kb", "median_segment_mb", "ref_median_segment_mb", "founder_share_tvd",
        "snprc_err_af_singleton", "snprc_err_af_rare_2-3", "snprc_err_af_low_4-6", "snprc_err_af_common_7+",
        "affinity_spearman", "parents_in_aff_top2", "pred_parents_in_share_top2",
        "snprc_err_dist_0-10kb", "snprc_err_dist_10-100kb", "snprc_err_dist_100-1000kb",
        "snprc_err_dist_1000-infkb", "decode_seconds"]
GW = ["err_per_kb", "err_per_kb_capped", "ceil_err_per_kb", "err_x_ceiling", "pct_exact", "pct_exact_called",
      "t50_err_per_kb", "t50_ceil_err_per_kb", "pct_uncalled_bp", "pct_mislocated", "sf_better_pct",
      "pick_rank_median", "wrongpick_rank_median", "wrongpick_excess_bp_median", "n_ranked_median", "excess_share",
      "wrongpick_err_share", "err_pct", "err_pct_capped", "ceil_err_pct_capped", "t50_err_pct_capped",
      "t50_ceil_err_pct_capped", "event_err_pct", "ceil_event_err_pct", "t50_event_err_pct", "t50_ceil_event_err_pct",
      "event_x_ceiling", "err_capped_x_ceiling", "pct_exact_event", "indel_bp_per_event", "rand_event_err_pct",
      "rand_err_pct_capped", "r2c_event_pct", "r2c_bp_pct", "pct_truth_unextractable", "pct_all_N", "pct_projfail_bp"]
GW_DIR = "genewindow_v2"        # gene-window scorer v2 (tile-on-failure projection; eval_genewindow.VERSIONS)


def clean(v):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else round(v, 7)


def core_block(df):
    out = {}
    for dep, dl in DEPTHS.items():
        d = {}
        for c in CLASSES:
            if (dep, c) in df.index:
                row = df.loc[(dep, c)]
                d[c] = {m: clean(row[m]) for m in CORE if m in row.index}
                pc, rc = row.get("pair_changes"), row.get("ref_pair_changes")
                d[c]["swr"] = clean(pc / rc) if rc else None
        for s in SCOPES:
            cs = [d[c] for c in CLASSES if c.startswith(s) and c in d]
            if not cs:
                continue
            agg = {}
            for m in CORE:
                vals = [x[m] for x in cs if x.get(m) is not None]
                agg[m] = clean(sum(vals) / len(vals)) if vals else None
            pc = [x.get("pair_changes") for x in cs]
            rc = [x.get("ref_pair_changes") for x in cs]
            agg["swr"] = clean(sum(pc) / sum(rc)) if None not in pc + rc and sum(rc) else None
            d[s] = agg
        if d:
            out[dl] = d
    return out


def gw_block(path):
    if not path.exists():
        return None
    g = pd.read_csv(path, sep="\t", dtype={"depth": str})
    out = {}
    for _, r in g.iterrows():
        dl = DEPTHS[float(r["depth"])]
        out.setdefault(dl, {})[r["group"]] = {m: clean(r[m]) for m in GW if m in r.index}
    return out


def gw_seconds(path):
    if not path.exists():
        return None
    r = pd.read_csv(path, sep="\t")
    t = r[r.step == "total"].wall_s
    return clean(t.iloc[0]) if len(t) else None


DEFAULT_BASELINE = "peel_hs"                      # promoted 2026-10-23: emE inputs + hybrid switch scale + two-pass peel (preset placedfix_hetdel_hs_peel; was hapswitch_hs); also the default of eval_suite.py dashboard / report_card.sh


def build(reg, baseline=DEFAULT_BASELINE, dest=OUT / "dashboard.html"):
    models = []
    for m, info in reg.items():
        f = OUT / m / "class_summary.tsv"
        if not f.exists():
            continue
        df = pd.read_csv(f, sep="\t").set_index(["depth", "dataset"])
        models.append({"name": m, "note": info.get("note", ""), "tags": info.get("tags", {}),
                       "core": core_block(df), "gw": gw_block(OUT / m / GW_DIR / "summary.tsv"),
                       "gw_seconds": gw_seconds(OUT / m / GW_DIR / "resources.tsv")})
    data = {"generated": time.strftime("%Y-%m-%d %H:%M"), "baseline": baseline if any(
        m["name"] == baseline for m in models) else (models[0]["name"] if models else None),
        "classes": CLASSES, "scopes": SCOPES, "models": models}
    tpl = (Path(__file__).parent / "eval_suite_dashboard_template.html").read_text()
    js = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    dest.write_text(tpl.replace("/*__DATA__*/null", js))
    return dest
