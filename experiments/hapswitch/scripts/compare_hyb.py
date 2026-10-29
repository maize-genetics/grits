#!/usr/bin/env python
"""Hybrid-class table (SNP+RC / all-sites / GW per event + ceiling/random, routing, coverage) vs a baseline,
plus a bit-identity check on every non-hybrid row.  compare_hyb.py BASE MODEL [MODEL...]"""
import sys
import pandas as pd
R = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/eval_suite"
models = sys.argv[1:]
ps = {m: pd.read_csv(f"{R}/{m}/per_sample.tsv", sep="\t") for m in models}
gw = {m: pd.read_csv(f"{R}/{m}/genewindow_v2/summary.tsv", sep="\t") for m in models}
gws = {m: pd.read_csv(f"{R}/{m}/genewindow_v2/per_sample.tsv", sep="\t") for m in models}
rows = []
for dep in (0.1, 1.0):
    for c in ("IDX-HYB", "MIX-HYB", "OUT-HYB"):
        for m in models:
            d = ps[m][(ps[m].dataset == c) & (ps[m].depth == dep)]
            g = gw[m][(gw[m].level == "class") & (gw[m].group == c) & (gw[m].depth == dep)].iloc[0]
            rows.append(dict(depth=dep, cls=c, model=m, snprc=100 * d.snprc_err.mean(), all=100 * d.all_err.mean(),
                             gw_event=g.event_err_pct, gw_ceil=g.ceil_event_err_pct, gw_rand=g.rand_event_err_pct,
                             cov=d.genome_covered.mean(), hyb=f"{int(d.router_hybrid.sum())}/{len(d)}",
                             pair_chg=d.pair_changes.mean(), ref_chg=d.ref_pair_changes.mean()))
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(3).to_string(index=False))
b = models[0]
for m in models[1:]:
    a, c = ps[b].set_index(["dataset", "individual", "depth"]), ps[m].set_index(["dataset", "individual", "depth"])
    o = a.index[~a.index.get_level_values(0).str.endswith("-HYB")]
    diff = (a.loc[o, ["snprc_err", "all_err"]] - c.loc[o, ["snprc_err", "all_err"]]).abs().max().max()
    ga, gc = gws[b].set_index(["class", "sample", "depth"]), gws[m].set_index(["class", "sample", "depth"])
    og = ga.index[~ga.index.get_level_values(0).str.endswith("-HYB")]
    gd = (ga.loc[og, "event_err_pct"] - gc.loc[og, "event_err_pct"]).abs().max()
    print(f"{m}: non-hybrid rows {len(o)}, max |diff| core {diff:.2e}, gw event {gd:.2e}")
print("\nper-row SNP+RC (base -> model) / GW event")
for dep in (0.1, 1.0):
    hy = ps[b].dataset.str.endswith("-HYB") & (ps[b].depth == dep)
    t = pd.concat({m: ps[m][ps[m].dataset.str.endswith("-HYB") & (ps[m].depth == dep)].set_index(["dataset", "individual"]).snprc_err * 100 for m in models}, axis=1)
    u = pd.concat({m: gws[m][gws[m]["class"].str.endswith("-HYB") & (gws[m].depth == dep)].rename(columns={"class": "dataset", "sample": "individual"}).set_index(["dataset", "individual"]).event_err_pct for m in models}, axis=1)
    print(dep); print(t.round(2).join(u.round(3), rsuffix="_gw").to_string())
