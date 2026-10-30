#!/usr/bin/env python
"""Peel calibration table: per hybrid class mean SNP+RC / all-sites error (%), n rows, compared-site ratio vs the
no-peel decode (pC_hs). calib_summary.py SUF (01 | 1x)"""
import os
import glob
import json
import sys
from collections import defaultdict

W = os.path.join(os.environ.get("GRITS_WORKDIR", "/nonexistent-GRITS_WORKDIR-unset"), "results")
suf, pre = sys.argv[1], "pC"
tags = sorted({p.split("/")[-1].rsplit("_", 2)[0] for p in glob.glob(f"{W}/{pre}_*_{suf}_*_fast")})
res = {}
for t in tags:
    d = defaultdict(list)
    for f in glob.glob(f"{W}/{t}_*_fast/*.json"):
        j = json.load(open(f))
        ds, ind = j["row_key"].split("__")[1:3]
        d[ds].append((ind, j["snprc_error_rate"], j["error_rate"], j["compared_sites"]))
    res[t] = d
base = res.get(f"{pre}_hs_{suf}")
cls = ["IDX-HYB", "MIX-HYB", "OUT-HYB"]
print(f"{'setting':<12}" + "".join(f"{c + ' snprc/all/n/cov':>34}" for c in cls))
for t, d in res.items():
    line = f"{t.replace(pre + '_', '').replace('_' + suf, ''):<12}"
    for c in cls:
        v = sorted(d.get(c, []))
        if not v:
            line += f"{'-':>34}"; continue
        cov = ""
        if base and len(base.get(c, [])) == len(v):
            b = dict((i, n) for i, _, _, n in base[c])
            cov = f"{sum(n for _, _, _, n in v) / sum(b[i] for i, *_ in v):.3f}"
        line += f"{100 * sum(x[1] for x in v) / len(v):>12.2f}{100 * sum(x[2] for x in v) / len(v):>8.2f}{len(v):>4}{cov:>8}  "
    print(line)
