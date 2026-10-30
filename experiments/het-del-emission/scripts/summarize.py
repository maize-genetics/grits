#!/usr/bin/env python
"""Per (tag, dataset) means from hetdel_eval.py result JSONs: SNP+RC err, all-sites err, coverage,
dosage-confusion row accuracies (truth 0/1/2 deleted haplotypes). Usage: summarize.py TAG [TAG ...]"""
import os
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path(os.environ.get("GRITS_WORKDIR", "/nonexistent-GRITS_WORKDIR-unset")) / "results"
print(f"{'tag':<22}{'dataset':<12}{'n':>3}{'snprc%':>9}{'all%':>8}{'cov%':>7}{'dos0%':>7}{'dos1%':>7}{'dos2%':>7}")
for tag in sys.argv[1:]:
    by = defaultdict(list)
    for f in sorted(glob.glob(str(RES / f"{tag}_*_fast" / f"{tag}__*.json"))):
        if not Path(f).parent.name in (f"{tag}_idx_fast", f"{tag}_out_fast", f"{tag}_mix_fast"):
            continue
        d = json.load(open(f))
        by[d["row_key"].split("__")[1]].append(d)
    for ds in sorted(by):
        rs = by[ds]
        c = np.array([r["dosage_confusion"] for r in rs if "dosage_confusion" in r], float)
        acc = (np.diagonal(c, axis1=1, axis2=2) / c.sum(2).clip(1)).mean(0) * 100 if len(c) else [np.nan] * 3
        print(f"{tag:<22}{ds:<12}{len(rs):>3}{100*np.mean([r['snprc_error_rate'] for r in rs]):>9.3f}"
              f"{100*np.mean([r['error_rate'] for r in rs]):>8.2f}{100*np.mean([r['genome_covered'] for r in rs]):>7.1f}"
              f"{acc[0]:>7.1f}{acc[1]:>7.1f}{acc[2]:>7.1f}")
