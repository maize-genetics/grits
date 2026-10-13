#!/usr/bin/env python
"""Confound check for feature_errors.py: SNP+RC error by (primary feature x arm-position bin), so
genic-vs-TE differences can be read within a fixed distance-from-centromere stratum (a recombination
proxy). Same per_record() (validated exact vs score_arrays) and annotation as feature_errors.py.
Output: crosstab_<part>.tsv (depth, dataset, individual, model, primary, arm, snprc_sites, snprc_err).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fast_snprc_score as fss  # noqa: E402
from feature_errors import MANIFEST, OUT, TAGS, bed_dir, ceiling_path, per_record  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--part", required=True)
    ap.add_argument("--depths", nargs="+", default=["0.1"])
    ap.add_argument("--models", nargs="+", default=["v6cfullfe_homo", "diploid_affinity", "ceiling"])
    a = ap.parse_args()
    ann = np.load(OUT / "panel_feature_annotation.npz")
    meta = json.loads((OUT / "feature_annotation_meta.json").read_text())
    nA = len(meta["ARM_NAMES"]); nP = len(meta["PRIMARY"])
    code = ann["primary"].astype(np.int32) * nA + ann["armbin"]
    panel = fss.Panel()
    panel.fa = np.asarray(panel.fa)
    panel.acls = np.asarray(panel.acls)
    rows_m = [r for r in csv.DictReader(open(MANIFEST), delimiter="\t")
              if r["coverage"] in a.depths and r["dataset_id"] in a.classes]
    fo = open(OUT / f"crosstab_{a.part}.tsv", "w")
    fo.write("depth\tdataset\tindividual\tmodel\tprimary\tarm\tsnprc_sites\tsnprc_err\n")
    for r in rows_m:
        dep, ds, ind = r["coverage"], r["dataset_id"], r["individual"]
        t1, t2 = fss.load_truth(r["truth_h1"]), fss.load_truth(r["truth_h2"])
        for m in a.models:
            if m == "ceiling":
                f1 = ceiling_path(panel, r["truth_h1"], t1["code"])
                f2 = ceiling_path(panel, r["truth_h2"], t2["code"])
            else:
                f1, f2 = fss.pair_from_bed(panel, bed_dir(TAGS[dep][m], ds, ind, dep))
            rows, match, snprc, _ = per_record(panel, t1, t2, f1, f2)
            sr = rows[snprc]
            s = np.bincount(code[sr], minlength=nP * nA)
            e = np.bincount(code[sr], weights=~match[snprc], minlength=nP * nA)
            for k in np.nonzero(s)[0]:
                fo.write(f"{dep}\t{ds}\t{ind}\t{m}\t{meta['PRIMARY'][k // nA]}\t{meta['ARM_NAMES'][k % nA]}\t"
                         f"{s[k]}\t{int(e[k])}\n")
            fo.flush()
        print("done", dep, ds, ind, flush=True)


if __name__ == "__main__":
    main()
