#!/usr/bin/env python
"""Split SNP+RefCall error by the regions old v6 called vs left null, for several models and the
panel ceiling.

Old v6 (26-state CRF with a null founder) leaves no call wherever it decodes null. Its BED defines,
per panel record, "called" vs "null". Every other model, and the panel ceiling (exact haploid Viterbi
per truth haplotype over the panel, mismatch cost 1, switch cost --lam; see panel_ceiling.py), is
scored on all records, on old v6's called records only, and on its null records only. This shows
whether full-coverage models already match old v6 where it was confident, and how low any panel-path
model could go in the regions it skipped.

Reads existing BEDs only. Output: TSV, one row per (depth, sample, model, subset).
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fast_snprc_score as fss  # noqa: E402
from panel_ceiling import best_path  # noqa: E402

MANIFEST = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv"
SCR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch")
TAGS = {"0.1": {"old_v6": "v6_B", "v6_nonull": "v6_B_nonull", "v6cfullfe": "v6cfullfe_B",
                "tern": "s200_ternary_routed", "da": "diploid_affinity_s200"},
        "1.0": {"old_v6": "v6_B_1x", "v6_nonull": "v6_B_nonull_1x", "v6cfullfe": "v6cfullfe_B_1x",
                "tern": "tern_1x", "da": "diploid_affinity_s200_1x"}}


def bed_dir(tag, ds, ind, dep):
    hits = glob.glob(str(SCR / f"{tag}_*_fast" / f"*__{ds}__{ind}__{dep}x" / "bed"))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=50.0)
    ap.add_argument("--classes", nargs="+", default=["OUT-INBRED", "OUT-HYB", "OUT-RIL2", "MIX-HYB",
                                                     "MIX-RIL2", "IDX-INBRED", "IDX-HYB", "IDX-RIL2"])
    a = ap.parse_args()
    panel = fss.Panel()
    panel.fa = np.asarray(panel.fa)
    rows = [r for r in csv.DictReader(open(MANIFEST), delimiter="\t")
            if r["coverage"] in TAGS and r["dataset_id"] in a.classes]
    rows.sort(key=lambda r: (a.classes.index(r["dataset_id"]), r["individual"], r["coverage"]))
    out = open(a.out, "w")
    out.write("depth\tdataset\tindividual\tmodel\tsubset\tsnprc_err\tsnprc_sites\tswitches_per_hap\n")
    paths = {}
    for r in rows:
        dep, ds, ind = r["coverage"], r["dataset_id"], r["individual"]
        dirs = {m: bed_dir(t, ds, ind, dep) for m, t in TAGS[dep].items()}
        if not all(dirs.values()):
            print("skip", dep, ds, ind, [m for m, d in dirs.items() if not d], flush=True)
            continue
        t1, t2 = fss.load_truth(r["truth_h1"]), fss.load_truth(r["truth_h2"])
        for tp, t in ((r["truth_h1"], t1), (r["truth_h2"], t2)):
            if tp not in paths:
                paths[tp] = best_path(panel, t["code"], a.lam)
        pred = {m: fss.pair_from_bed(panel, d) for m, d in dirs.items()}
        pred["ceiling"] = (paths[r["truth_h1"]][0], paths[r["truth_h2"]][0])
        sw = (paths[r["truth_h1"]][1] + paths[r["truth_h2"]][1]) / 2
        called = (pred["old_v6"][0] >= 0) & (pred["old_v6"][1] >= 0)
        for m, (f1, f2) in pred.items():
            for name, mask in (("all", None), ("v6_called", called), ("v6_null", ~called)):
                g1, g2 = f1.copy(), f2.copy()
                if mask is not None:
                    g1[~mask] = -1
                    g2[~mask] = -1
                res = fss.score_arrays(panel, t1, t2, g1, g2)
                e = res["snprc_error_rate"]
                out.write(f"{dep}\t{ds}\t{ind}\t{m}\t{name}\t{'' if e is None else e}\t"
                          f"{res['snprc_compared_sites']}\t{sw if m == 'ceiling' else ''}\n")
        out.flush()
        print("done", dep, ds, ind, flush=True)


if __name__ == "__main__":
    main()
