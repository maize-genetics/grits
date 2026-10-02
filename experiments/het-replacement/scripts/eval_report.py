#!/usr/bin/env python
"""
Stage 2.7 of experiments/het-replacement/PLAN.md: one report for one or more
checkpoints on the EVALUATION corpus (simulated_validation), from fast_eval_ckpt.py
outputs (exact to the full bed-to-vcf + comparator pipeline).

Per decoder and class (OUT / IDX / MIX x INBRED / HYB / RIL2):
  all-sites error, SNP+RefCall error, error outside truth deletions,
  deletion events fully correct, false-deletion events;
and for every hybrid class, the truth-vs-predicted table of #deleted haplotypes
(0 / 1 / 2) from the decoder's BEDs and the panel / truth caches.

Usage:
  eval_report.py --run TAG=CKPT[:ROUTE] ... [--compare TAG ...] [--depth 0.1]
    --run      evaluates CKPT with fast_eval_ckpt.py under TAG (skipped if its
               results exist); ":route" adds --route (per-sample homo/switch).
    --compare  existing fast_eval result tags to include as comparison rows
               (e.g. ternary_baseline_routed, read-HMM rows are labelled as comparison).
"""
import argparse
import csv
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "simval-corpus/scripts"
sys.path.insert(0, str(SCRIPTS))
import fast_snprc_score as fss  # noqa: E402

WORK = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
PY = "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/.pixi/envs/default/bin/python"
MANIFEST = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv"
SCOPES = ("out", "idx", "mix")
METRICS = [("error_rate", "all", 100), ("snprc_error_rate", "SNP+RC", 100),
           ("error_rate_outside_truth_del", "outsideDEL", 100),
           ("del_event_strict_acc", "DELevents", 100), ("false_del_events", "falseDEL", 1)]


def run_eval(tag, ckpt, route, depth, gpu, input_glob):
    if all(glob.glob(str(WORK / f"results/{tag}_{s}_fast/*.json")) for s in SCOPES):
        return
    cmd = [PY, str(SCRIPTS / "fast_eval_ckpt.py"), "--tag", tag, "--ckpt", ckpt, "--depth", depth,
           "--scope", *SCOPES, "--input-glob", input_glob] + (["--route"] if route else [])
    subprocess.run(cmd, check=True, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": gpu})


def load(tag):
    rows = {}
    for s in SCOPES:
        for f in glob.glob(str(WORK / f"results/{tag}_{s}_fast/*.json")):
            j = json.loads(Path(f).read_text())
            _t, ds, ind, _c = j["row_key"].split("__")
            rows[(ds, ind)] = j
    return rows


def dosage_table(tag, panel, ac, idx, fdel, man, want):
    M = np.zeros((3, 3))
    for (ds, ind), r in man.items():
        if ds != want:
            continue
        beds = [b for sc in SCOPES for b in glob.glob(str(WORK / f"scratch/{tag}_{sc}_fast/{tag}__{ds}__{ind}__*/bed"))]
        if not beds:
            continue
        d = []
        for tp in (r["truth_h1"], r["truth_h2"]):
            t = fss.load_truth(tp)
            c = t["code"]
            cls = np.where(c >= 0, ac[idx, np.clip(c, 0, None)], np.where(c == fss.T_NOVEL, t["ncls"], -1))
            d.append((cls == 3, (c != fss.T_NOINFO) & (c != fss.T_MISSING)))
        f1, f2 = (np.asarray(v) for v in fss.pair_from_bed(panel, beds[0]))
        ok = d[0][1] & d[1][1] & (f1 >= 0) & (f2 >= 0)
        tn = d[0][0].astype(int) + d[1][0]
        pn = fdel[idx, np.clip(f1, 0, None)].astype(int) + fdel[idx, np.clip(f2, 0, None)]
        np.add.at(M, (tn[ok], pn[ok]), 1)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=[])
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--depth", default="0.1")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--input-glob", default="scratch/lift_s200_local20k",
                    help="aligned eval-corpus row dirs (default: -s 200 lift + adaptive projection)")
    args = ap.parse_args()

    tags = []
    for spec in args.run:
        tag, rest = spec.split("=", 1)
        ckpt, _, flag = rest.partition(":")
        run_eval(tag, ckpt, flag == "route", args.depth, args.gpu, args.input_glob)
        tags.append(tag)
    tags += args.compare

    man = {(r["dataset_id"], r["individual"]): r for r in csv.DictReader(open(MANIFEST), delimiter="\t")
           if r["coverage"] == args.depth}
    panel = fss.Panel()
    fa = np.asarray(panel.fa)
    ac = np.asarray(panel.acls)
    idx = np.arange(fa.shape[0])
    fdel = np.zeros(fa.shape, bool)
    for f in range(fa.shape[1]):
        fdel[:, f] = (fa[:, f] >= 0) & (ac[idx, np.clip(fa[:, f], 0, None)] == 3)

    report = {}
    classes = sorted({ds for ds, _ in man if ds.split("-")[1] in ("INBRED", "HYB", "RIL2")})
    print(f"{'class':<11} {'decoder':<34} " + " ".join(f"{lab:>10}" for _, lab, _ in METRICS))
    for ds in classes:
        for tag in tags:
            rows = [v for (d, _), v in load(tag).items() if d == ds]
            if not rows:
                continue
            vals = {lab: float(np.mean([v[k] for v in rows if v.get(k) is not None]) * sc)
                    for k, lab, sc in METRICS if any(v.get(k) is not None for v in rows)}
            report.setdefault(ds, {})[tag] = dict(vals, n=len(rows))
            print(f"{ds:<11} {tag:<34} " + " ".join(f"{vals.get(lab, float('nan')):10.2f}" for _, lab, _ in METRICS))
    print("\nhybrid deletion dosage per class (rows = truth #deleted haplotypes, cols = predicted, row %):")
    for cls in [c for c in classes if c.endswith("-HYB")]:
        for tag in tags:
            M = dosage_table(tag, panel, ac, idx, fdel, man, cls)
            if not M.sum():
                continue
            report.setdefault("dosage", {}).setdefault(cls, {})[tag] = (M / np.maximum(M.sum(1, keepdims=True), 1)).tolist()
            print(f"  {cls:<8} {tag}")
            for d in range(3):
                if M[d].sum():
                    print(f"     truth {d}: share {100 * M[d].sum() / M.sum():5.1f}%   pred 0/1/2 = "
                          + "/".join(f"{100 * x:5.1f}" for x in M[d] / M[d].sum()))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
