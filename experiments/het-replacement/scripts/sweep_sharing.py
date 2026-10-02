#!/usr/bin/env python
"""Match-set statistics vs the calibration corpus over the SNP-matching parameters (read_snps L,
derived_sfs, sharing_theta): matching founders per row, consecutive-row difference / Jaccard,
and for hybrids the share of consecutive rows with no matching founder in common. Simulated
with the current realism flags (per-read SNPs, per-read bad at 0.031, dist structure)."""
import itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from python.crf.simulate_alleles import simulate  # noqa: E402
import gen_training_data as g  # noqa: E402
K = 25


def stats(d):
    M = d[:, :, :K] == 1
    live = (d[:, :, :K] > -2).all(2)
    a, b = M[:, 1:], M[:, :-1]; ok = live[:, 1:] & live[:, :-1]
    inter = (a & b).sum(2); uni = (a | b).sum(2)
    return {"differ": float((a != b).any(2)[ok].mean()),
            "jaccard": float((1 - inter / np.maximum(uni, 1))[ok].mean()),
            "disjoint": float(((inter == 0) & (uni > 0))[ok].mean()),
            "match_per_row": float(M[live].sum(-1).mean())}
p = json.load(open("../results/repl_params_v3.json"))["suggested_sim_params"]
repl = {k: p[k] for k in g.REPL_KEYS + g.REPL_STR_KEYS if k in p}; repl["repl_max_share"] = int(repl["repl_max_share"])
ds = json.load(open("../results/calib_dist_structure_s200.json"))
calib = json.load(open("../results/row_variation_check.json"))
tgt = {"inb_match": calib["calib_inbred"]["match_per_row"], "inb_differ": calib["calib_inbred"]["differ"],
       "inb_jac": calib["calib_inbred"]["jaccard"], "hyb_disjoint": calib["calib_hybrid"]["disjoint"],
       "hyb_jac": calib["calib_hybrid"]["jaccard"]}
print("calibration", {k: round(v, 3) for k, v in tgt.items()}, flush=True)
res = []
GRID = [tuple(float(v) if "." in v else int(v) for v in c.split(",")) for c in sys.argv[1:]] or list(itertools.product((8, 5, 3), (0.3, 0.15), (4.0, 2.5)))
for L, sfs, theta in GRID:
    row = {"read_snps": L, "derived_sfs": sfs, "sharing_theta": theta}
    for inb, kind in ((1.0, "inb"), (0.0, "hyb")):
        rec = dict(g.RECIPE, bad_frac=0.031, sharing_theta=theta)
        o, *_ = simulate(np.random.default_rng(3), windows=4, inbreeding=inb, indel_model="replacement", obs_table=p["obs_table"],
                         dist_structure=ds, per_read_snps=True, per_read_bad=True, read_snps=L, derived_sfs=sfs, **rec, **repl)
        s = stats(o[:, :o.shape[1] // 512 * 512].reshape(-1, 512, o.shape[-1]))
        row[f"{kind}_match"] = s["match_per_row"]; row[f"{kind}_differ"] = s["differ"]; row[f"{kind}_jac"] = s["jaccard"]
        row[f"{kind}_disjoint"] = s["disjoint"]
    row["err"] = float(sum(abs(row[k] - v) / max(abs(v), 1e-3) for k, v in tgt.items()))
    res.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k in ("read_snps", "derived_sfs", "sharing_theta", "err") or k in tgt}, flush=True)
best = min(res, key=lambda r: r["err"])
print("best", best)
Path("../results/sweep_sharing%s.json" % ("_refine" if sys.argv[1:] else "")).write_text(json.dumps({"target": tgt, "runs": res, "best": best}, indent=1))
