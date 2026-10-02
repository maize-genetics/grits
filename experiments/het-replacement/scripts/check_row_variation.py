#!/usr/bin/env python
"""Row-to-row variation of the ternary match sets, simulated vs CALIBRATION corpus: share of
consecutive rows whose match vectors differ, mean Jaccard distance of consecutive match sets,
share of consecutive rows with no matching founder in common. Per kind (inbred / hybrid)."""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
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


ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True); ap.add_argument("--dist-structure", required=True)
ap.add_argument("--calib-root", default="/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/calib_s200_local20k")
ap.add_argument("--windows", type=int, default=6); ap.add_argument("--out", required=True)
a = ap.parse_args()
res = {}
for kind, pat in (("inbred", "IDX-INBRED"), ("hybrid", "IDX-HYB"), ("out_inbred", "OUT-INBRED")):
    ds = [np.load(f, mmap_mode="r")[:300] for f in sorted(glob.glob(f"{a.calib_root}/calib__{pat}__*__0.1x/windowed_k25native_wcount.npy"))]
    res[f"calib_{kind}"] = stats(np.concatenate([np.asarray(x) for x in ds]))
p = json.loads(Path(a.params).read_text()); p = p.get("suggested_sim_params", p)
repl = {k: p[k] for k in g.REPL_KEYS + g.REPL_STR_KEYS if k in p and p[k] is not None}
repl["repl_max_share"] = int(repl["repl_max_share"])
dsj = json.loads(Path(a.dist_structure).read_text())
for pr in (False, True):
    for inb, kind in ((1.0, "inbred"), (0.0, "hybrid")):
        o, *_ = simulate(np.random.default_rng(3), windows=a.windows, inbreeding=inb, indel_model="replacement",
                         obs_table=p["obs_table"], dist_structure=dsj, per_read_snps=pr, **g.RECIPE, **repl)
        o = o[:, :o.shape[1] // 512 * 512].reshape(-1, 512, o.shape[-1])
        res[f"sim_{'perread' if pr else 'shared'}_{kind}"] = stats(o)
for k, v in res.items():
    print(f"{k:<24}", "  ".join(f"{m} {x:.3f}" for m, x in v.items()), flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))
