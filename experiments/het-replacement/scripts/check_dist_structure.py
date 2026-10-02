#!/usr/bin/env python
"""Simulated vs calibration anchor-distance structure: simulate a few windows with the build's
recipe (with and without --dist-structure) and compute measure_dist_structure's validation
stats in code space (mode split per state, distinct/n) plus per-state code quantiles and
consecutive-row persistence. Simulated data only; calibration numbers come from the JSON."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from python.crf.simulate_alleles import simulate  # noqa: E402
import gen_training_data as g  # noqa: E402

K = 25


def stats(tern, code):
    live = tern.max(1) > -2
    tern, code = tern[live], code[live].astype(np.int64)
    cnt = np.zeros((len(code), 128), np.int32)
    np.add.at(cnt, (np.repeat(np.arange(len(code)), K), code.ravel()), 1)
    m = cnt.argmax(1)
    side = np.sign(code - m[:, None])
    out = {}
    for s in (1, 0, -1):
        ms = tern == s
        n = ms.sum()
        dn = [len(np.unique(code[i][ms[i]])) / ms[i].sum() for i in np.nonzero(ms.sum(1) >= 3)[0][:20000]]
        out[str(s)] = {"at_mode": (ms & (side == 0)).sum() / n, "below": (ms & (side < 0)).sum() / n,
                       "above": (ms & (side > 0)).sum() / n, "distinct_per_n": float(np.mean(dn)),
                       "code_p10_50_90": np.percentile(code[ms], [10, 50, 90]).tolist()}
    out["founders_at_mode"] = float(cnt.max(1).mean())
    out["consecutive_same"] = float((code[1:] == code[:-1]).mean())
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True); ap.add_argument("--dist-structure", required=True)
ap.add_argument("--windows", type=int, default=6); ap.add_argument("--depth-factor", type=float, default=1.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
p = json.loads(Path(a.params).read_text()); p = p.get("suggested_sim_params", p)
repl = {k: p[k] for k in g.REPL_KEYS + g.REPL_STR_KEYS if k in p and p[k] is not None}
repl["repl_max_share"] = int(repl["repl_max_share"])
obs = dict(p["obs_table"]); f = a.depth_factor
recipe = dict(g.RECIPE)
if f != 1.0:
    recipe["min_cross"] = max(0, int(round(g.RECIPE["min_cross"] / f))); recipe["max_cross"] = max(1, int(round(g.RECIPE["max_cross"] / f)))
    recipe["ancestor_crossovers"] = 8.0 / f
    for k, sc in (("repl_mean_len", f), ("repl_shift_sites", f), ("repl_rate", 1.0 / f)):
        repl[k] = repl[k] * sc
    obs["bp_per_site"] = obs["bp_per_site"] / f
ds = json.loads(Path(a.dist_structure).read_text())
res = {"calibration": ds["validation"]}
for name, dstruct in (("sim_old", None), ("sim_new", ds)):
    o, *_ = simulate(np.random.default_rng(11), windows=a.windows, inbreeding=0.5, indel_model="replacement",
                     obs_table=obs, dist_structure=dstruct, **recipe, **repl)
    o = o.reshape(-1, o.shape[-1])
    res[name] = stats(o[:, :K], o[:, K + 2:2 * K + 2])
    print(name, json.dumps(res[name], default=float), flush=True)
print("calibration", json.dumps(ds["validation"]))
Path(a.out).write_text(json.dumps(res, indent=1, default=float))
