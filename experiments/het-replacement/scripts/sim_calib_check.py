#!/usr/bin/env python
"""
Stage 2.4 of experiments/het-replacement/PLAN.md: does simulated training data at the
calibrated parameters look like the calibration corpus?

Simulates the same individuals twice with one seed: once exactly (truth deletion state is
dist > 0) and once with the measured --obs-table (what the model will see). Reads are
identical between the two because the observation is drawn after row sampling, so the
exact run is the truth for the observed run. Compares against measure_real_stats.py output:
  * deleted fraction of founder-sites, #deleted-founders spectrum, pair both/exactly-one
  * ternary match/div/del shares, distance code p5/p50/p95 per state
  * P(-1 | deleted, not matched), P(-1 | present, not matched), P(match | deleted)
  * hybrids: relative row density at 0/1/2 deleted haplotypes
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from python.crf.simulate_alleles import simulate  # noqa: E402
from gen_training_data import RECIPE, REPL_KEYS, REPL_STR_KEYS  # noqa: E402

K = 25


def pct(x):
    return [int(v) for v in np.percentile(x, [5, 50, 95])] if x.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="measure_real_stats.py JSON")
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    real = json.loads(Path(args.params).read_text())
    sp = real["suggested_sim_params"]
    repl = {k: sp[k] for k in REPL_KEYS + REPL_STR_KEYS if k in sp}
    for kv in args.override:
        k, v = kv.split("=")
        repl[k] = v if k in REPL_STR_KEYS else float(v)
    repl["repl_max_share"] = int(repl["repl_max_share"])
    obs = sp["obs_table"]
    rep = {"repl": repl}
    for inb in (1.0, 0.0):
        kw = dict(windows=args.windows, inbreeding=inb, indel_model="replacement", **RECIPE, **repl)
        ex = simulate(np.random.default_rng(11), **kw)[0]
        ob = simulate(np.random.default_rng(11), obs_table=obs, **kw)[0]
        te, de, to = ex[..., :K], ex[..., K + 2:2 * K + 2], ob[..., :K]
        do = ob[..., K + 2:2 * K + 2]
        live = (te != -2).all(-1)
        dele = (de > 0) & live[..., None]
        pres = (de == 0) & live[..., None]
        nm = to != 1
        r = {"deleted_founder_site_frac": float(dele.sum() / (dele.sum() + pres.sum())),
             "tern_share": {s: float((to[live] == v).mean()) for s, v in (("match", 1), ("div", 0), ("del", -1))},
             "dist_code_p5_50_95": {s: pct(do[live][to[live] == v]) for s, v in (("del", -1), ("div", 0), ("match", 1))},
             "p_minus1_given_del_nonmatch": float((to[dele & nm] == -1).mean()),
             "p_minus1_given_present_nonmatch": float((to[pres & nm] == -1).mean()),
             "p_match_given_del": float((to[dele] == 1).mean())}
        nd = dele.sum(-1)[live]
        r["n_deleted_mean"] = float(nd.mean())
        r["pair_both_deleted"] = float(np.mean(nd / K * (nd - 1) / (K - 1)))
        r["pair_exactly_one"] = float(np.mean(2 * nd / K * (K - nd) / (K - 1)))
        if inb == 0.0:
            h1, h2 = ex[..., K].astype(int), ex[..., K + 1].astype(int)
            ok = live & (h1 >= 0) & (h2 >= 0)
            g = lambda f: np.take_along_axis(de > 0, np.clip(f, 0, K - 1)[..., None], 2)[..., 0]
            dos = g(h1).astype(int) + g(h2)
            rows = np.bincount(dos[ok], minlength=3).astype(float)
            r["row_share_by_dosage"] = (rows / rows.sum()).tolist()
        rep["inbred" if inb == 1.0 else "hybrid"] = r
    ro = real["ternary_obs"]
    cd = np.asarray(ro["del_by_len_counts"], float)
    cp = np.asarray(ro["present_counts"], float)
    feats = [s["features"] for s in real["samples"].values() if "features" in s]
    rep["real"] = {"deleted_bp_frac": sp.get("target_deleted_bp_frac"),
                   "panel": {k: v for k, v in real["panel"].items() if k != "n_deleted_spectrum"},
                   "tern_share": {s: float(np.mean([f["tern_frac"][s] for f in feats])) for s in ("match", "div", "del")},
                   "p_minus1_given_del_nonmatch": float(cd[:, 0].sum() / cd[:, :2].sum()),
                   "p_minus1_given_present_nonmatch": float(cp[0] / cp[:2].sum()),
                   "p_match_given_del": float(cd[:, 2].sum() / cd.sum()),
                   "hyb_rel_read_density_by_dosage": {k: float(np.mean([s["read_mix_by_dosage"][k]["rel_read_density"]
                                                        for n, s in real["samples"].items() if "HYB" in n and "IDX" not in n]))
                                                      for k in ("0", "1", "2")}}
    print(json.dumps(rep, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
