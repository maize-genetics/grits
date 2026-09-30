#!/usr/bin/env python
"""Pick bad_frac so simulated in-panel inbreds miss their true founder on the calibration-corpus
rate of rows (measure_true_founder_miss.py: 1.23%, mean run 1.04). Simulated only."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from python.crf.simulate_alleles import simulate  # noqa: E402
import gen_training_data as g  # noqa: E402


def runs(miss):
    L = []
    for w in miss:
        d = np.diff(np.r_[0, w.astype(np.int8), 0])
        L += list(np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0])
    return np.mean(L) if L else 0.0

K = 25
p = json.load(open("../results/repl_params_v3.json"))["suggested_sim_params"]
repl = {k: p[k] for k in g.REPL_KEYS + g.REPL_STR_KEYS if k in p}; repl["repl_max_share"] = int(repl["repl_max_share"])
ds = json.load(open("../results/calib_dist_structure_s200.json"))
res = {}
for per_read_bad in (False, True):
    for bf in ((0.05,) if not per_read_bad else (0.015, 0.02, 0.025)):
        rec = dict(g.RECIPE, bad_frac=bf)
        o, *rest = simulate(np.random.default_rng(4), windows=8, inbreeding=1.0, indel_model="replacement",
                            obs_table=p["obs_table"], dist_structure=ds, per_read_snps=True, per_read_bad=per_read_bad,
                            emit_row_provenance=True, **rec, **repl)
        pv = rest[-1]
        o = o[:, :o.shape[1] // 512 * 512].reshape(-1, 512, o.shape[-1]); pv = pv[:, :pv.shape[1] // 512 * 512].reshape(-1, 512)
        t = o[:, :, :K]; h = o[:, :, K].astype(int); live = (pv >= 0) & (h >= 0) & (h < K)
        tf = np.take_along_axis(t, np.clip(h, 0, K - 1)[..., None], 2)[..., 0]
        miss = (tf != 1) & live
        key = f"per_read_bad={per_read_bad} bad_frac={bf}"
        res[key] = {"miss_frac": float(miss.sum() / live.sum()), "mean_run_len": float(runs(miss)),
                    "bad_row_share": float(((pv & 16) > 0)[live].mean())}
        print(key, {k: round(v, 4) for k, v in res[key].items()}, flush=True)
Path("../results/fit_bad_frac.json").write_text(json.dumps(res, indent=1))
