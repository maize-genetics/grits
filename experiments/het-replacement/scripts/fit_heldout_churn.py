#!/usr/bin/env python
"""Calibrate the ancestral-lineage switch rate (simulate ancestor_crossovers) for held-out
augmentation: "closest-founder churn" = share of adjacent 128-row chunks of a window whose
best-matching founder (most MATCH rows) changes. Real: CALIBRATION corpus OUT-INBRED (out-of-panel)
and IDX-INBRED (in-panel). Simulated inbreds with the next build's realism flags: in-panel windows,
and windows whose true founder is hidden (its column dropped, as heldout_augment_v2.py does)."""
import glob, itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from python.crf.simulate_alleles import simulate  # noqa: E402
import gen_training_data as g  # noqa: E402
CH = 128


def churn(tern, tol=0.9):
    """tern [W,512,K] -> share of adjacent chunk pairs whose near-best founder sets (>= tol x the
    chunk's top match count) are disjoint. Robust to exact ties (simulated lineage-mates are exact
    copies; real relatives are near-copies), which a single argmax is not."""
    M = (tern == 1).reshape(tern.shape[0], -1, CH, tern.shape[2]).sum(2)     # [W,4,K]
    top = M >= np.maximum(tol * M.max(-1, keepdims=True), 1)
    return float((~(top[:, 1:] & top[:, :-1]).any(-1)).mean())


root = "/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/calib_s200_local20k"
res = {}
for cls in ("IDX-INBRED", "OUT-INBRED"):
    v = [churn(np.asarray(np.load(f, mmap_mode="r")[:, :, :25])) for f in sorted(glob.glob(f"{root}/calib__{cls}__*__0.1x/windowed_k25native_wcount.npy"))]
    res[f"calib_{cls}"] = float(np.mean(v))
print({k: round(v, 3) for k, v in res.items()}, flush=True)

p = json.load(open("../results/repl_params_v3.json"))["suggested_sim_params"]
repl = {k: p[k] for k in g.REPL_KEYS + g.REPL_STR_KEYS if k in p}; repl["repl_max_share"] = int(repl["repl_max_share"])
ds = json.load(open("../results/calib_dist_structure_s200.json"))
K2 = 27
for axc in [float(v) for v in sys.argv[1:]] or (8.0, 16.0, 32.0, 64.0):
    rec = dict(g.RECIPE, founders=K2, bad_frac=0.031, read_snps=5, derived_sfs=0.15, ancestor_crossovers=axc)
    o, *_ = simulate(np.random.default_rng(7), windows=10, inbreeding=1.0, indel_model="replacement", obs_table=p["obs_table"],
                     dist_structure=ds, per_read_snps=True, per_read_bad=True, **rec, **repl)
    o = o[:, :o.shape[1] // 512 * 512].reshape(o.shape[0], -1, 512, o.shape[-1])       # [ind,G,512,C]
    tern = o[..., :K2]; lab = o[..., K2].astype(int)
    inpanel, hidden = [], []
    for i in range(o.shape[0]):
        f, c = np.unique(lab[i][lab[i] >= 0], return_counts=True)
        h = f[np.argmax(c)]                                    # hide the individual's main founder
        maj = np.array([np.bincount(np.clip(w, 0, K2 - 1), minlength=K2).argmax() for w in lab[i]])
        vis = np.setdiff1d(np.arange(K2), [h])
        hw = maj == h
        if hw.any():
            hidden.append(churn(tern[i][hw][..., vis]))
        if (~hw).any():
            inpanel.append(churn(tern[i][~hw][..., vis]))
    r = {"sim_inpanel": float(np.mean(inpanel)), "sim_hidden": float(np.mean(hidden))}
    res[f"ancestor_crossovers={axc}"] = r
    print(f"ancestor_crossovers {axc:>5}: in-panel churn {r['sim_inpanel']:.3f} | hidden-founder churn {r['sim_hidden']:.3f}", flush=True)
Path("../results/fit_heldout_churn.json").write_text(json.dumps(res, indent=1))
