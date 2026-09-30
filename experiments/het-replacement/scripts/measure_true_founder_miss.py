#!/usr/bin/env python
"""Rows where the true founder does NOT match the read: calibration-corpus in-panel inbreds (true
founder = the line itself) vs simulated inbreds (true founder = row label; split by provenance
clean / bad-site / off-site). Also run-length of consecutive miss rows (scattered vs blocked)
and what the miss rows match instead."""
import glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from python.crf.train_diploid_indel import individual_split_rows  # noqa: E402
K, G = 25, 117
FOUNDERS = ["B73", "B97", "CML103", "CML228", "CML247", "CML277", "CML322", "CML333", "CML52", "CML69", "HP301",
            "Il14H", "Ki11", "Ki3", "Ky21", "M162W", "M37W", "Mo18W", "Ms71", "NC350", "NC358", "Oh43", "Oh7B", "P39", "Tzi8"]


def runs(miss):
    L = []
    for w in miss:
        d = np.diff(np.r_[0, w.astype(np.int8), 0]); s, e = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        L += list(e - s)
    return np.mean(L) if L else 0.0


out = {}
root = "/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/calib_s200_local20k"
for d in sorted(glob.glob(f"{root}/calib__IDX-INBRED__*__0.1x")):
    line = d.split("__")[2]
    x = np.load(d + "/windowed_k25native_wcount.npy", mmap_mode="r")
    x = np.asarray(x[::4])
    t = x[:, :, :K]; live = (t > -2).all(2)
    miss = (t[:, :, FOUNDERS.index(line)] != 1) & live
    other = ((t == 1).any(2) & miss).sum() / max(miss.sum(), 1)
    tern_miss = t[:, :, FOUNDERS.index(line)][miss]
    out[f"calib_{line}"] = {"miss_frac": float(miss.sum() / live.sum()), "other_founder_matches": float(other),
                            "mean_run_len": float(runs(miss)),
                            "true_founder_state_on_miss(-1/0)": [float((tern_miss == -1).mean()), float((tern_miss == 0).mean())]}
sim = np.load("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/data/training/maize_v3read_multidepth_fullscale_sliced.npy", mmap_mode="r")
prov = np.load("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/data/training/maize_v3read_multidepth_fullscale_sliced.prov.npy", mmap_mode="r")
n = len(sim) // G; per = n // 4
_, vr, _ = individual_split_rows(n, G, .1, .1, 0); vi = np.unique(vr // G)
for dep, di in (("0.1x", 0), ("1x", 2)):
    ids = vi[(vi >= di * per) & (vi < di * per + per // 2)][:4]
    x = np.concatenate([np.asarray(sim[i*G:(i+1)*G]) for i in ids]); p = np.concatenate([np.asarray(prov[i*G:(i+1)*G]) for i in ids]).astype(int)
    t = x[:, :, :K]; h = x[:, :, K].astype(int); live = (p >= 0) & (h >= 0) & (h < K)
    tf = np.take_along_axis(t, np.clip(h, 0, K - 1)[..., None], 2)[..., 0]
    miss = (tf != 1) & live
    clean = live & ((p & 8) == 0) & ((p & 16) == 0)
    r = {"miss_frac_all_rows": float(miss.sum() / live.sum()),
         "miss_frac_clean_rows": float((miss & clean).sum() / clean.sum()),
         "miss_frac_bad_site_rows": float((miss & ((p & 16) > 0)).sum() / max(((p & 16) > 0).sum(), 1)),
         "miss_frac_offsite_rows": float((miss & ((p & 8) > 0)).sum() / max(((p & 8) > 0).sum(), 1)),
         "share_of_rows_bad_site": float(((p & 16) > 0)[live].mean()), "mean_run_len": float(runs(miss))}
    out[f"sim_inbred_{dep}"] = r
for k, v in out.items():
    print(f"{k:<22}", {a: round(b, 4) if isinstance(b, float) else [round(c, 3) for c in b] for a, b in v.items()})
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
