"""Crossover head vs truth. Simulated validation windows: predicted lambda (xo_scale * xo_lam) vs the
true lineage-aware switch count, with depth and held-out mode (0 in-panel, 1 one own founder hidden,
2 both). Real 0.1x/1x samples: per-window lambda per sample (truth unknown per window; decoding only).
Writes an npz for xo_regression_plot.py."""
import argparse, csv, glob
from pathlib import Path
import numpy as np, torch
from python.crf.train_diploid_indel import (GRITSCRFDiploidIndel, IndelDiploidAffinityDataset, individual_split_rows,
                                            _founder_affinity)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--heldout-glob", required=True, help="per-depth <prefix>_d*.heldout.npy, in concat order")
ap.add_argument("--per-group", type=int, default=12, help="val individuals per depth x mode")
ap.add_argument("--out", required=True)
a = ap.parse_args()
K, G = 25, 117
DEP = ["0.1x", "0.5x", "1x", "2x"]
data = np.load(a.data, mmap_mode="r"); lin = np.load(a.data[:-4] + ".lin.npy", mmap_mode="r")
prov = np.load(a.data[:-4] + ".prov.npy", mmap_mode="r")
meta = np.concatenate([np.load(f) for f in [a.heldout_glob.replace("*", d) for d in ("0.1", "0.5", "1", "2")]])
n_ind = len(data) // G; per = n_ind // 4
_, vr, _ = individual_split_rows(n_ind, G, .1, .1, 0)
vi = np.unique(vr // G)
m = GRITSCRFDiploidIndel.load_from_checkpoint(a.ckpt, map_location="cpu", strict=False).cuda().eval()
res = {"sim_pred": [], "sim_true": [], "sim_depth": [], "sim_mode": [], "sim_outbred": []}
for di in range(4):
    for mode in (0, 1, 2):
        ids = [i for i in vi if di * per <= i < (di + 1) * per and meta[i, 0] == mode][:a.per_group]
        for i in ids:
            rows = np.arange(i * G, (i + 1) * G)
            dr = np.asarray(data[rows]); lr = np.asarray(lin[rows])
            ds = IndelDiploidAffinityDataset(dr, K, G, lr)
            bt = [ds[j] for j in range(G)]
            B = {k: torch.stack([x[k] for x in bt]).cuda() for k in bt[0]}
            with torch.no_grad():
                m(B["input_embeds"], None, B["ext_emb"], B.get("count"))
                lam = (m.xo_scale * m._heads["xo_lam"].float()).cpu().numpy()
                tg = m.head_targets(B["h1"], B["h2"], B["lin"], torch.tensor(np.asarray(prov[rows]).astype(np.int64)).cuda(), tern=B["input_embeds"][..., 0])
                nsw = (tg["switch"] & tg["switch_mask"]).sum(1).cpu().numpy()
            lab = dr[:, :, K:K + 2].astype(int)
            outbred = bool(((lab[..., 0] != lab[..., 1]) & (lab >= 0).all(-1)).any())
            res["sim_pred"].append(lam); res["sim_true"].append(nsw)
            res["sim_depth"].append(np.full(G, di)); res["sim_mode"].append(np.full(G, mode))
            res["sim_outbred"].append(np.full(G, outbred))
        print(DEP[di], "mode", mode, len(ids), "individuals", flush=True)
out = {k: np.concatenate(v) for k, v in res.items()}
ROOT = "/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/lift_s200_local20k"
for dep in ("0.1", "1.0"):
    for d in sorted(glob.glob(f"{ROOT}/eval__*__{dep}x")):
        key = d.split("eval__")[1].rsplit("__", 1)[0]
        x = np.load(d + "/windowed_k25native_wcount.npy")
        tern = x[:, :, :K].astype(np.float32); dist = x[:, :, K + 2:2 * K + 2].astype(np.float32)
        feats = torch.tensor(np.stack([tern, dist], -1)); count = torch.tensor(x[:, :, 2*K+2:3*K+2].astype(np.float32))
        ext = torch.tensor(_founder_affinity((tern == 1).astype(np.float32).reshape(-1, K))).unsqueeze(0).expand(len(x), -1, -1)
        lams = []
        for s in range(0, len(x), 256):
            with torch.no_grad():
                m(feats[s:s+256].cuda(), ext_emb=ext[s:s+256].cuda(), count=count[s:s+256].cuda())
                lams.append((m.xo_scale * m._heads["xo_lam"].float()).cpu().numpy())
        out[f"real_{dep}__{key}"] = np.concatenate(lams)
    print("real", dep, "done", flush=True)
np.savez(a.out, **out)
