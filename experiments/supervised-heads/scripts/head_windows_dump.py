"""Dump per-row supervised-head outputs + simulator truth for a few SIMULATED validation
windows (for head_windows_plot.py). Picks, per (depth, kind), the val window with the most
true lineage-aware switches (hybrid) or the most unclean rows (inbred)."""
import argparse
import numpy as np, torch
from python.crf.train_diploid_indel import (GRITSCRFDiploidIndel, individual_split_rows,
                                            IndelDiploidAffinityDataset, individual_affinity_target)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--picks", nargs="+", default=["0.1x:hybrid", "1x:hybrid", "1x:inbred"])
ap.add_argument("--scan", type=int, default=6, help="val individuals scanned per pick")
ap.add_argument("--out", required=True)
a = ap.parse_args()
K, G = 25, 117
data = np.load(a.data, mmap_mode="r"); lin = np.load(a.data[:-4] + ".lin.npy", mmap_mode="r")
prov = np.load(a.data[:-4] + ".prov.npy", mmap_mode="r")
n_ind = len(data) // G
_, vr, _ = individual_split_rows(n_ind, G, .1, .1, 0)
vi = np.unique(vr // G)
m = GRITSCRFDiploidIndel.load_from_checkpoint(a.ckpt, map_location="cpu", strict=False).cuda().eval()
DEP = ["0.1x", "0.5x", "1x", "2x"]
per = n_ind // 4
out = {}
for pi, pick in enumerate(a.picks):
    dname, kind = pick.split(":")
    di = DEP.index(dname)
    lo = di * per + (0 if kind == "inbred" else per // 2)
    ids = vi[(vi >= lo) & (vi < lo + per // 2)][:a.scan]
    best = None
    for i in ids:
        rows = np.arange(i * G, (i + 1) * G)
        dr = np.asarray(data[rows]); lr = np.asarray(lin[rows]); pr = np.asarray(prov[rows]).astype(np.int64)
        ds = IndelDiploidAffinityDataset(dr, K, G, lr)
        bt = [ds[j] for j in range(G)]
        B = {k: torch.stack([x[k] for x in bt]).cuda() for k in bt[0]}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            m(B["input_embeds"], None, B["ext_emb"], B.get("count"))
        H = {k: v.float() for k, v in m._heads.items()}
        tg = m.head_targets(B["h1"], B["h2"], B["lin"], torch.tensor(pr).cuda())
        nsw = (tg["switch"] & tg["switch_mask"]).sum(1).cpu().numpy()
        ncl = (tg["gate_mask"] & ~tg["gate"]).sum(1).cpu().numpy()
        score = nsw if kind == "hybrid" else ncl
        w = int(np.argmax(score))
        if best is not None and score[w] <= best[0]:
            continue
        t = lambda x: x[w].detach().cpu().numpy()
        best = (score[w], dict(
            tern=dr[w, :, :K].astype(np.int8), h1=t(B["h1"]), h2=t(B["h2"]), lin=lr[w],
            prov=pr[w], gate_p=t(torch.sigmoid(H["gate_logit"])), gate_y=t(tg["gate"]), gate_m=t(tg["gate_mask"]),
            sw_p=t(torch.sigmoid(-H["c"])), sw_y=t(tg["switch"]), sw_m=t(tg["switch_mask"]),
            xo_lam=np.float32((m.xo_scale.float() * H["xo_lam"])[w].item()), xo_true=np.float32(nsw[w]),
            het_p=t(torch.sigmoid(H["het_logit"])), het_y=t(tg["het"]), het_m=t(tg["het_mask"]),
            aff_win=t(torch.sigmoid(H["aff_logit"])),
            aff_ind=torch.sigmoid(H["aff_logit"]).mean(0).cpu().numpy(),
            aff_true=individual_affinity_target(dr, lr, K, G)[0],
            ind=np.int64(i), win=np.int64(w)))
    for k, v in best[1].items():
        out[f"{pi}/{k}"] = v
    out[f"{pi}/label"] = np.array(f"{dname} {kind}  (ind {best[1]['ind']}, window {best[1]['win']})")
    print(pick, "ind", best[1]["ind"], "win", best[1]["win"], "score", best[0],
          "xo_lam", best[1]["xo_lam"], "xo_true", best[1]["xo_true"], flush=True)
np.savez(a.out, n=len(a.picks), **out)
