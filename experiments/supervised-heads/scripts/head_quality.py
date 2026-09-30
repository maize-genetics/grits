"""Head quality on the SIMULATED validation split, before decoding (per depth, per kind):
gate AUC / average precision / precision-recall@0.5 vs the provenance clean flag; switch
calibration (expected switches per window sum_t p_t vs true lineage-aware switches); het
accuracy per row; pooled affinity vs truth (Pearson r, MAE). With --decode, also the CRF
tie accuracy and decoded switches/window (inbred/hybrid)."""
import argparse, json
import numpy as np, torch
from python.crf.train_diploid_indel import (GRITSCRFDiploidIndel, individual_split_rows,
                                            IndelDiploidAffinityDataset, individual_affinity_target)
from python.crf.simulate_alleles import prov_is_clean

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--per-kind", type=int, default=8, help="val individuals per kind per depth")
ap.add_argument("--decode", action="store_true"); ap.add_argument("--out", required=True)
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


def auc(score, y):
    o = np.argsort(score); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1)
    npos = y.sum(); nneg = len(y) - npos
    return float((r[y].sum() - npos * (npos + 1) / 2) / max(1, npos * nneg))


def ap_score(score, y):
    o = np.argsort(-score); yy = y[o]; tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / max(1, yy.sum()))


res = {}
for di, dname in enumerate(DEP):
    for kind in ("inbred", "hybrid"):
        lo = di * per + (0 if kind == "inbred" else per // 2)
        ids = vi[(vi >= lo) & (vi < lo + per // 2)][:a.per_kind]
        if len(ids) == 0:
            continue
        gs, gy, sw_pred, sw_true, het_p, het_y, aff_p, aff_t = [], [], [], [], [], [], [], []
        tie_ok, dec_sw, xo_p, xo_t = [], [], [], []
        for i in ids:
            rows = np.arange(i * G, (i + 1) * G)
            dr = np.asarray(data[rows]); lr = np.asarray(lin[rows])
            ds = IndelDiploidAffinityDataset(dr, K, G, lr)
            aff_t.append(individual_affinity_target(dr, lr, K, G)[0])
            bt = [ds[j] for j in range(G)]
            B = {k: torch.stack([x[k] for x in bt]).cuda() for k in bt[0]}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                m(B["input_embeds"], None, B["ext_emb"], B.get("count"))
            H = {k: v.float() for k, v in m._heads.items()}
            ap_ = torch.sigmoid(H["aff_logit"]).mean(0)
            aff_p.append(ap_.cpu().numpy())
            tg = m.head_targets(B["h1"], B["h2"], B["lin"], torch.tensor(np.asarray(prov[rows]).astype(np.int64)).cuda(),
                                tern=B["input_embeds"][..., 0])
            gm = tg["gate_mask"]
            gs.append(torch.sigmoid(H["gate_logit"])[gm].cpu().numpy()); gy.append(tg["gate"][gm].cpu().numpy())
            p = torch.sigmoid(-H["c"][:, 1:]) * tg["switch_mask"]
            sw_pred.append(p.sum(1).cpu().numpy()); sw_true.append((tg["switch"] & tg["switch_mask"]).sum(1).cpu().numpy())
            xo_p.append((m.xo_scale.float() * H["xo_lam"]).detach().cpu().numpy())
            hm = tg["het_mask"]
            het_p.append(torch.sigmoid(H["het_logit"])[hm].cpu().numpy()); het_y.append(tg["het"][hm].cpu().numpy())
            if a.decode:
                with torch.no_grad():
                    emis, g, c = m(B["input_embeds"], None, B["ext_emb"], B.get("count"),
                                   aff_pred=ap_.unsqueeze(0).expand(G, -1))
                    pred = m.crf_decode(emis, c)
                    al = m._allowed_pairs(B["h1"], B["h2"], B["lin"])
                    tie_ok.append(al.gather(2, pred[..., None]).float().mean().item())
                    dec_sw.append(m.nsw_pair[pred[:, :-1], pred[:, 1:]].sum(1).float().mean().item())
        gs, gy = np.concatenate(gs), np.concatenate(gy).astype(bool)
        hp, hy = np.concatenate(het_p), np.concatenate(het_y).astype(bool)
        sp, st = np.concatenate(sw_pred), np.concatenate(sw_true)
        AP, AT = np.array(aff_p), np.array(aff_t)
        XP = np.concatenate(xo_p)
        edges = [0, .02, .05, .1, .2, .5, 1, 2, 5, 1e9]
        xb = np.digitize(XP, edges) - 1
        curve = [[float(XP[xb == i].mean()), float(st[xb == i].mean()), int((xb == i).sum())]
                 for i in range(len(edges) - 1) if (xb == i).any()]
        pr = gs >= 0.5
        r = {"n_ind": len(ids), "gate_clean_frac": float(gy.mean()), "gate_auc": auc(gs, gy),
             "gate_avg_precision_clean": ap_score(gs, gy), "gate_avg_precision_noisy": ap_score(-gs, ~gy),
             "gate_precision@0.5": float(gy[pr].mean()) if pr.any() else None,
             "gate_recall@0.5": float(pr[gy].mean()), "gate_median": float(np.median(gs)),
             "xo_lambda_mean": float(XP.mean()), "xo_true_mean": float(st.mean()),
             "xo_poisson_nll": float((XP - st * np.log(np.maximum(XP, 1e-8))).mean()),
             "xo_calibration_curve[pred,true,n]": curve,
             "rowhead_switch_pred_per_window": float(sp.mean()), "switch_true_per_window": float(st.mean()),
             "het_acc": float(((hp >= 0.5) == hy).mean()), "het_true_frac": float(hy.mean()),
             "het_pred_mean": float(hp.mean()),
             "aff_pearson": float(np.corrcoef(AP.ravel(), AT.ravel())[0, 1]),
             "aff_mae": float(np.abs(AP - AT).mean()),
             "aff_top2_hit": float(np.mean([len(set(np.argsort(-p)[:2]) & set(np.flatnonzero(t >= t.max() - 1e-6))) > 0
                                            for p, t in zip(AP, AT)]))}
        if a.decode:
            r["tie_acc"] = float(np.mean(tie_ok)); r["decoded_switch_per_window"] = float(np.mean(dec_sw))
        res[f"{dname}/{kind}"] = r
        print(dname, kind, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(res, open(a.out, "w"), indent=1)
