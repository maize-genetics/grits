"""Anatomy of val errors for a GRITSCRFDiploidIndel checkpoint on lineage-labelled
simulated data: tie-aware accuracy per class (inbred/hybrid), Viterbi vs per-row
local argmax, and whether each Viterbi error is input-identifiable (does the
predicted pair's (tern,dist) columns differ from EVERY allowed pair's columns in
a +-w row neighbourhood?). Also g / c / switch-count summaries.

usage: val_error_anatomy.py --ckpt C --data D [--n-ind 40] [--w 16]"""
import argparse, json
import numpy as np, torch
from python.crf.train_diploid_indel import (GRITSCRFDiploidIndel, individual_split_rows,
                                            IndelDiploidAffinityDataset)
from python.crf.crf_kernels import _dcrf_viterbi

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--n-ind", type=int, default=40); ap.add_argument("--w", type=int, default=16)
ap.add_argument("--G", type=int, default=117); ap.add_argument("--K", type=int, default=25)
ap.add_argument("--out", default=None)
ap.add_argument("--split", default="val", choices=["train", "val"])
a = ap.parse_args()
dev = "cuda"
data = np.load(a.data, mmap_mode="r"); lin = np.load(a.data[:-4] + ".lin.npy", mmap_mode="r")
n_ind = len(data) // a.G
tr_rows, val_rows, _ = individual_split_rows(n_ind, a.G, 0.10, 0.10, 0)
if a.split == "train": val_rows = tr_rows
val_ind = np.unique(val_rows // a.G)
inb = val_ind[val_ind < n_ind // 2][: a.n_ind // 2]; hyb = val_ind[val_ind >= n_ind // 2][: a.n_ind // 2]
model = GRITSCRFDiploidIndel.load_from_checkpoint(a.ckpt, map_location="cpu", strict=False).to(dev).eval()
K = a.K; Kn = K + 1
res = {}
for name, inds in (("inbred", inb), ("hybrid", hyb)):
    rows = (inds[:, None] * a.G + np.arange(a.G)).ravel()
    ds = IndelDiploidAffinityDataset(data[rows], K, a.G, lin[rows])
    acc = dict(n=0, vit_ok=0, loc_ok=0, err=0, err_unident_win=0, err_unident_w=0,
               vit_wrong_loc_right=0, vit_right_loc_wrong=0, pred_sw=0, true_sw=0, homo_true=0,
               pred_homo=0, emis_true_gap=[], g=[], c=[], nwin=0, allowed_size=[])
    for s in range(0, len(ds), 64):
        b = [ds[i] for i in range(s, min(s + 64, len(ds)))]
        bt = {k: torch.stack([x[k] for x in b]).to(dev) for k in b[0]}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            emis, g, c = model(bt["input_embeds"], None, bt["ext_emb"], bt.get("count"))
        emis = emis.float(); c = c.float(); g = g.float()
        vit = _dcrf_viterbi(emis, c, model.nsw_pair, model.stay_bonus.float())
        loc = emis.argmax(-1)
        allowed = model._allowed_pairs(bt["h1"], bt["h2"], bt["lin"])        # [B,T,P]
        vok = allowed.gather(2, vit[..., None])[..., 0]
        lok = allowed.gather(2, loc[..., None])[..., 0]
        # input identity between founder columns (null = (0,-1))
        X = bt["input_embeds"]; B, T = X.shape[:2]
        pad = torch.zeros(B, T, 1, 2, device=dev); pad[..., 1] = -1
        Xp = torch.cat([X, pad], 2)                                          # [B,T,Kn,2]
        diff = (Xp[:, :, :, None, :] != Xp[:, :, None, :, :]).any(-1).float()  # [B,T,Kn,Kn]
        cs = torch.cat([torch.zeros_like(diff[:, :1]), diff.cumsum(1)], 1)
        lo = torch.clamp(torch.arange(T, device=dev) - a.w, min=0)
        hi = torch.clamp(torch.arange(T, device=dev) + a.w + 1, max=T)
        same_w = (cs[:, hi] - cs[:, lo]) == 0                                # [B,T,Kn,Kn]
        same_win = (diff.sum(1) == 0)[:, None].expand(B, T, Kn, Kn)
        pi, pj = model.pi, model.pj
        # founder-level: allowed sets per chromosome
        Kf = K
        def eqset(h):
            lh = bt["lin"].gather(2, h.clamp(max=Kf - 1).unsqueeze(-1)); real = (h < Kf).unsqueeze(-1)
            return torch.cat([(bt["lin"] == lh) & real, ~real], 2)          # [B,T,Kn]
        e1, e2 = eqset(bt["h1"]), eqset(bt["h2"])
        pa, pb = pi[vit], pj[vit]                                            # [B,T]
        def reach(same, f, e):   # does founder f have an input-identical founder in set e
            row = same.gather(2, f[..., None, None].expand(B, T, 1, Kn))[:, :, 0]  # [B,T,Kn]
            return (row & e).any(-1)
        for key, same in (("err_unident_w", same_w), ("err_unident_win", same_win)):
            u = (reach(same, pa, e1) & reach(same, pb, e2)) | (reach(same, pa, e2) & reach(same, pb, e1))
            acc[key] += int((u & ~vok).sum())
        hit = lambda e, f: e.gather(2, f[..., None])[..., 0]
        h1c = hit(e1, pa) | hit(e1, pb); h2c = hit(e2, pa) | hit(e2, pb)
        acc["err_one_chrom_right"] = acc.get("err_one_chrom_right", 0) + int((~vok & (h1c ^ h2c)).sum())
        acc["err_both_wrong"] = acc.get("err_both_wrong", 0) + int((~vok & ~h1c & ~h2c).sum())
        acc["err_pred_homo"] = acc.get("err_pred_homo", 0) + int((~vok & (pa == pb)).sum())
        # window-level evidence: truth pair vs the Viterbi-mode pair, rows explained
        # (some member has tern==1) by one pair and not the other
        tern = X[..., 0]; tern = torch.cat([tern, torch.zeros_like(tern[..., :1])], 2)
        m = (tern == 1)
        for w in range(B):
            vw = vit[w]; mode = torch.mode(vw).values
            tg = model.pair_table[bt["h1"][w], bt["h2"][w]]; tmode = torch.mode(tg).values
            ta, tb, qa, qb = pi[tmode], pj[tmode], pi[mode], pj[mode]
            et = m[w, :, ta] | m[w, :, tb]; ep = m[w, :, qa] | m[w, :, qb]
            wrong = float((~vok[w]).float().mean()) > 0.5
            key = "win_err" if wrong else "win_ok"
            acc.setdefault(key, []).append((int((et & ~ep).sum()), int((ep & ~et).sum()), float((~vok[w]).float().mean())))
            if wrong:
                # substitute founder: member of pred pair not in truth pair; relatedness = frac rows same lineage
                sub = [f for f in (int(qa), int(qb)) if f not in (int(ta), int(tb))]
                tru = [f for f in (int(ta), int(tb)) if f not in (int(qa), int(qb))]
                if len(sub) == 1 and len(tru) == 1 and sub[0] < K and tru[0] < K:
                    L = bt["lin"][w]
                    acc.setdefault("sub_lin_share", []).append(float((L[:, sub[0]] == L[:, tru[0]]).float().mean()))
        acc["n"] += vok.numel(); acc["vit_ok"] += int(vok.sum()); acc["loc_ok"] += int(lok.sum())
        acc["err"] += int((~vok).sum())
        acc["vit_wrong_loc_right"] += int((~vok & lok).sum()); acc["vit_right_loc_wrong"] += int((vok & ~lok).sum())
        acc["pred_sw"] += float(model.nsw_pair[vit[:, :-1], vit[:, 1:]].sum())
        tags = model.pair_table[bt["h1"], bt["h2"]]
        acc["true_sw"] += float(model.nsw_pair[tags[:, :-1], tags[:, 1:]].sum())
        acc["homo_true"] += int((bt["h1"] == bt["h2"]).sum()); acc["pred_homo"] += int((pa == pb).sum())
        acc["nwin"] += B
        best_allowed = emis.masked_fill(~allowed, -1e9).max(-1).values
        acc["emis_true_gap"].append((emis.max(-1).values - best_allowed).flatten().cpu().numpy()[::7])
        acc["g"].append(g.flatten().cpu().numpy()[::7]); acc["c"].append(c.flatten().cpu().numpy()[::7])
        acc["allowed_size"].append(allowed.sum(-1).flatten().cpu().numpy()[::7])
    n = acc["n"]
    gap = np.concatenate(acc["emis_true_gap"]); gg = np.concatenate(acc["g"]); cc = np.concatenate(acc["c"])
    asz = np.concatenate(acc["allowed_size"])
    res[name] = {
        "tie_acc_viterbi": acc["vit_ok"] / n, "tie_acc_local_argmax": acc["loc_ok"] / n,
        "err_frac": acc["err"] / n,
        "err_share_one_chrom_right": acc["err_one_chrom_right"] / max(1, acc["err"]),
        "err_share_both_wrong": acc["err_both_wrong"] / max(1, acc["err"]),
        "err_share_pred_homo": acc["err_pred_homo"] / max(1, acc["err"]),
        "err_input_identical_pm%d" % a.w: acc["err_unident_w"] / max(1, acc["err"]),
        "err_input_identical_window": acc["err_unident_win"] / max(1, acc["err"]),
        "vit_wrong_loc_right_frac_rows": acc["vit_wrong_loc_right"] / n,
        "vit_right_loc_wrong_frac_rows": acc["vit_right_loc_wrong"] / n,
        "pred_switches_per_window": acc["pred_sw"] / acc["nwin"], "true_label_switches_per_window": acc["true_sw"] / acc["nwin"],
        "true_homo_frac": acc["homo_true"] / n, "pred_homo_frac": acc["pred_homo"] / n,
        "emis_gap_best_minus_best_allowed_q50_q90": [float(np.quantile(gap, .5)), float(np.quantile(gap, .9))],
        "g_q10_50_90": [float(x) for x in np.quantile(gg, [.1, .5, .9])],
        "c_q10_50_90": [float(x) for x in np.quantile(cc, [.1, .5, .9])],
        "allowed_pairs_q10_50_90": [float(x) for x in np.quantile(asz, [.1, .5, .9])],
        "n_win_err": len(acc.get("win_err", [])), "n_win_ok": len(acc.get("win_ok", [])),
        "win_err_rows_truth_only_vs_pred_only_median": [float(np.median([x[i] for x in acc.get("win_err", [(0, 0, 0)])])) for i in (0, 1)],
        "win_err_frac_pred_explains_ge_truth": float(np.mean([x[1] >= x[0] for x in acc.get("win_err", [(0, 0, 0)])])),
        "win_ok_rows_truth_only_vs_pred_only_median": [float(np.median([x[i] for x in acc.get("win_ok", [(0, 0, 0)])])) for i in (0, 1)],
        "sub_lineage_share_q10_50_90": [float(x) for x in np.quantile(acc.get("sub_lin_share", [0]), [.1, .5, .9])],
        "stay_bonus": float(model.stay_bonus.detach()),
    }
    print(name, json.dumps(res[name], indent=1), flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
