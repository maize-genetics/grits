"""Whole-chromosome decode for the supervised-heads v6 model (route + homoroute, as production).

The encoder scores windows of T rows exactly as decode_custom / infer_real_founder_pairs do (same routing,
affinity, knockouts); its per-row pair emissions, per-row switch costs and per-window affinity prior are
then laid end to end along each contig and decoded by ONE Viterbi per contig, so a founder change at a
window boundary pays the usual switch cost instead of being a free restart. Overlapping last windows
contribute only the rows they own. All contigs are decoded together as a padded batch.

The affinity transition prior of a window is aw_w * (la0[newf1] + la0[newf2]) with la0 the genome-wide
read affinity (identical for every window) and aw_w = aff_w * (1 - aff_ood_s * ood_w): it is rebuilt per
row from the scalar aw_w instead of storing a [P,P] matrix per window (checked against model._trans_prior).
"""
import numpy as np
import torch

import hd_lib as H
from python.crf.train_diploid_indel import _prior_trans


def decode_wholechrom(model, inp, T=H.T0, batch_size=64, device="cuda", force_hybrid=None):
    data, contig, pos = inp["data"], inp["contig"], inp["pos"]
    N0, Tn, Wd = data.shape
    flat = data.reshape(-1, Wd)
    K = H.K
    tern0 = data[:, :, :K].astype(np.float32)
    M0 = (tern0 == H.TERN_MATCH)
    r = H.route_real_sample(M0, (pos + (contig << 40)).reshape(N0, Tn))
    hybrid = r["hybrid"] if force_hybrid is None else force_hybrid
    affinity = H._founder_affinity(M0.reshape(-1, K).astype(np.float32))
    nby = [int((contig == c).sum()) for c in np.unique(contig)]
    widx, own = H._rechunk(nby, T)
    tern = flat[:, :K].astype(np.float32)
    dist = flat[:, K + 2:2 * K + 2].astype(np.float32)
    cnt = flat[:, 2 * K + 2:3 * K + 2].astype(np.float32)
    feats = torch.tensor(np.stack([tern, dist], -1))
    count = torch.tensor(cnt)
    ext = torch.tensor(affinity, dtype=torch.float32).unsqueeze(0)
    aff_pred = torch.tensor(affinity[:, 0])
    model._knock = {"homo": not hybrid}
    model.het_prior = "off"
    model._seg_prior = None

    n = len(flat)
    P = model.pi.numel()
    E = torch.empty(n, P, dtype=torch.float32)          # per-row pair emission (owned rows)
    C = torch.empty(n, dtype=torch.float32)             # per-row switch cost into the row
    AW = np.empty(len(widx), np.float32)                # per-window affinity weight
    WOF = np.empty(n, np.int64)                         # window that owns each row
    tp0 = ip0 = None
    with torch.no_grad():
        for s in range(0, len(widx), batch_size):
            wi = torch.tensor(widx[s:s + batch_size])
            xb = feats[wi].to(device)
            B = xb.shape[0]
            emis_p, g, c = model(xb, ext_emb=ext.expand(B, -1, -1).to(device), count=count[wi].to(device),
                                 aff_pred=aff_pred.unsqueeze(0).expand(B, -1).to(device))
            aw = model.aff_w * torch.ones(B, device=device)
            if getattr(model, "extra_heads", False):
                ood = torch.sigmoid(model._heads["ood_logit"].float()).mean(1)
                aw = aw * (1.0 - model.aff_ood_s * ood).clamp(0.0, 1.0)
            if tp0 is None:
                k = int(torch.argmax(aw.abs()))
                tp0 = (model._trans_prior[k] / aw[k]).float()
                ip0 = (model._init_prior[k] / aw[k]).float()
            err = (model._trans_prior.float() - aw.view(-1, 1, 1) * tp0.unsqueeze(0)).abs().max().item()
            assert err < 1e-3, f"affinity prior is not aw * shared matrix (max err {err})"
            AW[s:s + B] = aw.float().cpu().numpy()
            o = own[s:s + B]
            rows = widx[s:s + B][o]
            E[rows] = emis_p.float().cpu()[torch.tensor(o)]
            C[rows] = c.float().cpu()[torch.tensor(o)]
            WOF[rows] = np.repeat(np.arange(s, s + B), o.sum(1))

    # one Viterbi per contig, contigs batched (padded to the longest)
    starts = np.cumsum([0] + nby[:-1])
    L = max(nby)
    nc = len(nby)
    nsw = model.nsw_pair.float()
    sb = model.stay_bonus.float()
    tp0 = tp0.to(device); ip0 = ip0.to(device)
    awt = torch.tensor(AW, device=device)
    rowi = np.zeros((nc, L), np.int64)
    valid = np.zeros((nc, L), bool)
    for ci, (st, m) in enumerate(zip(starts, nby)):
        rowi[ci, :m] = np.arange(st, st + m)
        valid[ci, :m] = True
    rowi_t = torch.tensor(rowi)
    valid_t = torch.tensor(valid, device=device)
    wof_t = torch.tensor(WOF, device=device)
    path = torch.zeros(nc, L, dtype=torch.long, device=device)
    bp = torch.zeros(L, nc, P, dtype=torch.int16, device=device)
    with torch.no_grad():
        r0 = rowi_t[:, 0]
        delta = E[r0].to(device) + awt[wof_t[r0.to(device)]].view(-1, 1) * ip0.view(1, -1)
        chunk = 4096
        for c0 in range(1, L, chunk):
            c1 = min(L, c0 + chunk)
            ri = rowi_t[:, c0:c1]
            Eb = E[ri.reshape(-1)].view(nc, c1 - c0, P).to(device)
            Cb = C[ri.reshape(-1)].view(nc, c1 - c0).to(device)
            Ab = awt[wof_t[ri.to(device)]]
            for j in range(c1 - c0):
                t = c0 + j
                tr = _prior_trans(Cb[:, j], nsw, sb, Ab[:, j].view(-1, 1, 1) * tp0.unsqueeze(0))
                best, idx = (delta.unsqueeze(2) + tr).max(dim=1)
                v = valid_t[:, t].unsqueeze(1)
                delta = torch.where(v, Eb[:, j] + best, delta)
                bp[t] = torch.where(v, idx, torch.arange(P, device=device).expand(nc, P)).to(torch.int16)
        last = delta.argmax(dim=1)
        path[:, L - 1] = last
        cur = last
        for t in range(L - 1, 0, -1):
            cur = bp[t].long().gather(1, cur.unsqueeze(1)).squeeze(1)
            path[:, t - 1] = cur
    pred = np.empty(n, np.int64)
    pth = path.cpu().numpy()
    for ci, (st, m) in enumerate(zip(starts, nby)):
        pred[st:st + m] = pth[ci, :m]
    lo = model.pi[torch.tensor(pred)].cpu().numpy().reshape(N0, Tn)
    hi = model.pj[torch.tensor(pred)].cpu().numpy().reshape(N0, Tn)
    return lo, hi, dict(route=r, hybrid=hybrid, n_win=len(widx), n_rows=n, contig_rows=nby)
