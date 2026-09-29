"""Empirical-Bayes het-prior weight, per sample and from its own reads only: with the het prior
normalized over pair states per row (w*log q - logsumexp w*log q, q = h for het pairs, 1-h for
homozygous), the heads CRF forward sum is the marginal likelihood of the sample's rows under
weight w (emissions are log-LRs, transitions log-probabilities). Report log Z per w on a grid,
the argmax, and the decoded homozygous-row fraction at each w. Diagnostic: nothing is fitted
to truth and no real data enters training."""
import sys, numpy as np, torch
from python.crf.train_diploid_indel import (GRITSCRFDiploidIndel, _founder_affinity, pooled_affinity,
                                            _prior_trans, crf_viterbi_prior)
ck, n_win, keys = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
S = "/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/lift_s200_local20k/eval__{}__0.1x/windowed_k25native_wcount.npy"
K = 25
W = [0.0, 0.25, 0.5, 1.0, 2.0]
m = GRITSCRFDiploidIndel.load_from_checkpoint(ck, map_location="cpu", strict=False).cuda().eval()
with torch.no_grad():
    m.het_w.fill_(0.0)
homo = m.homo_mask.float().view(-1)                                      # [P]


@torch.no_grad()
def forward_logz(emis, c, tp, ip):
    a = emis[:, 0] + ip
    for t in range(1, emis.shape[1]):
        a = emis[:, t] + torch.logsumexp(a.unsqueeze(2) + _prior_trans(c[:, t], m.nsw_pair.float(),
                                                                      m.stay_bonus.float(), tp), dim=1)
    return torch.logsumexp(a, 1)


print("sample                    " + "".join(f"  logZ w={w:<4}" for w in W) + "  | best w | homo frac at w=" + ",".join(map(str, W)))
for key in keys:
    data = np.load(S.format(key))
    idx = np.linspace(0, len(data) - 1, min(n_win, len(data))).astype(int)
    data = data[idx]
    tern = data[:, :, :K].astype(np.float32); dist = data[:, :, K + 2:2 * K + 2].astype(np.float32)
    feats = torch.tensor(np.stack([tern, dist], -1)); count = torch.tensor(data[:, :, 2*K+2:3*K+2].astype(np.float32))
    ext = torch.tensor(_founder_affinity((tern == 1).astype(np.float32).reshape(-1, K))).unsqueeze(0).expand(len(data), -1, -1)
    ap = pooled_affinity(m, feats, count, ext, "cuda", 64)
    lz = np.zeros(len(W)); hf = np.zeros(len(W))
    for s in range(0, len(data), 64):
        B = min(64, len(data) - s)
        with torch.no_grad():
            emis0, g, c = m(feats[s:s+B].cuda(), ext_emb=ext[s:s+B].cuda(), count=count[s:s+B].cuda(),
                            aff_pred=ap.unsqueeze(0).expand(B, -1))
            emis0 = emis0.float(); c = c.float()
            tp, ip = m._trans_prior.float(), m._init_prior.float()
            h = torch.sigmoid(m._heads["het_logit"].float()).clamp(1e-4, 1 - 1e-4).unsqueeze(-1)   # [B,T,1]
            lq = homo * torch.log1p(-h) + (1 - homo) * torch.log(h)                               # [B,T,P]
            for i, w in enumerate(W):
                pri = w * lq - torch.logsumexp(w * lq, -1, keepdim=True)
                e = emis0 + pri
                lz[i] += forward_logz(e, c, tp, ip).sum().item()
                pred = crf_viterbi_prior(e, c, m.nsw_pair, m.stay_bonus, tp, ip)
                hf[i] += (m.pi[pred] == m.pj[pred]).float().sum().item()
    hf /= len(data) * data.shape[1]
    rel = lz - lz.max()
    print(f"{key:<26}" + "".join(f"{v:>12.1f}" for v in rel) + f"  | {W[int(np.argmax(lz))]:>6} | " + ",".join(f"{x:.2f}" for x in hf), flush=True)
