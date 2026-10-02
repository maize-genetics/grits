"""Supervised heads on REAL eval rows (diagnostic only, nothing is fitted): het-head mean, gate
mean, window crossover lambda and decoded homozygous-row fraction per sample."""
import sys, numpy as np, torch
from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, _founder_affinity, pooled_affinity
ck, keys = sys.argv[1], sys.argv[2:]
S = "/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/lift_s200_local20k/eval__{}__0.1x/windowed_k25native_wcount.npy"
K = 25
m = GRITSCRFDiploidIndel.load_from_checkpoint(ck, map_location="cpu", strict=False).cuda().eval()
for key in keys:
    data = np.load(S.format(key))
    tern = data[:, :, :K].astype(np.float32); dist = data[:, :, K + 2:2 * K + 2].astype(np.float32)
    feats = torch.tensor(np.stack([tern, dist], -1)); count = torch.tensor(data[:, :, 2*K+2:3*K+2].astype(np.float32))
    ext = torch.tensor(_founder_affinity((tern == 1).astype(np.float32).reshape(-1, K))).unsqueeze(0).expand(len(data), -1, -1)
    ap = pooled_affinity(m, feats, count, ext, "cuda", 128)
    h, g, lam, homo, ood, dos = [], [], [], [], [], []
    for s in range(0, len(data), 128):
        B = min(128, len(data) - s)
        with torch.no_grad():
            emis, gg, c = m(feats[s:s+B].cuda(), ext_emb=ext[s:s+B].cuda(), count=count[s:s+B].cuda(),
                            aff_pred=ap.unsqueeze(0).expand(B, -1))
            pred = m.crf_decode(emis, c)
        h.append(torch.sigmoid(m._heads["het_logit"].float()).cpu().numpy().ravel()); g.append(gg.float().cpu().numpy().ravel())
        lam.append((m.xo_scale * m._heads["xo_lam"].float()).detach().cpu().numpy()); homo.append((m.pi[pred] == m.pj[pred]).float().cpu().numpy().ravel())
        if "ood_logit" in m._heads:
            ood.append(torch.sigmoid(m._heads["ood_logit"].float()).cpu().numpy().ravel())
            dos.append(torch.softmax(m._heads["dos_logit"].float(), -1).reshape(-1, 3).cpu().numpy())
    h = np.concatenate(h)
    print(f"{key:<24} het mean {h.mean():.3f} median {np.median(h):.3f} | gate mean {np.concatenate(g).mean():.3f} | "
          f"xo_lam/window {np.concatenate(lam).mean():.3f} | decoded homozygous rows {np.concatenate(homo).mean():.3f}"
          + (f" | out-of-panel mean {np.concatenate(ood).mean():.3f} | dosage 0/1/2 {np.concatenate(dos).mean(0).round(3).tolist()}"
             if ood else ""), flush=True)
