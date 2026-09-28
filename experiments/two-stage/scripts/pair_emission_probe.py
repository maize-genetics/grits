"""Zero-training probe: re-decode a checkpoint's own per-founder emissions with
different pair-emission combiners (additive = the model's, mixture = logsumexp),
tie-aware accuracy per individual type on val windows. CPU ok."""
import sys, numpy as np, torch
from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, individual_split_rows, IndelDiploidAffinityDataset
from python.crf.crf_kernels import _dcrf_viterbi
ck, dp = sys.argv[1], sys.argv[2]; n_ind = int(sys.argv[3]); stride = int(sys.argv[4])
dev = "cuda" if torch.cuda.is_available() else "cpu"
data = np.load(dp, mmap_mode="r"); lin = np.load(dp[:-4] + ".lin.npy", mmap_mode="r")
G, K = 117, 25
_, vr, _ = individual_split_rows(len(data) // G, G, .1, .1, 0)
vi = np.unique(vr // G); H = len(data) // G // 2
m = GRITSCRFDiploidIndel.load_from_checkpoint(ck, map_location="cpu", strict=False).to(dev).eval()
for name, inds in (("inbred", vi[vi < H][:n_ind]), ("hybrid", vi[vi >= H][:n_ind])):
    rows = (inds[:, None] * G + np.arange(0, G, stride)).ravel()
    aff_rows = (inds[:, None] * G + np.arange(G)).ravel()
    ds = IndelDiploidAffinityDataset(data[aff_rows], K, G, lin[aff_rows])
    sel = [i for i in range(len(ds)) if i % G % stride == 0]
    acc = {}
    for s in range(0, len(sel), 16):
        b = [ds[i] for i in sel[s:s + 16]]
        bt = {k: torch.stack([x[k] for x in b]).to(dev) for k in b[0]}
        with torch.no_grad():
            ep, g, c, raw = m(bt["input_embeds"], None, bt["ext_emb"], bt.get("count"), return_raw=True)
            ef = (g.unsqueeze(-1) * raw).float()
            ei, ej = ef[..., m.pi], ef[..., m.pj]
            pen = m.homo_penalty * m.homo_mask
            variants = {"additive(model)": ep.float(),
                        "mixture_lse": torch.logaddexp(ei, ej) - pen,
                        "mixture_lse_x2": 2 * torch.logaddexp(ei, ej) - pen}
            al = m._allowed_pairs(bt["h1"], bt["h2"], bt["lin"])
            for k, e in variants.items():
                v = _dcrf_viterbi(e, c.float(), m.nsw_pair, m.stay_bonus.float())
                ok = al.gather(2, v[..., None]).float().mean().item()
                acc.setdefault(k, []).append((ok, len(b)))
    print(name, {k: round(sum(a * n for a, n in v) / sum(n for _, n in v), 4) for k, v in acc.items()}, flush=True)
