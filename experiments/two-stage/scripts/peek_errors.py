import sys, numpy as np, torch
from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, individual_split_rows, IndelDiploidAffinityDataset
from python.crf.crf_kernels import _dcrf_viterbi
ck, dp = sys.argv[1], sys.argv[2]
data = np.load(dp, mmap_mode="r"); lin = np.load(dp[:-4] + ".lin.npy", mmap_mode="r")
G, K = 117, 25
_, vr, _ = individual_split_rows(len(data) // G, G, .1, .1, 0)
vi = np.unique(vr // G); hyb = vi[vi >= len(data) // G // 2][:6]
rows = (hyb[:, None] * G + np.arange(G)).ravel()
ds = IndelDiploidAffinityDataset(data[rows], K, G, lin[rows])
m = GRITSCRFDiploidIndel.load_from_checkpoint(ck, map_location="cpu", strict=False).cuda().eval()
from collections import Counter
subs = Counter(); shown = 0
for s in range(0, len(ds), 64):
    b = [ds[i] for i in range(s, min(s + 64, len(ds)))]
    bt = {k: torch.stack([x[k] for x in b]).cuda() for k in b[0]}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        e, g, c = m(bt["input_embeds"], None, bt["ext_emb"], bt.get("count"))
    v = _dcrf_viterbi(e.float(), c.float(), m.nsw_pair, m.stay_bonus.float())
    for w in range(len(b)):
        h1, h2 = bt["h1"][w], bt["h2"][w]
        tp = m.pair_table[h1, h2]; vm = torch.mode(v[w]).values; tm = torch.mode(tp).values
        if vm == tm: continue
        t = (int(m.pi[tm]), int(m.pj[tm])); p = (int(m.pi[vm]), int(m.pj[vm]))
        subs[(t, p)] += 1
        if shown < 6:
            tern = bt["input_embeds"][w, :, :, 0]
            mc = (tern == 1).float().sum(0).cpu().numpy().astype(int)
            print("win", s + w, "truth", t, "pred", p, "match counts/founder", mc.tolist(),
                  "ext_emb true/pred:", bt["ext_emb"][w][[x for x in t if x < K]].cpu().numpy().round(3).tolist(),
                  bt["ext_emb"][w][[x for x in p if x < K]].cpu().numpy().round(3).tolist())
            shown += 1
print(subs.most_common(15))
