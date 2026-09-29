"""Pooled per-individual affinity-head prediction [N/G, K] for a simulated set (stage-2 input):
mean over each individual's G windows of sigmoid(aff_logit), frozen stage-1 encoder."""
import sys
import numpy as np, torch
from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, IndelDiploidAffinityDataset
ck, dp, out = sys.argv[1:4]
G, K = 117, 25
data = np.load(dp, mmap_mode="r")
m = GRITSCRFDiploidIndel.load_from_checkpoint(ck, map_location="cpu", strict=False).cuda().eval()
n_ind = len(data) // G
res = np.zeros((n_ind, K), dtype=np.float32)
step = 20
for i0 in range(0, n_ind, step):
    rows = np.arange(i0 * G, min(n_ind, i0 + step) * G)
    ds = IndelDiploidAffinityDataset(np.asarray(data[rows]), K, G)
    acc = []
    for s in range(0, len(ds), 234):
        b = [ds[i] for i in range(s, min(s + 234, len(ds)))]
        X = torch.stack([x["input_embeds"] for x in b]).cuda()
        e = torch.stack([x["ext_emb"] for x in b]).cuda()
        c = torch.stack([x["count"] for x in b]).cuda() if "count" in b[0] else None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            m(X, None, e, c)
        acc.append(torch.sigmoid(m._heads["aff_logit"].float()).cpu().numpy())
    acc = np.concatenate(acc).reshape(-1, G, K)
    res[i0:i0 + len(acc)] = acc.mean(1)
    if i0 % 400 == 0:
        print(i0, flush=True)
np.save(out, res)
print("wrote", out, res.shape)
