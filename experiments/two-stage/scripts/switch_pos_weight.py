"""pos_weight = #non-switch / #switch row pairs over the TRAIN split (lineage-aware, as aux_targets)."""
import sys, numpy as np, torch
from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, individual_split_rows
dp = sys.argv[1]; G, K = 117, 25
data = np.load(dp, mmap_mode="r"); lin = np.load(dp[:-4] + ".lin.npy", mmap_mode="r")
tr, _, _ = individual_split_rows(len(data) // G, G, .1, .1, 0)
m = GRITSCRFDiploidIndel(num_parents=K, d_model=8, n_heads=1, n_layers=1).cuda()
H = len(data) // G // 2 * G
tot = {"inbred": [0, 0], "hybrid": [0, 0]}
for s in range(0, len(tr), 2048):
    r = tr[s:s + 2048]
    lab = torch.tensor(np.asarray(data[r, :, K:K + 2]), dtype=torch.long).cuda()
    h1 = torch.where(lab[..., 0] < 0, K, lab[..., 0]); h2 = torch.where(lab[..., 1] < 0, K, lab[..., 1])
    L = torch.tensor(np.asarray(lin[r]), dtype=torch.long).cuda()
    X = torch.zeros(len(r), data.shape[1], K, 2, device="cuda")
    _, _, sw = m.aux_targets(X, h1, h2, L)
    for name, sel in (("inbred", r < H), ("hybrid", r >= H)):
        sel = torch.tensor(sel, device="cuda")
        tot[name][0] += int(sw[sel].sum()); tot[name][1] += sw[sel].numel()
p = sum(v[0] for v in tot.values()); n = sum(v[1] for v in tot.values())
for k, (a, b) in tot.items():
    print(k, "switches", a, "pairs", b, "per window", a / (b / (data.shape[1] - 1)))
print("pos_weight (non-switch/switch) =", (n - p) / p)
