"""prov vs support gate labels on sh5 simulated data, by row kind."""
import numpy as np, torch
from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, individual_split_rows
from python.crf.simulate_alleles import PROV_KIND_MASK
K, G = 25, 117
D = "/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/data/training/maize_v3read_multidepth_fullscale_sliced"
data = np.load(D + ".npy", mmap_mode="r"); prov = np.load(D + ".prov.npy", mmap_mode="r")
m = GRITSCRFDiploidIndel(num_parents=K, supervised_heads="stage1", gate_target="support", founder_affinity=True, time_local_emis=True, emission="likelihood_dist")
n = len(data) // G; per = n // 4
_, vr, _ = individual_split_rows(n, G, .1, .1, 0); vi = np.unique(vr // G)
for dep, di in (("0.1x", 0), ("1x", 2)):
    for kind, off in (("inbred", 0), ("hybrid", per // 2)):
        ids = vi[(vi >= di * per + off) & (vi < di * per + off + per // 2)][:3]
        x = np.concatenate([np.asarray(data[i*G:(i+1)*G]) for i in ids]); p = torch.tensor(np.concatenate([np.asarray(prov[i*G:(i+1)*G]) for i in ids]).astype(np.int64))
        h1 = torch.tensor(x[:, :, K].astype(np.int64)); h2 = torch.tensor(x[:, :, K + 1].astype(np.int64))
        h1 = torch.where(h1 < 0, K, h1); h2 = torch.where(h2 < 0, K, h2)
        lin = torch.tensor(np.load(D + ".lin.npy", mmap_mode="r")[np.concatenate([np.arange(i*G, (i+1)*G) for i in ids])].astype(np.int64))
        tern = torch.tensor(x[:, :, :K].astype(np.float32))
        m.gate_target = "prov"; a = m.head_targets(h1, h2, lin, p)
        m.gate_target = "support"; b = m.head_targets(h1, h2, lin, p, tern=tern)
        kd = (p & PROV_KIND_MASK); off_ = (p >= 0) & ((p & 8) > 0); bad = (p >= 0) & ((p & 16) > 0); col = (p >= 0) & (kd <= 1) & ~bad
        f = lambda g, msk: float(g["gate"][msk].float().mean())
        print(f"{dep} {kind:<6} clean frac prov {f(a, a['gate_mask']):.3f} support {f(b, b['gate_mask']):.3f} | "
              f"support-clean among: collinear {f(b, col):.3f}  bad-site {f(b, bad):.3f}  off-site replacement {f(b, off_):.3f} "
              f"(off-site = {float(off_[p >= 0].float().mean()):.3f} of rows)")
