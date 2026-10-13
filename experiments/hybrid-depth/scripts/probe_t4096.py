"""Memory / speed probe for the v6cfullfe recipe at T rows per window (read-only on the data).

Builds real training batches in memory by concatenating each individual's consecutive 512-row
windows of maize_v6hoc_multidepth (same layout the re-sliced file would have) and times a
stage-1 step (encoder + supervised heads, bf16 autocast, backward + AdamW) and a stage-2 step
(frozen encoder, CRF NLL with prior) at several batch sizes. Reports peak GPU memory and s/step.
usage: probe_t4096.py T B1 [B2 ...]"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/grits_workdir/hybrid-depth-wt/src")
from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, _founder_affinity, TERN_MATCH  # noqa: E402

D = "/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/data/training/maize_v6hoc_multidepth_fullscale_sliced"
TERN = ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
        "diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt")
K, G, T0 = 25, 117, 512
T = int(sys.argv[1]); Bs = [int(b) for b in sys.argv[2:]]
X = np.load(D + ".npy", mmap_mode="r"); Lh = np.load(D + ".lin.npy", mmap_mode="r")
P = np.load(D + ".prov.npy", mmap_mode="r"); O = np.load(D + ".ood.npy", mmap_mode="r")
per = T // T0


def batch(B, seed):
    rng = np.random.default_rng(seed)
    inds = rng.choice(4000, B, replace=False)
    xs, ls, ps, os_, ex = [], [], [], [], []
    for i in inds:
        w0 = i * G + rng.integers(0, G - per + 1)
        xs.append(np.asarray(X[w0:w0 + per]).reshape(T, -1)); ls.append(np.asarray(Lh[w0:w0 + per]).reshape(T, K))
        ps.append(np.asarray(P[w0:w0 + per]).reshape(T)); os_.append(np.asarray(O[w0:w0 + per]).reshape(T))
        M = (np.asarray(X[i * G:(i + 1) * G, :, :K]) == TERN_MATCH).astype(np.float32)
        ex.append(_founder_affinity(M.reshape(-1, K)))
    x = np.stack(xs)
    h = x[:, :, K:K + 2].astype(np.int64); h = np.where(h < 0, K, h)
    dev = "cuda"
    return dict(input_embeds=torch.tensor(np.stack([x[:, :, :K], x[:, :, K + 2:2 * K + 2]], -1), dtype=torch.float32, device=dev),
                h1=torch.tensor(h[..., 0], device=dev), h2=torch.tensor(h[..., 1], device=dev),
                count=torch.tensor(x[:, :, 2 * K + 2:].astype(np.float32), device=dev),
                lin=torch.tensor(np.stack(ls).astype(np.int64), device=dev),
                ext_emb=torch.tensor(np.stack(ex), dtype=torch.float32, device=dev),
                prov=torch.tensor(np.stack(ps).astype(np.int64), device=dev),
                ood=torch.tensor(np.stack(os_).astype(np.int64), device=dev))


def mk(stage):
    m = GRITSCRFDiploidIndel(num_parents=25, d_model=256, n_heads=8, n_layers=6, time_local_emis=True,
                             homo_penalty=3.0, founder_affinity=True, tie_aware_loss=True, pair_emission="mixture",
                             emission="likelihood", supervised_heads=stage, het_prior="off", no_distance=True,
                             aff_source="reads", gate_target="support", extra_heads=True,
                             no_null=(stage == "stage2"), freeze_emission=(stage == "stage2"))
    sd = torch.load(TERN, map_location="cpu", weights_only=False)["state_dict"]
    m.load_state_dict(sd, strict=False)
    return m.cuda().train()


for stage in ("stage1", "stage2"):
    m = mk(stage)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    for B in Bs:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            ts = []
            for it in range(6):
                b = batch(B, it)
                torch.cuda.synchronize(); t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = m._step(b)[0]
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); ts.append(time.time() - t0)
            print(f"T={T} {stage} B={B}: {np.mean(ts[2:]):.3f} s/step  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB  "
                  f"loss {float(loss):.3f}  rows/s {B * T / np.mean(ts[2:]):,.0f}", flush=True)
        except torch.OutOfMemoryError:
            print(f"T={T} {stage} B={B}: OOM", flush=True)
            opt.zero_grad(set_to_none=True)
    del m, opt
