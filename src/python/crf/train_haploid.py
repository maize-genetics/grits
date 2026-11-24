"""
E1 haploid training — GRITS-CRF on maize inbred data.

Maize inbred .npy files are [T, 25]: cols 0-23 = founder read counts, col 24 = label.
The CRF runs over K=24 founder states (nsw[i,j] = i!=j).
The encoder is architecturally identical to the diploid path; only the state layer differs.

Usage:
    pixi run --environment gpu python src/python/crf/train_haploid.py --workdir /workdir/esb33
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

torch.set_float32_matmul_precision("medium")
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader

from python.crf.train_crf import FounderPathEncoder
from python.crf.callbacks import EMACallback


# --------------------------------------------------------------------------- #
#  Vectorized CRF — JIT-compiled to remove Python loop overhead over T        #
# --------------------------------------------------------------------------- #

@torch.jit.script
def _crf_nll(emis: torch.Tensor, c: torch.Tensor, nsw: torch.Tensor,
             stay_bonus: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
    """
    CRF NLL without materializing the full [B,T,K,K] transition tensor.

    Partition (log_Z): sequential forward pass, O(B*K^2) per step.
    Score: fully vectorized — tr[b,t,tags[t-1],tags[t]] reduces to a
    1-D switch mask, no K×K materialization needed.
    emis: [B,T,K]  c: [B,T]  nsw: [K,K]  tags: [B,T]
    """
    B, T, K = emis.shape
    # fp32 partition: the encoder runs bf16, but this 512-step sequential
    # logsumexp accumulation needs >2-digit precision or it detonates at scale.
    # No matmuls here, so casting inputs keeps the whole recursion in fp32.
    emis = emis.float()
    c = c.float()
    nsw = nsw.float()
    stay_bonus = stay_bonus.float()
    stay_mask = (nsw == 0).float()

    # Forward pass (partition)
    a = emis[:, 0]
    for t in range(1, T):
        tr_t = -c[:, t, None, None] * nsw[None] + stay_bonus * stay_mask[None]
        a = emis[:, t] + torch.logsumexp(a.unsqueeze(2) + tr_t, dim=1)
    log_Z = torch.logsumexp(a, dim=1)

    # Score of true path — fully vectorized, no K×K materialization
    bi = torch.arange(B, device=emis.device)
    t_idx = torch.arange(T, device=emis.device)
    emis_score = emis[bi.unsqueeze(1), t_idx.unsqueeze(0), tags].sum(1)  # [B]
    # tr[b,t,i,j] = 0 if i==j else -c[b,t]; nsw[i,j] = 1 if i!=j
    switches = (tags[:, :-1] != tags[:, 1:]).float()  # [B, T-1]
    tr_score = (-c[:, 1:] * switches + stay_bonus * (1.0 - switches)).sum(1)  # [B]

    return (log_Z - emis_score - tr_score).mean()


@torch.jit.script
def _crf_viterbi(emis: torch.Tensor, c: torch.Tensor, nsw: torch.Tensor,
                 stay_bonus: torch.Tensor) -> torch.Tensor:
    """Viterbi decode without materializing [B,T,K,K]. emis:[B,T,K] -> [B,T]"""
    B, T, K = emis.shape
    stay_mask = (nsw == 0).float()
    delta = emis[:, 0]
    bp = torch.zeros(T - 1, B, K, dtype=torch.long, device=emis.device)
    for t in range(1, T):
        tr_t = -c[:, t, None, None] * nsw[None] + stay_bonus * stay_mask[None]
        sc = delta.unsqueeze(2) + tr_t
        best, idx = sc.max(dim=1)
        delta = emis[:, t] + best
        bp[t - 1] = idx
    path = torch.zeros(B, T, dtype=torch.long, device=emis.device)
    path[:, T - 1] = delta.argmax(dim=1)
    for t in range(T - 2, -1, -1):
        path[:, t] = bp[t].gather(1, path[:, t + 1].unsqueeze(1)).squeeze(1)
    return path


def _window_corr(c: torch.Tensor, log_rate: torch.Tensor) -> torch.Tensor:
    """Mean per-window Pearson correlation between transition cost c_t and the
    (log) hidden recombination rate. Site 0 is dropped (no transition there),
    matching eval_recomb.py. Scale/shift-invariant in c, so it supervises the
    *ranking* (hot ⇒ cheap to switch) without pinning c's absolute scale.
    Minimizing the returned value drives c_t → anti-correlated with the rate.
    """
    c = c[:, 1:]
    r = log_rate[:, 1:]
    cc = c - c.mean(1, keepdim=True)
    rr = r - r.mean(1, keepdim=True)
    num = (cc * rr).sum(1)
    den = torch.sqrt((cc * cc).sum(1) * (rr * rr).sum(1) + 1e-8)
    return (num / (den + 1e-8)).mean()


class NeuralCRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.stay_bonus = nn.Parameter(torch.tensor(2.0))

    def nll(self, emis: torch.Tensor, c: torch.Tensor, nsw: torch.Tensor,
            tags: torch.Tensor) -> torch.Tensor:
        return _crf_nll(emis, c, nsw, self.stay_bonus, tags)

    @torch.no_grad()
    def viterbi(self, emis: torch.Tensor, c: torch.Tensor,
                nsw: torch.Tensor) -> torch.Tensor:
        return _crf_viterbi(emis, c, nsw, self.stay_bonus)


INBRED_PREFIXES = [
    'B97', 'CML103', 'CML228', 'CML247', 'CML277', 'CML322', 'CML333',
    'CML52', 'CML69', 'HP301', 'Il14H', 'Ki11', 'Ki3', 'Ky21',
    'M162W', 'M37W', 'Mo18W', 'Ms71', 'NC350', 'NC358',
    'Oh43', 'Oh7B', 'P39', 'Tzi8',
]


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class LabeledDatasetHaploid(Dataset):
    """
    Reads 25-col inbred .npy matrices [T, 25]:
      cols 0:24  = founder read counts  (features)
      col  24    = founder label in [0, num_parents)  (-1 → num_parents = unknown)
    Windows into [window_size, num_parents] feature + [window_size] label tensors.
    """

    def __init__(self, file_dir, file_names, window_size=512, num_parents=24, step_size=128):
        self.file_dir = file_dir
        self.file_names = file_names
        self.window_size = window_size
        self.num_parents = num_parents
        self.step_size = step_size
        self.windows = self._build_windows()
        print(f"HaploidDataset: {len(self.windows)} windows from {len(file_names)} files")

    def _build_windows(self):
        windows = []
        for i, fname in enumerate(self.file_names):
            T = np.load(f"{self.file_dir}/{fname}", mmap_mode='r').shape[0]
            n = max(0, (T - self.window_size) // self.step_size)
            windows.extend((i, j) for j in range(n))
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        fi, wi = self.windows[idx]
        s = wi * self.step_size
        mat = np.load(
            f"{self.file_dir}/{self.file_names[fi]}",
            mmap_mode='r', allow_pickle=True
        )[s : s + self.window_size]

        features = torch.tensor(mat[:, :self.num_parents], dtype=torch.float)
        labels = torch.tensor(mat[:, self.num_parents].astype(np.int64), dtype=torch.long)
        labels[labels == -1] = self.num_parents  # unknown → last index
        return {"input_embeds": features, "labels": labels}


# --------------------------------------------------------------------------- #
#  Pre-windowed dataset  (N, 512, 25) — primary data source                   #
# --------------------------------------------------------------------------- #

class PreWindowedHaploidDataset(Dataset):
    """
    Wraps a pre-windowed int8 array; each row is already one training sample.

    Two layouts are accepted, both predicting a single founder path (col K):
      (N, T, K+1)  haploid : cols 0:K features, col K label
      (N, T, K+2)  diploid : cols 0:K features, col K = H1, col K+1 = H2
                             — H1 is the target; H2 is ignored (inbred data has
                             H1==H2, so the H1 path is well-defined).
    Label values >= K are clamped to K (unknown state).
    """
    def __init__(self, data: np.ndarray, num_parents: int = 24):
        self.data = data
        self.K = num_parents
        # K+3 layout carries the hidden true recomb rate in col K+2. It is NEVER a
        # model input; exposed here only as an auxiliary training target for c_t.
        self.has_rate = data.shape[-1] == num_parents + 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]                                    # (T, K+1 / K+2 / K+3)
        features = torch.tensor(row[:, :self.K], dtype=torch.float32)
        labels = row[:, self.K].astype(np.int64)               # col K = H1 path
        labels = np.clip(labels, 0, self.K)                    # noise >= K → unknown
        out = {"input_embeds": features,
               "labels": torch.tensor(labels, dtype=torch.long)}
        if self.has_rate:
            rate = row[:, self.K + 2].astype(np.float32)       # hidden true rate
            out["log_rate"] = torch.tensor(
                np.log(np.clip(rate, 1.0, None)), dtype=torch.float32)
        return out


def make_splits(path: str, num_parents: int, val_frac: float, test_frac: float,
                limit_n: int = 0):
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    if limit_n and limit_n < len(data):
        data = data[:limit_n]  # file is pre-shuffled; head slice is a random subset
    N = len(data)
    ncol = data.shape[-1]
    if ncol not in (num_parents + 1, num_parents + 2, num_parents + 3):
        raise ValueError(
            f"{Path(path).name} has {ncol} cols; expected K+1={num_parents + 1} "
            f"(haploid), K+2={num_parents + 2} (diploid), or K+3={num_parents + 3} "
            f"(diploid + recomb track) for K={num_parents}")
    layout = {num_parents + 1: "haploid",
              num_parents + 2: "diploid (predicting H1)",
              num_parents + 3: "diploid + hidden recomb track (eval-only, "
                               "cols>K+1 not fed to model)"}[ncol]
    n_test  = int(N * test_frac)
    n_val   = int(N * val_frac)
    n_train = N - n_val - n_test
    train_ds = PreWindowedHaploidDataset(data[:n_train],                num_parents)
    val_ds   = PreWindowedHaploidDataset(data[n_train:n_train + n_val], num_parents)
    test_ds  = PreWindowedHaploidDataset(data[n_train + n_val:],        num_parents)
    print(f"Dataset {Path(path).name}: N={N:,}  cols={ncol}  layout={layout}")
    print(f"  train={n_train:,}  val={n_val:,}  test={n_test:,}")
    return train_ds, val_ds, test_ds


def _founder_affinity(feats_block):
    """E5 relatedness signal: per-founder genome-wide affinity for one individual.

    feats_block [W, T, K] binary match indicators over ALL of an individual's
    windows. Returns [K, 2] = (raw match rate, founders-mean-centered rate). The
    raw rate is high for founders the individual descends from (and their
    consistent IBD-mates) and at background for the rest; centering across founders
    sharpens that contrast. Both features are bounded ([0,1] and [-1,1]) — a
    standardized (÷sd) version blows up for individuals whose founders share a
    near-uniform rate and overflows the bf16 CRF partition. Uses observations only
    (no labels), so it is computable identically at inference.
    """
    r = feats_block.reshape(-1, feats_block.shape[-1]).mean(0).astype(np.float32)  # [K]
    return np.stack([r, r - r.mean()], axis=-1)                     # [K, 2] bounded


class IndividualRelatednessDataset(Dataset):
    """E5: PreWindowedHaploidDataset + a per-individual founder-affinity ext_emb.

    Windows are grouped into individuals of G consecutive rows (the sim writes
    them contiguously). For each individual the genome-wide founder affinity is
    estimated once from its own features and attached to every one of its windows
    as `ext_emb` [K, ext_dim], conditioning the encoder so it can favour founders
    the individual actually carries and break within-window IBD ties.
    """
    def __init__(self, data: np.ndarray, num_parents: int, windows_per_individual: int):
        self.data = data
        self.K = num_parents
        self.G = windows_per_individual
        N = len(data)
        if N % self.G:
            raise ValueError(f"split rows {N} not divisible by windows/ind {self.G}")
        n_ind = N // self.G
        feats = np.asarray(data[:, :, :num_parents]).reshape(n_ind, self.G,
                                                             data.shape[1], num_parents)
        self.affinity = np.stack(
            [_founder_affinity(feats[i]) for i in range(n_ind)]).astype(np.float32)
        self.ext_dim = self.affinity.shape[-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        features = torch.tensor(row[:, :self.K], dtype=torch.float32)
        labels = np.clip(row[:, self.K].astype(np.int64), 0, self.K)
        return {"input_embeds": features,
                "labels": torch.tensor(labels, dtype=torch.long),
                "ext_emb": torch.tensor(self.affinity[idx // self.G],
                                        dtype=torch.float32)}


def make_individual_splits(path, num_parents, val_frac, test_frac,
                           windows_per_individual, limit_n=0):
    """Like make_splits but splits by whole individuals (G contiguous windows),

    so an individual's relatedness estimate never spans the train/test boundary
    and test individuals are entirely unseen. Returns E5 relatedness datasets.
    """
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    G = windows_per_individual
    if limit_n:
        data = data[:(limit_n // G) * G]
    N = len(data)
    n_ind = N // G
    n_test = int(n_ind * test_frac) * G
    n_val = int(n_ind * val_frac) * G
    n_train = N - n_val - n_test
    mk = lambda d: IndividualRelatednessDataset(d, num_parents, G)
    train_ds = mk(data[:n_train])
    val_ds = mk(data[n_train:n_train + n_val])
    test_ds = mk(data[n_train + n_val:])
    print(f"Dataset {Path(path).name}: N={N:,} individuals={n_ind} "
          f"windows/ind={G} ext_dim={train_ds.ext_dim}")
    print(f"  train={n_train:,}  val={n_val:,}  test={n_test:,}")
    return train_ds, val_ds, test_ds


# --------------------------------------------------------------------------- #
#  Lightning module                                                             #
# --------------------------------------------------------------------------- #

class GRITSCRFHaploid(pl.LightningModule):
    """
    Haploid CRF: same FounderPathEncoder as the diploid model, but the CRF
    runs over K founder states with nsw[i,j] = (i != j).
    Diploid promotion (E4b) loads this encoder unchanged into the pair-state wrapper.
    """

    def __init__(self, num_parents=24, d_model=256, n_heads=8, n_layers=6,
                 lr=3e-4, weight_decay=1e-5, gate_reg=0.05, time_local_emis=False,
                 warmup_steps=0, recomb_aux_weight=0.0, window_c=False,
                 no_transition=False, ext_dim=0, cosine_decay=False,
                 spike_skip=False, spike_mult=8.0):
        super().__init__()
        self.save_hyperparameters()
        self.num_parents = num_parents
        self.ext_dim = ext_dim
        self.cosine_decay = cosine_decay
        # Loss-spike step-skip: drop optimizer steps whose raw grad-norm is
        # non-finite or >> the running median, so one bad batch can't kick the
        # model out of the converged basin (RESULTS E8 1M oscillation).
        self.spike_skip = spike_skip
        self.spike_mult = spike_mult
        self._gnorm_ema = -1.0
        self._gnorm_seen = 0
        self._n_skipped = 0
        self.lr = lr
        self.weight_decay = weight_decay
        self.gate_reg = gate_reg
        self.warmup_steps = warmup_steps
        self.recomb_aux_weight = recomb_aux_weight
        self.no_transition = no_transition

        K = num_parents + 1  # +1 for unknown state
        self.encoder = FounderPathEncoder(d_model, n_heads, n_layers,
                                          ext_dim=ext_dim,
                                          time_local_emis=time_local_emis,
                                          window_c=window_c)
        self.crf = NeuralCRF()
        if no_transition:
            # Pure per-site emission classifier: zero & freeze the switch terms
            # so tr_t == 0 (Viterbi → per-site argmax; partition → independent
            # softmax). c is also zeroed at decode/score time (see decode()).
            nn.init.zeros_(self.crf.stay_bonus)
            self.crf.stay_bonus.requires_grad_(False)

        # nsw[i,j] = number of haplotype switches between state i and state j.
        # For haploid K states: 0 if i==j, 1 otherwise.
        nsw = (1.0 - torch.eye(K))
        self.register_buffer("nsw", nsw)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GRITSCRFHaploid: K={K} states, {n_params:,} params")

    def forward(self, X, ext_emb=None):
        B, T, K_feat = X.shape
        K = self.num_parents + 1
        X_pad = torch.cat([X, torch.zeros(B, T, 1, device=X.device)], dim=-1)  # [B,T,K]
        founder_mask = torch.ones(B, K, device=X.device)
        if ext_emb is not None:
            # pad the unknown-founder row (state K-1) with a zero relatedness vector
            ext_emb = torch.cat(
                [ext_emb, torch.zeros(B, 1, ext_emb.shape[-1], device=X.device)], dim=1)
        emis_f, g, c = self.encoder(X_pad, founder_mask, ext_emb=ext_emb)  # emis_f [B,T,K]
        return emis_f, g, c

    def _c_eff(self, c):
        """Transition cost actually used by the CRF: zeroed for the
        no-transition arm so the model is a pure per-site emission classifier."""
        return torch.zeros_like(c) if self.no_transition else c

    @torch.no_grad()
    def decode(self, emis_f, c):
        """Viterbi path, respecting the no_transition arm. Shared by validation
        and the standalone test evaluator so all decoding paths are identical."""
        return self.crf.viterbi(emis_f, self._c_eff(c), self.nsw)

    def _step(self, batch):
        X, tags = batch["input_embeds"], batch["labels"]
        emis_f, g, c = self(X, batch.get("ext_emb"))
        crf_loss = self.crf.nll(emis_f, self._c_eff(c), self.nsw, tags)
        gate_loss = self.gate_reg * (1.0 - g).mean()
        loss = crf_loss + gate_loss
        # Auxiliary supervision: push c_t to anti-correlate with the hidden recomb
        # rate (target only — the rate is never fed into the forward pass). corr in
        # [-1,1]; minimizing it drives hot regions toward low switch cost.
        corr = c.new_zeros(())
        if "log_rate" in batch:
            corr = _window_corr(c, batch["log_rate"])
            if self.recomb_aux_weight > 0:
                loss = loss + self.recomb_aux_weight * corr
        return loss, crf_loss, g, c, emis_f, corr

    def training_step(self, batch, batch_idx):
        loss, crf_loss, g, c, _, corr = self._step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/crf_loss", crf_loss)
        self.log("train/gate", g.mean())
        self.log("train/recomb", c.mean())
        self.log("train/recomb_corr", corr, prog_bar=True)
        self.log("train/stay_bonus", self.crf.stay_bonus)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, crf_loss, g, c, emis_f, corr = self._step(batch)
        tags = batch["labels"]
        pred = self.decode(emis_f, c)
        acc = (pred == tags).float().mean()
        self.log("val/loss", loss, prog_bar=True)
        self.log("val_loss", loss)                     # slash-free alias: ModelCheckpoint's
                                                         # filename= can't safely interpolate a
                                                         # metric name containing "/" (Lightning
                                                         # treats it as a path separator and
                                                         # scatters checkpoints into a stray
                                                         # val/ subdir) — see callbacks below.
        self.log("val/crf_loss", crf_loss)
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/gate", g.mean())
        self.log("val/recomb_corr", corr, prog_bar=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        # Runs after backward, BEFORE gradient clipping — so we see the raw
        # spike. Skip the step (zero grads → no-op update) when the global
        # grad-norm is non-finite or >> the running EMA of recent good norms.
        if not self.spike_skip:
            return
        norms = [p.grad.detach().norm() for p in self.parameters()
                 if p.grad is not None]
        if not norms:
            return
        g = float(torch.norm(torch.stack(norms)))
        self._gnorm_seen += 1
        warming = self._gnorm_seen <= 50         # establish the EMA first
        thresh = self.spike_mult * self._gnorm_ema if self._gnorm_ema > 0 else float("inf")
        spike = (not math.isfinite(g)) or (not warming and g > thresh)
        # Always update the EMA (capped on a spike) so it tracks the true level
        # and cannot lock low — otherwise heavy-tailed norms death-spiral into
        # skipping nearly every step (the e8-1M-final 57%-skip bug).
        g_ema = thresh if (spike and math.isfinite(thresh)) else (g if math.isfinite(g) else thresh)
        if math.isfinite(g_ema):
            self._gnorm_ema = g_ema if self._gnorm_ema <= 0 else 0.98 * self._gnorm_ema + 0.02 * g_ema
        if spike:
            for p in self.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            self._n_skipped += 1
            self.log("train/skipped", float(self._n_skipped), prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        if self.warmup_steps > 0:
            w = self.warmup_steps
            if self.cosine_decay:
                # Linear warmup then cosine decay to ~0. A constant lr lets late
                # epochs kick the model back out of the converged basin (the 1M
                # epoch-scale oscillation, RESULTS E8); decaying the step size
                # to zero lets it settle.
                total = max(int(self.trainer.estimated_stepping_batches), w + 1)
                import math
                def lam(step: int) -> float:
                    if step < w:
                        return (step + 1) / w
                    prog = (step - w) / max(1, total - w)
                    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
            else:
                # Linear warmup then constant.
                lam = lambda step: min(1.0, (step + 1) / w)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=5)
        return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val/loss"}


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def list_inbred_files(data_dir, val_chr="10"):
    """Split maize inbred files into train / val by chromosome."""
    all_files = [
        f for f in os.listdir(data_dir)
        if f.endswith(".npy") and any(f.startswith(p + "_") for p in INBRED_PREFIXES)
    ]
    train = [f for f in all_files if f"_chr{val_chr}_" not in f]
    val   = [f for f in all_files if f"_chr{val_chr}_" in f]
    return sorted(train), sorted(val)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir",     default="/workdir/esb33")
    # Pre-windowed single-file mode (preferred)
    p.add_argument("--data",        default=None,
                   help="Pre-windowed (N,512,25) .npy file; uses 80/10/10 sequential split")
    p.add_argument("--val-frac",    type=float, default=0.10)
    p.add_argument("--test-frac",   type=float, default=0.10)
    p.add_argument("--limit-n",     type=int, default=0,
                   help="Cap total observations before splitting (0 = all). "
                        "File is pre-shuffled, so this is a random subset.")
    # Legacy per-file mode
    p.add_argument("--data-dir",    default="/workdir/smm477/ML-data/training")
    p.add_argument("--val-chr",     default="10")
    p.add_argument("--window-size", type=int, default=512)
    p.add_argument("--step-size",   type=int, default=128)
    # Shared
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--d-model",     type=int, default=256)
    p.add_argument("--n-heads",     type=int, default=8)
    p.add_argument("--n-layers",    type=int, default=6)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--gate-reg",    type=float, default=0.05)
    p.add_argument("--time-local-emis", action="store_true",
                   help="Per-site founder emission key (for within-window founder switches)")
    p.add_argument("--window-c", action="store_true",
                   help="Single transition potential per window (pool features "
                        "over T) instead of a per-site c_t.")
    p.add_argument("--no-transition", action="store_true",
                   help="Disable transition potentials (stay_bonus=0, c=0): the "
                        "CRF reduces to a per-site emission classifier.")
    p.add_argument("--relatedness", action="store_true",
                   help="E5: condition the encoder on a per-individual founder "
                        "affinity (relatedness) vector estimated genome-wide from "
                        "the reads. Requires a sim built with "
                        "--windows-per-individual and the matching value here.")
    p.add_argument("--windows-per-individual", type=int, default=100,
                   help="E5: G consecutive windows per individual (must match the "
                        "value used to build the sim).")
    p.add_argument("--run-name", default="e1-haploid",
                   help="Sub-dir name for checkpoints and TB logs (one per arm).")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Linear LR warmup steps (0 = plateau scheduler)")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Gradient-norm clip value (lower = more bf16-stable)")
    p.add_argument("--cosine-decay", action="store_true",
                   help="Warmup then cosine-decay LR to ~0 (settles the basin "
                        "oscillation; needs --warmup-steps > 0)")
    p.add_argument("--spike-skip", action="store_true",
                   help="Skip optimizer steps with non-finite or spiking grad-norm")
    p.add_argument("--spike-mult", type=float, default=8.0,
                   help="Skip a step if raw grad-norm > this × running EMA")
    p.add_argument("--recomb-aux-weight", type=float, default=0.0,
                   help="Weight on the recomb_head auxiliary loss: per-window "
                        "Pearson corr(c_t, hidden log-rate), minimized so c_t "
                        "anti-correlates with the rate. Needs the K+3 layout "
                        "(--recomb-span>1 sim). 0 = off.")
    p.add_argument("--precision",   default="bf16-mixed",
                   help="Lightning precision (bf16-mixed avoids fp16 logsumexp overflow)")
    p.add_argument("--max-epochs",  type=int, default=50)
    p.add_argument("--patience",    type=int, default=10)
    p.add_argument("--devices",     type=int, default=1,
                   help="Number of GPUs (default 1; use 1 to avoid DDP overhead)")
    p.add_argument("--resume", default=None,
                   help="Path to a .ckpt to resume training from (optimizer/scheduler "
                        "state included; passed as Trainer.fit(ckpt_path=...)).")
    p.add_argument("--ema", action="store_true",
                   help="Track a weight EMA and swap it in for validation/checkpoint "
                        "(tames late-epoch oscillation/collapse).")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay rate (only used with --ema).")
    args = p.parse_args()

    workdir = Path(args.workdir)
    ckpt_dir = workdir / "checkpoints" / args.run_name
    log_dir  = workdir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ext_dim = 0
    if args.relatedness:
        train_ds, val_ds, _ = make_individual_splits(
            args.data, args.num_parents, args.val_frac, args.test_frac,
            args.windows_per_individual, limit_n=args.limit_n)
        ext_dim = train_ds.ext_dim
    elif args.data:
        train_ds, val_ds, _ = make_splits(
            args.data, args.num_parents, args.val_frac, args.test_frac,
            limit_n=args.limit_n)
    else:
        train_files, val_files = list_inbred_files(args.data_dir, args.val_chr)
        print(f"Train files: {len(train_files)}  |  Val files (chr{args.val_chr}): {len(val_files)}")
        train_ds = LabeledDatasetHaploid(args.data_dir, train_files,
                                         args.window_size, args.num_parents, args.step_size)
        val_ds   = LabeledDatasetHaploid(args.data_dir, val_files,
                                         args.window_size, args.num_parents, args.step_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=True)

    model = GRITSCRFHaploid(
        num_parents=args.num_parents,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        lr=args.lr,
        gate_reg=args.gate_reg,
        time_local_emis=args.time_local_emis,
        warmup_steps=args.warmup_steps,
        recomb_aux_weight=args.recomb_aux_weight,
        window_c=args.window_c,
        no_transition=args.no_transition,
        ext_dim=ext_dim,
        cosine_decay=args.cosine_decay,
        spike_skip=args.spike_skip,
        spike_mult=args.spike_mult,
    )

    callbacks = [
        # monitor the slash-free "val_loss" alias (logged alongside "val/loss"): a
        # filename= token containing "/" makes Lightning scatter checkpoints into a
        # stray val/ subdir instead of writing directly under ckpt_dir.
        ModelCheckpoint(dirpath=str(ckpt_dir), monitor="val_loss", mode="min",
                        save_top_k=3, save_last=True,
                        filename="e1-{epoch:02d}-{val_loss:.3f}"),
        EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),
    ]
    if args.ema:
        callbacks.append(EMACallback(args.ema_decay))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=TensorBoardLogger(str(log_dir), name=args.run_name),
        accelerator="auto",
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    print(f"Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
