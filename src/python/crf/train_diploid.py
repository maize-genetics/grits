"""
E4 diploid training — joint pair-state CRF on the shared FounderPathEncoder.

The encoder is unchanged (one emis_f [B,T,K] per founder). The diploid layer
forms pair-state emissions emis_p[b,t,(i,j)] = emis_f[i] + emis_f[j] over the
P = K(K+1)/2 unordered founder pairs, and decodes a pair path with a CRF whose
transition cost is -c * (#chromosomes that switch), nsw in {0,1,2}. A two-
chromosome switch therefore costs exp(-2c) = exp(-c)^2 — two INDEPENDENT
chromosome switches, matching the generative sim (no hard ban).

Reuses the state-count-agnostic forward/Viterbi structure from train_haploid
with a pair switch matrix. Only [B,P,P] is materialized per timestep (never
[B,T,P,P]).

Data: (N, T, K+2) — cols 0:K features, col K = H1, col K+1 = H2 (make_splits).

Usage:
    pixi run --environment gpu python src/python/crf/train_diploid.py \
        --data /workdir/esb33/data/training/sim_diploid_512.npy \
        --time-local-emis --lr 1e-4 --warmup-steps 500 --precision bf16-mixed \
        --max-epochs 5 --run-name diploid-pair
"""

import argparse
import math
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
from python.crf.train_haploid import make_splits
from python.crf.callbacks import EMACallback
# Pair-state CRF kernels — extracted verbatim into crf_kernels.py (2026-09,
# experiments/simulator-indels/TRAINING_PLAN.md §2) so the new indel-aware
# model (train_diploid_indel.py) shares them instead of duplicating them.
# Re-imported into this module's namespace (not just used locally) so every
# existing `from python.crf.train_diploid import _dcrf_viterbi`-style call
# site elsewhere in the repo keeps working unchanged.
from python.crf.crf_kernels import (
    _dcrf_nll, _dcrf_viterbi, _dcrf_marginal, _dcrf_viterbi_factored,
    build_pair_tables,
)


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class PreWindowedDiploidDataset(Dataset):
    """(N,T,K+2): cols 0:K features, col K = H1, col K+1 = H2. Returns the
    feature window plus the two haplotype labels (pair index built in the
    module to keep the K×K table in one place)."""
    def __init__(self, data, num_parents=24):
        self.data = data
        self.K = num_parents

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        feats = torch.tensor(row[:, :self.K], dtype=torch.float32)
        h1 = np.clip(row[:, self.K].astype(np.int64), 0, self.K)
        h2 = np.clip(row[:, self.K + 1].astype(np.int64), 0, self.K)
        return {"input_embeds": feats,
                "h1": torch.tensor(h1, dtype=torch.long),
                "h2": torch.tensor(h2, dtype=torch.long)}


def make_diploid_splits(path, num_parents, val_frac, test_frac, limit_n=0):
    """Same deterministic head-slice split as make_splits, diploid dataset."""
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    if limit_n and limit_n < len(data):
        data = data[:limit_n]
    N = len(data)
    n_test = int(N * test_frac)
    n_val = int(N * val_frac)
    n_tr = N - n_val - n_test
    mk = lambda a: PreWindowedDiploidDataset(a, num_parents)
    print(f"Diploid {Path(path).name}: N={N:,} cols={data.shape[-1]}  "
          f"train={n_tr:,} val={n_val:,} test={n_test:,}")
    return mk(data[:n_tr]), mk(data[n_tr:n_tr + n_val]), mk(data[n_tr + n_val:])


# E7: per-individual heterozygosity proxy → adaptive homozygous penalty. Inbred
# individuals (F=1) have identical gametes, so consecutive single-gamete reads stay
# on the same founder and their match-founder sets overlap; outbred individuals
# interleave two gametes, so adjacent reads' match sets disagree more. Aggregated
# over an individual's windows this tracks (1-F) almost exactly (corr -0.99), and
# is computed from reads only — so the het prior can be set per individual.
HET_INBRED, HET_OUTBRED = 0.23, 0.50      # proxy at F=1 and F=0 (sim calibration)


def _het_scale(feats_block):
    """Map an individual's windows [W,T,K] (0/1) to homo-penalty scale in [0,1]:
    0 for inbred (allow homozygous), 1 for fully outbred (full het prior)."""
    a, b = feats_block[:, :-1], feats_block[:, 1:]
    inter = (a * b).sum(-1)
    uni = ((a + b) > 0).sum(-1)
    jac = np.where(uni > 0, inter / np.maximum(uni, 1), 1.0)
    het = float((1.0 - jac).mean())
    return float(np.clip((het - HET_INBRED) / (HET_OUTBRED - HET_INBRED), 0.0, 1.0))


class DiploidIndividualDataset(PreWindowedDiploidDataset):
    """Diploid dataset + a per-individual adaptive homozygous-penalty scale, from
    the genome-wide het proxy (reads only). Windows are grouped in blocks of G."""
    def __init__(self, data, num_parents, windows_per_individual):
        super().__init__(data, num_parents)
        G = windows_per_individual
        if len(data) % G:
            raise ValueError(f"rows {len(data)} not divisible by windows/ind {G}")
        self.G = G
        feats = np.asarray(data[:, :, :num_parents]).reshape(
            len(data) // G, G, data.shape[1], num_parents).astype(np.float32)
        self.scale = np.array([_het_scale(feats[i]) for i in range(len(feats))],
                              dtype=np.float32)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out["homo_scale"] = torch.tensor(self.scale[idx // self.G], dtype=torch.float32)
        return out


def make_diploid_individual_splits(path, num_parents, val_frac, test_frac, G, limit_n=0):
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    if limit_n:
        data = data[:(limit_n // G) * G]
    N = len(data)
    n_ind = N // G
    n_test = int(n_ind * test_frac) * G
    n_val = int(n_ind * val_frac) * G
    n_tr = N - n_val - n_test
    mk = lambda a: DiploidIndividualDataset(a, num_parents, G)
    print(f"Diploid(individual) {Path(path).name}: N={N:,} individuals={n_ind} "
          f"train={n_tr:,} val={n_val:,} test={n_test:,}")
    return mk(data[:n_tr]), mk(data[n_tr:n_tr + n_val]), mk(data[n_tr + n_val:])


def _founder_affinity(feats_block):
    """Per-founder genome-wide affinity for one individual (E5 relatedness signal).
    feats_block [W,T,K] (binary match) -> [K,2] = (raw match rate, founders-mean-
    centered rate). High for founders the individual descends from (and their IBD-
    mates), at background for the rest; centering sharpens the contrast. Both bounded.
    Reads only (no labels), so identical at inference."""
    r = feats_block.reshape(-1, feats_block.shape[-1]).mean(0).astype(np.float32)  # [K]
    return np.stack([r, r - r.mean()], axis=-1)                     # [K,2] bounded


def estimate_inbreeding_coef(affinity_rate):
    """Genome-wide zygosity estimate from _founder_affinity's raw match-rate
    column [K]: background-corrected gap between the top-1 and top-2 founder.
    Real reads carry a substantial non-zero match rate even for unrelated
    founders (pangenome background co-support, not signal) -- the top-2 rate
    alone isn't near zero for a true hybrid, so the gap must be normalized
    against the rate of an uninvolved (rank>=3) founder, not against 0.
    1.0 = one dominant founder (inbred), 0.0 = two comparably-elevated
    founders (outbred/hybrid). Verified on real IDX-INBRED/IDX-HYB samples --
    see tests/python/crf/test_real_data_affinity.py."""
    order = np.argsort(-affinity_rate)
    r1, r2 = affinity_rate[order[0]], affinity_rate[order[1]]
    bg = np.median(affinity_rate[order[2:]])
    denom = r1 - bg
    if denom <= 1e-6:
        return 0.0
    return float(1.0 - np.clip((r2 - bg) / denom, 0.0, 1.0))


def homo_scale_from_affinity(affinity_rate, inbred_thresh=0.5):
    """--homo-scale value to condition the model's fixed homo_penalty on this
    individual's actual zygosity, instead of applying it uniformly regardless
    of evidence (infer_wholegenome_real.py's documented gap). A smooth scale
    (1 - est_F) under-corrects: real per-site emission margins are often
    smaller than the residual penalty even at est_F~0.9, so a hard threshold
    is used instead -- verified end-to-end on real data (100% inbred / 98.9%
    hybrid pair_acc vs 1.5% / 98.9% with the fixed penalty)."""
    return 0.0 if estimate_inbreeding_coef(affinity_rate) > inbred_thresh else 1.0


class DiploidAffinityDataset(PreWindowedDiploidDataset):
    """Diploid dataset + a per-individual founder-affinity ext_emb [K,2], attached
    to every window of the individual (windows grouped in blocks of G). Conditions
    the encoder to favour founders the individual actually carries and break the
    within-window IBD ties the local emission cannot."""
    def __init__(self, data, num_parents, windows_per_individual):
        super().__init__(data, num_parents)
        G = windows_per_individual
        if len(data) % G:
            raise ValueError(f"rows {len(data)} not divisible by windows/ind {G}")
        self.G = G
        feats = np.asarray(data[:, :, :num_parents]).reshape(
            len(data) // G, G, data.shape[1], num_parents).astype(np.float32)
        self.affinity = np.stack(
            [_founder_affinity(feats[i]) for i in range(len(feats))]).astype(np.float32)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out["ext_emb"] = torch.tensor(self.affinity[idx // self.G], dtype=torch.float32)
        return out


def make_diploid_affinity_splits(path, num_parents, val_frac, test_frac, G, limit_n=0):
    """Individual-aligned split (same boundaries as make_diploid_individual_splits,
    so the test set matches the head-slice splits) with founder-affinity ext_emb."""
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    if limit_n:
        data = data[:(limit_n // G) * G]
    N = len(data)
    n_ind = N // G
    n_test = int(n_ind * test_frac) * G
    n_val = int(n_ind * val_frac) * G
    n_tr = N - n_val - n_test
    mk = lambda a: DiploidAffinityDataset(a, num_parents, G)
    print(f"Diploid(affinity) {Path(path).name}: N={N:,} individuals={n_ind} "
          f"train={n_tr:,} val={n_val:,} test={n_test:,}")
    return mk(data[:n_tr]), mk(data[n_tr:n_tr + n_val]), mk(data[n_tr + n_val:])


# --------------------------------------------------------------------------- #
#  Lightning module                                                            #
# --------------------------------------------------------------------------- #

class GRITSCRFDiploid(pl.LightningModule):
    def __init__(self, num_parents=24, d_model=256, n_heads=8, n_layers=6,
                 lr=1e-4, weight_decay=1e-5, gate_reg=0.05, time_local_emis=False,
                 warmup_steps=0, homo_penalty=0.0,
                 cosine_decay=False, spike_skip=False, spike_mult=8.0,
                 loss_spike_mult=5.0,
                 learned_het=False, founder_affinity=False):
        super().__init__()
        self.save_hyperparameters()
        self.num_parents = num_parents
        self.lr = lr
        self.weight_decay = weight_decay
        self.gate_reg = gate_reg
        self.warmup_steps = warmup_steps
        self.homo_penalty = homo_penalty
        # Stability recipe ported from E8 haploid (RESULTS E8): cosine-decay-to-0
        # settles the basin oscillation, spike-skip drops freak-gradient steps.
        self.cosine_decay = cosine_decay
        self.spike_skip = spike_skip
        self.spike_mult = spike_mult
        self._gnorm_ema = -1.0
        self._gnorm_seen = 0
        self._n_skipped = 0
        # A localized partition-NLL spike can corrupt the encoder without moving the
        # GLOBAL grad-norm enough to trip spike_mult (the collapse-to-degenerate
        # failure). Skip on a CRF-loss spike too: catch the freak batch directly.
        self.loss_spike_mult = loss_spike_mult
        self._loss_ema = -1.0
        self._loss_seen = 0
        self._loss_spike = False
        # E7-diag fix: learned per-locus het prior replaces the fixed homo_penalty.
        self.learned_het = learned_het
        # crf-relatedness: per-individual founder-affinity prior (ext_bias). A
        # genome-wide per-founder presence signal that biases emissions toward the
        # founders the individual carries and breaks within-window IBD ties.
        self.founder_affinity = founder_affinity
        ext_dim = 2 if founder_affinity else 0

        K = num_parents + 1                          # +1 unknown, matches encoder
        self.encoder = FounderPathEncoder(d_model, n_heads, n_layers, ext_dim=ext_dim,
                                          time_local_emis=time_local_emis,
                                          learned_het=learned_het)
        self.stay_bonus = nn.Parameter(torch.tensor(2.0))

        pi, pj, pair_table, nsw = build_pair_tables(K)
        self.register_buffer("pi", pi)
        self.register_buffer("pj", pj)
        self.register_buffer("pair_table", pair_table)
        self.register_buffer("nsw_pair", nsw)
        # Het prior: with one read/site the emission emis_f[i]+emis_f[j] is
        # maximized by the homozygous pair of the observed founder, so without a
        # counter-force the decode collapses to all-homozygous. Subtract a
        # constant from homozygous pair-states (i==j), matching diploid_hmm.
        self.register_buffer("homo_mask", (pi == pj).float())
        self.P = pi.numel()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GRITSCRFDiploid: K={K} states, P={self.P} pair-states, "
              f"{n_params:,} params")

    def forward(self, X, homo_scale=None, ext_emb=None):
        B, T, K_feat = X.shape
        K = self.num_parents + 1
        X_pad = torch.cat([X, torch.zeros(B, T, 1, device=X.device)], dim=-1)
        founder_mask = torch.ones(B, K, device=X.device)
        if ext_emb is not None:                                  # pad the null founder
            ext_emb = torch.cat(
                [ext_emb, torch.zeros(B, 1, ext_emb.shape[-1], device=X.device)], dim=1)
        if self.learned_het:
            emis_f, g, c, het = self.encoder(X_pad, founder_mask, ext_emb=ext_emb,
                                             emit_het=True)
        else:
            emis_f, g, c = self.encoder(X_pad, founder_mask, ext_emb=ext_emb)  # [B,T,K]
        emis_p = emis_f[..., self.pi] + emis_f[..., self.pj]     # [B,T,P]
        if self.learned_het:
            # E7-diag fix: a PER-LOCUS, encoder-driven homozygous penalty. The
            # Transformer sees the sustained-alternation pattern of a het region and
            # raises this where it's heterozygous; ~0 in homozygous regions. This is
            # the emission-side het signal the transition cost provably cannot give.
            het_pen = F.softplus(het).unsqueeze(-1)              # [B,T,1] >= 0
            emis_p = emis_p - het_pen * self.homo_mask
        elif self.homo_penalty != 0.0:
            # E7: with a per-individual scale (0=inbred → no penalty, 1=outbred →
            # full het prior), the homozygous penalty adapts to each sample's
            # inbreeding; without it, the fixed scalar applies to all.
            pen = self.homo_penalty
            if homo_scale is not None:
                pen = pen * homo_scale.view(B, 1, 1)
            emis_p = emis_p - pen * self.homo_mask
        return emis_p, g, c

    def _pair_labels(self, h1, h2):
        return self.pair_table[h1, h2]                           # [B,T]

    def _step(self, batch):
        X, h1, h2 = batch["input_embeds"], batch["h1"], batch["h2"]
        emis_p, g, c = self(X, batch.get("homo_scale"), batch.get("ext_emb"))
        tags = self._pair_labels(h1, h2)
        crf = _dcrf_nll(emis_p, c, self.nsw_pair, self.stay_bonus, tags)
        loss = crf + self.gate_reg * (1.0 - g).mean()
        return loss, crf, g, c, emis_p, tags

    def training_step(self, batch, _):
        loss, crf, g, c, _, _ = self._step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/crf_loss", crf)
        self.log("train/gate", g.mean())
        if self.spike_skip:                                    # flag CRF-loss spikes
            self._loss_seen += 1
            cv = float(crf.detach())
            warming = self._loss_seen <= 50
            thresh = (self.loss_spike_mult * self._loss_ema
                      if self._loss_ema > 0 else float("inf"))
            self._loss_spike = (math.isfinite(cv) and not warming and cv > thresh)
            # cap the EMA update on a spike so it can't be dragged up by the freak
            upd = min(cv, thresh) if math.isfinite(thresh) else cv
            if math.isfinite(upd):
                self._loss_ema = (upd if self._loss_ema <= 0
                                  else 0.98 * self._loss_ema + 0.02 * upd)
        return loss

    @torch.no_grad()
    def _accuracy(self, emis_p, c, h1, h2):
        pred = _dcrf_viterbi(emis_p, c, self.nsw_pair, self.stay_bonus)
        pair_true = self.pair_table[h1, h2]
        pair_acc = (pred == pair_true).float().mean()
        # per-haplotype: both stored sorted (pi<=pj), compare to sorted truth
        pred_lo, pred_hi = self.pi[pred], self.pj[pred]
        t_lo = torch.minimum(h1, h2)
        t_hi = torch.maximum(h1, h2)
        hap_acc = ((pred_lo == t_lo).float() + (pred_hi == t_hi).float()).mean() / 2
        return pair_acc, hap_acc

    def validation_step(self, batch, _):
        loss, crf, g, c, emis_p, _ = self._step(batch)
        pair_acc, hap_acc = self._accuracy(emis_p, c, batch["h1"], batch["h2"])
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/pair_acc", pair_acc, prog_bar=True)
        self.log("val_pair_acc", pair_acc)              # slash-free alias: ModelCheckpoint's
                                                          # filename= can't safely interpolate a
                                                          # metric name containing "/" (Lightning
                                                          # treats it as a path separator and
                                                          # scatters checkpoints into a stray
                                                          # val/ subdir) — see callbacks below.
        self.log("val/hap_acc", hap_acc, prog_bar=True)
        self.log("val/gate", g.mean())
        return loss

    def on_before_optimizer_step(self, optimizer):
        # See train_haploid: runs before clipping; skip steps whose raw global
        # grad-norm is non-finite or >> the running EMA of good norms.
        if not self.spike_skip:
            return
        norms = [p.grad.detach().norm() for p in self.parameters()
                 if p.grad is not None]
        if not norms:
            return
        g = float(torch.norm(torch.stack(norms)))
        self._gnorm_seen += 1
        warming = self._gnorm_seen <= 50
        thresh = self.spike_mult * self._gnorm_ema if self._gnorm_ema > 0 else float("inf")
        spike = (not math.isfinite(g)) or (not warming and g > thresh)
        # Always update the EMA (capped on a spike) so it can't lock low — see
        # train_haploid for the death-spiral this prevents.
        g_ema = thresh if (spike and math.isfinite(thresh)) else (g if math.isfinite(g) else thresh)
        if math.isfinite(g_ema):
            self._gnorm_ema = g_ema if self._gnorm_ema <= 0 else 0.98 * self._gnorm_ema + 0.02 * g_ema
        # also skip if this batch's CRF NLL spiked (set in training_step): a
        # localized partition blow-up the global grad-norm may not surface.
        if spike or self._loss_spike:
            for p in self.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            self._n_skipped += 1
            self.log("train/skipped", float(self._n_skipped), prog_bar=True)
        self._loss_spike = False

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        if self.warmup_steps > 0:
            w = self.warmup_steps
            if self.cosine_decay:
                total = max(int(self.trainer.estimated_stepping_batches), w + 1)
                def lam(step: int) -> float:
                    if step < w:
                        return (step + 1) / w
                    prog = (step - w) / max(1, total - w)
                    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
            else:
                lam = lambda s: min(1.0, s / w)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5)
        return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val/loss"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--workdir", default="/workdir/esb33")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--limit-n", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gate-reg", type=float, default=0.05)
    p.add_argument("--time-local-emis", action="store_true")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Gradient-norm clip value")
    p.add_argument("--cosine-decay", action="store_true",
                   help="Warmup then cosine-decay LR to ~0 (E8 stability recipe; "
                        "needs --warmup-steps > 0)")
    p.add_argument("--spike-skip", action="store_true",
                   help="Skip optimizer steps with non-finite or spiking grad-norm")
    p.add_argument("--spike-mult", type=float, default=8.0,
                   help="Skip a step if raw grad-norm > this × running EMA")
    p.add_argument("--loss-spike-mult", type=float, default=5.0,
                   help="With --spike-skip, also skip a step if its CRF NLL > this × "
                        "running EMA (catches localized partition spikes the global "
                        "grad-norm misses)")
    p.add_argument("--ema", action="store_true",
                   help="Maintain a weight EMA and use it for validation + the saved "
                        "checkpoint (tames late-epoch oscillation/collapse)")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay (effective window ~1/(1-decay) steps)")
    p.add_argument("--homo-penalty", type=float, default=0.0,
                   help="Subtract from homozygous pair emissions (het prior); "
                        "counters the all-homozygous collapse of single-read diploid.")
    p.add_argument("--adaptive-homo", action="store_true",
                   help="E7: scale --homo-penalty per individual by a genome-wide "
                        "het proxy (0 for inbred lines, 1 for outbred), so one model "
                        "serves a mixed-inbreeding panel. Needs --windows-per-individual.")
    p.add_argument("--learned-het", action="store_true",
                   help="E7-diag fix: per-locus encoder-driven homozygous penalty "
                        "(replaces fixed --homo-penalty). The emission-side het signal.")
    p.add_argument("--windows-per-individual", type=int, default=100)
    p.add_argument("--founder-affinity", action="store_true",
                   help="crf-relatedness: condition the encoder on a per-individual "
                        "founder-affinity prior (ext_bias). Needs "
                        "--windows-per-individual.")
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--val-check-interval", type=int, default=0,
                   help="Validate every N training steps (0 = once per epoch). Maps "
                        "the within-epoch peak/drift at fine resolution.")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--run-name", default="diploid-pair")
    p.add_argument("--resume", default=None,
                   help="Path to a .ckpt to resume training from (optimizer/scheduler "
                        "state included; passed as Trainer.fit(ckpt_path=...)).")
    return p.parse_args()


def main():
    args = parse_args()
    workdir = Path(args.workdir)
    ckpt_dir = workdir / "checkpoints" / args.run_name
    log_dir = workdir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.founder_affinity:
        train_ds, val_ds, _ = make_diploid_affinity_splits(
            args.data, args.num_parents, args.val_frac, args.test_frac,
            args.windows_per_individual, limit_n=args.limit_n)
    elif args.adaptive_homo:
        train_ds, val_ds, _ = make_diploid_individual_splits(
            args.data, args.num_parents, args.val_frac, args.test_frac,
            args.windows_per_individual, limit_n=args.limit_n)
    else:
        train_ds, val_ds, _ = make_diploid_splits(
            args.data, args.num_parents, args.val_frac, args.test_frac,
            limit_n=args.limit_n)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = GRITSCRFDiploid(
        num_parents=args.num_parents, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, lr=args.lr, gate_reg=args.gate_reg,
        time_local_emis=args.time_local_emis, warmup_steps=args.warmup_steps,
        homo_penalty=args.homo_penalty, cosine_decay=args.cosine_decay,
        spike_skip=args.spike_skip, spike_mult=args.spike_mult,
        loss_spike_mult=args.loss_spike_mult,
        learned_het=args.learned_het, founder_affinity=args.founder_affinity)

    # Checkpoint/stop on val/pair_acc (max): the CRF partition NLL can spike on
    # long-block data even as Viterbi accuracy stays good, so selecting on loss
    # can discard the best model. Accuracy is the quantity we report.
    callbacks = [
        # monitor the slash-free "val_pair_acc" alias (logged alongside "val/pair_acc"):
        # a filename= token containing "/" makes Lightning scatter checkpoints into a
        # stray val/ subdir instead of writing directly under ckpt_dir.
        ModelCheckpoint(dirpath=str(ckpt_dir), monitor="val_pair_acc",
                        mode="max", save_top_k=2, save_last=True,
                        filename="d-{epoch:02d}-{val_pair_acc:.4f}"),
        EarlyStopping(monitor="val_pair_acc", mode="max", patience=args.patience),
    ]
    if args.ema:
        callbacks.append(EMACallback(args.ema_decay))
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, callbacks=callbacks,
        logger=TensorBoardLogger(str(log_dir), name=args.run_name),
        accelerator="auto", devices=args.devices, precision=args.precision,
        val_check_interval=(args.val_check_interval or None),
        gradient_clip_val=args.grad_clip)
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    print(f"Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
