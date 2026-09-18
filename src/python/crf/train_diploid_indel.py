"""
Indel-aware diploid training — ternary+distance ("2K+2") input format.

New model line, not an upgrade of GRITSCRFDiploid: the input representation
is materially different (2 channels per cell instead of 1, with different
semantics — see experiments/simulator-indels/TRAINING_PLAN.md §1), so this
gets its own training script, its own LightningModule (GRITSCRFDiploidIndel),
and its own checkpoint lineage. diploid-affinity-sim512-h3 stays deployed and
untouched — a comparison target, not a checkpoint this script ever loads.

Data layout (experiments/simulator-indels/PLAN.md §2.3), width 2K+2:
    cols 0:K      ternary read-sharing state per founder, {-1,0,1} =
                  deletion/diverged/match relative to reference (producer may
                  also emit TERN_PAD=-2 on the rare short-window edge case)
    col  K        H1 founder label (0..K-1, or LABEL_PAD=-1 if unlabeled)
    col  K+1      H2 founder label (0..K-1, or LABEL_PAD=-1 if unlabeled)
    cols K+2:2K+2 distance to nearest reference anchor per founder, int8
                  log-coded (DIST_PAD=-1 = no anchor, else 0..DIST_SAT-1)

The TERN_*/DIST_*/LABEL_PAD constants below are literal copies of the
producer's contract (simulate_alleles.py, branch simulator-indel-modeling),
not an import from that file: this script's worktree branches off that
file's last-committed state, which predates those constants landing there —
importing them would create a runtime dependency on a file that is actively
being edited by a parallel effort. Keep these values in sync with
simulate_alleles.py's own TERN_DEL/TERN_DIV/TERN_MATCH/TERN_PAD/DIST_PAD/
DIST_SAT/LABEL_PAD/DIST_LOG_SCALE block once both land on the same branch.

Usage:
    pixi run -- python src/python/crf/train_diploid_indel.py \
        --data /workdir/esb33/data/training/sim_diploid_indel.npy \
        --time-local-emis --lr 1e-4 --warmup-steps 500 --precision bf16-mixed \
        --max-epochs 5 --run-name diploid-indel-pair
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

from python.crf.train_crf import IndelFounderPathEncoder
from python.crf.crf_kernels import _dcrf_nll, _dcrf_viterbi, build_pair_tables
from python.crf.train_diploid import _founder_affinity, homo_scale_from_affinity
from python.crf.callbacks import EMACallback

# --- Indel-mode sentinels, mirroring simulate_alleles.py's contract -------
TERN_DEL, TERN_DIV, TERN_MATCH, TERN_PAD = -1, 0, 1, -2
DIST_PAD = -1
DIST_SAT = 127
LABEL_PAD = -1


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class IndelDiploidDataset(Dataset):
    """(N,T,2K+2): cols 0:K ternary, col K=H1, col K+1=H2, cols K+2:2K+2
    distance. Returns the (ternary,distance) feature window plus the two
    haplotype labels. Unlike PreWindowedDiploidDataset's np.clip(label,0,K)
    (which wrongly maps LABEL_PAD=-1 onto founder 0), unlabeled positions are
    explicitly remapped to the null-founder index K, matching the convention
    ropebwt_npy_to_matrix.py already uses for real data (gA[gA<0]=K)."""
    def __init__(self, data, num_parents=24):
        self.data = data
        self.K = num_parents

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        K = self.K
        tern = row[:, :K].astype(np.float32)
        dist = row[:, K + 2:2 * K + 2].astype(np.float32)
        feats = torch.tensor(np.stack([tern, dist], axis=-1), dtype=torch.float32)  # [T,K,2]
        h1_raw = row[:, K].astype(np.int64)
        h2_raw = row[:, K + 1].astype(np.int64)
        h1 = np.where(h1_raw < 0, K, h1_raw)
        h2 = np.where(h2_raw < 0, K, h2_raw)
        return {"input_embeds": feats,
                "h1": torch.tensor(h1, dtype=torch.long),
                "h2": torch.tensor(h2, dtype=torch.long)}


def _check_width(path, data, num_parents):
    expected = 2 * num_parents + 2
    if data.shape[-1] != expected:
        raise ValueError(f"{path}: expected width {expected} (2K+2, K={num_parents}), "
                         f"got {data.shape[-1]}")


def make_indel_diploid_splits(path, num_parents, val_frac, test_frac, limit_n=0):
    """Same deterministic head-slice split as train_diploid.make_diploid_splits,
    applied to the 2K+2 layout."""
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    _check_width(path, data, num_parents)
    if limit_n and limit_n < len(data):
        data = data[:limit_n]
    N = len(data)
    n_test = int(N * test_frac)
    n_val = int(N * val_frac)
    n_tr = N - n_val - n_test
    mk = lambda a: IndelDiploidDataset(a, num_parents)
    print(f"IndelDiploid {Path(path).name}: N={N:,} cols={data.shape[-1]}  "
          f"train={n_tr:,} val={n_val:,} test={n_test:,}")
    return mk(data[:n_tr]), mk(data[n_tr:n_tr + n_val]), mk(data[n_tr + n_val:])


def _het_scale_indel(feats_block, het_inbred, het_outbred):
    """Same Jaccard-of-adjacent-match-sets proxy as train_diploid._het_scale,
    computed over the MATCH view (ternary==1) — bit-identical arithmetic to
    the existing binary path, since ternary==1 IS today's binary presence
    (verified against ropebwt3-phg's rb3_lift_ternary_state). Only the two
    calibration endpoints are parameterized rather than hardcoded: the
    simulator now models coverage/deletions, which shifts this proxy's
    distribution even though the definition of "supported" hasn't changed."""
    a, b = feats_block[:, :-1], feats_block[:, 1:]
    inter = (a * b).sum(-1)
    uni = ((a + b) > 0).sum(-1)
    jac = np.where(uni > 0, inter / np.maximum(uni, 1), 1.0)
    het = float((1.0 - jac).mean())
    return float(np.clip((het - het_inbred) / (het_outbred - het_inbred), 0.0, 1.0))


class IndelDiploidIndividualDataset(IndelDiploidDataset):
    """IndelDiploidDataset + a per-individual adaptive homozygous-penalty
    scale, from the genome-wide het proxy over the MATCH view (ternary==1).
    Windows grouped in blocks of G, mirroring DiploidIndividualDataset."""
    def __init__(self, data, num_parents, windows_per_individual,
                 het_inbred=0.23, het_outbred=0.50):
        super().__init__(data, num_parents)
        G = windows_per_individual
        if len(data) % G:
            raise ValueError(f"rows {len(data)} not divisible by windows/ind {G}")
        self.G = G
        tern = np.asarray(data[:, :, :num_parents])
        M = (tern == TERN_MATCH).astype(np.float32).reshape(
            len(data) // G, G, data.shape[1], num_parents)
        self.scale = np.array(
            [_het_scale_indel(M[i], het_inbred, het_outbred) for i in range(len(M))],
            dtype=np.float32)
        print(f"IndelDiploid(individual) het_scale: mean={self.scale.mean():.4f} "
              f"std={self.scale.std():.4f} min={self.scale.min():.4f} "
              f"max={self.scale.max():.4f}  (het_inbred={het_inbred}, "
              f"het_outbred={het_outbred} — recalibrate once real indel-mode "
              f"data's distribution is known, TRAINING_PLAN.md §3)")

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out["homo_scale"] = torch.tensor(self.scale[idx // self.G], dtype=torch.float32)
        return out


def make_indel_diploid_individual_splits(path, num_parents, val_frac, test_frac, G,
                                         het_inbred=0.23, het_outbred=0.50, limit_n=0):
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    _check_width(path, data, num_parents)
    if limit_n:
        data = data[:(limit_n // G) * G]
    N = len(data)
    n_ind = N // G
    n_test = int(n_ind * test_frac) * G
    n_val = int(n_ind * val_frac) * G
    n_tr = N - n_val - n_test
    mk = lambda a: IndelDiploidIndividualDataset(a, num_parents, G, het_inbred, het_outbred)
    print(f"IndelDiploid(individual) {Path(path).name}: N={N:,} individuals={n_ind} "
          f"train={n_tr:,} val={n_val:,} test={n_test:,}")
    return mk(data[:n_tr]), mk(data[n_tr:n_tr + n_val]), mk(data[n_tr + n_val:])


class IndelDiploidAffinityDataset(IndelDiploidDataset):
    """IndelDiploidDataset + a per-individual founder-affinity ext_emb [K,2],
    from train_diploid._founder_affinity (reused verbatim, bit-identical — it
    has no calibration constants to shift, just mean/centered-mean over the
    MATCH view) attached to every window of the individual."""
    def __init__(self, data, num_parents, windows_per_individual):
        super().__init__(data, num_parents)
        G = windows_per_individual
        if len(data) % G:
            raise ValueError(f"rows {len(data)} not divisible by windows/ind {G}")
        self.G = G
        tern = np.asarray(data[:, :, :num_parents])
        M = (tern == TERN_MATCH).astype(np.float32).reshape(
            len(data) // G, G, data.shape[1], num_parents)
        self.affinity = np.stack(
            [_founder_affinity(M[i]) for i in range(len(M))]).astype(np.float32)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out["ext_emb"] = torch.tensor(self.affinity[idx // self.G], dtype=torch.float32)
        return out


def make_indel_diploid_affinity_splits(path, num_parents, val_frac, test_frac, G, limit_n=0):
    """Individual-aligned split, same boundaries as
    make_indel_diploid_individual_splits (mirrors train_diploid.py's
    make_diploid_affinity_splits intent)."""
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    _check_width(path, data, num_parents)
    if limit_n:
        data = data[:(limit_n // G) * G]
    N = len(data)
    n_ind = N // G
    n_test = int(n_ind * test_frac) * G
    n_val = int(n_ind * val_frac) * G
    n_tr = N - n_val - n_test
    mk = lambda a: IndelDiploidAffinityDataset(a, num_parents, G)
    print(f"IndelDiploid(affinity) {Path(path).name}: N={N:,} individuals={n_ind} "
          f"train={n_tr:,} val={n_val:,} test={n_test:,}")
    return mk(data[:n_tr]), mk(data[n_tr:n_tr + n_val]), mk(data[n_tr + n_val:])


# --------------------------------------------------------------------------- #
#  Lightning module                                                            #
# --------------------------------------------------------------------------- #

class GRITSCRFDiploidIndel(pl.LightningModule):
    """Mirrors GRITSCRFDiploid's structure exactly — same loss, same CRF
    kernels (imported from crf_kernels.py), same training-loop glue.
    TRAINING_PLAN.md §2: EMACallback and the Trainer/ModelCheckpoint/
    EarlyStopping wiring are reusable AS WRITTEN, duplicated here as
    boilerplate rather than imported, matching that document's
    recommendation for a fully independent training script. Only the encoder
    class and the null-founder pad value (§3) differ from GRITSCRFDiploid."""
    def __init__(self, num_parents=24, d_model=256, n_heads=8, n_layers=6,
                 lr=1e-4, weight_decay=1e-5, gate_reg=0.05, time_local_emis=False,
                 warmup_steps=0, homo_penalty=0.0,
                 cosine_decay=False, spike_skip=False, spike_mult=8.0,
                 loss_spike_mult=5.0,
                 learned_het=False, founder_affinity=False, fast_cells=False):
        super().__init__()
        self.save_hyperparameters()
        self.num_parents = num_parents
        self.lr = lr
        self.weight_decay = weight_decay
        self.gate_reg = gate_reg
        self.warmup_steps = warmup_steps
        self.homo_penalty = homo_penalty
        self.cosine_decay = cosine_decay
        self.spike_skip = spike_skip
        self.spike_mult = spike_mult
        self._gnorm_ema = -1.0
        self._gnorm_seen = 0
        self._n_skipped = 0
        self.loss_spike_mult = loss_spike_mult
        self._loss_ema = -1.0
        self._loss_seen = 0
        self._loss_spike = False
        self.learned_het = learned_het
        self.founder_affinity = founder_affinity
        ext_dim = 2 if founder_affinity else 0

        K = num_parents + 1                          # +1 unknown, matches encoder
        self.encoder = IndelFounderPathEncoder(
            d_model, n_heads, n_layers, ext_dim=ext_dim,
            time_local_emis=time_local_emis, learned_het=learned_het,
            fast_cells=fast_cells)
        self.stay_bonus = nn.Parameter(torch.tensor(2.0))

        pi, pj, pair_table, nsw = build_pair_tables(K)
        self.register_buffer("pi", pi)
        self.register_buffer("pj", pj)
        self.register_buffer("pair_table", pair_table)
        self.register_buffer("nsw_pair", nsw)
        self.register_buffer("homo_mask", (pi == pj).float())
        self.P = pi.numel()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GRITSCRFDiploidIndel: K={K} states, P={self.P} pair-states, "
              f"{n_params:,} params")

    def forward(self, X, homo_scale=None, ext_emb=None):
        B, T, K_feat, _ = X.shape
        K = self.num_parents + 1
        # Null-founder pad: (TERN_DIV, DIST_PAD), not zeros — DIST_PAD=-1 is
        # the genuine "no anchor" sentinel, and TERN_DIV=0 is the closest
        # analogue to today's "no read support". founder_mask does NOT
        # exclude this column (all-ones, matching GRITSCRFDiploid), so it IS
        # a live decode state and the pad value matters numerically —
        # TRAINING_PLAN.md §3.
        pad = torch.empty(B, T, 1, 2, device=X.device, dtype=X.dtype)
        pad[..., 0] = TERN_DIV
        pad[..., 1] = DIST_PAD
        X_pad = torch.cat([X, pad], dim=2)
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
            het_pen = F.softplus(het).unsqueeze(-1)              # [B,T,1] >= 0
            emis_p = emis_p - het_pen * self.homo_mask
        elif self.homo_penalty != 0.0:
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
        if self.spike_skip:
            self._loss_seen += 1
            cv = float(crf.detach())
            warming = self._loss_seen <= 50
            thresh = (self.loss_spike_mult * self._loss_ema
                      if self._loss_ema > 0 else float("inf"))
            self._loss_spike = (math.isfinite(cv) and not warming and cv > thresh)
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
        self.log("val_pair_acc", pair_acc)              # slash-free alias, see train_diploid.py
        self.log("val/hap_acc", hap_acc, prog_bar=True)
        self.log("val/gate", g.mean())
        return loss

    def on_before_optimizer_step(self, optimizer):
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
        g_ema = thresh if (spike and math.isfinite(thresh)) else (g if math.isfinite(g) else thresh)
        if math.isfinite(g_ema):
            self._gnorm_ema = g_ema if self._gnorm_ema <= 0 else 0.98 * self._gnorm_ema + 0.02 * g_ema
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


def infer_real_founder_pairs(model, data, num_parents, device=None, batch_size=64):
    """THE single real-data inference recipe for GRITSCRFDiploidIndel --
    every eval script (real-data unit tests, depth sweeps, per-checkpoint
    comparisons) should call this instead of re-deriving the ext_emb /
    homo_scale / batch / decode loop itself. Real-world ML teams call this
    "training/serving skew" risk: if feature computation isn't provably the
    same code path everywhere it's used, eval numbers silently drift from
    what a shipped model actually does. This session accumulated a dozen
    near-duplicate scratch scripts before consolidating here -- exactly the
    failure mode a shared function prevents.

    data: [N,T,2*num_parents+2] real ternary+distance array (the H1/H2
    label columns are ignored -- real inference never has embedded truth).
    Uses the checkpoint's one established recipe: genome-wide founder
    affinity as ext_emb, homo_scale_from_affinity (train_diploid.py, the
    p90-of-per-window-inbreeding-estimate classifier) as homo_scale. No
    other variant/override point -- if a different homo_scale is ever
    needed, change it here, once, not in each caller.

    Returns (pred_lo, pred_hi): [N,T] int arrays, the low/high founder
    index of the predicted pair at every real site."""
    K = num_parents
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    N, T, W = data.shape
    if W != 2 * K + 2:
        raise ValueError(f"data width {W} != 2*num_parents+2={2 * K + 2}")

    tern = data[:, :, :K].astype(np.float32)
    dist = data[:, :, K + 2:2 * K + 2].astype(np.float32)
    feats = torch.tensor(np.stack([tern, dist], axis=-1), dtype=torch.float32)
    M = (tern == TERN_MATCH).astype(np.float32)
    affinity = _founder_affinity(M.reshape(-1, K))
    homo_scale = homo_scale_from_affinity(M)
    ext_emb = torch.tensor(affinity, dtype=torch.float32).unsqueeze(0).expand(N, -1, -1)

    preds_lo, preds_hi = [], []
    with torch.no_grad():
        for s in range(0, N, batch_size):
            xb = feats[s:s + batch_size].to(device)
            eb = ext_emb[s:s + batch_size].to(device)
            hs = torch.full((xb.shape[0],), homo_scale, device=device)
            emis_p, _g, c = model(xb, ext_emb=eb, homo_scale=hs)
            pred = _dcrf_viterbi(emis_p, c, model.nsw_pair, model.stay_bonus)
            preds_lo.append(model.pi[pred].cpu().numpy())
            preds_hi.append(model.pj[pred].cpu().numpy())
    return np.concatenate(preds_lo), np.concatenate(preds_hi)


def _window_error_slots(true_lo_w0, true_hi_w0, maj_lo, maj_hi):
    """Multiset difference between one window's true founder pair and its
    majority predicted pair -- e.g. true=(B73,B73) vs pred=(B73,Oh43)
    yields [(B73,Oh43)], the one mismatched slot. A plain set difference
    would collapse this to nothing when the true pair is homozygous
    (duplicate elements), so this uses collections.Counter instead."""
    from collections import Counter
    true_ms = Counter([int(true_lo_w0), int(true_hi_w0)])
    pred_ms = Counter([int(maj_lo), int(maj_hi)])
    wrong_true = list((true_ms - pred_ms).elements())
    wrong_pred = list((pred_ms - true_ms).elements())
    return list(zip(wrong_true, wrong_pred))


def score_ibd_adjusted_accuracy(pred_lo, pred_hi, true_lo, true_hi, valid,
                                 raw_counts, row_idx, num_parents, ibd_thresh=0.8):
    """The scoring half of ibd_adjusted_accuracy, taking predictions
    directly instead of a model+data pair -- lets this metric be applied to
    ANY model's predictions on the same real sample (e.g. comparing the
    older non-indel GRITSCRFDiploid against GRITSCRFDiploidIndel apples-to-
    apples: same truth, same raw-support check, same ibd_thresh, only the
    predictions differ). See ibd_adjusted_accuracy's docstring for the full
    mechanism description; this is the reusable core of it.

    IMPORTANT CALIBRATION CAVEAT (found via a genome-wide genotype-level
    rescore, not just founder-decode -- do not re-litigate without new
    evidence): the ibd_thresh=0.8 raw-read-count-ratio test verifies tied
    READ-MAPPING support between the true and predicted founder, NOT tied
    SNP GENOTYPE. A window can show tied raw counts (repetitive sequence,
    paralogy, anchor/liftover ambiguity -- the same mechanism that produces
    "PLACED" vs "EXACT" reads) without the two founders actually agreeing
    at most individual SNP positions inside that window. Confirmed on real
    IDX-HYB Oh43xIl14H: this metric reports ~flat ~99.6-99.9% accuracy at
    every depth (0.01x-2.0x), but the genome-wide SNP+RefCall-restricted
    genotype comparator (compare_gvcf_truth_diploid.py --snp-refcall-
    metrics, which independently excludes all indel-classified sites) shows
    real, depth-INCREASING SNP-level error (0.64%->2.65% SNP-class error,
    0.1x->2.0x) that tracks the model's RAW (non-adjusted) founder error far
    more closely than this "IBD-adjusted" number. Root cause: as real depth
    rises, the CRF's Viterbi decode becomes more CONFIDENT in the whole-
    window majority call (via stay_bonus reinforcing within-window
    consistency) -- more confidence, not more correctness, when the
    underlying read-mapping signal was ambiguous to begin with. So: treat
    this function's "credited" windows as "real, non-random read-mapping
    confound identified" (a legitimate, still-useful signal -- these are
    NOT arbitrary/random model errors), not as "verified correct at the
    genotype level." Don't assume ibd_adjusted_acc approximates true
    SNP-level accuracy, especially at higher real depth.

    pred_lo/pred_hi/true_lo/true_hi/valid: [N,T]. raw_counts: raw.npy's
    count block (or anything indexable as raw_counts[rows, :num_parents]).
    row_idx: list of N real-row-index arrays, one per window.

    Returns (pair_acc, ibd_adjusted_acc, n_windows_credited, n_windows_checked).
    """
    pair_acc = float(((pred_lo == true_lo) & (pred_hi == true_hi))[valid].mean())

    N, T = pred_lo.shape
    credited = np.zeros((N, T), dtype=bool)
    n_checked, n_credited_windows = 0, 0
    for w in range(N):
        err_w = ((pred_lo[w] != true_lo[w]) | (pred_hi[w] != true_hi[w])) & valid[w]
        if not err_w.any():
            continue
        if (true_lo[w] != true_lo[w, 0]).any() or (true_hi[w] != true_hi[w, 0]).any():
            continue  # internal truth switch -- leave unadjusted (conservative)
        pairs, counts = np.unique(np.stack([pred_lo[w], pred_hi[w]], axis=1), axis=0, return_counts=True)
        maj_lo, maj_hi = pairs[np.argmax(counts)]
        slots = _window_error_slots(true_lo[w, 0], true_hi[w, 0], maj_lo, maj_hi)
        if not slots:
            continue  # majority call is actually right even though some sites in the window differ
        mean_support = np.asarray(raw_counts[row_idx[w], :num_parents]).astype(np.float64).mean(axis=0)
        n_checked += 1
        ok = all(mean_support[tf] > 1e-9 and mean_support[pf] / mean_support[tf] >= ibd_thresh
                 for tf, pf in slots)
        if ok:
            credited[w] = err_w
            n_credited_windows += 1

    adjusted_correct = ((pred_lo == true_lo) & (pred_hi == true_hi)) | credited
    ibd_adj_acc = float(adjusted_correct[valid].mean())
    return pair_acc, ibd_adj_acc, n_credited_windows, n_checked


def ibd_adjusted_accuracy(model, data, num_parents, true_lo, true_hi, valid,
                           raw_counts, row_idx, device=None, ibd_thresh=0.8):
    """Second real-data accuracy metric alongside plain pair_acc: credits
    window-level errors that real raw read support cannot actually
    distinguish from the true call (a real, non-random read-MAPPING
    confound -- verified this session: 95.3% of real RIL2 errors and 87.7%
    of real HYB errors show the wrong founder's real support statistically
    tied with or exceeding the true founder's, concentrated in B73-
    involving pairs with confirmed local IBD tracts on chr5/6/7/8).

    CAVEAT (read score_ibd_adjusted_accuracy's docstring in full before
    treating this as "true accuracy"): "tied raw read support" is NOT the
    same claim as "tied SNP genotype". A genome-wide genotype-level rescore
    found this metric reports flat ~99.6-99.9% for real HYB at every depth
    while the actual SNP+RefCall-restricted genotype error rises real and
    sharply with depth (0.64%->2.65% SNP-class error, 0.1x->2.0x) -- this
    metric is a legitimate "is the error explainable by a real mapping
    confound, not random noise" signal, not a genotype-level correctness
    guarantee, and its accuracy degrades specifically as real depth rises
    (the CRF gets more CONFIDENT in an already-ambiguous majority call, not
    more correct).

    For each 512-site window, compares the model's majority predicted pair
    to the true pair (using infer_real_founder_pairs -- the one canonical
    inference path). If they disagree AND the window has a single
    well-defined true pair (no internal truth switch), checks the
    mismatched founder(s)' real raw read support (mean over the window's
    real rows, from `raw_counts` = raw.npy's [:, :num_parents] count block)
    against the true founder(s)'. If the wrong founder's support is
    >=ibd_thresh (default 0.8) of the true founder's, the whole window is
    credited as correct for this metric. Internal-truth-switch windows, or
    windows where support can't be assessed, are left unadjusted (scored
    exactly as pair_acc).

    Thin wrapper around score_ibd_adjusted_accuracy (the reusable scoring
    core -- use that directly to score a DIFFERENT model's predictions on
    the same sample/truth/raw-support for an apples-to-apples comparison).

    data: [N,T,2*num_parents+2] real array. true_lo/true_hi/valid: [N,T]
    per-site (pass constant arrays for INBRED/HYB, real oracle per-site
    arrays for kinds like RIL2 whose true pair varies along the genome).
    row_idx: list of N real-row-index arrays, one per window, matching
    data's window boundaries (see ropebwt_npy_to_matrix.py's --window-size
    chunking -- same convention test_real_data_affinity.py's
    _ril2_truth_windows uses).

    Returns (pair_acc, ibd_adjusted_acc, n_windows_credited, n_windows_checked).
    """
    pred_lo, pred_hi = infer_real_founder_pairs(model, data, num_parents, device=device)
    return score_ibd_adjusted_accuracy(pred_lo, pred_hi, true_lo, true_hi, valid,
                                        raw_counts, row_idx, num_parents, ibd_thresh=ibd_thresh)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--workdir", default="/workdir/esb33")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--limit-n", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--accumulate-grad-batches", type=int, default=1,
                   help="Accumulate gradients over N micro-batches per optimizer "
                        "step (Lightning's own mechanism), for an effective batch "
                        "of --batch-size * N without the memory cost of a larger "
                        "real batch. --spike-skip's gradient-norm check "
                        "(on_before_optimizer_step) already fires once per "
                        "effective step under accumulation, no changes needed "
                        "there; its loss-based check (per micro-batch, in "
                        "training_step) reflects only the LAST micro-batch of "
                        "each group rather than every one -- a minor, accepted "
                        "reduction in that one signal's granularity, not a "
                        "correctness issue.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gate-reg", type=float, default=0.05)
    p.add_argument("--time-local-emis", action="store_true")
    p.add_argument("--fast-cells", action="store_true",
                   help="Exact 516-row cell-embedding lookup instead of the "
                        "per-cell MLP (inference-time speedup; off by default "
                        "during training, mirrors FounderPathEncoder.binary_cells).")
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
                   help="Scale --homo-penalty per individual by a genome-wide het "
                        "proxy over the ternary MATCH view (0 for inbred lines, 1 "
                        "for outbred). Needs --windows-per-individual.")
    p.add_argument("--het-inbred", type=float, default=0.23,
                   help="Het-proxy calibration floor (inbred/F=1 endpoint). "
                        "TRAINING_PLAN.md §3: needs fresh empirical values once real "
                        "indel-mode simulator output exists — the 0.23 default is "
                        "copied from the binary model's calibration, a placeholder, "
                        "not a measurement on this format.")
    p.add_argument("--het-outbred", type=float, default=0.50,
                   help="Het-proxy calibration ceiling (outbred/F=0 endpoint); "
                        "see --het-inbred.")
    p.add_argument("--learned-het", action="store_true",
                   help="Per-locus encoder-driven homozygous penalty (replaces "
                        "fixed --homo-penalty). The emission-side het signal.")
    p.add_argument("--windows-per-individual", type=int, default=100)
    p.add_argument("--founder-affinity", action="store_true",
                   help="Condition the encoder on a per-individual founder-affinity "
                        "prior (ext_bias) over the ternary MATCH view. Needs "
                        "--windows-per-individual.")
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--val-check-interval", type=int, default=0,
                   help="Validate every N training steps (0 = once per epoch). "
                        "Counts micro-batches, not effective (post-accumulation) "
                        "optimizer steps -- with --accumulate-grad-batches > 1, "
                        "scale this down proportionally to keep the same "
                        "validation cadence relative to real weight updates.")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--run-name", default="diploid-indel-pair")
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
        train_ds, val_ds, _ = make_indel_diploid_affinity_splits(
            args.data, args.num_parents, args.val_frac, args.test_frac,
            args.windows_per_individual, limit_n=args.limit_n)
    elif args.adaptive_homo:
        train_ds, val_ds, _ = make_indel_diploid_individual_splits(
            args.data, args.num_parents, args.val_frac, args.test_frac,
            args.windows_per_individual, het_inbred=args.het_inbred,
            het_outbred=args.het_outbred, limit_n=args.limit_n)
    else:
        train_ds, val_ds, _ = make_indel_diploid_splits(
            args.data, args.num_parents, args.val_frac, args.test_frac,
            limit_n=args.limit_n)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = GRITSCRFDiploidIndel(
        num_parents=args.num_parents, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, lr=args.lr, gate_reg=args.gate_reg,
        time_local_emis=args.time_local_emis, warmup_steps=args.warmup_steps,
        homo_penalty=args.homo_penalty, cosine_decay=args.cosine_decay,
        spike_skip=args.spike_skip, spike_mult=args.spike_mult,
        loss_spike_mult=args.loss_spike_mult,
        learned_het=args.learned_het, founder_affinity=args.founder_affinity,
        fast_cells=args.fast_cells)

    # Checkpoint/stop on val/pair_acc (max), matching train_diploid.py: the CRF
    # partition NLL can spike on long-block data even as Viterbi accuracy stays
    # good, so selecting on loss can discard the best model.
    callbacks = [
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
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accumulate_grad_batches)
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    print(f"Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
