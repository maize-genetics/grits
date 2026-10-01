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
from python.crf.crf_kernels import _dcrf_marginal, _dcrf_nll, _dcrf_viterbi, build_pair_tables
from python.crf.train_diploid import _estimate_inbreeding_coef_batch, _founder_affinity, homo_scale_from_affinity
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
    """(N,T,2K+2 or 3K+2): cols 0:K ternary, col K=H1, col K+1=H2, cols
    K+2:2K+2 distance, optional cols 2K+2:3K+2 real per-cell read-support
    count (experiments/depth-confidence-fix/ -- --emit-read-counts on
    either producer). Width is checked per-item from the data itself, not
    a constructor flag, so old (2K+2) and new (3K+2) files are both
    transparently supported -- "count" is simply absent from the returned
    dict for old data, and GRITSCRFDiploidIndel.forward already treats a
    missing count as "no signal" (matches its zero-initialized count_proj).
    Returns the (ternary,distance[,count]) feature window plus the two
    haplotype labels. Unlike PreWindowedDiploidDataset's np.clip(label,0,K)
    (which wrongly maps LABEL_PAD=-1 onto founder 0), unlabeled positions are
    explicitly remapped to the null-founder index K, matching the convention
    ropebwt_npy_to_matrix.py already uses for real data (gA[gA<0]=K)."""
    def __init__(self, data, num_parents=24, lin=None):
        self.data = data
        self.K = num_parents
        self.lin = lin          # [N,T,K] ancestral lineage per founder at each row (--tie-aware-loss)

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
        out = {"input_embeds": feats,
               "h1": torch.tensor(h1, dtype=torch.long),
               "h2": torch.tensor(h2, dtype=torch.long)}
        if row.shape[-1] >= 3 * K + 2:
            count = row[:, 2 * K + 2:3 * K + 2].astype(np.float32)
            out["count"] = torch.tensor(count, dtype=torch.float32)
        if self.lin is not None:
            out["lin"] = torch.tensor(np.asarray(self.lin[idx], dtype=np.int64))
        return out


def _check_width(path, data, num_parents):
    K = num_parents
    if data.shape[-1] not in (2 * K + 2, 3 * K + 2):
        raise ValueError(f"{path}: expected width {2 * K + 2} (2K+2) or "
                         f"{3 * K + 2} (3K+2, with read-support counts) for "
                         f"K={K}, got {data.shape[-1]}")


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
                 het_inbred=0.23, het_outbred=0.50, lin=None):
        super().__init__(data, num_parents, lin)
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


def individual_split_rows(n_ind, G, val_frac, test_frac, seed=0, legacy_tail=False):
    """Row indices (train, val, test) for an individual-aligned split. Whole
    individuals (blocks of G windows) go to one split. Seeded shuffle by
    default: every training .npy is a file-order concatenation of inbred and
    outbred blocks, so the old tail split (legacy_tail=True) gave val/test a
    single individual type -- 100% het for lowrate/regionfix, 100%
    homozygous for heldout_augment -- making val_pair_acc (and checkpoint
    selection) one-class. Indices are sorted within each split so memmap
    reads stay sequential."""
    n_test = int(n_ind * test_frac)
    n_val = int(n_ind * val_frac)
    n_tr = n_ind - n_val - n_test
    order = (np.arange(n_ind) if legacy_tail
             else np.random.default_rng(seed).permutation(n_ind))
    to_rows = lambda ids: (np.sort(ids)[:, None] * G + np.arange(G)).ravel()
    return (to_rows(order[:n_tr]), to_rows(order[n_tr:n_tr + n_val]),
            to_rows(order[n_tr + n_val:]))


def _load_lineage(lin_path, data):
    if lin_path is None:
        return None
    lin = np.load(lin_path, mmap_mode="r")
    if lin.shape[1] != data.shape[1] or len(lin) < len(data):
        raise ValueError(f"{lin_path}: shape {lin.shape} does not match data rows {data.shape[:2]}")
    return lin[:len(data)]


def make_indel_diploid_individual_splits(path, num_parents, val_frac, test_frac, G,
                                         het_inbred=0.23, het_outbred=0.50, limit_n=0,
                                         split_seed=0, legacy_tail_split=False, lin_path=None):
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    _check_width(path, data, num_parents)
    if limit_n:
        data = data[:(limit_n // G) * G]
    N = len(data)
    n_ind = N // G
    splits = individual_split_rows(n_ind, G, val_frac, test_frac, split_seed, legacy_tail_split)
    lin = _load_lineage(lin_path, data)
    mk = lambda rows: IndelDiploidIndividualDataset(data[rows], num_parents, G,
                                                    het_inbred, het_outbred,
                                                    None if lin is None else lin[rows])
    print(f"IndelDiploid(individual) {Path(path).name}: N={N:,} individuals={n_ind} "
          f"train={len(splits[0]):,} val={len(splits[1]):,} test={len(splits[2]):,} "
          f"split={'legacy-tail' if legacy_tail_split else f'shuffled(seed={split_seed})'}")
    return tuple(mk(r) for r in splits)


class IndelDiploidAffinityDataset(IndelDiploidDataset):
    """IndelDiploidDataset + a per-individual founder-affinity ext_emb [K,2],
    from train_diploid._founder_affinity (reused verbatim, bit-identical — it
    has no calibration constants to shift, just mean/centered-mean over the
    MATCH view) attached to every window of the individual."""
    def __init__(self, data, num_parents, windows_per_individual, lin=None,
                 train_homo_scale=False, prov=None, aff_target=None, aff_pred=None, ood=None):
        super().__init__(data, num_parents, lin)
        self.prov = prov                  # [N,T] int8 row provenance (--supervised-heads)
        self.ood = ood                    # [N,T] int8 held-out sidecar (bit 0/1: hap1/hap2 hidden)
        self.aff_target = aff_target      # [N/G,K] per-individual or [N,K] per-window affinity
        self.aff_pred = aff_pred          # [N/G,K] pooled affinity-head prediction (stage2)
        G = windows_per_individual
        if len(data) % G:
            raise ValueError(f"rows {len(data)} not divisible by windows/ind {G}")
        self.G = G
        tern = np.asarray(data[:, :, :num_parents])
        M = (tern == TERN_MATCH).astype(np.float32).reshape(
            len(data) // G, G, data.shape[1], num_parents)
        self.affinity = np.stack(
            [_founder_affinity(M[i]) for i in range(len(M))]).astype(np.float32)
        # --train-homo-scale: the individual's true kind as its homozygous-penalty scale
        # (0 = inbred, every labelled row h1==h2; 1 = hybrid), the same 0/1 the router
        # applies at inference. Without it the full homo_penalty hits inbreds too.
        self.homo_scale = None
        if train_homo_scale:
            h = np.asarray(data[:, :, num_parents:num_parents + 2]).reshape(len(M), -1, 2)
            lab = (h[..., 0] >= 0) & (h[..., 1] >= 0)
            het = ((h[..., 0] != h[..., 1]) & lab).any(1)
            self.homo_scale = het.astype(np.float32)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out["ext_emb"] = torch.tensor(self.affinity[idx // self.G], dtype=torch.float32)
        if self.homo_scale is not None:
            out["homo_scale"] = torch.tensor(self.homo_scale[idx // self.G], dtype=torch.float32)
        if self.prov is not None:
            out["prov"] = torch.tensor(np.asarray(self.prov[idx], dtype=np.int64))
        if self.ood is not None:
            out["ood"] = torch.tensor(np.asarray(self.ood[idx], dtype=np.int64))
        if self.aff_target is not None:
            j = idx if len(self.aff_target) == len(self.data) else idx // self.G
            out["aff_target"] = torch.tensor(self.aff_target[j], dtype=torch.float32)
        if self.aff_pred is not None:
            out["aff_pred"] = torch.tensor(self.aff_pred[idx // self.G], dtype=torch.float32)
        return out


def individual_affinity_target(data, lin, num_parents, G):
    """[N/G,K] float32: per individual, the fraction of its labelled rows where founder f is
    lineage-equivalent (same ancestral lineage at the row) to one of the two true founders."""
    K = num_parents
    n_ind = len(data) // G
    out = np.zeros((n_ind, K), dtype=np.float32)
    for i in range(n_ind):
        lab = np.asarray(data[i * G:(i + 1) * G, :, K:K + 2]).astype(np.int64)    # [G,T,2]
        L = np.asarray(lin[i * G:(i + 1) * G]).astype(np.int64)                  # [G,T,K]
        real = (lab[..., 0] >= 0) & (lab[..., 1] >= 0)
        h1, h2 = np.clip(lab[..., 0], 0, K - 1), np.clip(lab[..., 1], 0, K - 1)
        l1 = np.take_along_axis(L, h1[..., None], -1)
        l2 = np.take_along_axis(L, h2[..., None], -1)
        mem = ((L == l1) | (L == l2)) & real[..., None]
        out[i] = mem.sum((0, 1)) / max(1, real.sum())
    return out


def make_indel_diploid_affinity_splits(path, num_parents, val_frac, test_frac, G, limit_n=0,
                                       split_seed=0, legacy_tail_split=False, lin_path=None,
                                       train_homo_scale=False, prov_path=None, aff_targets=False,
                                       aff_pred_path=None, aff_target_scope="individual", ood_path=None):
    """Individual-aligned split, same boundaries as
    make_indel_diploid_individual_splits (mirrors train_diploid.py's
    make_diploid_affinity_splits intent)."""
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    _check_width(path, data, num_parents)
    if limit_n:
        data = data[:(limit_n // G) * G]
    N = len(data)
    n_ind = N // G
    splits = individual_split_rows(n_ind, G, val_frac, test_frac, split_seed, legacy_tail_split)
    lin = _load_lineage(lin_path, data)
    prov = None if prov_path is None else np.load(prov_path, mmap_mode="r")[:N]
    aff_pred = None if aff_pred_path is None else np.load(aff_pred_path)[:n_ind]
    ood = None if ood_path is None else np.load(ood_path, mmap_mode="r")[:N]

    def mk(rows):
        ids = rows[::G] // G
        lr = None if lin is None else lin[rows]
        dr = data[rows]
        return IndelDiploidAffinityDataset(
            dr, num_parents, G, lr, train_homo_scale=train_homo_scale,
            prov=None if prov is None else prov[rows],
            # window scope: each window's own founder fractions -- all a window can see; their mean
            # over the individual's (equal-length) windows is the individual target exactly
            aff_target=(individual_affinity_target(dr, lr, num_parents,
                                                   1 if aff_target_scope == "window" else G)
                        if aff_targets else None),
            aff_pred=None if aff_pred is None else aff_pred[ids],
            ood=None if ood is None else ood[rows])
    print(f"IndelDiploid(affinity) {Path(path).name}: N={N:,} individuals={n_ind} "
          f"train={len(splits[0]):,} val={len(splits[1]):,} test={len(splits[2]):,} "
          f"split={'legacy-tail' if legacy_tail_split else f'shuffled(seed={split_seed})'}")
    return tuple(mk(r) for r in splits)


# --------------------------------------------------------------------------- #
#  Lightning module                                                            #
# --------------------------------------------------------------------------- #

def _dcrf_nll_tied(emis, c, nsw, stay_bonus, allowed):
    """Pair-state CRF NLL crediting every path whose pair state at each row is in
    `allowed` [B,T,P] (--tie-aware-loss): log Z - log Z_allowed, the second forward
    pass run with disallowed states masked out. With `allowed` one-hot on the label
    this equals _dcrf_nll."""
    emis = emis.float()
    c = c.float()
    nsw = nsw.float()
    stay_bonus = stay_bonus.float()
    stay_mask = (nsw == 0).float()
    neg = torch.tensor(-1e9, dtype=emis.dtype, device=emis.device)
    emis_a = torch.where(allowed, emis, neg)
    a, b = emis[:, 0], emis_a[:, 0]
    for t in range(1, emis.shape[1]):
        tr_t = -c[:, t, None, None] * nsw[None] + stay_bonus * stay_mask[None]
        a = emis[:, t] + torch.logsumexp(a.unsqueeze(2) + tr_t, dim=1)
        b = emis_a[:, t] + torch.logsumexp(b.unsqueeze(2) + tr_t, dim=1)
    return (torch.logsumexp(a, dim=1) - torch.logsumexp(b, dim=1)).mean()


def _new_founder_tables(pi, pj, nsw):
    """[P,P] long x2: the founder(s) newly entered on the transition p -> q (multiset
    q minus p; K = none). nsw=1 -> one new founder, nsw=2 -> two, nsw=0 -> none."""
    from collections import Counter
    P = pi.numel()
    K = int(max(pi.max(), pj.max())) + 1
    n1 = torch.full((P, P), K, dtype=torch.long)
    n2 = torch.full((P, P), K, dtype=torch.long)
    pairs = list(zip(pi.tolist(), pj.tolist()))
    for a, pa in enumerate(pairs):
        cp = Counter(pa)
        for b, qb in enumerate(pairs):
            new = list((Counter(qb) - cp).elements())
            if nsw[a, b] == 0:
                continue
            new = new[:int(nsw[a, b])]
            if len(new) > 0:
                n1[a, b] = new[0]
            if len(new) > 1:
                n2[a, b] = new[1]
    return n1, n2


def _prior_trans(c_t, nsw, stay_bonus, trans_prior, seg_t=None):
    """seg_t [B,P]: a per-destination log-prior paid only on entering a new pair state (a
    founder change), i.e. once per segment (--het-prior segment)."""
    stay_mask = (nsw == 0).float()
    tr = -c_t[:, None, None] * nsw[None] + stay_bonus * stay_mask[None] + trans_prior
    if seg_t is not None:
        tr = tr + (1.0 - stay_mask)[None] * seg_t[:, None, :]
    return tr


def crf_nll_prior(emis, c, nsw, stay_bonus, allowed, trans_prior, init_prior, seg=None):
    """Tie-aware pair CRF NLL (as _dcrf_nll_tied) with a per-sample transition prior
    trans_prior [B,P,P] (added to every p->q step) and initial prior init_prior [B,P];
    seg [B,T,P] optional per-segment log-prior (paid at row 0 and on each state change)."""
    emis = emis.float(); c = c.float(); nsw = nsw.float()
    sb = stay_bonus.float(); tp = trans_prior.float()
    neg = torch.tensor(-1e9, dtype=emis.dtype, device=emis.device)
    emis_a = torch.where(allowed, emis, neg)
    ip = init_prior.float() + (seg[:, 0].float() if seg is not None else 0.0)
    a = emis[:, 0] + ip
    b = emis_a[:, 0] + ip
    for t in range(1, emis.shape[1]):
        tr = _prior_trans(c[:, t], nsw, sb, tp, None if seg is None else seg[:, t].float())
        a = emis[:, t] + torch.logsumexp(a.unsqueeze(2) + tr, dim=1)
        b = emis_a[:, t] + torch.logsumexp(b.unsqueeze(2) + tr, dim=1)
    return (torch.logsumexp(a, dim=1) - torch.logsumexp(b, dim=1)).mean()


@torch.no_grad()
def crf_viterbi_prior(emis, c, nsw, stay_bonus, trans_prior, init_prior, seg=None):
    emis = emis.float(); c = c.float(); nsw = nsw.float()
    sb = stay_bonus.float(); tp = trans_prior.float()
    B, T, P = emis.shape
    delta = emis[:, 0] + init_prior.float() + (seg[:, 0].float() if seg is not None else 0.0)
    bp = torch.zeros(T - 1, B, P, dtype=torch.long, device=emis.device)
    for t in range(1, T):
        best, idx = (delta.unsqueeze(2) + _prior_trans(
            c[:, t], nsw, sb, tp, None if seg is None else seg[:, t].float())).max(dim=1)
        delta = emis[:, t] + best
        bp[t - 1] = idx
    path = torch.zeros(B, T, dtype=torch.long, device=emis.device)
    path[:, T - 1] = delta.argmax(dim=1)
    for t in range(T - 2, -1, -1):
        path[:, t] = bp[t].gather(1, path[:, t + 1].unsqueeze(1)).squeeze(1)
    return path


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
                 learned_het=False, founder_affinity=False, fast_cells=False,
                 tie_aware_loss=False, pair_emission="sum", emission="learned",
                 supervised_heads="off", xo_placement=False, het_prior="row", no_distance=False,
                 aff_source="head", gate_target="prov", extra_heads=False):
        super().__init__()
        self.save_hyperparameters()
        self.tie_aware_loss = tie_aware_loss
        self.no_distance = no_distance
        # CRF founder prior: "head" = pooled supervised affinity head; "reads" = the sample's
        # own genome-wide founder match rate (_founder_affinity, diploid-affinity's ext_emb),
        # with no affinity head trained in stage 1
        if aff_source not in ("head", "reads"):
            raise ValueError(f"aff_source must be head/reads, got {aff_source!r}")
        self.aff_source = aff_source
        # gate head target: "prov" = simulator provenance (collinear or insertion read, or a
        # replacement read at its own site, not on a bad site); "support" = the row matches the
        # true founder of the haplotype it was read from (an off-site replacement read that still
        # carries that founder's signal counts as trustworthy; a corrupted read that happens to
        # match it too is harmless)
        if gate_target not in ("prov", "support"):
            raise ValueError(f"gate_target must be prov/support, got {gate_target!r}")
        self.gate_target = gate_target
        if pair_emission not in ("sum", "mixture"):
            raise ValueError(f"pair_emission must be 'sum' or 'mixture', got {pair_emission!r}")
        self.pair_emission = pair_emission
        if emission not in ("learned", "likelihood", "likelihood_dist"):
            raise ValueError(f"emission must be 'learned', 'likelihood' or 'likelihood_dist', got {emission!r}")
        self.emission = emission
        # --emission likelihood: the CRF emission is a read likelihood computed from the input,
        # not an encoder founder score. One learned log-likelihood per ternary state, shared by
        # every founder (deleted / present-no-match / match); init ~ log(1e-3), log(1e-2), 0.
        # The encoder then only supplies the per-row gate g (site weight) and switch cost c.
        self.tern_loglik = nn.Parameter(torch.tensor([-6.9, -4.6, 0.0]))
        # --emission likelihood_dist: one log-likelihood per (ternary state, anchor-distance band),
        # still shared by every founder. Bands over the log distance code round(8*log2(1+bp)):
        # 0 | 1-40 (<30bp) | 41-56 | 57-72 | 73-80 | 81-88 (1-2kb) | 89-104 (2-8kb) | >=105 | -1 none.
        # Initialised to the per-state values so training starts where 'likelihood' does.
        self.register_buffer("dist_band_edges", torch.tensor([0.5, 40.5, 56.5, 72.5, 80.5, 88.5, 104.5]))
        self.tern_dist_loglik = nn.Parameter(
            torch.tensor([-6.9, -4.6, 0.0]).unsqueeze(1).repeat(1, 9))
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
        # --supervised-heads (experiments/supervised-heads/): the encoder is trained DIRECTLY
        # on simulator-truth targets (stage1) -- gate g = P(row clean), switch p = sigmoid(-c),
        # het h = P(true founders are distinct lineages), per-founder affinity a = fraction of
        # the individual's rows where the founder is a true (lineage-equivalent) founder -- and
        # then (stage2) only the CRF scalars below are fitted with the encoder frozen.
        # Created last so that flags-off models keep their exact init RNG stream.
        if supervised_heads not in ("off", "stage1", "stage2"):
            raise ValueError(f"supervised_heads must be off/stage1/stage2, got {supervised_heads!r}")
        self.supervised_heads = supervised_heads
        if supervised_heads != "off":
            if emission not in ("likelihood", "likelihood_dist"):
                raise ValueError("--supervised-heads needs --emission likelihood or likelihood_dist")
            self.sup_het_head = nn.Linear(d_model, 1)
            self.sup_aff_head = nn.Linear(d_model, 1)
            # window-level crossover count: lambda_w = softplus(xo_head(mean_t H)); the CRF's
            # per-row switch probability spreads lambda_w over the window's T-1 transitions
            # (uniformly, or with --xo-placement in proportion to the per-row switch head)
            self.sup_xo_head = nn.Linear(d_model, 1)
            self.xo_placement = xo_placement
            # --extra-heads: per-row deletion dosage (0/1/2 haplotypes lacking the B73 sequence)
            # and per-row out-of-panel probability (the row's true founder is not in the panel).
            # CRF use, weights 0 in (A) and fitted in stage 2 (B): dosage adds
            # dos_w * g * log P(dosage = #founders of the pair reading -1) per row; the window's
            # mean out-of-panel score scales the switch probability by exp(ood_w * score)
            self.extra_heads = extra_heads
            if extra_heads:
                self.sup_dos_head = nn.Linear(d_model, 3)
                self.sup_ood_head = nn.Linear(d_model, 1)
                self.dos_w = nn.Parameter(torch.tensor(0.0))
                self.ood_w = nn.Parameter(torch.tensor(0.0))
                # founder prior only for the in-panel share of the window: aff_w * max(0, 1 - s*ood);
                # a sample-wide affinity prior pins hybrids to their two parents but penalises the
                # locally right, genome-wide rare founders of an out-of-panel mosaic (sh5 fit B with
                # the prior off: hybrids +0.5-0.7pp, OUT inbred/RIL -0.6-0.7pp). s = 1 in (A), fitted in B
                self.aff_ood_s = nn.Parameter(torch.tensor(1.0))
            # het-head prior: "row" adds w_het*log(h | 1-h) to every row's emission (summed over
            # the window it can swamp the reads when the head is off on real data); "segment"
            # pays it once per segment (row 0 and each state change); "off" drops it
            if het_prior not in ("row", "segment", "off"):
                raise ValueError(f"het_prior must be row/segment/off, got {het_prior!r}")
            self.het_prior = het_prior
            self._seg_prior = None
            # CRF scalars in the heads model, at their simulator-derived values ("A"):
            # emission = log(g*mean(exp(l_i), exp(l_j)) + 1-g) with l a log-LR table;
            # het prior w_het*log(h | 1-h); transitions stay 0 / switch -c per founder change
            # with c = c_scale*c_head + c_offset (c_head = -logit p), plus w_aff*log(a_x/sum a)
            # for each newly entered founder x and at the first row.
            self.het_w = nn.Parameter(torch.tensor(1.0))
            self.aff_w = nn.Parameter(torch.tensor(1.0))
            self.c_scale = nn.Parameter(torch.tensor(1.0))      # unused since the xo head
            self.c_offset = nn.Parameter(torch.tensor(math.log(2.0)))
            # window crossover calibration, fitted in log space: a raw scale can cross 0, where
            # p = xo_scale*lambda/(T-1) hits its clamp and the gradient dies (sh3 fit B: -0.19)
            self.log_xo_scale = nn.Parameter(torch.tensor(0.0))
            with torch.no_grad():
                self.stay_bonus.fill_(0.0)
            nf1, nf2 = _new_founder_tables(pi, pj, nsw)
            self.register_buffer("newf1", nf1)
            self.register_buffer("newf2", nf2)
            if supervised_heads == "stage2":
                # stay_bonus stays at 0: a bonus on every stay is (up to multi-founder
                # changes) the same as a higher switch cost, so fitting both is degenerate
                keep = {"tern_loglik", "tern_dist_loglik", "het_w", "aff_w", "log_xo_scale",
                        "dos_w", "ood_w", "aff_ood_s"}
                for n_, q in self.named_parameters():
                    q.requires_grad_(n_ in keep)
        self._heads = None
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GRITSCRFDiploidIndel: K={K} states, P={self.P} pair-states, "
              f"{n_params:,} params")

    def forward(self, X, homo_scale=None, ext_emb=None, count=None, aff_pred=None, het_pred=None):
        if getattr(self, "no_distance", False):      # ablation: the ternary array alone
            X = torch.stack([X[..., 0], torch.zeros_like(X[..., 1])], dim=-1)
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
        if count is not None:                                    # 0 = no read support, neutral
            count = torch.cat(
                [count, torch.zeros(B, T, 1, device=X.device, dtype=count.dtype)], dim=2)
        sup = self.supervised_heads != "off"
        if self.learned_het:
            emis_f, g, c, het = self.encoder(X_pad, founder_mask, ext_emb=ext_emb,
                                             emit_het=True, count=count)
        elif sup:
            emis_f, g, c, H, cells = self.encoder(X_pad, founder_mask, ext_emb=ext_emb,
                                                  count=count, return_hidden=True)
            Kf = self.num_parents
            gate_logit = self.encoder.gate_head(H).squeeze(-1)                  # [B,T]
            het_logit = self.sup_het_head(H).squeeze(-1)                       # [B,T]
            pooled = (cells[:, :, :Kf, :] * H.unsqueeze(2)).mean(1)            # [B,Kf,d]
            aff_logit = self.sup_aff_head(pooled).squeeze(-1)                  # [B,Kf]
            xo_lam = F.softplus(self.sup_xo_head(H.float().mean(1)).squeeze(-1))   # [B]
            self._heads = dict(gate_logit=gate_logit, het_logit=het_logit,
                               aff_logit=aff_logit, c=c, xo_lam=xo_lam)
            if getattr(self, "extra_heads", False):
                self._heads["dos_logit"] = self.sup_dos_head(H).float()           # [B,T,3]
                self._heads["ood_logit"] = self.sup_ood_head(H).squeeze(-1)       # [B,T]
        else:
            emis_f, g, c = self.encoder(X_pad, founder_mask, ext_emb=ext_emb,
                                        count=count)  # [B,T,K]
        if self.emission == "likelihood":
            tern_idx = (X_pad[..., 0].round().clamp(-1, 1) + 1).long()          # -1/0/1 -> 0/1/2
            emis_f = g.unsqueeze(-1) * self.tern_loglik[tern_idx]            # [B,T,K]
        elif self.emission == "likelihood_dist":
            tern_idx = (X_pad[..., 0].round().clamp(-1, 1) + 1).long()
            d = X_pad[..., 1]
            band = torch.bucketize(d, self.dist_band_edges)                      # 0..7
            band = torch.where(d < 0, torch.full_like(band, 8), band)            # no anchor -> 8
            emis_f = g.unsqueeze(-1) * self.tern_dist_loglik[tern_idx, band]
        if sup:
            return self._heads_crf(X_pad, g, c, het_logit, aff_logit, aff_pred, ext_emb, het_pred)
        if self.pair_emission == "mixture":
            # each row comes from one haplotype or the other: a het pair is credited when
            # EITHER founder explains the row, so two founders explaining different reads
            # beat two relatives explaining the same ones; homozygous (i,i) scores emis_f[i]
            emis_p = (torch.logaddexp(emis_f[..., self.pi], emis_f[..., self.pj])
                      - math.log(2.0))                               # [B,T,P]
        else:
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

    @property
    def xo_scale(self):
        return self.log_xo_scale.exp()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # checkpoints from before the log-space parameterization store a raw xo_scale
        k = prefix + "xo_scale"
        if k in state_dict:
            state_dict[prefix + "log_xo_scale"] = torch.log(state_dict.pop(k).float().clamp_min(1e-6))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _heads_crf(self, X_pad, g, c_head, het_logit, aff_logit, aff_pred, ext_emb=None, het_pred=None):
        """Supervised-heads CRF inputs. Returns (emis_p [B,T,P], g, c [B,T]) and stores the
        transition prior self._trans_prior [B,P,P] and initial prior self._init_prior [B,P]
        (consumed by crf_nll_prior / crf_viterbi_prior). homo_scale is not used."""
        B, T = g.shape
        Kf = self.num_parents
        tern_idx = (X_pad[..., 0].round().clamp(-1, 1) + 1).long()
        if self.emission == "likelihood_dist":
            d = X_pad[..., 1]
            band = torch.bucketize(d, self.dist_band_edges)
            band = torch.where(d < 0, torch.full_like(band, 8), band)
            ll = self.tern_dist_loglik[tern_idx, band].float()                  # [B,T,K] log-LR
        else:
            ll = self.tern_loglik[tern_idx].float()
        # null founder: log-LR 0 (uninformative) -- it is never a row's source in the simulator
        ll = torch.cat([ll[..., :Kf], torch.zeros_like(ll[..., :1])], -1)
        mix = torch.logaddexp(ll[..., self.pi], ll[..., self.pj]) - math.log(2.0)   # [B,T,P]
        gf = g.float().clamp(1e-6, 1 - 1e-6).unsqueeze(-1)
        emis_p = torch.logaddexp(torch.log(gf) + mix, torch.log1p(-gf))
        h = torch.sigmoid(het_logit.float()).clamp(1e-4, 1 - 1e-4).unsqueeze(-1)
        if het_pred is not None:            # a pooled (window / chromosome / sample) het rate per window
            h = het_pred.float().clamp(1e-4, 1 - 1e-4).view(-1, 1, 1).expand_as(h)
        lq = self.het_w * (self.homo_mask * torch.log1p(-h) + (1 - self.homo_mask) * torch.log(h))
        self._seg_prior = None
        hp = getattr(self, "het_prior", "row")
        if hp == "row":
            emis_p = emis_p + lq
        elif hp == "segment":
            self._seg_prior = lq                                                 # [B,T,P]
        if getattr(self, "aff_source", "head") == "reads":
            if ext_emb is None:
                raise ValueError("aff_source='reads' needs ext_emb (the genome-wide match rate)")
            a = ext_emb[:, :Kf, 0].float()                                     # raw match rate
        else:
            a = aff_pred if aff_pred is not None else torch.sigmoid(aff_logit.float())
        a = torch.cat([a.float().clamp(1e-3, 1.0), torch.full_like(a[:, :1], 1e-3)], 1)
        la = torch.log(a / a[:, :Kf].sum(1, keepdim=True))                     # [B,K]
        la0 = torch.cat([la, torch.zeros_like(la[:, :1])], 1)                   # index K = none
        aw = self.aff_w * torch.ones(la.shape[0], device=la.device)             # [B]
        if getattr(self, "extra_heads", False):
            ood_score = torch.sigmoid(self._heads["ood_logit"].float()).mean(1)  # [B] window score
            aw = aw * (1.0 - self.aff_ood_s * ood_score).clamp(0.0, 1.0)
        self._trans_prior = aw.view(-1, 1, 1) * (la0[:, self.newf1] + la0[:, self.newf2])   # [B,P,P]
        self._init_prior = aw.view(-1, 1) * (la[:, self.pi] + la[:, self.pj])  # [B,P]
        # switch probability per transition t-1 -> t (t >= 1) from the window crossover count
        lam = self.xo_scale * self._heads["xo_lam"].float()                  # [B]
        if getattr(self, "extra_heads", False):
            ood_w = torch.sigmoid(self._heads["ood_logit"].float()).mean(1)     # [B] window score
            lam = lam * torch.exp(self.ood_w * ood_w)
            deleted = (X_pad[..., 0].round() == -1).long()                      # [B,T,K]
            dos_obs = deleted[..., self.pi] + deleted[..., self.pj]             # [B,T,P] 0..2
            logp = F.log_softmax(self._heads["dos_logit"].float(), -1)          # [B,T,3]
            emis_p = emis_p + self.dos_w * g.float().unsqueeze(-1) * logp.gather(2, dos_obs)
        if self.xo_placement:
            w = torch.sigmoid(-c_head.float()[:, 1:])
            w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
        else:
            w = torch.full_like(c_head.float()[:, 1:], 1.0 / (T - 1))
        p = (lam.unsqueeze(1) * w).clamp(1e-12, 1 - 1e-4)
        p = torch.cat([p[:, :1], p], 1)                                      # c[:,0] unused
        # stay log(1-p) / change log(p/2) + log(a_x/sum a); minus log(1-p) on every
        # transition at t (path-invariant): stay 0, change -(-logit p + log 2)
        c = -torch.log(p) + torch.log1p(-p) + self.c_offset
        return emis_p, g, c

    def crf_decode(self, emis_p, c, stay_bonus=None, marginal=False):
        """Viterbi for this model: the prior-aware kernel in heads mode, else _dcrf_viterbi."""
        sb = self.stay_bonus if stay_bonus is None else stay_bonus
        if self.supervised_heads != "off":
            return crf_viterbi_prior(emis_p, c, self.nsw_pair, sb, self._trans_prior,
                                     self._init_prior, self._seg_prior)
        dec = _dcrf_marginal if marginal else _dcrf_viterbi
        return dec(emis_p, c, self.nsw_pair, sb)

    def _pair_labels(self, h1, h2):
        return self.pair_table[h1, h2]                           # [B,T]

    def _allowed_pairs(self, h1, h2, lin):
        """[B,T,P] bool: pair states made of founders sharing the true founders'
        ancestral lineage at each row (IBD-equivalent, indistinguishable in the
        input). The null label (K-1) is equivalent only to itself."""
        Kf = self.num_parents
        def eq(h):
            lh = lin.gather(2, h.clamp(max=Kf - 1).unsqueeze(-1))       # [B,T,1]
            real = (h < Kf).unsqueeze(-1)
            return torch.cat([(lin == lh) & real, ~real], dim=2)       # [B,T,K]
        e1, e2 = eq(h1), eq(h2)
        return (e1[..., self.pi] & e2[..., self.pj]) | (e1[..., self.pj] & e2[..., self.pi])

    def head_targets(self, h1, h2, lin, prov=None, tern=None, ood=None):
        """Simulator-truth targets for --supervised-heads (masks: True = row counts).
          gate   [B,T]   row clean per simulate_alleles.prov_is_clean (needs prov)
          switch [B,T-1] lineage-aware switch between rows t and t+1 (no pair state
                         allowed at both rows)
          het    [B,T]   the two true founders are distinct ancestral lineages at t
        Rows with a null (padded) label are masked out of het and switch."""
        Kf = self.num_parents
        def eq(h):
            lh = lin.gather(2, h.clamp(max=Kf - 1).unsqueeze(-1))
            real = (h < Kf).unsqueeze(-1)
            return torch.cat([(lin == lh) & real, ~real], dim=2)
        e1, e2 = eq(h1), eq(h2)
        allowed = (e1[..., self.pi] & e2[..., self.pj]) | (e1[..., self.pj] & e2[..., self.pi])
        switch = ~(allowed[:, 1:] & allowed[:, :-1]).any(-1)
        real = (h1 < Kf) & (h2 < Kf)
        l1 = lin.gather(2, h1.clamp(max=Kf - 1).unsqueeze(-1)).squeeze(-1)
        l2 = lin.gather(2, h2.clamp(max=Kf - 1).unsqueeze(-1)).squeeze(-1)
        het = l1 != l2
        out = dict(switch=switch, switch_mask=real[:, 1:] & real[:, :-1], het=het, het_mask=real)
        if prov is not None:
            from python.crf.simulate_alleles import PROV_OFF_SITE, PROV_BAD_SITE
            out["gate"] = (prov >= 0) & ((prov & (PROV_OFF_SITE | PROV_BAD_SITE)) == 0)
            out["gate_mask"] = prov >= 0
            if getattr(self, "extra_heads", False):
                from python.crf.simulate_alleles import PROV_DEL_H1, PROV_DEL_H2
                out["dos"] = ((prov & PROV_DEL_H1) > 0).long() + ((prov & PROV_DEL_H2) > 0).long()
                out["dos_mask"] = (prov >= 0) & real
                out["ood"] = (ood > 0) if ood is not None else torch.zeros_like(prov, dtype=torch.bool)
                out["ood_mask"] = prov >= 0
            if getattr(self, "gate_target", "prov") == "support":
                if tern is None:
                    raise ValueError("gate_target='support' needs the ternary rows")
                from python.crf.simulate_alleles import PROV_KIND_MASK
                hap2 = (prov & PROV_KIND_MASK) % 2 == 1
                hs = torch.where(hap2, h2, h1)
                real_src = hs < Kf
                sup = tern.gather(2, hs.clamp(max=Kf - 1).unsqueeze(-1)).squeeze(-1).round() == 1
                out["gate"] = (prov >= 0) & real_src & sup
                out["gate_mask"] = (prov >= 0) & real_src
        return out

    def head_loss(self, heads, tg, aff_target=None):
        """Mean masked BCE per head (fp32); returns total and the parts."""
        def bce(logit, y, m):
            l = F.binary_cross_entropy_with_logits(logit.float(), y.float(), reduction="none")
            m = m.float()
            return (l * m).sum() / m.sum().clamp_min(1.0)
        n_sw = (tg["switch"] & tg["switch_mask"]).float().sum(1)
        lam = heads["xo_lam"].float().clamp_min(1e-8)
        parts = {"switch": bce(-heads["c"][:, 1:], tg["switch"], tg["switch_mask"]),
                 "xo": (lam - n_sw * torch.log(lam)).mean(),                 # Poisson NLL
                 "het": bce(heads["het_logit"], tg["het"], tg["het_mask"])}
        if "gate" in tg:
            parts["gate"] = bce(heads["gate_logit"], tg["gate"], tg["gate_mask"])
        if aff_target is not None:
            parts["aff"] = F.binary_cross_entropy_with_logits(heads["aff_logit"].float(),
                                                              aff_target.float())
        if "dos" in tg and "dos_logit" in heads:
            ce = F.cross_entropy(heads["dos_logit"].float().flatten(0, 1), tg["dos"].flatten(),
                                 reduction="none").view_as(tg["dos"])
            m = tg["dos_mask"].float()
            parts["dos"] = (ce * m).sum() / m.sum().clamp_min(1.0)
            parts["ood"] = bce(heads["ood_logit"], tg["ood"], tg["ood_mask"])
        return sum(parts.values()), parts

    def _step(self, batch):
        X, h1, h2 = batch["input_embeds"], batch["h1"], batch["h2"]
        emis_p, g, c = self(X, batch.get("homo_scale"), batch.get("ext_emb"),
                            batch.get("count"), batch.get("aff_pred"))
        tags = self._pair_labels(h1, h2)
        self._head_parts = None
        if self.supervised_heads == "stage1":
            tg = self.head_targets(h1, h2, batch["lin"], batch.get("prov"), tern=X[..., 0],
                                   ood=batch.get("ood"))
            loss, self._head_parts = self.head_loss(self._heads, tg, batch.get("aff_target"))
            return loss, loss, g, c, emis_p, tags
        if self.supervised_heads != "off":
            crf = crf_nll_prior(emis_p, c, self.nsw_pair, self.stay_bonus,
                                self._allowed_pairs(h1, h2, batch["lin"]),
                                self._trans_prior, self._init_prior, self._seg_prior)
            return crf, crf, g, c, emis_p, tags
        if self.tie_aware_loss and "lin" in batch:
            crf = _dcrf_nll_tied(emis_p, c, self.nsw_pair, self.stay_bonus,
                                 self._allowed_pairs(h1, h2, batch["lin"]))
        else:
            crf = _dcrf_nll(emis_p, c, self.nsw_pair, self.stay_bonus, tags)
        loss = crf + self.gate_reg * (1.0 - g).mean()
        return loss, crf, g, c, emis_p, tags

    def _log_heads(self, prefix):
        if getattr(self, "_head_parts", None):
            for k, v in self._head_parts.items():
                self.log(f"{prefix}/head_{k}", v)

    def training_step(self, batch, _):
        loss, crf, g, c, _, _ = self._step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/crf_loss", crf)
        self.log("train/gate", g.mean())
        self._log_heads("train")
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
        pred = self.crf_decode(emis_p, c)
        pair_true = self.pair_table[h1, h2]
        pair_acc = (pred == pair_true).float().mean()
        pred_lo, pred_hi = self.pi[pred], self.pj[pred]
        t_lo = torch.minimum(h1, h2)
        t_hi = torch.maximum(h1, h2)
        hap_acc = ((pred_lo == t_lo).float() + (pred_hi == t_hi).float()).mean() / 2
        # [homo_correct, homo_n, het_correct, het_n, pred_homo, n], summed over
        # the epoch: val_pair_acc alone hides a one-class val set or a model
        # that always calls homozygous (the original held-out-augment).
        correct, homo = (pred == pair_true), (h1 == h2)
        if getattr(self, "_val_class_counts", None) is None:
            self._val_class_counts = torch.zeros(6, dtype=torch.float64, device=pred.device)
        self._val_class_counts += torch.stack([
            (correct & homo).sum(), homo.sum(), (correct & ~homo).sum(), (~homo).sum(),
            (pred_lo == pred_hi).sum(), torch.tensor(pred.numel(), device=pred.device),
        ]).double()
        return pair_acc, hap_acc

    def on_validation_epoch_start(self):
        self._val_class_counts = torch.zeros(6, dtype=torch.float64, device=self.device)

    def on_validation_epoch_end(self):
        hc, hn, tc, tn, ph, n = self._val_class_counts.tolist()
        nan = float("nan")
        self.log("val/pair_acc_homo", hc / hn if hn else nan)
        self.log("val/pair_acc_het", tc / tn if tn else nan)
        self.log("val/homo_frac", hn / n if n else nan)
        self.log("val/pred_homo_frac", ph / n if n else nan)

    def validation_step(self, batch, _):
        loss, crf, g, c, emis_p, _ = self._step(batch)
        if self.supervised_heads == "stage1":
            # heads only: the CRF scalars are not set yet, so no decode; select on the summed
            # validation head losses (gate + switch + xo + het + affinity)
            self.log("val/loss", loss, prog_bar=True)
            self.log("val_head_total", loss)
            self._log_heads("val")
            return loss
        pair_acc, hap_acc = self._accuracy(emis_p, c, batch["h1"], batch["h2"])
        if "lin" in batch:
            with torch.no_grad():
                pred = self.crf_decode(emis_p, c)
                ok = self._allowed_pairs(batch["h1"], batch["h2"], batch["lin"]).gather(
                    2, pred.unsqueeze(-1)).float().mean()
            self.log("val/pair_acc_tie", ok, prog_bar=True)
            self.log("val_pair_acc_tie", ok)            # slash-free alias for checkpointing
            with torch.no_grad():
                okw = self._allowed_pairs(batch["h1"], batch["h2"], batch["lin"]).gather(
                    2, pred.unsqueeze(-1)).squeeze(-1).float()
                inb = (batch["h1"] == batch["h2"]).all(1)
                if inb.any():
                    self.log("val/tie_acc_inbred", okw[inb].mean(), batch_size=int(inb.sum()))
                if (~inb).any():
                    self.log("val/tie_acc_hybrid", okw[~inb].mean(), batch_size=int((~inb).sum()))
        self.log("val/loss", loss, prog_bar=True)
        self.log("val_loss", loss)
        self.log("val/pair_acc", pair_acc, prog_bar=True)
        self.log("val_pair_acc", pair_acc)              # slash-free alias, see train_diploid.py
        self.log("val/hap_acc", hap_acc, prog_bar=True)
        self.log("val/gate", g.mean())
        self._log_heads("val")
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


# Sample router thresholds -- midpoints of the gaps seen on the 30 real 0.1x
# IDX/OUT simval samples (2026-09-24): same-bin disjoint fraction homozygous
# <= 0.272 vs hybrid >= 0.370; p90 IDX homozygous >= 0.723, IDX hybrid <= 0.323,
# every OUT sample 0.43-0.64. Validate on MIX / other depths before trusting.
ROUTE_DISJ_HYBRID = 0.32
ROUTE_P90_INPANEL_HOMO = 0.68
ROUTE_P90_INPANEL_HYB = 0.375
ROUTE_OUT_SWITCH_SCALE = 0.03


def route_real_sample(M, row_bin, informative_max=12):
    """Per-sample decode settings from two read-level statistics.

    disj: among consecutive rows in the SAME refmap bin that each match
    1..informative_max founders, the fraction whose matching-founder sets are
    disjoint. Nearby reads of an inbred (in-panel or held-out) come from one
    haplotype and share founders; in a hybrid half the pairs straddle the two
    haplotypes. Unlike p90 it does not depend on any panel founder matching
    the sample closely, so it works for held-out lines.

    p90: homo_scale_from_affinity's statistic. In-panel samples have windows
    where one founder (inbred, high p90) or two founders (hybrid, low p90)
    explain nearly every read; held-out samples sit in between.

    Returns dict(disj, p90, hybrid, in_panel, homo_scale, switch_scale):
    homo_scale 1 for hybrids else 0; switch_scale 1.0 in-panel, else
    ROUTE_OUT_SWITCH_SCALE (held-out genomes are fine mosaics of the panel and
    need many more founder switches than the model's in-panel prior allows).
    M: [N,T,K] bool MATCH view; row_bin: [N,T] int64 key unique per
    (contig, refmap bin)."""
    Mf = M.reshape(-1, M.shape[-1]).astype(bool)
    b = np.asarray(row_bin).reshape(-1)
    nm = Mf.sum(1)
    inf = (nm >= 1) & (nm <= informative_max)
    pair = (b[1:] == b[:-1]) & inf[1:] & inf[:-1]
    disj = float(((Mf[1:] & Mf[:-1]).sum(1) == 0)[pair].mean()) if pair.any() else float("nan")
    win_rate = M.astype(np.float32).mean(axis=1)
    p90 = float(np.percentile(_estimate_inbreeding_coef_batch(win_rate), 90))
    hybrid = bool(disj > ROUTE_DISJ_HYBRID)
    in_panel = bool(p90 < ROUTE_P90_INPANEL_HYB) if hybrid else bool(p90 > ROUTE_P90_INPANEL_HOMO)
    return dict(disj=disj, p90=p90, hybrid=hybrid, in_panel=in_panel,
                homo_scale=1.0 if hybrid else 0.0,
                switch_scale=1.0 if in_panel else ROUTE_OUT_SWITCH_SCALE)


def infer_real_founder_pairs(model, data, num_parents, device=None, batch_size=256,
                             homo_scale=None, switch_scale=None, route=False,
                             row_bin=None, decode="viterbi", aff_region=0, het_scale=None, het_region=5):
    """THE single real-data inference recipe for GRITSCRFDiploidIndel --
    every eval script (real-data unit tests, depth sweeps, per-checkpoint
    comparisons) should call this instead of re-deriving the ext_emb /
    homo_scale / batch / decode loop itself. Real-world ML teams call this
    "training/serving skew" risk: if feature computation isn't provably the
    same code path everywhere it's used, eval numbers silently drift from
    what a shipped model actually does. This session accumulated a dozen
    near-duplicate scratch scripts before consolidating here -- exactly the
    failure mode a shared function prevents.

    data: [N,T,2*num_parents+2] or [N,T,3*num_parents+2] real ternary(+
    distance)(+count) array (the H1/H2 label columns are ignored -- real
    inference never has embedded truth). The wider 3K+2 form (real
    per-cell read-support counts, experiments/depth-confidence-fix/ --
    emit via ropebwt_npy_to_matrix.py --emit-read-counts) is detected
    from the array's own width -- no separate flag or parameter needed;
    old 2K+2 real-data conversions keep working with count=None (the
    model's zero-initialized count_proj then contributes nothing, so this
    is safe by construction, not just "supported"). Uses the checkpoint's
    one established recipe: genome-wide founder affinity as ext_emb,
    homo_scale_from_affinity (train_diploid.py, the p90-of-per-window-
    inbreeding-estimate classifier) as homo_scale. No other variant/
    override point -- if a different homo_scale is ever needed, change it
    here, once, not in each caller. The one exception is the explicit
    homo_scale= argument, for diagnostics only (e.g. an oracle inbred/hybrid
    call, to measure what the p90 classifier's mistakes cost on held-out
    samples); None = the normal classifier. switch_scale (diagnostic, default
    1.0 = as trained) multiplies the whole transition score (the encoder's
    per-site switch cost c AND stay_bonus), making founder switches cheaper
    (<1) or dearer (>1) at decode time only; None = 1.0. route=True picks
    both from route_real_sample (needs row_bin); explicit homo_scale /
    switch_scale arguments still override it. decode="marginal" takes the
    per-site posterior argmax (_dcrf_marginal) instead of the Viterbi path --
    it maximizes expected per-site accuracy, the quantity the genotype score
    measures.

    Returns (pred_lo, pred_hi): [N,T] int arrays, the low/high founder
    index of the predicted pair at every real site."""
    K = num_parents
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    N, T, W = data.shape
    if W not in (2 * K + 2, 3 * K + 2):
        raise ValueError(f"data width {W} not in {{2K+2={2 * K + 2}, "
                         f"3K+2 (with read-support counts)={3 * K + 2}}}")
    has_count = W == 3 * K + 2

    tern = data[:, :, :K].astype(np.float32)
    dist = data[:, :, K + 2:2 * K + 2].astype(np.float32)
    feats = torch.tensor(np.stack([tern, dist], axis=-1), dtype=torch.float32)
    count = (torch.tensor(data[:, :, 2 * K + 2:3 * K + 2].astype(np.float32),
                          dtype=torch.float32) if has_count else None)
    M = (tern == TERN_MATCH).astype(np.float32)
    affinity = _founder_affinity(M.reshape(-1, K))
    auto_scale = homo_scale_from_affinity(M)
    if route:
        if row_bin is None:
            raise ValueError("route=True needs row_bin")
        r = route_real_sample(M == 1, row_bin)
        print(f"  route: disj={r['disj']:.4f} p90={r['p90']:.3f} hybrid={r['hybrid']} "
              f"in_panel={r['in_panel']}", flush=True)
        homo_scale = r["homo_scale"] if homo_scale is None else homo_scale
        switch_scale = r["switch_scale"] if switch_scale is None else switch_scale
    if homo_scale is None:
        homo_scale = auto_scale
    if switch_scale is None:
        switch_scale = 1.0
    print(f"  homo_scale={homo_scale} (p90 classifier chose {auto_scale}) "
          f"switch_scale={switch_scale}", flush=True)
    ext_emb = torch.tensor(affinity, dtype=torch.float32).unsqueeze(0).expand(N, -1, -1)

    aff_pred = aff_win = None
    if getattr(model, "supervised_heads", "off") != "off":
        # the heads decide the homozygous penalty (per row), the switch cost and the affinity
        # prior: the homo_scale and switch_scale chosen above are NOT used for this model
        if getattr(model, "aff_source", "head") == "reads":
            aff_pred = torch.tensor(affinity[:, 0])      # prior comes from ext_emb in the model
        elif aff_region != 0:
            # regional prior: each window's affinity = mean head output over the windows within
            # +-aff_region on the same contig (follows a mosaic sample's local founders);
            # aff_region < 0: each window's OWN head output (its estimate of the global affinity,
            # no pooling across windows)
            if row_bin is None:
                raise ValueError("aff_region needs row_bin (contig of each window)")
            aff_win = regional_affinity(model, feats, count, ext_emb, device, batch_size,
                                        np.asarray(row_bin)[:, 0] >> 40, max(aff_region, 0))
            aff_pred = aff_win.mean(0)
        else:
            aff_pred = pooled_affinity(model, feats, count, ext_emb, device, batch_size)
        print(f"  supervised heads: router homo_scale/switch_scale unused, pooled affinity "
              f"top founders {np.argsort(-aff_pred.cpu().numpy())[:4].tolist()}"
              + (f" (regional prior, +-{aff_region} windows)" if aff_region > 0 else
                 " (per-window own prior)" if aff_region < 0 else ""), flush=True)

    het_win = None
    if het_scale is not None and getattr(model, "supervised_heads", "off") != "off":
        # pooled het rate as a once-per-segment prior: each window's het fraction (mean of the
        # per-row het head), averaged over the whole sample, its chromosome, or +-het_region windows
        if row_bin is None and het_scale != "global":
            raise ValueError("het_scale chrom/region needs row_bin (contig of each window)")
        hw = window_het_fraction(model, feats, count, ext_emb, device, batch_size)       # [N]
        contig = np.zeros(N, np.int64) if row_bin is None else np.asarray(row_bin)[:, 0] >> 40
        het_win = pool_windows(hw, contig, {"global": -1, "chrom": 0, "region": het_region}[het_scale])
        model.het_prior = "segment"
        print(f"  het prior per segment from {het_scale} het rate: mean {float(het_win.mean()):.3f} "
              f"(window het fraction mean {float(hw.mean()):.3f})", flush=True)

    preds_lo, preds_hi = [], []
    with torch.no_grad():
        for s in range(0, N, batch_size):
            xb = feats[s:s + batch_size].to(device)
            eb = ext_emb[s:s + batch_size].to(device)
            hs = torch.full((xb.shape[0],), homo_scale, device=device)
            cb = count[s:s + batch_size].to(device) if count is not None else None
            ab = (aff_win[s:s + batch_size].to(device) if aff_win is not None else
                  None if aff_pred is None else aff_pred.unsqueeze(0).expand(xb.shape[0], -1))
            hp = None if het_win is None else het_win[s:s + batch_size].to(device)
            emis_p, _g, c = model(xb, ext_emb=eb, homo_scale=hs, count=cb, aff_pred=ab, het_pred=hp)
            if aff_pred is not None:        # heads model: its own calibrated switch cost
                pred = model.crf_decode(emis_p, c)
            else:
                dec = _dcrf_marginal if decode == "marginal" else _dcrf_viterbi
                pred = dec(emis_p, c * switch_scale, model.nsw_pair,
                           model.stay_bonus * switch_scale)
            preds_lo.append(model.pi[pred].cpu().numpy())
            preds_hi.append(model.pj[pred].cpu().numpy())
    return np.concatenate(preds_lo), np.concatenate(preds_hi)


@torch.no_grad()
@torch.no_grad()
def window_het_fraction(model, feats, count, ext_emb, device, batch_size):
    """[N]: each window's het fraction = mean over its rows of the supervised het head."""
    out = []
    for s in range(0, len(feats), batch_size):
        cb = count[s:s + batch_size].to(device) if count is not None else None
        model(feats[s:s + batch_size].to(device), ext_emb=ext_emb[s:s + batch_size].to(device), count=cb)
        out.append(torch.sigmoid(model._heads["het_logit"].float()).mean(1).cpu())
    return torch.cat(out)


def pool_windows(v, contig, radius):
    """Pool a per-window value [N] (windows in contig order): radius < 0 -> whole-sample mean,
    0 -> mean over the window's contig, > 0 -> mean over +-radius windows of the same contig."""
    if radius < 0:
        return torch.full_like(v, float(v.mean()))
    out = torch.empty_like(v)
    for c in np.unique(contig):
        idx = np.nonzero(contig == c)[0]
        x = v[idx]
        if radius == 0:
            out[idx] = x.mean()
            continue
        cs = torch.cat([torch.zeros(1), x.cumsum(0)])
        lo = np.clip(np.arange(len(idx)) - radius, 0, len(idx))
        hi = np.clip(np.arange(len(idx)) + radius + 1, 0, len(idx))
        out[idx] = (cs[hi] - cs[lo]) / torch.tensor(hi - lo, dtype=torch.float32)
    return out


def regional_affinity(model, feats, count, ext_emb, device, batch_size, contig, radius):
    """[N,K]: per window, the mean affinity-head probability over the windows of the same contig
    within +-radius windows (windows are in contig order, as windowed_k25native_wcount.npy)."""
    rows = []
    for s in range(0, len(feats), batch_size):
        xb = feats[s:s + batch_size].to(device)
        eb = ext_emb[s:s + batch_size].to(device)
        cb = count[s:s + batch_size].to(device) if count is not None else None
        with torch.no_grad():
            model(xb, ext_emb=eb, count=cb)
        rows.append(torch.sigmoid(model._heads["aff_logit"].float()).cpu())
    a = torch.cat(rows)                                              # [N,K]
    out = torch.empty_like(a)
    contig = np.asarray(contig)
    for c in np.unique(contig):
        idx = np.nonzero(contig == c)[0]
        cs = torch.cat([torch.zeros(1, a.shape[1]), a[idx].cumsum(0)])
        lo = np.clip(np.arange(len(idx)) - radius, 0, len(idx))
        hi = np.clip(np.arange(len(idx)) + radius + 1, 0, len(idx))
        out[idx] = (cs[hi] - cs[lo]) / torch.tensor(hi - lo, dtype=torch.float32)[:, None]
    return out


@torch.no_grad()
def pooled_affinity(model, feats, count, ext_emb, device, batch_size=256):
    """Supervised-heads affinity for one sample: mean over ALL its windows of the per-window
    affinity-head probabilities -> [K] (the per-sample pooled prior)."""
    acc, n = None, 0
    for s in range(0, len(feats), batch_size):
        xb = feats[s:s + batch_size].to(device)
        eb = ext_emb[s:s + batch_size].to(device)
        cb = count[s:s + batch_size].to(device) if count is not None else None
        model(xb, ext_emb=eb, count=cb)
        a = torch.sigmoid(model._heads["aff_logit"].float()).sum(0)
        acc = a if acc is None else acc + a
        n += xb.shape[0]
    return acc / n


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

    # Vectorized pre-filter (experiments/depth-confidence-fix/, same class of
    # fix as homo_scale_from_affinity's earlier O(N) loop): compute has_err/
    # has_switch for ALL N windows at once, and batch-gather raw_counts for
    # every remaining candidate in ONE fancy-index call instead of N separate
    # per-window mmap reads -- verified bit-exact against the original
    # per-window loop on real data (bench_score_ibd_adjusted.py). The
    # per-window majority-vote/_window_error_slots logic is deliberately left
    # as a loop over just the (usually much smaller) candidate set -- low
    # risk, and not the actual bottleneck (0.02-0.34s even at N=28k;
    # infer_real_founder_pairs's Viterbi decode dominates real eval time by
    # 2+ orders of magnitude).
    N, T = pred_lo.shape
    correct = (pred_lo == true_lo) & (pred_hi == true_hi)
    has_err = (~correct & valid).any(axis=1)
    has_switch = (true_lo != true_lo[:, :1]).any(axis=1) | (true_hi != true_hi[:, :1]).any(axis=1)
    candidates = np.flatnonzero(has_err & ~has_switch)

    credited = np.zeros((N, T), dtype=bool)
    n_checked, n_credited_windows = 0, 0

    if candidates.size:
        idx2d = np.stack([row_idx[w] for w in candidates], axis=0)              # [C,T]
        mean_support_all = np.asarray(raw_counts[idx2d, :num_parents]).astype(np.float64).mean(axis=1)  # [C,K]

        for i, w in enumerate(candidates):
            err_w = (~correct[w]) & valid[w]
            pairs, counts = np.unique(np.stack([pred_lo[w], pred_hi[w]], axis=1), axis=0, return_counts=True)
            maj_lo, maj_hi = pairs[np.argmax(counts)]
            slots = _window_error_slots(true_lo[w, 0], true_hi[w, 0], maj_lo, maj_hi)
            if not slots:
                continue  # majority call is actually right even though some sites in the window differ
            mean_support = mean_support_all[i]
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


# Hyperparameters that change what a warm-started checkpoint's weights MEAN
# (emission-head wiring, pair-emission offset, feature conditioning, shapes):
# a mismatch silently fine-tunes the weights under a different model. Found
# the hard way -- every low-rate and held-out-augment retrain warm-started
# diploid-indel-v3-k25-overlay-affinity (time_local_emis=True,
# homo_penalty=3.0) without those flags, and the bisection in
# experiments/depth-confidence-fix/ showed that alone collapses training
# (val_pair_acc 0.03 vs 0.50) regardless of the data.
WARM_START_MODEL_HPARAMS = ("num_parents", "d_model", "n_heads", "n_layers",
                            "time_local_emis", "homo_penalty", "learned_het",
                            "founder_affinity")
# Optimization-schedule hparams: legitimately changeable, but worth a warning.
WARM_START_SCHEDULE_HPARAMS = ("warmup_steps", "spike_skip")


def warm_start_hparam_mismatches(ckpt_hparams, run_hparams):
    """Compare a warm-start checkpoint's saved hyper_parameters against this
    run's. Returns (model_mismatches, schedule_mismatches), each a list of
    (name, ckpt_value, run_value). Keys absent from the checkpoint (older
    checkpoints predating a flag) are skipped rather than guessed."""
    model_mm, sched_mm = [], []
    for keys, out in ((WARM_START_MODEL_HPARAMS, model_mm),
                      (WARM_START_SCHEDULE_HPARAMS, sched_mm)):
        for k in keys:
            if k in ckpt_hparams and k in run_hparams and ckpt_hparams[k] != run_hparams[k]:
                out.append((k, ckpt_hparams[k], run_hparams[k]))
    return model_mm, sched_mm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--workdir", default="/workdir/esb33")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--limit-n", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=0,
                   help="Seed for the individual-level train/val/test shuffle.")
    p.add_argument("--legacy-tail-split", action="store_true",
                   help="Old unshuffled split (val/test = file-order tail). Only for "
                        "reproducing pre-2026-09-23 runs: on concatenated inbred+outbred "
                        "data it makes val a single individual type.")
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
    p.add_argument("--warm-start-ckpt", default=None,
                   help="Path to a .ckpt to load ONLY the model weights from "
                        "(strict=False, e.g. an older checkpoint that predates "
                        "density_proj -- experiments/depth-confidence-fix/), then "
                        "train fresh (new optimizer/scheduler/epoch state) -- unlike "
                        "--resume, which requires an exact architecture match and "
                        "restores full trainer state. Mutually exclusive with --resume.")
    p.add_argument("--train-homo-scale", action="store_true",
                   help="with --founder-affinity: scale the homozygous penalty per simulated individual "
                        "by its true kind (0 inbred, 1 hybrid), as the router does at inference; "
                        "default applies the full penalty to every individual")
    p.add_argument("--emission", choices=["learned", "likelihood", "likelihood_dist"], default="learned",
                   help="CRF emission source: learned (encoder founder scores, default) or likelihood "
                        "(per-row read log-likelihood from the ternary state, 3 shared learned values; "
                        "the encoder only supplies the gate and switch cost)")
    p.add_argument("--pair-emission", choices=["sum", "mixture"], default="sum",
                   help="pair-state emission from per-founder scores: sum (e_i + e_j, default) or "
                        "mixture (logaddexp(e_i, e_j) - log 2: each row explained by either haplotype)")
    p.add_argument("--supervised-heads", choices=["off", "stage1", "stage2"], default="off",
                   help="stage1: train the encoder only on simulator-truth head targets (gate=row "
                        "clean from <data>.prov.npy, switch, het, per-individual affinity); stage2: "
                        "freeze everything but the CRF scalars and fit them with the CRF loss "
                        "(warm start from the stage-1 ckpt, --aff-pred for the pooled affinity)")
    p.add_argument("--extra-heads", action="store_true",
                   help="--supervised-heads: add the deletion-dosage and out-of-panel heads (targets: "
                        "--prov-deletion-bits provenance and <data>.ood.npy from heldout_augment_v2.py)")
    p.add_argument("--gate-target", choices=["prov", "support"], default="prov",
                   help="--supervised-heads gate target: simulator provenance (default) or 'the row "
                        "matches the true founder of the haplotype it was read from'")
    p.add_argument("--aff-source", choices=["head", "reads"], default="head",
                   help="--supervised-heads CRF founder prior: pooled affinity head, or the sample's "
                        "genome-wide read match rate (no affinity head/loss in stage 1)")
    p.add_argument("--aff-target-scope", choices=["individual", "window"], default="individual",
                   help="stage1 affinity-head target: the individual's genome-wide founder fractions "
                        "(default) or each window's own (pooled over windows at decode = the former)")
    p.add_argument("--no-distance", action="store_true",
                   help="ablation: replace the anchor-distance channel with a constant 0 (train and "
                        "inference; saved in the checkpoint), so the encoder sees only the ternary array")
    p.add_argument("--het-prior", choices=["row", "segment", "off"], default="row",
                   help="--supervised-heads: het-head prior on every row (row), once per segment "
                        "(segment: row 0 and each pair-state change) or not at all (off)")
    p.add_argument("--xo-placement", action="store_true",
                   help="supervised heads: spread the window crossover count over rows in proportion "
                        "to the per-row switch head (default: uniformly over the window)")
    p.add_argument("--prov-labels", default=None,
                   help="[N,T] int8 row-provenance sidecar (default <data minus .npy>.prov.npy)")
    p.add_argument("--aff-pred", default=None,
                   help="stage2: [N/G,K] pooled affinity-head predictions (pool_affinity.py)")
    p.add_argument("--limit-train-batches", type=float, default=1.0,
                   help="fraction of training batches per epoch (Lightning), e.g. for stage2")
    p.add_argument("--tie-aware-loss", action="store_true",
                   help="credit any founder pair IBD-equivalent (same ancestral lineage at the "
                        "row) to the labelled pair; needs per-row lineage labels (--lineage-labels)")
    p.add_argument("--lineage-labels", default=None,
                   help="[N,T,K] int8 per-row lineage sidecar (default with --tie-aware-loss: "
                        "<data minus .npy>.lin.npy, written by gen_training_data.py)")
    p.add_argument("--allow-hparam-mismatch", action="store_true",
                   help="Proceed even if --warm-start-ckpt was trained with different "
                        "model hparams (time_local_emis, homo_penalty, ...) than this "
                        "run's flags. Off by default: such a mismatch silently changes "
                        "what the loaded weights mean.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.resume and args.warm_start_ckpt:
        raise SystemExit("--resume and --warm-start-ckpt are mutually exclusive")
    workdir = Path(args.workdir)
    ckpt_dir = workdir / "checkpoints" / args.run_name
    log_dir = workdir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    lin_path = None
    if args.tie_aware_loss:
        lin_path = args.lineage_labels or str(Path(args.data).with_suffix("")) + ".lin.npy"
        if not Path(lin_path).exists():
            raise SystemExit(f"--tie-aware-loss: lineage labels not found at {lin_path}")
        if not (args.founder_affinity or args.adaptive_homo):
            raise SystemExit("--tie-aware-loss needs the individual-aligned splits "
                             "(--founder-affinity or --adaptive-homo)")
    prov_path = None
    if args.supervised_heads != "off":
        if not (args.founder_affinity and args.tie_aware_loss):
            raise SystemExit("--supervised-heads needs --founder-affinity and --tie-aware-loss "
                             "(individual splits + lineage labels)")
        if args.supervised_heads == "stage1":
            prov_path = args.prov_labels or str(Path(args.data).with_suffix("")) + ".prov.npy"
            if not Path(prov_path).exists():
                raise SystemExit(f"--supervised-heads stage1: provenance not found at {prov_path}")
    if args.founder_affinity:
        train_ds, val_ds, _ = make_indel_diploid_affinity_splits(
            args.data, args.num_parents, args.val_frac, args.test_frac,
            args.windows_per_individual, limit_n=args.limit_n,
            split_seed=args.split_seed, legacy_tail_split=args.legacy_tail_split,
            lin_path=lin_path, train_homo_scale=args.train_homo_scale,
            prov_path=prov_path,
            aff_targets=args.supervised_heads == "stage1" and args.aff_source == "head",
            aff_pred_path=args.aff_pred, aff_target_scope=args.aff_target_scope,
            ood_path=(str(Path(args.data).with_suffix("")) + ".ood.npy"
                      if args.extra_heads and args.supervised_heads == "stage1" else None))
    elif args.adaptive_homo:
        train_ds, val_ds, _ = make_indel_diploid_individual_splits(
            args.data, args.num_parents, args.val_frac, args.test_frac,
            args.windows_per_individual, het_inbred=args.het_inbred,
            het_outbred=args.het_outbred, limit_n=args.limit_n,
            split_seed=args.split_seed, legacy_tail_split=args.legacy_tail_split,
            lin_path=lin_path)
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
        fast_cells=args.fast_cells, tie_aware_loss=args.tie_aware_loss,
        pair_emission=args.pair_emission, emission=args.emission,
        supervised_heads=args.supervised_heads, xo_placement=args.xo_placement,
        het_prior=args.het_prior, no_distance=args.no_distance, aff_source=args.aff_source,
        gate_target=args.gate_target, extra_heads=args.extra_heads)

    if args.warm_start_ckpt:
        # Weight-only load (strict=False): the source checkpoint may predate
        # params this architecture has since gained (e.g. density_proj --
        # experiments/depth-confidence-fix/), which stay at this model's own
        # init (zero, for density_proj) rather than being an error. Optimizer/
        # scheduler/epoch state is intentionally NOT restored -- this is a
        # fresh training run seeded with old weights, not a resumed one.
        ck = torch.load(args.warm_start_ckpt, map_location="cpu", weights_only=False)
        model_mm, sched_mm = warm_start_hparam_mismatches(
            ck.get("hyper_parameters", {}), dict(model.hparams))
        fmt = lambda mm: ", ".join(f"{k}: ckpt={a!r} run={b!r}" for k, a, b in mm)
        if sched_mm:
            print(f"WARNING --warm-start-ckpt schedule hparams differ: {fmt(sched_mm)}")
        if model_mm:
            msg = (f"--warm-start-ckpt model hparams differ from this run's flags: "
                   f"{fmt(model_mm)}")
            if not args.allow_hparam_mismatch:
                raise SystemExit(msg + " (pass the checkpoint's flags, or "
                                 "--allow-hparam-mismatch if intentional)")
            print("WARNING " + msg)
        sd = ck["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"--warm-start-ckpt {args.warm_start_ckpt}: "
              f"missing={missing} unexpected={unexpected}")

    # Checkpoint/stop on val/pair_acc (max), matching train_diploid.py: the CRF
    # partition NLL can spike on long-block data even as Viterbi accuracy stays
    # good, so selecting on loss can discard the best model.
    sel_metric = "val_pair_acc_tie" if args.tie_aware_loss else "val_pair_acc"
    sel_mode = "max"
    if args.supervised_heads == "stage1":          # heads only: select on the summed head losses
        sel_metric, sel_mode = "val_head_total", "min"
    callbacks = [
        # tie-aware runs select on tie-aware accuracy: exact-pair accuracy is
        # arbitrary among lineage-equivalent founders there
        ModelCheckpoint(dirpath=str(ckpt_dir), monitor=sel_metric,
                        mode=sel_mode, save_top_k=2, save_last=True,
                        filename="d-{epoch:02d}-{" + sel_metric + ":.4f}"),
        EarlyStopping(monitor=sel_metric, mode=sel_mode, patience=args.patience),
    ]
    if args.ema:
        callbacks.append(EMACallback(args.ema_decay))
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, callbacks=callbacks,
        logger=TensorBoardLogger(str(log_dir), name=args.run_name),
        accelerator="auto", devices=args.devices, precision=args.precision,
        val_check_interval=(args.val_check_interval or None),
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=args.limit_train_batches)
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    print(f"Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
