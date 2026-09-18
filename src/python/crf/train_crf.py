"""
GRITS-CRF Training Script with PyTorch Lightning
Diploid founder path imputation using neural CRF
"""

import os
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

NEG_INF = -1e4


# --------------------------------------------------------------------------- #
#  Dataset with diploid pair conversion                                        #
# --------------------------------------------------------------------------- #

# Create a shared function for consistent pair indexing
def create_diploid_pairs(num_parents):
    """Create consistent pair mapping used by both dataset and model"""
    pairs = []
    pair_to_idx = {}

    K = num_parents + 1  # +1 for unknown state
    for i in range(K):
        for j in range(i, K):
            idx = len(pairs)
            pairs.append((i, j))
            pair_to_idx[(i, j)] = idx
            if i != j:
                pair_to_idx[(j, i)] = idx

    return pairs, pair_to_idx

class LabeledDatasetDiploid(Dataset):
    def __init__(self, file_dir, input_file_names, window_size=512, num_parents=24, step_size=128):
        self.file_dir = file_dir
        self.input_file_names = input_file_names
        self.window_size = window_size
        self.num_parents = num_parents
        self.step_size = step_size

        # Precompute pair indices for diploid states
        # self.pair_idx, self.pairs = self._create_pair_index()
        # self.windows = self.__generate_windows__()
        # self.n_windows = len(self.windows)

        # Use shared function
        self.pairs, self.pair_idx = create_diploid_pairs(num_parents)
        print(f"Created {len(self.pairs)} pair states for {num_parents} parents")

        self.windows = self.__generate_windows__()
        self.n_windows = len(self.windows)

        print(f"Dataset created: {self.n_windows} windows, {len(self.pairs)} pair states")

    # def _create_pair_index(self):
    #     """Create mapping from unordered founder pairs to indices"""
    #     pairs = []
    #     pair_to_idx = {}
    #
    #     # Include all unordered pairs {i,j} where i <= j
    #     for i in range(self.num_parents + 1):  # +1 for unknown state
    #         for j in range(i, self.num_parents + 1):
    #             idx = len(pairs)
    #             pairs.append((i, j))
    #             pair_to_idx[(i, j)] = idx
    #             if i != j:  # Also map {j,i} to same index for unordered pairs
    #                 pair_to_idx[(j, i)] = idx
    #
    #     return pair_to_idx, pairs

    def _diploid_to_pair_idx(self, diploid_labels):
        """Convert [T, 2] diploid labels to [T] pair indices"""
        """Convert [T, 2] diploid labels to [T] pair indices with bounds checking"""
        T = diploid_labels.shape[0]
        pair_indices = torch.zeros(T, dtype=torch.long)

        for t in range(T):
            i, j = diploid_labels[t, 0].item(), diploid_labels[t, 1].item()

            # Add bounds checking
            if (i, j) not in self.pair_idx:
                print(f"Warning: Invalid pair ({i}, {j}) at position {t}")
                # Fallback to unknown state pair
                unknown_idx = self.num_parents
                pair_indices[t] = self.pair_idx[(unknown_idx, unknown_idx)]
            else:
                pair_indices[t] = self.pair_idx[(i, j)]

        return pair_indices
        # T = diploid_labels.shape[0]
        # pair_indices = torch.zeros(T, dtype=torch.long)
        #
        # for t in range(T):
        #     i, j = diploid_labels[t, 0].item(), diploid_labels[t, 1].item()
        #     # Look up the pair index for this unordered pair
        #     pair_indices[t] = self.pair_idx[(i, j)]
        #
        # return pair_indices

    def __generate_windows__(self):
        windows = []
        for idx in range(len(self.input_file_names)):
            filelen = np.load(f"{self.file_dir}/{self.input_file_names[idx]}").shape[0]
            num_windows = (filelen - self.window_size) // self.step_size
            windows.extend([(idx, idy) for idy in range(num_windows)])
        return windows

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        # retrieve window index from list
        file_idx, pos_idx = self.windows[idx]

        # convert to position start and end
        pos_start = pos_idx * self.step_size
        pos_end = pos_start + self.window_size

        # grab segment from mmaped numpy
        ip = np.load(
            f"{self.file_dir}/{self.input_file_names[file_idx]}",
            allow_pickle=True,
            mmap_mode='r'
        )[pos_start:pos_end]

        if ip.shape[1] != self.num_parents + 2:  # copy haploid labels
            matrix = np.concatenate([ip, ip[:, self.num_parents:self.num_parents + 1]], axis=1)
        else:  # diploid labels
            matrix = ip[:, 0:self.num_parents + 2]

        # Convert labels and handle unknown states
        labels = torch.tensor(matrix[:, self.num_parents:self.num_parents + 2], dtype=torch.int64)
        labels[labels == -1] = self.num_parents

        # Convert diploid labels to pair indices
        pair_labels = self._diploid_to_pair_idx(labels)

        return {
            "input_embeds": torch.tensor(matrix[:, :-2], dtype=torch.float),
            "labels": pair_labels,
            "diploid_labels": labels  # Keep for debugging
        }


# --------------------------------------------------------------------------- #
#  Model components (from original encoder_crf.py)                             #
# --------------------------------------------------------------------------- #

class FounderPathEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=4, ext_dim=0,
                 time_local_emis=False, window_c=False, learned_het=False,
                 binary_cells=False):
        super().__init__()
        self.d_model = d_model
        self.time_local_emis = time_local_emis
        self.window_c = window_c
        # Binary fast-path: for a 0/1 match matrix, cell(log1p(X)) has only TWO
        # distinct outputs, so the per-cell MLP (the d x d matmul over B*T*K cells,
        # the bulk of encoder FLOPs) collapses to a 2-row table lookup — exact, no
        # accuracy change. Off by default (general read-count inputs); set at
        # inference for binary data. See _embed_cells.
        self.binary_cells = binary_cells
        # E7-diag fix: a PER-LOCUS het logit from the window context. The diploid
        # wrapper turns it into a per-site homozygous penalty — the emission-side
        # signal that the transition cost provably cannot supply (RESULTS E7-diag).
        # Gated so it adds no params unless requested (old checkpoints still load).
        if learned_het:
            self.het_head = nn.Linear(d_model, 1)
            nn.init.zeros_(self.het_head.weight)
            nn.init.constant_(self.het_head.bias, -5.0)   # softplus(-5)~0 => starts off
        else:
            self.het_head = None
        self.cell = nn.Sequential(nn.Linear(1, d_model), nn.GELU(),
                                  nn.Linear(d_model, d_model))
        # E5 relatedness: a DIRECT per-founder additive emission bias from the
        # affinity vector — added straight to each founder's emission logit (a
        # genome-wide founder-presence prior), constant over sites. This is far
        # easier to learn than shifting the emission key (which only helps if the
        # shift aligns with the per-site hidden state). Zero-init so it starts as a
        # no-op (identical to the no-ext baseline) and grows gently.
        self.ext_bias = nn.Linear(ext_dim, 1) if ext_dim > 0 else None
        if self.ext_bias is not None:
            nn.init.zeros_(self.ext_bias.weight)
            nn.init.zeros_(self.ext_bias.bias)
        self.fpool = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fquery = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                         batch_first=True, activation="gelu")
        self.pos_encoder = nn.TransformerEncoder(enc, n_layers)
        self.gate_head = nn.Linear(d_model, 1)
        self.recomb_head = nn.Linear(d_model + 3, 1)
        self.scale = d_model ** -0.5

    @staticmethod
    def _posenc(T, d, device):
        pe = torch.zeros(T, d, device=device)
        pos = torch.arange(T, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2, device=device).float()
                        * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def _embed_cells(self, X):
        """Per-(site,founder) embedding [B,T,K,d] from cell(log1p(X)). With
        binary_cells, X is assumed 0/1 and the MLP collapses to a 2-row lookup
        (exact): cell() applied to {log1p(0), log1p(1)} then gathered by X."""
        if self.binary_cells:
            vals = torch.tensor([[0.0], [math.log1p(1.0)]],
                                device=X.device, dtype=X.dtype)
            table = self.cell(vals)                              # [2, d]
            return table[X.long()]                               # [B,T,K,d]
        return self.cell(torch.log1p(X).unsqueeze(-1))

    def forward(self, X, founder_mask, dbp=None, ext_emb=None, emit_het=False):
        B, T, K = X.shape
        cells = self._embed_cells(X)

        cf = cells.reshape(B * T, K, self.d_model)
        q = self.fquery.expand(B * T, 1, self.d_model)
        kpad = ~founder_mask.bool().unsqueeze(1).expand(B, T, K).reshape(B * T, K)
        h, _ = self.fpool(q, cf, cf, key_padding_mask=kpad)
        h = h.reshape(B, T, self.d_model) + self._posenc(T, self.d_model, X.device)
        H = self.pos_encoder(h)

        if self.time_local_emis:
            # Per-site founder key: emission at site t scores founder f using
            # its cell embedding AT t, not averaged over the window. Required
            # when the active founder switches within a window (recombination).
            cf_local = cells.masked_fill(~founder_mask.bool().view(B, 1, K, 1), 0.0)
            emis = torch.einsum("btd,btkd->btk", H, cf_local) * self.scale
        else:
            e = cells.mean(dim=1)
            e = e.masked_fill(~founder_mask.bool().unsqueeze(-1), 0.0)
            emis = torch.einsum("btd,bkd->btk", H, e) * self.scale
        if self.ext_bias is not None and ext_emb is not None:
            # E5: genome-wide founder-presence prior — one learned scalar per founder
            # from its affinity, added to every site's emission logit.
            emis = emis + self.ext_bias(ext_emb).squeeze(-1).unsqueeze(1)   # [B,1,K]
        emis = emis.masked_fill(~founder_mask.bool().unsqueeze(1), NEG_INF)

        g = torch.sigmoid(self.gate_head(H)).squeeze(-1)
        valid = founder_mask.bool().unsqueeze(1)
        emis = torch.where(valid, g.unsqueeze(-1) * emis.clamp(min=NEG_INF / 2),
                           torch.full_like(emis, NEG_INF))

        depth = torch.log1p(X.sum(-1, keepdim=True))
        ent = self._entropy(emis, founder_mask).unsqueeze(-1)
        if dbp is None:
            dbp = torch.ones(B, T, 1, device=X.device)
        feats = torch.cat([H, depth, ent, torch.log1p(dbp)], dim=-1)
        if self.window_c:
            # Single transition potential per window: pool features over T, emit
            # one c, broadcast back to [B,T]. The CRF still receives [B,T].
            c = F.softplus(self.recomb_head(feats.mean(dim=1))).squeeze(-1)  # [B]
            c = c.unsqueeze(1).expand(B, T)
        else:
            c = F.softplus(self.recomb_head(feats)).squeeze(-1)              # [B,T]
        if emit_het:
            het = (self.het_head(H).squeeze(-1) if self.het_head is not None
                   else torch.zeros(B, T, device=X.device))                 # [B,T] logit
            return emis, g, c, het
        return emis, g, c

    def _entropy(self, emis, founder_mask):
        p = torch.softmax(emis.masked_fill(~founder_mask.bool().unsqueeze(1), NEG_INF), dim=-1)
        return -(p.clamp_min(1e-9).log() * p).sum(-1)


class IndelFounderPathEncoder(nn.Module):
    """Sibling of FounderPathEncoder for the ternary+distance ("indel-aware")
    input format: per (site,founder) cell is (ternary in {-2,-1,0,1}, distance
    code in {-1,0..127}) instead of a single binary value. Structurally
    identical to FounderPathEncoder elsewhere (pooling/transformer/heads) —
    only the cell embedding and the recomb-head's depth/del_frac features
    differ. A sibling class, not a refactor of FounderPathEncoder: this keeps
    the deployed diploid-affinity-sim512-h3 checkpoint's code path completely
    untouched (experiments/simulator-indels/TRAINING_PLAN.md §2, open
    engineering-judgment call — sibling class chosen for now).

    The cell alphabet is finite: 4 ternary states x 129 distance codes = 516.
    fast_cells replaces the per-cell MLP with an exact 516-row lookup table —
    the same idea as FounderPathEncoder's binary_cells (2-row table) but for
    this wider alphabet.
    """
    N_TERN = 4           # {-2,-1,0,1} one-hot'd, index = value + 2
    N_DIST = 129          # {-1,0..127}, index = value + 1
    N_STATES = N_TERN * N_DIST   # 516

    def __init__(self, d_model=128, n_heads=4, n_layers=4, ext_dim=0,
                 time_local_emis=False, window_c=False, learned_het=False,
                 fast_cells=False):
        super().__init__()
        self.d_model = d_model
        self.time_local_emis = time_local_emis
        self.window_c = window_c
        self.fast_cells = fast_cells
        if learned_het:
            self.het_head = nn.Linear(d_model, 1)
            nn.init.zeros_(self.het_head.weight)
            nn.init.constant_(self.het_head.bias, -5.0)   # softplus(-5)~0 => starts off
        else:
            self.het_head = None
        # cell input: [onehot4(ternary) | distance-scalar] -> 5 dims. log1p is
        # unsafe here (log1p(-1)=-inf on the ternary channel, DIST_PAD=-1 on
        # the distance channel), so the cell embed is redesigned rather than
        # reusing FounderPathEncoder's log1p(X) transform.
        self.cell = nn.Sequential(nn.Linear(5, d_model), nn.GELU(),
                                  nn.Linear(d_model, d_model))
        self.ext_bias = nn.Linear(ext_dim, 1) if ext_dim > 0 else None
        if self.ext_bias is not None:
            nn.init.zeros_(self.ext_bias.weight)
            nn.init.zeros_(self.ext_bias.bias)
        self.fpool = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fquery = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                         batch_first=True, activation="gelu")
        self.pos_encoder = nn.TransformerEncoder(enc, n_layers)
        self.gate_head = nn.Linear(d_model, 1)
        # +4: depth, del_frac, entropy, log1p(dbp) -- one wider than
        # FounderPathEncoder's +3 (depth, entropy, log1p(dbp)): del_frac is a
        # genuinely new recomb-head input (PLAN.md §2.6 — indels locally
        # suppress crossover; local deletion density is the observable).
        self.recomb_head = nn.Linear(d_model + 4, 1)
        self.scale = d_model ** -0.5
        # density_proj: per-WINDOW (not per-timestep) evidence-density bias
        # -- experiments/depth-confidence-fix/. A per-row version was tried
        # first and rejected (the plan's own probe gate: per-row gap/stack
        # does not separate correct from incorrect predictions within a
        # fixed depth, only the whole-window aggregate does), so this is
        # added once per window and broadcasts over T, same zero-init
        # convention as ext_bias/het_head above -- at init the model is
        # numerically identical to a checkpoint trained without this input,
        # which is what makes warm-starting from diploid-indel-v3-k25-
        # overlay-affinity safe. Constructed LAST (after every other layer
        # above) so its weight-init RNG draws don't shift any other layer's
        # random initialization under a fixed seed -- nn.Linear's constructor
        # always draws from the RNG even though these weights are zeroed
        # immediately after; verified against
        # TestOverfitSmoke.test_loss_falls_and_accuracy_rises, whose fixed-
        # seed trajectory changed when density_proj was constructed earlier.
        self.density_proj = nn.Linear(2, d_model)
        nn.init.zeros_(self.density_proj.weight)
        nn.init.zeros_(self.density_proj.bias)

    @staticmethod
    def _posenc(T, d, device):
        pe = torch.zeros(T, d, device=device)
        pos = torch.arange(T, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2, device=device).float()
                        * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    @staticmethod
    def _cell_input(tern, dist):
        """tern, dist: matching-shape int/float tensors -> [...,5] float input
        for self.cell. tern in {-2,-1,0,1} one-hot'd (index=tern+2); dist in
        {-1,0..127} mapped to a scalar in [-1,1], with -1 the explicit
        no-anchor sentinel -- NOT 0, which would instead claim "sitting
        exactly on an anchor" (the opposite fact)."""
        onehot = F.one_hot(tern.long() + 2, num_classes=4).float()
        dist_f = dist.float()
        dist_scalar = torch.where(dist_f < 0, torch.full_like(dist_f, -1.0),
                                  dist_f / 127.0).unsqueeze(-1)
        return torch.cat([onehot, dist_scalar], dim=-1)

    def _embed_cells(self, X):
        """Per-(site,founder) embedding [B,T,K,d] from (ternary, distance).
        X: [B,T,K,2], X[...,0]=ternary, X[...,1]=distance code. With
        fast_cells, the finite 516-state alphabet collapses the per-cell MLP
        to an exact table lookup, mirroring FounderPathEncoder.binary_cells."""
        tern, dist = X[..., 0], X[..., 1]
        if self.fast_cells:
            tern_idx = torch.arange(4, device=X.device) - 2           # {-2,-1,0,1}
            dist_idx = torch.arange(-1, 128, device=X.device)         # {-1,0..127}
            tt, dd = torch.meshgrid(tern_idx, dist_idx, indexing="ij")
            table = self.cell(self._cell_input(tt.reshape(-1), dd.reshape(-1)))  # [516,d]
            state_id = (tern.long() + 2) * self.N_DIST + (dist.long() + 1)       # [B,T,K]
            return table[state_id]
        return self.cell(self._cell_input(tern, dist))

    def forward(self, X, founder_mask, dbp=None, ext_emb=None, emit_het=False,
                window_density=None):
        B, T, K, _ = X.shape
        cells = self._embed_cells(X)

        cf = cells.reshape(B * T, K, self.d_model)
        q = self.fquery.expand(B * T, 1, self.d_model)
        kpad = ~founder_mask.bool().unsqueeze(1).expand(B, T, K).reshape(B * T, K)
        h, _ = self.fpool(q, cf, cf, key_padding_mask=kpad)
        h = h.reshape(B, T, self.d_model) + self._posenc(T, self.d_model, X.device)
        if window_density is not None:
            # [B,2] raw int8-range codes (compute_window_density's alphabet,
            # DIST_PAD=-1 sentinel) -> [-1,1] scalars, same normalization
            # _cell_input uses for the per-cell distance channel -- then a
            # single vector per window, broadcast over T (unsqueeze(1)),
            # zero at init so this is a pure additive bias once trained.
            wd = window_density.float()
            wd = torch.where(wd < 0, torch.full_like(wd, -1.0), wd / 127.0)
            h = h + self.density_proj(wd).unsqueeze(1)
        H = self.pos_encoder(h)

        tern = X[..., 0]
        if self.time_local_emis:
            # Per-site founder key: emission at site t scores founder f using
            # its cell embedding AT t, not averaged over the window. Required
            # when the active founder switches within a window (recombination).
            cf_local = cells.masked_fill(~founder_mask.bool().view(B, 1, K, 1), 0.0)
            emis = torch.einsum("btd,btkd->btk", H, cf_local) * self.scale
        else:
            e = cells.mean(dim=1)
            e = e.masked_fill(~founder_mask.bool().unsqueeze(-1), 0.0)
            emis = torch.einsum("btd,bkd->btk", H, e) * self.scale
        if self.ext_bias is not None and ext_emb is not None:
            emis = emis + self.ext_bias(ext_emb).squeeze(-1).unsqueeze(1)   # [B,1,K]
        emis = emis.masked_fill(~founder_mask.bool().unsqueeze(1), NEG_INF)

        g = torch.sigmoid(self.gate_head(H)).squeeze(-1)
        valid = founder_mask.bool().unsqueeze(1)
        emis = torch.where(valid, g.unsqueeze(-1) * emis.clamp(min=NEG_INF / 2),
                           torch.full_like(emis, NEG_INF))

        # depth: count of MATCHING founders per site (ternary==1) -- the
        # ternary analogue of FounderPathEncoder's log1p(X.sum(-1)). A raw sum
        # over {-2,-1,0,1} can go negative and make log1p produce NaN, so this
        # counts matches specifically rather than summing the raw channel.
        depth = torch.log1p((tern == 1).float().sum(-1, keepdim=True))
        # del_frac: fraction of founders reading as a real deletion at this
        # site -- new recomb-head input, no analogue in FounderPathEncoder.
        del_frac = (tern == -1).float().mean(-1, keepdim=True)
        ent = self._entropy(emis, founder_mask).unsqueeze(-1)
        if dbp is None:
            dbp = torch.ones(B, T, 1, device=X.device)
        feats = torch.cat([H, depth, del_frac, ent, torch.log1p(dbp)], dim=-1)
        if self.window_c:
            c = F.softplus(self.recomb_head(feats.mean(dim=1))).squeeze(-1)  # [B]
            c = c.unsqueeze(1).expand(B, T)
        else:
            c = F.softplus(self.recomb_head(feats)).squeeze(-1)              # [B,T]
        if emit_het:
            het = (self.het_head(H).squeeze(-1) if self.het_head is not None
                   else torch.zeros(B, T, device=X.device))                 # [B,T] logit
            return emis, g, c, het
        return emis, g, c

    def _entropy(self, emis, founder_mask):
        p = torch.softmax(emis.masked_fill(~founder_mask.bool().unsqueeze(1), NEG_INF), dim=-1)
        return -(p.clamp_min(1e-9).log() * p).sum(-1)


class NeuralCRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.stay_bonus = nn.Parameter(torch.tensor(2.0))

    def _trans(self, c, nsw):
        B, T = c.shape
        tr = -c[:, :, None, None] * nsw[None, None]
        tr = tr + (nsw == 0).float()[None, None] * self.stay_bonus
        return tr

    def partition(self, emis, tr):
        B, T, S = emis.shape
        a = emis[:, 0]
        for t in range(1, T):
            a = emis[:, t] + torch.logsumexp(a.unsqueeze(2) + tr[:, t], dim=1)
        return torch.logsumexp(a, dim=1)

    def score(self, emis, tr, tags):
        B, T, S = emis.shape
        bi = torch.arange(B, device=emis.device)
        s = emis[bi, 0, tags[:, 0]]
        for t in range(1, T):
            s = s + tr[bi, t, tags[:, t - 1], tags[:, t]] + emis[bi, t, tags[:, t]]
        return s

    def nll(self, emis, tr, tags):
        return (self.partition(emis, tr) - self.score(emis, tr, tags)).mean()

    @torch.no_grad()
    def viterbi(self, emis, tr):
        B, T, S = emis.shape
        delta = emis[:, 0]
        bp = []
        for t in range(1, T):
            sc = delta.unsqueeze(2) + tr[:, t]
            best, idx = sc.max(dim=1)
            delta = emis[:, t] + best
            bp.append(idx)
        path = [delta.argmax(dim=1)]
        for idx in reversed(bp):
            path.append(idx.gather(1, path[-1].unsqueeze(1)).squeeze(1))
        return torch.stack(path[::-1], dim=1)


# --------------------------------------------------------------------------- #
#  Diploid utilities                                                            #
# --------------------------------------------------------------------------- #

def diploid_pair_index(K, device="cpu"):
    pairs = [(i, j) for i in range(K) for j in range(i, K)]
    idx = torch.tensor(pairs, device=device)
    P = len(pairs)
    nsw = torch.zeros(P, P, device=device)
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            nsw[a, b] = min((i != k) + (j != l), (i != l) + (j != k))
    return idx, nsw


# def diploid_emissions(emis_f, pair_idx):
#     i, j = pair_idx[:, 0], pair_idx[:, 1]
#     return emis_f[..., i] + emis_f[..., j]


def diploid_emissions(emis_f, pair_idx):
    """
    Convert founder emissions [B,T,K] to pair emissions [B,T,P]
    """
    B, T, K = emis_f.shape
    P = pair_idx.shape[0]

    # Extract founder indices
    i_idx = pair_idx[:, 0]  # [P]
    j_idx = pair_idx[:, 1]  # [P]

    # Check bounds
    if i_idx.max() >= K or j_idx.max() >= K:
        raise ValueError(f"Pair indices exceed founder count: max_i={i_idx.max()}, max_j={j_idx.max()}, K={K}")

    # Vectorized indexing: [B, T, P]
    emis_i = emis_f[:, :, i_idx]  # [B, T, P]
    emis_j = emis_f[:, :, j_idx]  # [B, T, P]

    return emis_i + emis_j


# --------------------------------------------------------------------------- #
#  Lightning Module                                                             #
# --------------------------------------------------------------------------- #

class GRITSCRFModel(pl.LightningModule):
    def __init__(self, num_parents=24, d_model=128, n_heads=4, n_layers=4,
                 lr=3e-4, weight_decay=1e-5, gate_reg=0.05):
        super().__init__()
        self.save_hyperparameters()

        self.num_parents = num_parents
        self.lr = lr
        self.weight_decay = weight_decay
        self.gate_reg = gate_reg

        # Model components
        self.encoder = FounderPathEncoder(d_model, n_heads, n_layers)
        self.crf = NeuralCRF()

        # Diploid state space
        K_total = num_parents + 1  # +1 for unknown state
        pairs, _ = create_diploid_pairs(num_parents)

        # Create diploid index tensors
        pair_idx_list = []
        for i, j in pairs:
            pair_idx_list.append([i, j])

        self.register_buffer("pair_idx", torch.tensor(pair_idx_list))
        # self.register_buffer("pair_idx", diploid_pair_index(K)[0])
        P = len(pairs)
        nsw = torch.zeros(P, P)
        for a, (i, j) in enumerate(pairs):
            for b, (k, l) in enumerate(pairs):
                nsw[a, b] = min((i != k) + (j != l), (i != l) + (j != k))
        self.register_buffer("nsw", nsw)

        print(f"Model: {P} pair states, pair_idx shape: {self.pair_idx.shape}")
        # self.register_buffer("nsw", diploid_pair_index(K)[1])

        # Metrics
        self.train_losses = []
        self.val_losses = []

    def forward(self, X):
        B, T, K = X.shape

        # Pad input to include unknown state (index 24)
        X_padded = torch.cat([
            X,
            torch.zeros(B, T, 1, device=X.device, dtype=X.dtype)
        ], dim=-1)  # Now [B, T, 25]

        # founder_mask = torch.ones(B, K, device=X.device)
        founder_mask = torch.ones(B, K+1, device=X.device)
        # emis_f, g, c = self.encoder(X, founder_mask)
        emis_f, g, c = self.encoder(X_padded, founder_mask)

        # Debug: Check shapes before diploid_emissions
        print(f"Debug: emis_f shape: {emis_f.shape}, pair_idx shape: {self.pair_idx.shape}")
        print(f"Debug: pair_idx max values: i={self.pair_idx[:, 0].max()}, j={self.pair_idx[:, 1].max()}")

        emis_p = diploid_emissions(emis_f, self.pair_idx)
        tr = self.crf._trans(c, self.nsw)
        return emis_p, tr, g, c

    def training_step(self, batch, batch_idx):
        X = batch["input_embeds"]
        tags = batch["labels"]

        emis_p, tr, g, c = self(X)

        B, T, K = X.shape
        max_pair_idx = self.pair_idx.shape[0] - 1

        if tags.max() > max_pair_idx:
            print(f"ERROR: Label index {tags.max()} exceeds max pair index {max_pair_idx}")
            print(f"Batch shape: {X.shape}, Tags shape: {tags.shape}")
            print(f"Unique labels: {tags.unique()}")
            return None

        try:
            emis_p, tr, g, c = self(X)
            crf_loss = self.crf.nll(emis_p, tr, tags)
            gate_loss = self.gate_reg * (1.0 - g).mean()
            total_loss = crf_loss + gate_loss

            if batch_idx % 50 == 0:
                # Check if parameters are actually updating
                total_grad_norm = 0
                for p in self.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.norm().item() ** 2
                total_grad_norm = total_grad_norm ** 0.5

                self.log("train/grad_norm", total_grad_norm)

                if total_grad_norm < 1e-6:
                    print("WARNING: Very small gradients - model may not be learning")

            # Logging
            self.log("train/crf_loss", crf_loss, prog_bar=True)
            self.log("train/gate_loss", gate_loss)
            self.log("train/total_loss", total_loss)
            self.log("train/mean_gate", g.mean())
            self.log("train/mean_recomb", c.mean())
            self.log("train/stay_bonus", self.crf.stay_bonus)
            return total_loss
        except RuntimeError as e:
            print(f"Runtime error in training step:")
            print(f"X shape: {X.shape}")
            print(f"tags shape: {tags.shape}, min: {tags.min()}, max: {tags.max()}")
            print(f"emis_p shape: {emis_p.shape if 'emis_p' in locals() else 'not created'}")
            raise e

        # # CRF loss
        # crf_loss = self.crf.nll(emis_p, tr, tags)
        #
        # # Gate regularization (keep gate near 1 by default)
        # gate_loss = self.gate_reg * (1.0 - g).mean()
        #
        # total_loss = crf_loss + gate_loss
        #
        # # Logging
        # self.log("train/crf_loss", crf_loss, prog_bar=True)
        # self.log("train/gate_loss", gate_loss)
        # self.log("train/total_loss", total_loss)
        # self.log("train/mean_gate", g.mean())
        # self.log("train/mean_recomb", c.mean())
        # self.log("train/stay_bonus", self.crf.stay_bonus)
        #
        # return total_loss

    def validation_step(self, batch, batch_idx):
        X = batch["input_embeds"]
        tags = batch["labels"]

        emis_p, tr, g, c = self(X)
        crf_loss = self.crf.nll(emis_p, tr, tags)
        gate_loss = self.gate_reg * (1.0 - g).mean()
        total_loss = crf_loss + gate_loss

        if batch_idx == 0:  # Log first batch details each epoch
            # Check if model is actually changing predictions
            with torch.no_grad():
                pred_paths = self.crf.viterbi(emis_p, tr)

            # Log prediction diversity
            unique_predictions = [len(torch.unique(pred_paths[i])) for i in range(min(4, pred_paths.shape[0]))]
            self.log("val/pred_diversity", sum(unique_predictions) / len(unique_predictions))

            # Log label diversity for comparison
            unique_labels = [len(torch.unique(tags[i])) for i in range(min(4, tags.shape[0]))]
            self.log("val/label_diversity", sum(unique_labels) / len(unique_labels))

            # Are predictions just stuck on one state?
            mode_predictions = [torch.mode(pred_paths[i])[0].item() for i in range(min(4, pred_paths.shape[0]))]
            print(f"Dominant predicted states: {mode_predictions}")

        # Path accuracy
        pred_paths = self.crf.viterbi(emis_p, tr)
        path_acc = (pred_paths == tags).float().mean()

        print(str(tags[0:10]))

        self.log("val/crf_loss", crf_loss, prog_bar=True)
        self.log("val/total_loss", total_loss)
        self.log("val_total_loss", total_loss)          # slash-free alias so the
                                                          # ModelCheckpoint filename= token
                                                          # below actually substitutes
                                                          # (a "/" in the metric name is a
                                                          # path separator to Lightning).
        self.log("val/path_acc", path_acc, prog_bar=True)
        self.log("val/mean_gate", g.mean())

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
            # Remove verbose=True - it was deprecated
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/total_loss"
        }


# --------------------------------------------------------------------------- #
#  Training script                                                              #
# --------------------------------------------------------------------------- #

def main():
    # Configuration
    config = {
        # "workdir": "/path/to/your/workdir",  # UPDATE THIS — per CLAUDE.md convention,
        "workdir": "/workdir/esb33",  # checkpoints are written to <workdir>/checkpoints/
        # "data_dir": "/path/to/your/data",  # UPDATE THIS
        "data_dir": "/workdir/smm477/ML-data/training",  # UPDATE THIS
        # "train_files": ["train_file1.npy", "train_file2.npy"],  # UPDATE THIS
        "train_files": ["B97_chr10_matrix_data-1_0.1x.npy", "B97_chr9_matrix_data-1_1x.npy"],  # UPDATE THIS
        # "val_files": ["val_file1.npy"],  # UPDATE THIS
        "val_files": ["B97_chr6_matrix_data-1_1x.npy"],  # UPDATE THIS
        "num_parents": 24,
        "window_size": 512,
        "step_size": 128,
        "batch_size": 8,
        "num_workers": 4,
        # "d_model": 128,
        "d_model": 64,
        "n_heads": 4,
        # "n_layers": 4,
        "n_layers": 2,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "gate_reg": 0.05,
        "max_epochs": 100,
        "patience": 10
    }

    # Datasets
    train_dataset = LabeledDatasetDiploid(
        config["data_dir"],
        config["train_files"],
        window_size=config["window_size"],
        num_parents=config["num_parents"],
        step_size=config["step_size"]
    )

    val_dataset = LabeledDatasetDiploid(
        config["data_dir"],
        config["val_files"],
        window_size=config["window_size"],
        num_parents=config["num_parents"],
        step_size=config["step_size"]
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # In your main() function, before training:
    print("Checking dataset...")
    sample = train_dataset[0]
    print(f"Input shape: {sample['input_embeds'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Label range: {sample['labels'].min()} to {sample['labels'].max()}")
    print(f"Unique labels: {sample['labels'].unique()}")

    # Check a batch
    sample_batch = next(iter(train_loader))
    print(f"Batch input shape: {sample_batch['input_embeds'].shape}")
    print(f"Batch labels shape: {sample_batch['labels'].shape}")
    print(f"Batch label range: {sample_batch['labels'].min()} to {sample_batch['labels'].max()}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # Model
    model = GRITSCRFModel(
        num_parents=config["num_parents"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        gate_reg=config["gate_reg"]
    )

    # Callbacks
    ckpt_dir = Path(config["workdir"]) / "checkpoints" / "legacy-diploid"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val_total_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="grits-crf-{epoch:02d}-{val_total_loss:.3f}"
    )

    early_stop_callback = EarlyStopping(
        monitor="val/total_loss",
        mode="min",
        patience=config["patience"],
        verbose=True
    )

    # Logger
    logger = TensorBoardLogger("logs", name="grits_crf")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator="auto",
        devices="auto",
        precision=16,  # Mixed precision for speed
        gradient_clip_val=1.0
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()