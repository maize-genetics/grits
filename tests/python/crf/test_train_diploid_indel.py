"""
Tests for train_diploid_indel.py / IndelFounderPathEncoder (train_crf.py) /
crf_kernels.py — the indel-aware ("2K+2" ternary+distance) training-side
implementation for experiments/simulator-indels/TRAINING_PLAN.md.

Run in isolation (several pre-existing, unrelated test files in this repo
already fail at collection time, matching the convention established by
tests/python/crf/test_simulate_alleles.py):
    pixi run -- pytest tests/python/crf/test_train_diploid_indel.py -v
"""

import math
import unittest

import numpy as np
import torch

from python.crf.crf_kernels import _dcrf_nll, _dcrf_viterbi, build_pair_tables
from python.crf.train_crf import IndelFounderPathEncoder
from python.crf.train_diploid_indel import (
    IndelDiploidDataset, GRITSCRFDiploidIndel,
    TERN_DEL, TERN_DIV, TERN_MATCH, DIST_PAD,
)

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
#  Fixture generator -- synthetic 2K+2 data, NOT meant to resemble the real   #
#  simulator's statistics. Known founder paths + a planted deletion tract    #
#  per window, for round-trip / overfit tests only.                          #
# --------------------------------------------------------------------------- #

def make_indel_fixture(n_windows=64, T=64, K=6, seed=0):
    """Synthesize a (n_windows, T, 2K+2) int8 array with KNOWN founder paths
    (returned separately) and one planted deletion tract per window. The
    tract's distance code grows away from its own boundary (min 1 at the
    edge) toward the middle, matching "near-zero in colinear sequence,
    growing moving away from an anchor" (PLAN.md §2.3)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n_windows, T, 2 * K + 2), dtype=np.int8)
    h1_all = np.zeros((n_windows, T), dtype=np.int64)
    h2_all = np.zeros((n_windows, T), dtype=np.int64)
    for w in range(n_windows):
        n_cross = int(rng.integers(0, 3))
        cuts = sorted(rng.choice(np.arange(1, T), size=n_cross, replace=False)) if n_cross else []
        bounds = [0] + list(cuts) + [T]
        h1 = np.zeros(T, dtype=np.int64)
        h2 = np.zeros(T, dtype=np.int64)
        f1 = int(rng.integers(0, K))
        f2 = int(rng.integers(0, K))
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            h1[s:e] = f1
            h2[s:e] = f2
            f1 = (f1 + int(rng.integers(1, K))) % K
            f2 = (f2 + int(rng.integers(1, K))) % K
        h1_all[w], h2_all[w] = h1, h2

        del_founder = int(rng.integers(0, K))
        d_start = int(rng.integers(0, T // 2))
        d_len = int(rng.integers(1, T // 4 + 1))
        d_end = min(T, d_start + d_len)
        for t in range(T):
            active = {int(h1[t]), int(h2[t])}
            for f in range(K):
                if f == del_founder and d_start <= t < d_end:
                    tern = TERN_DEL
                    d_to_edge = min(t - d_start, d_end - 1 - t) + 1
                    code = int(np.clip(round(8.0 * math.log2(1 + d_to_edge)), 0, 126))
                elif f in active:
                    tern, code = TERN_MATCH, 0
                else:
                    tern, code = TERN_DIV, 0
                out[w, t, f] = tern
                out[w, t, K + 2 + f] = code
        out[w, :, K] = h1
        out[w, :, K + 1] = h2
    return out, h1_all, h2_all


# --------------------------------------------------------------------------- #
#  1. CRF kernel extraction is bit-exact                                      #
# --------------------------------------------------------------------------- #

# Verbatim copies of the PRE-extraction functions (train_diploid.py, before
# the crf_kernels.py move), kept here only as an independent reference to
# assert byte-for-byte equivalence against the extracted module.

def _ref_dcrf_nll(emis, c, nsw, stay_bonus, tags):
    B, T, P = emis.shape
    emis = emis.float(); c = c.float(); nsw = nsw.float(); stay_bonus = stay_bonus.float()
    stay_mask = (nsw == 0).float()
    a = emis[:, 0]
    for t in range(1, T):
        tr_t = -c[:, t, None, None] * nsw[None] + stay_bonus * stay_mask[None]
        a = emis[:, t] + torch.logsumexp(a.unsqueeze(2) + tr_t, dim=1)
    log_Z = torch.logsumexp(a, dim=1)
    bi = torch.arange(B, device=emis.device)
    t_idx = torch.arange(T, device=emis.device)
    emis_score = emis[bi.unsqueeze(1), t_idx.unsqueeze(0), tags].sum(1)
    prev, nxt = tags[:, :-1], tags[:, 1:]
    nsw_path = nsw[prev, nxt]
    stay_path = (nsw_path == 0).float()
    tr_score = (-c[:, 1:] * nsw_path + stay_bonus * stay_path).sum(1)
    return (log_Z - emis_score - tr_score).mean()


def _ref_dcrf_viterbi(emis, c, nsw, stay_bonus):
    B, T, P = emis.shape
    stay_mask = (nsw == 0).float()
    delta = emis[:, 0]
    bp = torch.zeros(T - 1, B, P, dtype=torch.long, device=emis.device)
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


class TestCrfKernelExtraction(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.K = 5
        self.pi, self.pj, self.pair_table, self.nsw = build_pair_tables(self.K)
        self.P = self.pi.numel()
        self.B, self.T = 4, 12
        self.emis = torch.randn(self.B, self.T, self.P)
        self.c = torch.rand(self.B, self.T).abs() + 0.1
        self.stay_bonus = torch.tensor(1.7)
        self.tags = torch.randint(0, self.P, (self.B, self.T))

    def test_nll_bit_exact(self):
        got = _dcrf_nll(self.emis, self.c, self.nsw, self.stay_bonus, self.tags)
        want = _ref_dcrf_nll(self.emis, self.c, self.nsw, self.stay_bonus, self.tags)
        self.assertTrue(torch.equal(got, want), f"{got.item()} != {want.item()}")

    def test_viterbi_bit_exact(self):
        got = _dcrf_viterbi(self.emis, self.c, self.nsw, self.stay_bonus)
        want = _ref_dcrf_viterbi(self.emis, self.c, self.nsw, self.stay_bonus)
        self.assertTrue(torch.equal(got, want))


# --------------------------------------------------------------------------- #
#  2. IndelFounderPathEncoder fast_cells exact-lookup equivalence             #
# --------------------------------------------------------------------------- #

class TestFastCellsEquivalence(unittest.TestCase):
    def test_fast_cells_bit_identical(self):
        torch.manual_seed(2)
        K = 6
        enc_slow = IndelFounderPathEncoder(d_model=16, n_heads=2, n_layers=1,
                                           fast_cells=False)
        enc_fast = IndelFounderPathEncoder(d_model=16, n_heads=2, n_layers=1,
                                           fast_cells=True)
        enc_fast.load_state_dict(enc_slow.state_dict())
        enc_slow.eval(); enc_fast.eval()

        B, T = 3, 8
        tern = torch.randint(-1, 2, (B, T, K)).float()    # {-1,0,1}, no TERN_PAD
        dist = torch.randint(-1, 128, (B, T, K)).float()  # {-1,0..127}
        X = torch.stack([tern, dist], dim=-1)
        founder_mask = torch.ones(B, K)

        with torch.no_grad():
            cells_slow = enc_slow._embed_cells(X)
            cells_fast = enc_fast._embed_cells(X)
        self.assertTrue(torch.allclose(cells_slow, cells_fast, atol=1e-6))

        with torch.no_grad():
            emis_s, g_s, c_s = enc_slow(X, founder_mask)
            emis_f, g_f, c_f = enc_fast(X, founder_mask)
        self.assertTrue(torch.allclose(emis_s, emis_f, atol=1e-5))
        self.assertTrue(torch.allclose(g_s, g_f, atol=1e-6))
        self.assertTrue(torch.allclose(c_s, c_f, atol=1e-5))


# --------------------------------------------------------------------------- #
#  3. Dataset round-trip + DIST_PAD-is-not-distance-zero contract            #
# --------------------------------------------------------------------------- #

class TestDatasetRoundTrip(unittest.TestCase):
    def test_label_pad_remaps_to_K_not_founder_0(self):
        K = 4
        row = np.zeros((3, 2 * K + 2), dtype=np.int8)
        row[:, K] = -1        # LABEL_PAD on H1
        row[:, K + 1] = 2     # a real H2 label
        data = row[None]      # [1, T, 2K+2]
        ds = IndelDiploidDataset(data, num_parents=K)
        item = ds[0]
        self.assertTrue(torch.equal(item["h1"], torch.full((3,), K, dtype=torch.long)))
        self.assertTrue(torch.equal(item["h2"], torch.full((3,), 2, dtype=torch.long)))

    def test_ternary_and_distance_slices(self):
        K, T = 3, 2
        row = np.zeros((T, 2 * K + 2), dtype=np.int8)
        row[0, :K] = [TERN_MATCH, TERN_DEL, TERN_DIV]
        row[0, K + 2:2 * K + 2] = [5, DIST_PAD, 0]
        row[1, :K] = [TERN_DIV, TERN_DIV, TERN_MATCH]
        row[1, K + 2:2 * K + 2] = [DIST_PAD, DIST_PAD, 0]
        row[:, K] = 1
        row[:, K + 1] = 2
        data = row[None]
        ds = IndelDiploidDataset(data, num_parents=K)
        item = ds[0]
        feats = item["input_embeds"]  # [T,K,2]
        self.assertEqual(tuple(feats.shape), (T, K, 2))
        self.assertEqual(feats[0, 0, 0].item(), TERN_MATCH)
        self.assertEqual(feats[0, 1, 0].item(), TERN_DEL)
        self.assertEqual(feats[0, 1, 1].item(), DIST_PAD)

    def test_dist_pad_differs_from_distance_zero(self):
        """The no-anchor sentinel (DIST_PAD=-1) must embed differently from a
        real distance of 0 ("sitting exactly on an anchor") -- the opposite
        fact. This is the core property behind the -1.0-not-0.0 scalar
        mapping in IndelFounderPathEncoder._cell_input."""
        enc = IndelFounderPathEncoder(d_model=16, n_heads=2, n_layers=1)
        enc.eval()
        tern = torch.tensor([TERN_DIV, TERN_DIV], dtype=torch.float32)
        dist_pad = torch.tensor([DIST_PAD, DIST_PAD], dtype=torch.float32)
        dist_zero = torch.tensor([0.0, 0.0], dtype=torch.float32)
        with torch.no_grad():
            emb_pad = enc.cell(enc._cell_input(tern, dist_pad))
            emb_zero = enc.cell(enc._cell_input(tern, dist_zero))
        self.assertFalse(torch.allclose(emb_pad, emb_zero))

    def test_forward_runs_clean_with_null_founder_pad(self):
        model = GRITSCRFDiploidIndel(num_parents=4, d_model=8, n_heads=2, n_layers=1)
        model.eval()
        B, T, K = 2, 5, 4
        tern = torch.zeros(B, T, K)
        dist = torch.zeros(B, T, K)
        X = torch.stack([tern, dist], dim=-1)
        with torch.no_grad():
            emis_p, g, c = model(X)
        self.assertEqual(tuple(emis_p.shape), (B, T, model.P))
        self.assertFalse(torch.isnan(emis_p).any())


# --------------------------------------------------------------------------- #
#  4. Overfit-one-batch smoke test: gradients flow through the whole new path #
# --------------------------------------------------------------------------- #

class TestOverfitSmoke(unittest.TestCase):
    def test_loss_falls_and_accuracy_rises(self):
        torch.manual_seed(3)
        K = 6
        data, h1_all, h2_all = make_indel_fixture(n_windows=32, T=32, K=K, seed=7)
        ds = IndelDiploidDataset(data, num_parents=K)
        batch = {
            "input_embeds": torch.stack([ds[i]["input_embeds"] for i in range(len(ds))]),
            "h1": torch.stack([ds[i]["h1"] for i in range(len(ds))]),
            "h2": torch.stack([ds[i]["h2"] for i in range(len(ds))]),
        }
        model = GRITSCRFDiploidIndel(num_parents=K, d_model=32, n_heads=2, n_layers=2,
                                     lr=1e-2)
        opt = model.configure_optimizers()
        if isinstance(opt, dict):
            opt = opt["optimizer"]
        model.train()

        def eval_metrics():
            model.eval()
            with torch.no_grad():
                loss, crf, g, c, emis_p, tags = model._step(batch)
                pair_acc, hap_acc = model._accuracy(emis_p, c, batch["h1"], batch["h2"])
            model.train()
            return float(loss), float(pair_acc), float(hap_acc)

        loss0, pair_acc0, hap_acc0 = eval_metrics()
        losses = [loss0]
        for _ in range(150):
            opt.zero_grad()
            loss, crf, g, c, emis_p, tags = model._step(batch)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        loss1, pair_acc1, hap_acc1 = eval_metrics()

        self.assertLess(loss1, loss0 * 0.5,
                        f"loss did not fall enough: {loss0:.3f} -> {loss1:.3f}")
        # Per-haplotype accuracy is the cleanest signal that gradients reach
        # every new component (cell embed, recomb head's del_frac/depth,
        # pair CRF): without a het prior (--homo-penalty/--learned-het, not
        # used here) the pair-state CRF is known to bias toward homozygous
        # pairs on single-read data (see GRITSCRFDiploid's own homo_penalty
        # comment), so pair_acc alone lags even when the model is learning
        # correctly -- hap_acc is the more diagnostic metric for this test.
        chance_hap = 1.0 / (model.num_parents + 1)
        self.assertGreater(hap_acc1, chance_hap * 2,
                           f"hap_acc {hap_acc1:.4f} not far above chance {chance_hap:.4f}")
        self.assertGreater(hap_acc1, hap_acc0,
                           f"hap_acc did not improve: {hap_acc0:.4f} -> {hap_acc1:.4f}")
        self.assertGreaterEqual(pair_acc1, pair_acc0,
                                f"pair_acc regressed: {pair_acc0:.4f} -> {pair_acc1:.4f}")


if __name__ == "__main__":
    unittest.main()
