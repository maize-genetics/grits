import unittest

import numpy as np
import torch

from python.crf.train_diploid_indel import (DEPTH_KNOBS, GRITSCRFDiploidIndel, infer_real_founder_pairs,
                                            rows_per_mb)


def _model(K=5):
    torch.manual_seed(0)
    m = GRITSCRFDiploidIndel(num_parents=K, d_model=16, n_heads=2, n_layers=1, emission="likelihood_dist",
                             pair_emission="mixture", time_local_emis=True, founder_affinity=True,
                             tie_aware_loss=True, supervised_heads="stage2")
    return m.eval()


def _data(K=5, N=4, T=16, step=2000, seed=1):
    rng = np.random.default_rng(seed)
    tern = rng.integers(-1, 2, (N, T, K)).astype(np.float32)
    dist = rng.integers(-1, 90, (N, T, K)).astype(np.float32)
    count = rng.integers(0, 5, (N, T, K)).astype(np.float32)
    data = np.concatenate([tern, np.zeros((N, T, 2), np.float32), dist, count], -1)
    pos = np.arange(N * T, dtype=np.int64) * step
    contig = (np.arange(N * T) >= N * T // 2).astype(np.int64)
    return data, (pos + (contig << 40)).reshape(N, T)


class TestDepthKnobs(unittest.TestCase):
    def test_rows_per_mb(self):
        _, rb = _data(step=2000)                  # 32 rows per contig, 31 * 2 kb span each
        self.assertAlmostEqual(rows_per_mb(rb), 64 / (2 * 31 * 2000 / 1e6))

    def test_none_identical_and_params_restored(self):
        m, (data, rb) = _model(), _data()
        before = {k: getattr(m, k).detach().clone() for k in ("tern_loglik", "aff_w")}
        a = infer_real_founder_pairs(m, data, 5, device="cpu", route=True, row_bin=rb)
        b = infer_real_founder_pairs(m, data, 5, device="cpu", route=True, row_bin=rb, depth_knobs="v1")
        c = infer_real_founder_pairs(m, data, 5, device="cpu", route=True, row_bin=rb)
        for x, y in zip(a, c):
            np.testing.assert_array_equal(x, y)
        for k, v in before.items():
            self.assertTrue(torch.equal(getattr(m, k), v), k)
        self.assertEqual(len(b), 2)

    def test_knobs_equal_manual_override(self):
        m, (data, rb) = _model(), _data()
        kn = dict(ll_del=-7.5, ll_div=-7.0, aff_ref=rows_per_mb(rb) / 4, aff_a=1.0)
        a = infer_real_founder_pairs(m, data, 5, device="cpu", route=True, row_bin=rb, depth_knobs=kn)
        with torch.no_grad():
            m.tern_loglik[0], m.tern_loglik[1] = -7.5, -7.0
            m.aff_w.mul_(4.0)
        b = infer_real_founder_pairs(m, data, 5, device="cpu", route=True, row_bin=rb)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_v1_preset_keys(self):
        self.assertEqual(set(DEPTH_KNOBS["v1"]) - {"ll_del", "ll_div", "ll_match", "aff_ref", "aff_a",
                                                   "aff_min", "aff_max"}, set())
