import os
import unittest
from unittest import mock

import numpy as np
import torch

import python.crf.train_diploid_indel as tdi
from python.crf.train_diploid_indel import (GRITSCRFDiploidIndel, infer_real_founder_pairs, peel_decode,
                                            peel_for_sample, phase_pair_path)

WORK = os.environ.get("GRITS_WORKDIR", "/nonexistent-GRITS_WORKDIR-unset")
CKPT = f"{WORK}/indel_baseline/checkpoints/v6cfullfe-stage2B/last.ckpt"
ROWS = f"{WORK}/scratch/editmatch/model_eval_E/rows_0.1"


def _model(K=4):
    torch.manual_seed(0)
    m = GRITSCRFDiploidIndel(num_parents=K, d_model=16, n_heads=2, n_layers=1, emission="likelihood_dist",
                             pair_emission="mixture", time_local_emis=True, founder_affinity=True,
                             tie_aware_loss=True, supervised_heads="stage2")
    return m.eval()


def _data(K=4, N=4, T=16, step=2000, seed=1):
    rng = np.random.default_rng(seed)
    tern = rng.integers(-1, 2, (N, T, K)).astype(np.float32)
    dist = rng.integers(-1, 90, (N, T, K)).astype(np.float32)
    count = rng.integers(0, 5, (N, T, K)).astype(np.float32)
    data = np.concatenate([tern, np.zeros((N, T, 2), np.float32), dist, count], -1)
    pos = np.arange(N * T, dtype=np.int64) * step
    contig = (np.arange(N * T) >= N * T // 2).astype(np.int64)
    return data, (pos + (contig << 40)).reshape(N, T)


def _route(hybrid, in_panel=False):
    real = tdi.route_real_sample

    def f(M, row_bin, informative_max=12):
        r = real(M, row_bin, informative_max)
        r["hybrid"], r["in_panel"] = hybrid, in_panel
        return r
    return mock.patch.object(tdi, "route_real_sample", f)


class TestPeelParts(unittest.TestCase):
    def test_phase(self):
        lo = torch.tensor([[0, 0, 1, 1, 2]])
        hi = torch.tensor([[3, 3, 3, 3, 3]])      # (0,3) (0,3) (1,3) (1,3) (2,3): track b = 3 throughout
        a, b = phase_pair_path(lo, hi)
        self.assertEqual(b.tolist(), [[3] * 5])
        self.assertEqual(a.tolist(), [[0, 0, 1, 1, 2]])
        lo, hi = torch.tensor([[1, 2, 2]]), torch.tensor([[2, 3, 3]])   # (1,2) -> (2,3): 2 continues track b
        a, b = phase_pair_path(lo, hi)
        self.assertEqual(b.tolist(), [[2, 2, 2]])
        self.assertEqual(a.tolist(), [[1, 3, 3]])

    def test_levels(self):
        _, rb = _data(step=2000)                       # ~516 rows/Mb
        self.assertEqual(peel_for_sample(dict(beta=0.5), rb), (0.5, 1.0, 1, False))
        self.assertEqual(peel_for_sample(dict(levels=[(1000, 0.3, 1.5, 2, True)]), rb), (0.3, 1.5, 2, True))
        self.assertIsNone(peel_for_sample(dict(levels=[(10, 0.3, 1.0, 1)]), rb))


class TestPeelSynthetic(unittest.TestCase):
    """True pair (0,1). Founder 2 is a close relative of founder 0 (it MATCHes 95% of haplotype-0 reads and
    none of haplotype 1's); haplotype 1 yields only a quarter of the rows, and homozygous pairs pay a small het prior. The 50/50 pair mixture credits (0,2)
    with ~log 2 on every haplotype-0 row, which outweighs the haplotype-1 rows only founder 1 explains, so the
    pair decode calls (0,2): haplotype 0 right, haplotype 1 wrong. Peel fixes founder 0 (it has unique support,
    2 has none), down-weights the rows 0 explains and re-decodes the other haplotype: (0,1)."""

    def setUp(self):
        K = 4
        self.m = m = _model(K)
        T = 200
        tern = torch.zeros(1, T, K)
        hap1 = torch.arange(T) % 4 == 0
        tern[0, ~hap1, 0] = 1
        tern[0, hap1, 1] = 1
        tern[0, ~hap1 & (torch.arange(T) % 20 != 1), 2] = 1
        ll = torch.where(tern == 1, torch.tensor(0.5), torch.tensor(-1.0))
        ll = torch.cat([ll, torch.full((1, T, 1), -1.0)], -1)               # null founder: explains nothing
        self.emis = torch.logaddexp(ll[..., m.pi], ll[..., m.pj]) - np.log(2.0)
        self.emis = self.emis - 0.1 * (m.pi == m.pj).float()               # a hybrid's per-row het prior
        P = m.pi.numel()
        m._trans_prior, m._init_prior, m._seg_prior = torch.zeros(1, P, P), torch.zeros(1, P), None
        self.c = torch.full((1, T), 3.0)
        self.tern = tern

    def test_pair_decode_wrong_peel_right(self):
        m = self.m
        p1 = m.crf_decode(self.emis, self.c)
        self.assertTrue(torch.all(p1 == int(m.pair_table[0, 2])))
        p2 = peel_decode(m, self.emis, self.c, self.tern, p1, beta=0.0, gamma=1.0, iters=1)
        self.assertTrue(torch.all(p2 == int(m.pair_table[0, 1])))
        # a second (alternating) pass fixes 1 and re-decodes haplotype 0: 0 still wins
        p3 = peel_decode(m, self.emis, self.c, self.tern, p1, beta=0.0, gamma=1.0, iters=2)
        self.assertTrue(torch.all(p3 == int(m.pair_table[0, 1])))

    def test_unit_weights_reproduce_pass1(self):
        p1 = self.m.crf_decode(self.emis, self.c)
        for it in (1, 2, 3):
            p2 = peel_decode(self.m, self.emis, self.c, self.tern, p1, beta=1.0, gamma=1.0, iters=it)
            self.assertTrue(torch.equal(p1, p2))


class TestPeelInference(unittest.TestCase):
    def test_off_unit_and_hybrid_only(self):
        m, (data, rb) = _model(), _data()
        kw = dict(device="cpu", route=True, row_bin=rb)
        with _route(True):
            off = infer_real_founder_pairs(m, data, 4, **kw)
            unit = infer_real_founder_pairs(m, data, 4, peel=dict(beta=1.0, gamma=1.0, iters=2), **kw)
        for x, y in zip(off, unit):
            np.testing.assert_array_equal(x, y)
        calls = []
        real = tdi.peel_decode

        def spy(*a, **k):
            calls.append(1)
            return real(*a, **k)
        with mock.patch.object(tdi, "peel_decode", spy):
            with _route(False):
                infer_real_founder_pairs(m, data, 4, peel=dict(beta=0.2), **kw)
            self.assertEqual(len(calls), 0)
            with _route(True, in_panel=True):
                infer_real_founder_pairs(m, data, 4, peel=dict(beta=0.2, heldout_only=True), **kw)
            self.assertEqual(len(calls), 0)
            with _route(True):
                infer_real_founder_pairs(m, data, 4, peel=dict(beta=0.2), **kw)
            self.assertGreater(len(calls), 0)

    def test_needs_route(self):
        m, (data, rb) = _model(), _data()
        with self.assertRaises(ValueError):
            infer_real_founder_pairs(m, data, 4, device="cpu", row_bin=rb, peel=dict(beta=0.5))


@unittest.skipUnless(os.path.exists(CKPT) and os.path.isdir(ROWS), "emE checkpoint / rows not available")
class TestPeelRealRow(unittest.TestCase):
    """emE checkpoint, first 24 windows of a real OUT-HYB 0.1x row (promoted preset placedfix_hetdel_hs):
    off == beta=gamma=1 bit-identically; a real setting changes the hybrid and leaves a homozygous row alone."""

    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../experiments/simval-corpus/scripts"))
        import heldout_assembly_eval as hae
        cls.m = GRITSCRFDiploidIndel.load_from_checkpoint(CKPT, map_location="cpu", strict=False).eval()
        cls.rows = {}
        for name in ("eval__OUT-HYB__Tx303xA188__0.1x", "eval__OUT-INBRED__Tx303__0.1x"):
            d = f"{ROWS}/{name}"
            data = np.load(f"{d}/windowed_k25native_wcount.npy", mmap_mode="r")
            layout = hae.load_contig_layout(f"{d}/raw.npy.bins.tsv", data.shape[1], bin_size=256)
            rb = np.concatenate([np.asarray(pos, np.int64) + (ci << 40) for ci, (_c, pos, _n) in enumerate(layout)]
                                ).reshape(data.shape[0], data.shape[1])
            cls.rows[name] = (np.ascontiguousarray(data[:24]), rb[:24])

    def _run(self, name, **kw):
        data, rb = self.rows[name]
        return infer_real_founder_pairs(self.m, data, 25, device="cpu", preset="placedfix_hetdel_hs", route=True,
                                        row_bin=rb, knockout="homoroute", **kw)

    def test_real_hybrid_off_identical(self):
        off = self._run("eval__OUT-HYB__Tx303xA188__0.1x")
        unit = self._run("eval__OUT-HYB__Tx303xA188__0.1x", peel=dict(beta=1.0, gamma=1.0, iters=2))
        for x, y in zip(off, unit):
            np.testing.assert_array_equal(x, y)
        on = self._run("eval__OUT-HYB__Tx303xA188__0.1x", peel=dict(beta=0.0, gamma=1.0, iters=1))
        self.assertTrue(any((x != y).any() for x, y in zip(off, on)))

    def test_real_homozygous_untouched(self):
        off = self._run("eval__OUT-INBRED__Tx303__0.1x")
        on = self._run("eval__OUT-INBRED__Tx303__0.1x", peel=dict(beta=0.0, gamma=2.0, iters=2))
        for x, y in zip(off, on):
            np.testing.assert_array_equal(x, y)


if __name__ == "__main__":
    unittest.main()
