import os
import unittest
from unittest import mock

import numpy as np
import torch

import python.crf.train_diploid_indel as tdi
from python.crf.crf_kernels import build_pair_tables
from python.crf.train_diploid_indel import (GRITSCRFDiploidIndel, crf_viterbi_prior, hap_switch_for_sample,
                                            hap_switch_nsw, infer_real_founder_pairs)

WORK = os.environ.get("GRITS_WORKDIR", "/nonexistent-GRITS_WORKDIR-unset")
CKPT = f"{WORK}/indel_baseline/checkpoints/v6cfullfe-stage2B/last.ckpt"
ROWS = f"{WORK}/scratch/editmatch/model_eval_E/rows_0.1"


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


def _route(hybrid):
    real = tdi.route_real_sample

    def f(M, row_bin, informative_max=12):
        r = real(M, row_bin, informative_max)
        r["hybrid"] = hybrid
        return r
    return mock.patch.object(tdi, "route_real_sample", f)


class TestHapSwitchNsw(unittest.TestCase):
    def test_values(self):
        _, _, _, nsw = build_pair_tables(4)
        self.assertIs(hap_switch_nsw(nsw, 1, 1), nsw)
        o = hap_switch_nsw(nsw, 0.5, 3.0)
        self.assertTrue(torch.equal(o[nsw == 0], torch.zeros_like(o[nsw == 0])))
        self.assertTrue(torch.all(o[nsw == 1] == 0.5))
        self.assertTrue(torch.all(o[nsw == 2] == 6.0))
        with self.assertRaises(ValueError):
            hap_switch_nsw(nsw, 0.0, 1.0)

    def test_levels(self):
        _, rb = _data(step=2000)                       # ~516 rows/Mb
        hs = dict(levels=[(100, 0.1, 0.1), (1000, 0.5, 2.0)])
        self.assertEqual(hap_switch_for_sample(hs, rb), (0.5, 2.0, False))
        self.assertEqual(hap_switch_for_sample(dict(levels=[(1000, 0.6, 0.6, True)]), rb), (0.6, 0.6, True))
        self.assertEqual(hap_switch_for_sample(dict(levels=[(10, 0.1, 0.1)]), rb), (1.0, 1.0, False))
        self.assertEqual(hap_switch_for_sample(dict(one=0.3)), (0.3, 1.0, False))


class TestHapSwitchSynthetic(unittest.TestCase):
    def test_one_haplotype_switch_recovered(self):
        """True pair (0,1) then (0,2): one haplotype switches 1 -> 2 at t=30. The reads favour founder 2 over
        1 by 0.3 nats per row after the switch, 30 rows = 9 nats < the default switch cost c=12, so the
        trained pricing stays on (0,1); a cheaper one-haplotype switch recovers (0,2). Making two-haplotype
        switches dearer must not matter here (the true change is a single switch)."""
        K = 4
        pi, pj, pt, nsw = build_pair_tables(K)
        P = pi.numel()
        T = 60
        emis = torch.zeros(1, T, P)
        emis[0, :, int(pt[0, 1])] += 0.3
        emis[0, 30:, int(pt[0, 1])] -= 0.3
        emis[0, 30:, int(pt[0, 2])] += 0.3
        c = torch.full((1, T), 12.0)
        tp = torch.zeros(1, P, P)
        ip = torch.zeros(1, P)
        truth = torch.tensor([int(pt[0, 1])] * 30 + [int(pt[0, 2])] * 30)
        base = crf_viterbi_prior(emis, c, nsw, torch.tensor(0.0), tp, ip)[0]
        self.assertTrue(torch.all(base == int(pt[0, 1])))
        for two in (1.0, 5.0):
            hs = crf_viterbi_prior(emis, c, hap_switch_nsw(nsw, 0.5, two), torch.tensor(0.0), tp, ip)[0]
            self.assertTrue(torch.equal(hs, truth), two)

    def test_double_switch_priced_separately(self):
        """(0,1) -> (2,3) both haplotypes switch: the trained pricing jumps directly (one nsw=2 step); with a
        dear double switch the decoder gets there via a one-shared pair (two nsw=1 steps), never a direct jump."""
        K = 4
        pi, pj, pt, nsw = build_pair_tables(K)
        P = pi.numel()
        T = 40
        emis = torch.zeros(1, T, P)
        emis[0, :20, int(pt[0, 1])] = 0.5
        emis[0, 20:, int(pt[2, 3])] = 0.5
        c = torch.full((1, T), 3.0)
        z = torch.zeros(1, P, P), torch.zeros(1, P)
        free = crf_viterbi_prior(emis, c, nsw, torch.tensor(0.0), *z)[0]
        self.assertEqual(int(free[25]), int(pt[2, 3]))
        dear = crf_viterbi_prior(emis, c, hap_switch_nsw(nsw, 1.0, 10.0), torch.tensor(0.0), *z)[0]
        steps = lambda path: nsw[path[:-1], path[1:]]                                   # noqa: E731
        self.assertEqual(int((steps(free) == 2).sum()), 1)
        self.assertEqual(int((steps(dear) == 2).sum()), 0)
        self.assertEqual(int((steps(dear) == 1).sum()), 2)
        self.assertEqual(int(dear[-1]), int(pt[2, 3]))


class TestHapSwitchInference(unittest.TestCase):
    def test_off_and_unit_identical_and_hybrid_only(self):
        m, (data, rb) = _model(), _data()
        seen = []
        real = tdi.crf_viterbi_prior

        def spy(emis, c, nsw, *a, **k):
            seen.append(nsw)
            return real(emis, c, nsw, *a, **k)
        with _route(True), mock.patch.object(tdi, "crf_viterbi_prior", spy):
            off = infer_real_founder_pairs(m, data, 5, device="cpu", route=True, row_bin=rb)
            unit = infer_real_founder_pairs(m, data, 5, device="cpu", route=True, row_bin=rb,
                                            hap_switch=dict(one=1.0, two=1.0))
            on = infer_real_founder_pairs(m, data, 5, device="cpu", route=True, row_bin=rb,
                                          hap_switch=dict(one=0.01, two=0.01))
        self.assertIsNone(getattr(m, "_nsw_dec", None))
        for x, y in zip(off, unit):
            np.testing.assert_array_equal(x, y)
        self.assertTrue(all(x is m.nsw_pair for x in seen[:2]))
        self.assertTrue(torch.all(seen[-1][m.nsw_pair == 1] == 0.01))
        seen.clear()
        with _route(False), mock.patch.object(tdi, "crf_viterbi_prior", spy):
            off_h = infer_real_founder_pairs(m, data, 5, device="cpu", route=True, row_bin=rb)
            on_h = infer_real_founder_pairs(m, data, 5, device="cpu", route=True, row_bin=rb,
                                            hap_switch=dict(one=0.01, two=0.01))
        for x, y in zip(off_h, on_h):
            np.testing.assert_array_equal(x, y)
        self.assertTrue(all(x is m.nsw_pair for x in seen))

    def test_heldout_only_skips_in_panel_hybrids(self):
        m, (data, rb) = _model(), _data()
        real = tdi.route_real_sample
        for in_panel, expect_on in ((True, False), (False, True)):
            def f(M, row_bin, informative_max=12):
                r = real(M, row_bin, informative_max)
                r["hybrid"], r["in_panel"] = True, in_panel
                return r
            seen = []
            rv = tdi.crf_viterbi_prior

            def spy(emis, c, nsw, *a, **k):
                seen.append(nsw)
                return rv(emis, c, nsw, *a, **k)
            with mock.patch.object(tdi, "route_real_sample", f), mock.patch.object(tdi, "crf_viterbi_prior", spy):
                infer_real_founder_pairs(m, data, 5, device="cpu", route=True, row_bin=rb,
                                         hap_switch=dict(one=0.5, two=0.5, heldout_only=True))
            self.assertEqual(seen[0] is not m.nsw_pair, expect_on)

    def test_needs_route(self):
        m, (data, rb) = _model(), _data()
        with self.assertRaises(ValueError):
            infer_real_founder_pairs(m, data, 5, device="cpu", row_bin=rb, hap_switch=dict(one=0.5))


@unittest.skipUnless(os.path.exists(CKPT) and os.path.isdir(ROWS), "emE checkpoint / rows not available")
class TestHapSwitchRealRow(unittest.TestCase):
    """emE checkpoint on the first 24 windows of a real OUT-HYB 0.1x row: off == one=two=1 bit-identically,
    and the option changes nothing on a homozygous-routed row."""

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
        return infer_real_founder_pairs(self.m, data, 25, device="cpu", preset="placedfix_hetdel", route=True,
                                        row_bin=rb, knockout="homoroute", **kw)

    def test_real_hybrid_off_identical(self):
        off = self._run("eval__OUT-HYB__Tx303xA188__0.1x")
        unit = self._run("eval__OUT-HYB__Tx303xA188__0.1x", hap_switch=dict(one=1.0, two=1.0))
        for x, y in zip(off, unit):
            np.testing.assert_array_equal(x, y)

    def test_real_homozygous_untouched(self):
        off = self._run("eval__OUT-INBRED__Tx303__0.1x")
        on = self._run("eval__OUT-INBRED__Tx303__0.1x", hap_switch=dict(one=0.2, two=0.2))
        for x, y in zip(off, on):
            np.testing.assert_array_equal(x, y)


if __name__ == "__main__":
    unittest.main()
