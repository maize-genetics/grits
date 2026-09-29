import math
import unittest

import numpy as np
import torch

from python.crf.crf_kernels import _dcrf_viterbi
from python.crf.train_diploid_indel import (GRITSCRFDiploidIndel, _dcrf_nll_tied, _new_founder_tables,
                                            crf_nll_prior, crf_viterbi_prior,
                                            individual_affinity_target)


def _model(K=5, **kw):
    torch.manual_seed(0)
    kw.setdefault("emission", "likelihood_dist")
    kw.setdefault("pair_emission", "mixture")
    m = GRITSCRFDiploidIndel(num_parents=K, d_model=16, n_heads=2, n_layers=1,
                             time_local_emis=True, founder_affinity=True, tie_aware_loss=True, **kw)
    return m.eval()


def _batch(K=5, B=2, T=9, seed=1):
    g = torch.Generator().manual_seed(seed)
    tern = torch.randint(-1, 2, (B, T, K), generator=g).float()
    dist = torch.randint(-1, 90, (B, T, K), generator=g).float()
    return {"input_embeds": torch.stack([tern, dist], -1),
            "h1": torch.randint(0, K, (B, T), generator=g),
            "h2": torch.randint(0, K, (B, T), generator=g),
            "ext_emb": torch.randn(B, K, 2, generator=g),
            "lin": torch.randint(0, 3, (B, T, K), generator=g),
            "prov": torch.randint(-1, 30, (B, T), generator=g),
            "aff_target": torch.rand(B, K, generator=g)}


class TestOffUnchanged(unittest.TestCase):
    def test_default_init_stream_and_outputs_unchanged(self):
        a, b = _model(), _model(supervised_heads="off")
        for (n1, p1), (n2, p2) in zip(a.state_dict().items(), b.state_dict().items()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.equal(p1, p2))
        h = _model(supervised_heads="stage1")
        sd = h.state_dict()
        for n, p in a.state_dict().items():      # heads model: same init for shared params
            if n != "stay_bonus":
                self.assertTrue(torch.equal(p, sd[n]), n)


class TestKernels(unittest.TestCase):
    def test_zero_prior_matches_existing_kernels(self):
        m = _model()
        B, T = 3, 8
        torch.manual_seed(2)
        emis = torch.randn(B, T, m.P)
        c = torch.rand(B, T) * 3
        allowed = torch.rand(B, T, m.P) < 0.3
        allowed[..., 0] = True
        zt, zi = torch.zeros(B, m.P, m.P), torch.zeros(B, m.P)
        sb = torch.tensor(1.3)
        self.assertAlmostEqual(float(crf_nll_prior(emis, c, m.nsw_pair, sb, allowed, zt, zi)),
                               float(_dcrf_nll_tied(emis, c, m.nsw_pair, sb, allowed)), places=4)
        self.assertTrue(torch.equal(crf_viterbi_prior(emis, c, m.nsw_pair, sb, zt, zi),
                                    _dcrf_viterbi(emis, c, m.nsw_pair, sb)))

    def test_new_founder_tables(self):
        m = _model(K=4)
        pairs = list(zip(m.pi.tolist(), m.pj.tolist()))
        idx = {p: i for i, p in enumerate(pairs)}
        n1, n2 = _new_founder_tables(m.pi, m.pj, m.nsw_pair)
        none = 5
        self.assertEqual((int(n1[idx[(0, 1)], idx[(0, 1)]]), int(n2[idx[(0, 1)], idx[(0, 1)]])), (none, none))
        self.assertEqual(int(n1[idx[(0, 1)], idx[(1, 2)]]), 2)
        self.assertEqual(int(n2[idx[(0, 1)], idx[(1, 2)]]), none)
        self.assertEqual(int(n1[idx[(0, 0)], idx[(0, 3)]]), 3)
        self.assertEqual(sorted([int(n1[idx[(0, 1)], idx[(2, 3)]]), int(n2[idx[(0, 1)], idx[(2, 3)]])]), [2, 3])
        self.assertEqual(sorted([int(n1[idx[(0, 0)], idx[(2, 2)]]), int(n2[idx[(0, 0)], idx[(2, 2)]])]), [2, 2])


class TestHeadsModel(unittest.TestCase):
    def test_simulator_derived_transition_values(self):
        m = _model(supervised_heads="stage1")
        b = _batch()
        B, K = 2, 5
        uniform = torch.full((B, K), 0.4)
        m(b["input_embeds"], None, b["ext_emb"], None, aff_pred=uniform)
        pairs = list(zip(m.pi.tolist(), m.pj.tolist()))
        idx = {p: i for i, p in enumerate(pairs)}
        tp = m._trans_prior
        # uniform affinity: entering founder x costs log(1/K); staying costs nothing
        self.assertAlmostEqual(float(tp[0, idx[(0, 1)], idx[(1, 2)]]), math.log(1 / K), places=5)
        self.assertAlmostEqual(float(tp[0, idx[(0, 1)], idx[(2, 3)]]), 2 * math.log(1 / K), places=5)
        self.assertEqual(float(tp[0, idx[(0, 1)], idx[(0, 1)]]), 0.0)
        self.assertEqual(float(m.stay_bonus), 0.0)
        # c = -logit(p) + log 2
        emis, g, c = m(b["input_embeds"], None, b["ext_emb"], None, aff_pred=uniform)
        self.assertTrue(torch.allclose(c, m._heads["c"].float() + math.log(2.0), atol=1e-6))

    def test_gate_mixture_emission(self):
        m = _model(supervised_heads="stage1")
        b = _batch()
        with torch.no_grad():
            m.het_w.fill_(0.0)
        emis, g, c = m(b["input_embeds"], None, b["ext_emb"], None)
        # a fully noisy row (g -> 0) carries no pair information
        X = b["input_embeds"]
        with torch.no_grad():
            m.encoder.gate_head.bias.fill_(-50.0)
            m.encoder.gate_head.weight.zero_()
        emis0, _, _ = m(X, None, b["ext_emb"], None)
        self.assertLess(float(emis0.abs().max()), 1e-5)

    def test_head_targets(self):
        m = _model(K=5, supervised_heads="stage1")
        lin = torch.tensor([[0, 0, 1, 2, 2]]).expand(3, 5).unsqueeze(0).clone()
        h1 = torch.tensor([[0, 1, 2]])
        h2 = torch.tensor([[1, 3, 5]])
        prov = torch.tensor([[0, 8 | 4, -1]])
        tg = m.head_targets(h1, h2, lin, prov)
        self.assertEqual(tg["het"][0].tolist(), [False, True, True])
        self.assertEqual(tg["het_mask"][0].tolist(), [True, True, False])
        self.assertEqual(tg["switch"][0].tolist(), [True, True])
        self.assertEqual(tg["gate"][0].tolist(), [True, False, False])
        self.assertEqual(tg["gate_mask"][0].tolist(), [True, True, False])

    def test_stage1_loss_trains_heads_only(self):
        m = _model(supervised_heads="stage1").train()
        b = _batch()
        loss, *_ = m._step(b)
        loss.backward()
        self.assertIsNotNone(m.sup_het_head.weight.grad)
        self.assertIsNotNone(m.sup_aff_head.weight.grad)
        self.assertIsNotNone(m.encoder.gate_head.weight.grad)
        self.assertIsNotNone(m.encoder.recomb_head.weight.grad)
        for n in ("stay_bonus", "het_w", "aff_w", "c_scale", "c_offset", "tern_dist_loglik"):
            self.assertIsNone(dict(m.named_parameters())[n].grad, n)

    def test_stage2_trains_scalars_only(self):
        m = _model(supervised_heads="stage2").train()
        train = {n for n, p in m.named_parameters() if p.requires_grad}
        self.assertEqual(train, {"tern_loglik", "tern_dist_loglik", "stay_bonus", "het_w", "aff_w",
                                 "c_scale", "c_offset"})
        b = _batch()
        b["aff_pred"] = torch.rand(2, 5)
        loss, *_ = m._step(b)
        loss.backward()
        self.assertIsNotNone(m.aff_w.grad)
        self.assertIsNotNone(m.c_offset.grad)


class TestAffinityTarget(unittest.TestCase):
    def test_fraction_of_rows(self):
        K, G, T = 4, 2, 3
        data = np.zeros((G, T, 2 * K + 2), dtype=np.int8)
        data[..., K] = 0
        data[..., K + 1] = 2
        data[1, 2, K] = -1                              # one unlabelled row
        lin = np.tile(np.array([0, 0, 1, 2], dtype=np.int8), (G, T, 1))
        lin[0, 0] = [0, 3, 1, 2]                        # founder 1 leaves both true lineages at one row
        a = individual_affinity_target(data, lin, K, G)
        np.testing.assert_allclose(a[0], [1.0, 4 / 5, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
