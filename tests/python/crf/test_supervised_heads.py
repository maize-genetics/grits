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
        # c = -logit(p_t) + log 2 with p_t = lambda_w / (T-1)
        emis, g, c = m(b["input_embeds"], None, b["ext_emb"], None, aff_pred=uniform)
        lam = m._heads["xo_lam"].float()
        p = lam / 8
        self.assertTrue(torch.allclose(c[:, 1:], (-torch.log(p) + torch.log1p(-p) + math.log(2.0))[:, None].expand(-1, 8),
                                       atol=1e-5))

    def test_switch_probs_sum_to_lambda(self):
        for place in (False, True):
            m = _model(supervised_heads="stage1", xo_placement=place)
            b = _batch()
            emis, g, c = m(b["input_embeds"], None, b["ext_emb"], None)
            p = torch.sigmoid(-(c[:, 1:] - math.log(2.0)))
            self.assertTrue(torch.allclose(p.sum(1), m._heads["xo_lam"].float(), rtol=1e-4))
            if place:   # placement follows the per-row switch head
                w = torch.sigmoid(-m._heads["c"].float()[:, 1:])
                self.assertTrue(torch.allclose(p / p.sum(1, keepdim=True), w / w.sum(1, keepdim=True), atol=1e-5))

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
        self.assertIsNotNone(m.sup_xo_head.weight.grad)
        for n in ("stay_bonus", "het_w", "aff_w", "c_scale", "c_offset", "log_xo_scale", "tern_dist_loglik"):
            self.assertIsNone(dict(m.named_parameters())[n].grad, n)

    def test_stage2_trains_scalars_only(self):
        m = _model(supervised_heads="stage2").train()
        train = {n for n, p in m.named_parameters() if p.requires_grad}
        self.assertEqual(train, {"tern_loglik", "tern_dist_loglik", "het_w", "aff_w", "log_xo_scale"})
        b = _batch()
        b["aff_pred"] = torch.rand(2, 5)
        loss, *_ = m._step(b)
        loss.backward()
        self.assertIsNotNone(m.aff_w.grad)
        self.assertIsNotNone(m.log_xo_scale.grad)

    def test_xo_scale_positive_and_old_checkpoints_convert(self):
        m = _model(supervised_heads="stage2")
        with torch.no_grad():
            m.log_xo_scale.fill_(-50.0)
        self.assertGreater(float(m.xo_scale), 0.0)
        sd = {k: v for k, v in m.state_dict().items() if k != "log_xo_scale"}
        sd["xo_scale"] = torch.tensor(2.5)
        m.load_state_dict(sd)
        self.assertAlmostEqual(float(m.xo_scale), 2.5, places=5)


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


class TestHetPriorModes(unittest.TestCase):
    def test_segment_prior_paid_once_per_segment(self):
        from python.crf.train_diploid_indel import crf_viterbi_prior, _prior_trans
        torch.manual_seed(0)
        m = _model(supervised_heads="stage2", het_prior="segment").eval()
        P = m.nsw_pair.shape[0]
        B, T = 2, 7
        emis = torch.randn(B, T, P)
        c = torch.full((B, T), 3.0)
        tp = torch.zeros(B, P, P)
        ip = torch.zeros(B, P)
        seg = torch.randn(B, T, P)
        path = crf_viterbi_prior(emis, c, m.nsw_pair, m.stay_bonus, tp, ip, seg)
        def score(pth):
            sc = emis[:, 0].gather(1, pth[:, :1]).squeeze(1) + seg[:, 0].gather(1, pth[:, :1]).squeeze(1)
            for t in range(1, T):
                tr = _prior_trans(c[:, t], m.nsw_pair.float(), m.stay_bonus.float(), tp, seg[:, t])
                sc = sc + tr[torch.arange(B), pth[:, t - 1], pth[:, t]] + emis[:, t].gather(1, pth[:, t:t + 1]).squeeze(1)
            return sc
        best = score(path)
        for _ in range(200):                                  # Viterbi beats random paths
            self.assertTrue((score(torch.randint(0, P, (B, T))) <= best + 1e-4).all())
        # a constant path pays seg only at row 0
        const = torch.zeros(B, T, dtype=torch.long)
        exp = emis[:, :, 0].sum(1) + seg[:, 0, 0] + T * 0 + (T - 1) * float(m.stay_bonus)
        self.assertTrue(torch.allclose(score(const), exp, atol=1e-4))

    def test_modes_change_emission(self):
        b = _batch()
        out = {}
        for mode in ("row", "segment", "off"):
            m = _model(supervised_heads="stage2", het_prior=mode).eval()
            torch.manual_seed(0)
            with torch.no_grad():
                emis, g, c = m(b["input_embeds"], None, b["ext_emb"], b.get("count"), aff_pred=torch.rand(2, 5))
            out[mode] = (emis, m._seg_prior)
        self.assertIsNone(out["row"][1])
        self.assertIsNone(out["off"][1])
        self.assertIsNotNone(out["segment"][1])
        self.assertTrue(torch.allclose(out["segment"][0], out["off"][0]))


class TestWindowAffinityTarget(unittest.TestCase):
    def test_window_targets_pool_to_individual(self):
        from python.crf.train_diploid_indel import individual_affinity_target
        rng = np.random.default_rng(0)
        G, T, K = 4, 16, 5
        data = np.zeros((2 * G, T, 2 * K + 2), np.int8)
        data[:, :, K:K + 2] = rng.integers(0, K, size=(2 * G, T, 2))
        lin = rng.integers(0, 3, size=(2 * G, T, K))
        ind = individual_affinity_target(data, lin, K, G)
        win = individual_affinity_target(data, lin, K, 1)
        self.assertEqual(win.shape, (2 * G, K))
        np.testing.assert_allclose(win.reshape(2, G, K).mean(1), ind, atol=1e-6)


class TestReadsAffinitySource(unittest.TestCase):
    def test_prior_from_read_match_rate(self):
        b = _batch()
        m = _model(supervised_heads="stage2", aff_source="reads").eval()
        with torch.no_grad():
            m(b["input_embeds"], None, b["ext_emb"], b.get("count"), aff_pred=torch.rand(2, 5))
        a = b["ext_emb"][:, :, 0].float().clamp(1e-3, 1.0)
        la = torch.log(a / a.sum(1, keepdim=True))
        exp = float(m.aff_w) * (la[:, m.pi.clamp(max=4)] + la[:, m.pj.clamp(max=4)])
        real = (m.pi < 5) & (m.pj < 5)
        self.assertTrue(torch.allclose(m._init_prior[:, real], exp[:, real], atol=1e-5))


class TestSupportGateTarget(unittest.TestCase):
    def test_support_label(self):
        m = _model(supervised_heads="stage1", gate_target="support")
        K = m.num_parents
        h1 = torch.tensor([[0, 0, 1]]); h2 = torch.tensor([[2, 2, 2]])
        lin = torch.arange(K).view(1, 1, K).expand(1, 3, K).clone()
        # row kinds: 0 hap1 collinear, 5 hap2 replacement off-site (5|8), 0 hap1 on a bad site (0|16)
        prov = torch.tensor([[0, 5 | 8, 0 | 16]])
        tern = torch.zeros(1, 3, K)
        tern[0, 0, 0] = 1            # row 0 matches hap1's founder 0
        tern[0, 1, 2] = 1            # row 1 (hap2) matches hap2's founder 2
        tern[0, 2, 3] = 1            # row 2 (hap1, founder 1) matches only founder 3
        tg = m.head_targets(h1, h2, lin, prov, tern=tern)
        self.assertEqual(tg["gate"].tolist(), [[True, True, False]])
        m.gate_target = "prov"
        self.assertEqual(m.head_targets(h1, h2, lin, prov)["gate"].tolist(), [[True, False, False]])


class TestExtraHeads(unittest.TestCase):
    def test_targets_loss_and_crf_neutral_at_zero_weight(self):
        b = _batch()
        m = _model(supervised_heads="stage1", extra_heads=True).train()
        B, T = b["h1"].shape
        b["prov"] = torch.tensor([[0, 1 | 32, 0 | 32 | 64] + [0] * (T - 3)] * B)
        b["ood"] = torch.zeros(B, T, dtype=torch.long); b["ood"][:, 1] = 2
        loss, *_ = m._step(b)
        self.assertIn("dos", m._head_parts); self.assertIn("ood", m._head_parts)
        loss.backward()
        self.assertIsNotNone(m.sup_dos_head.weight.grad)
        self.assertIsNotNone(m.sup_ood_head.weight.grad)
        tg = m.head_targets(b["h1"], b["h2"], b["lin"], b["prov"], tern=b["input_embeds"][..., 0], ood=b["ood"])
        self.assertEqual(tg["dos"][0, :3].tolist(), [0, 1, 2])
        self.assertTrue(bool(tg["ood"][0, 1]) and not bool(tg["ood"][0, 0]))
        # with dos_w = ood_w = 0 the CRF inputs equal the model without the extra heads
        m2 = _model(supervised_heads="stage2", extra_heads=True).eval()
        with torch.no_grad():
            m2.aff_ood_s.fill_(0.0)            # affinity scaling off -> identical to no extra heads
        m3 = _model(supervised_heads="stage2").eval()
        m3.load_state_dict({k: v for k, v in m2.state_dict().items() if k in m3.state_dict()})
        with torch.no_grad():
            e2, _, c2 = m2(b["input_embeds"], None, b["ext_emb"], b.get("count"), aff_pred=torch.rand(B, 5))
            e3, _, c3 = m3(b["input_embeds"], None, b["ext_emb"], b.get("count"), aff_pred=torch.rand(B, 5))
        self.assertTrue(torch.allclose(e2, e3, atol=1e-5)); self.assertTrue(torch.allclose(c2, c3, atol=1e-5))


class TestOodAffinityScaling(unittest.TestCase):
    def test_prior_scaled_by_in_panel_share(self):
        b = _batch()
        m = _model(supervised_heads="stage2", extra_heads=True).eval()
        B = b["h1"].shape[0]
        with torch.no_grad():
            m.sup_ood_head.weight.zero_(); m.sup_ood_head.bias.fill_(50.0)     # out-of-panel score 1
            m(b["input_embeds"], None, b["ext_emb"], b.get("count"), aff_pred=torch.rand(B, 5))
        self.assertTrue(torch.allclose(m._init_prior, torch.zeros_like(m._init_prior), atol=1e-5))
        with torch.no_grad():
            m.sup_ood_head.bias.fill_(-50.0)                                  # in-panel: full prior
            m(b["input_embeds"], None, b["ext_emb"], b.get("count"), aff_pred=torch.rand(B, 5))
        self.assertGreater(float(m._init_prior.abs().max()), 1e-3)


class TestPoolWindows(unittest.TestCase):
    def test_scales(self):
        from python.crf.train_diploid_indel import pool_windows
        v = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5, 0.5])
        contig = np.array([0, 0, 0, 0, 1, 1])
        self.assertTrue(torch.allclose(pool_windows(v, contig, -1), torch.full((6,), 0.5)))
        self.assertTrue(torch.allclose(pool_windows(v, contig, 0), torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])))
        r = pool_windows(v, contig, 1)
        self.assertTrue(torch.allclose(r[:4], torch.tensor([0.5, 1 / 3, 2 / 3, 0.5])))
