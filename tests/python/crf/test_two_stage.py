import unittest

import torch

from python.crf.crf_kernels import _dcrf_nll
from python.crf.train_crf import IndelFounderPathEncoder
from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, _dcrf_nll_tied


def _model(K=5, **kw):
    torch.manual_seed(0)
    m = GRITSCRFDiploidIndel(num_parents=K, d_model=16, n_heads=2, n_layers=1,
                             time_local_emis=True, founder_affinity=True, **kw)
    return m.eval()      # no dropout: repeated forwards must be comparable


def _batch(K=5, B=2, T=9, lin=True, seed=1):
    g = torch.Generator().manual_seed(seed)
    tern = torch.randint(-1, 2, (B, T, K), generator=g).float()
    dist = torch.randint(-1, 20, (B, T, K), generator=g).float()
    b = {"input_embeds": torch.stack([tern, dist], -1),
         "h1": torch.randint(0, K, (B, T), generator=g),
         "h2": torch.randint(0, K, (B, T), generator=g),
         "ext_emb": torch.randn(B, K, 2, generator=g)}
    if lin:
        b["lin"] = torch.randint(0, 3, (B, T, K), generator=g)
    return b


class TestFlagsOffUnchanged(unittest.TestCase):
    def test_loss_matches_reference_when_off(self):
        for tie in (False, True):
            m = _model(tie_aware_loss=tie)
            b = _batch()
            loss, crf, g, c, emis, tags = m._step(b)
            e2, g2, c2 = m(b["input_embeds"], None, b["ext_emb"], None)
            if tie:
                ref = _dcrf_nll_tied(e2, c2, m.nsw_pair, m.stay_bonus,
                                     m._allowed_pairs(b["h1"], b["h2"], b["lin"]))
            else:
                ref = _dcrf_nll(e2, c2, m.nsw_pair, m.stay_bonus, tags)
            ref = ref + m.gate_reg * (1.0 - g2).mean()
            self.assertEqual(float(loss), float(ref))

    def test_return_raw_does_not_change_outputs(self):
        m = _model()
        b = _batch()
        a = m(b["input_embeds"], None, b["ext_emb"], None)
        r = m(b["input_embeds"], None, b["ext_emb"], None, return_raw=True)
        for x, y in zip(a, r[:3]):
            self.assertTrue(torch.equal(x, y))
        self.assertEqual(r[3].shape, (2, 9, 6))

    def test_raw_times_gate_is_emission(self):
        torch.manual_seed(0)
        enc = IndelFounderPathEncoder(16, 2, 1, ext_dim=2, time_local_emis=True).eval()
        X = _batch()["input_embeds"]
        mask = torch.ones(2, 5)
        emis, g, c, raw = enc(X, mask, return_raw=True)
        self.assertTrue(torch.allclose(emis, g.unsqueeze(-1) * raw, atol=1e-6))


class TestAuxTargets(unittest.TestCase):
    def test_founder_gate_switch_targets(self):
        K = 5
        m = _model(K)
        # lineages: founders 0,1 share lineage 0; 3,4 share lineage 2; 2 alone
        lin = torch.tensor([[0, 0, 1, 2, 2]]).expand(4, K).unsqueeze(0).clone()
        h1 = torch.tensor([[0, 1, 1, 2]])       # 0->1 is a lineage-mate (no switch), 1->2 is a switch
        h2 = torch.tensor([[3, 3, 3, 5]])       # 5 = null label at the last row
        tern = torch.tensor([[[1, 1, 0, 0, 0],   # explained by founder 0/1, discriminative
                              [0, 0, 1, 0, 0],   # only founder 2 matches: not explained
                              [1, 1, 1, 1, 0],   # 4 of 5 match: not discriminative
                              [0, 0, 1, 0, 0]]]).float()
        X = torch.stack([tern, torch.zeros_like(tern)], -1)
        founder, gate, switch = m.aux_targets(X, h1, h2, lin)
        self.assertEqual(founder[0, 0].tolist(), [True, True, False, True, True, False])
        self.assertEqual(founder[0, 3].tolist(), [False, False, True, False, False, True])
        self.assertEqual(gate[0].tolist(), [True, False, False, True])
        self.assertEqual(switch[0].tolist(), [False, False, True])
        # without lineage: exact labels, and a 0->1 label change IS a switch
        founder, _, switch = m.aux_targets(X, h1, h2, None)
        self.assertEqual(founder[0, 0].tolist(), [True, False, False, True, False, False])
        self.assertEqual(switch[0].tolist(), [True, False, True])

    def test_aux_loss_finite_and_weighted(self):
        m = _model(aux_w_founder=1.0, aux_w_gate=0.0, aux_w_switch=0.0)
        b = _batch()
        e, g, c, raw = m(b["input_embeds"], None, b["ext_emb"], None, return_raw=True)
        tot, (lf, lg, ls) = m.aux_loss(raw, g, c, m.aux_targets(b["input_embeds"], b["h1"], b["h2"], b["lin"]))
        self.assertTrue(all(torch.isfinite(x) for x in (lf, lg, ls)))
        self.assertAlmostEqual(float(tot), float(lf), places=6)


class TestStages(unittest.TestCase):
    def test_stage1_is_aux_only_and_skips_crf(self):
        m = _model(stage1_aux_only=True, tie_aware_loss=True)
        b = _batch()
        loss, crf, *_ = m._step(b)
        e, g, c, raw = m(b["input_embeds"], None, b["ext_emb"], None, return_raw=True)
        ref, _ = m.aux_loss(raw, g, c, m.aux_targets(b["input_embeds"], b["h1"], b["h2"], b["lin"]))
        self.assertAlmostEqual(float(loss), float(ref), places=6)
        loss.backward()
        self.assertIsNone(m.stay_bonus.grad)          # CRF parameter not in the graph
        self.assertIsNotNone(m.encoder.recomb_head.weight.grad)
        self.assertIsNotNone(m.encoder.gate_head.weight.grad)

    def test_aux_weight_adds_to_crf_loss(self):
        m0 = _model(tie_aware_loss=True)
        m1 = _model(tie_aware_loss=True, aux_loss_weight=0.5)
        b = _batch()
        l0 = m0._step(b)[0]
        l1 = m1._step(b)[0]
        e, g, c, raw = m1(b["input_embeds"], None, b["ext_emb"], None, return_raw=True)
        aux, _ = m1.aux_loss(raw, g, c, m1.aux_targets(b["input_embeds"], b["h1"], b["h2"], b["lin"]))
        self.assertAlmostEqual(float(l1), float(l0 + 0.5 * aux), places=5)

    def test_freeze_trunk_leaves_heads_and_stay(self):
        m = _model(freeze_encoder_trunk=True)
        train = {n for n, p in m.named_parameters() if p.requires_grad}
        self.assertIn("stay_bonus", train)
        self.assertTrue(any(n.startswith("encoder.gate_head") for n in train))
        self.assertTrue(any(n.startswith("encoder.recomb_head") for n in train))
        self.assertTrue(any(n.startswith("encoder.ext_bias") for n in train))
        self.assertFalse(any(n.startswith(("encoder.cell", "encoder.pos_encoder", "encoder.fpool",
                                           "encoder.count_proj", "encoder.fquery")) for n in train))


if __name__ == "__main__":
    unittest.main()
