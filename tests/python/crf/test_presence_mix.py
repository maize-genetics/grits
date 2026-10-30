import math
import unittest

import torch

from python.crf.train_diploid_indel import GRITSCRFDiploidIndel


def _model(**kw):
    torch.manual_seed(0)
    return GRITSCRFDiploidIndel(num_parents=5, d_model=16, n_heads=2, n_layers=1, time_local_emis=True,
                                founder_affinity=True, tie_aware_loss=True, emission="likelihood",
                                pair_emission="mixture", supervised_heads="stage2", extra_heads=True,
                                **kw).eval()


def _x(seed=1, B=2, T=40, K=6):
    g = torch.Generator().manual_seed(seed)
    tern = torch.randint(-1, 2, (B, T, K), generator=g).float()
    return torch.stack([tern, torch.zeros_like(tern)], -1)


class TestPresenceMix(unittest.TestCase):
    def test_rho_one_is_plain_mixture_and_homozygous_unchanged(self):
        m = _model()
        X = _x()
        ll = m.tern_loglik[(X[..., 0] + 1).long()]
        plain = torch.logaddexp(ll[..., m.pi], ll[..., m.pj]) - math.log(2.0)
        self.assertTrue(torch.allclose(m._presence_mix(X, ll, 0.0, 4), plain, atol=1e-6))
        pm = m._presence_mix(X, ll, math.log(0.2), 4)
        homo = m.homo_mask.bool()
        self.assertTrue(torch.allclose(pm[..., homo], plain[..., homo], atol=1e-5))
        self.assertFalse(torch.allclose(pm[..., ~homo], plain[..., ~homo]))

    def test_deleted_founder_no_longer_halves_present_founders_reads(self):
        m = _model()
        X = torch.zeros(1, 1, 6, 2)
        X[0, 0, 0, 0] = -1           # founder 0 deleted here
        X[0, 0, 1, 0] = 1            # founder 1 matches the read
        ll = m.tern_loglik[(X[..., 0] + 1).long()]
        het = int(((m.pi == 0) & (m.pj == 1)).nonzero())
        hom = int(((m.pi == 1) & (m.pj == 1)).nonzero())
        plain = m._presence_mix(X, ll, 0.0, 0)[0, 0]
        pm = m._presence_mix(X, ll, math.log(0.01), 0)[0, 0]
        self.assertAlmostEqual(float(plain[hom] - plain[het]), math.log(2.0), delta=0.05)
        self.assertLess(float(pm[hom] - pm[het]), 0.02)

    def test_decode_override_and_fitted_param(self):
        m = _model()
        self.assertFalse(hasattr(m, "log_del_rho"))
        f = _model(del_rho_fit=0.3)
        self.assertAlmostEqual(float(f.log_del_rho.exp()), 0.3, places=5)
        self.assertTrue(f.log_del_rho.requires_grad)       # fitted in stage 2
        self.assertFalse(f.encoder.gate_head.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()


class TestInferencePreset(unittest.TestCase):
    def test_placedfix_hetdel_preset(self):
        import python.crf.train_diploid_indel as tdi
        self.assertEqual(tdi.INFERENCE_PRESETS["placedfix_hetdel"],
                         dict(del_rho=0.1, del_win=0, route_disj_hybrid=0.25))

    def test_preset_sets_and_restores(self):
        import python.crf.train_diploid_indel as tdi
        seen = {}

        def fake(model, data, k, *a, **kw):
            seen.update(rho=model.del_rho, win=model.del_win, disj=tdi.ROUTE_DISJ_HYBRID)
            return None, None

        class M:
            pass
        m, orig, before = M(), tdi._infer_with_depth_knobs, tdi.ROUTE_DISJ_HYBRID
        tdi._infer_with_depth_knobs = fake
        try:
            tdi.infer_real_founder_pairs(m, None, 25, preset="placedfix_hetdel")
        finally:
            tdi._infer_with_depth_knobs = orig
        self.assertEqual(seen, dict(rho=0.1, win=0, disj=0.25))
        self.assertIsNone(m.del_rho)
        self.assertEqual(tdi.ROUTE_DISJ_HYBRID, before)
