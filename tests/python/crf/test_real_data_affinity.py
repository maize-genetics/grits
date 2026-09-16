"""
Real-data gate for the founder-affinity mechanism (_founder_affinity,
estimate_inbreeding_coef, homo_scale_from_affinity in train_diploid.py):
real IDX-INBRED/IDX-HYB samples with known true founders, not simulated
data. This is the check the model-comparison work in this session's
checkpoints depends on -- if it fails, the affinity signal itself (or the
checkpoint's use of it) is broken and no downstream accuracy comparison
against that checkpoint is trustworthy.

Skips entirely if the real converted-npy corpus isn't present on this
machine (it lives under the grits_workdir, not the repo).

Run in isolation, same convention as test_train_diploid_indel.py:
    pixi run -- pytest tests/python/crf/test_real_data_affinity.py -v
"""

import os
import unittest

import numpy as np
import torch

from python.crf.train_diploid import (
    _founder_affinity, estimate_inbreeding_coef, homo_scale_from_affinity)

REALDIR = "/workdir/zrm22/HackathonJun2026/grits_workdir/data/real"
CKPT = ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
        "diploid-indel-v3-overlay-affinity/d-epoch=04-val_pair_acc=0.6799.ckpt")
K = 24

# (dataset, individual-file-stem, (true founder 1, true founder 2))
SAMPLES = [
    ("IDX-INBRED", "B73", ("B73", "B73")),
    ("IDX-INBRED", "Oh43", ("Oh43", "Oh43")),
    ("IDX-INBRED", "Il14H", ("Il14H", "Il14H")),
    ("IDX-INBRED", "B97", ("B97", "B97")),
    ("IDX-INBRED", "CML103", ("CML103", "CML103")),
    ("IDX-HYB", "B73xOh43", ("B73", "Oh43")),
    ("IDX-HYB", "B73xCML103", ("B73", "CML103")),
    ("IDX-HYB", "Oh43xIl14H", ("Oh43", "Il14H")),
    ("IDX-HYB", "B97xCML103", ("B97", "CML103")),
    ("IDX-HYB", "Il14HxB97", ("Il14H", "B97")),
]
# panel index of each named founder in the K=24 (post drop-idx-23) real npy layout
IDX = {"B73": 0, "B97": 1, "CML103": 2, "Il14H": 11, "Oh43": 21}
NAME_OF = {v: k for k, v in IDX.items()}


def _sample_path(ds, ind):
    return os.path.join(REALDIR, f"{ind}_{ds}_0.1x_ternary_k24.npy")


def _have_real_data():
    return all(os.path.exists(_sample_path(ds, ind)) for ds, ind, _ in SAMPLES)


def _have_ckpt():
    return os.path.exists(CKPT)


def _load_affinity_rate(ds, ind):
    data = np.load(_sample_path(ds, ind))
    tern = data[:, :, :K].astype(np.float32)
    M = (tern == 1).astype(np.float32).reshape(-1, K)
    return _founder_affinity(M)[:, 0]


@unittest.skipUnless(_have_real_data(), f"real converted npy corpus not found under {REALDIR}")
class TestRealDataAffinity(unittest.TestCase):
    """_founder_affinity, computed on real ropebwt3 reads (no labels), must
    correctly identify which founder(s) an individual actually carries."""

    def test_affinity_top_founders_match_true_founders(self):
        for ds, ind, (p1, p2) in SAMPLES:
            with self.subTest(sample=f"{ds}/{ind}"):
                rate = _load_affinity_rate(ds, ind)
                order = np.argsort(-rate)
                true_set = {p1, p2}
                top1_name = NAME_OF.get(int(order[0]))
                top2_name = NAME_OF.get(int(order[1]))
                if p1 == p2:
                    self.assertEqual(top1_name, p1,
                                      f"{ds}/{ind}: top-1 affinity founder {top1_name} != true {p1}")
                else:
                    self.assertEqual({top1_name, top2_name}, true_set,
                                      f"{ds}/{ind}: top-2 affinity founders "
                                      f"{{{top1_name},{top2_name}}} != true {true_set}")

    def test_estimated_inbreeding_coefficient_is_accurate(self):
        """Background-corrected top1/top2 affinity gap must separate real
        fully-inbred lines (true F=1) from real F1 hybrids (true F=0)."""
        for ds, ind, (p1, p2) in SAMPLES:
            with self.subTest(sample=f"{ds}/{ind}"):
                rate = _load_affinity_rate(ds, ind)
                est_F = estimate_inbreeding_coef(rate)
                true_F = 1.0 if p1 == p2 else 0.0
                self.assertLess(abs(est_F - true_F), 0.3,
                                 f"{ds}/{ind}: estimated F={est_F:.3f} vs true F={true_F}")

    def test_inbred_and_hybrid_populations_cleanly_separated(self):
        inbred_F = [estimate_inbreeding_coef(_load_affinity_rate(ds, ind))
                    for ds, ind, (p1, p2) in SAMPLES if p1 == p2]
        hyb_F = [estimate_inbreeding_coef(_load_affinity_rate(ds, ind))
                 for ds, ind, (p1, p2) in SAMPLES if p1 != p2]
        self.assertGreater(min(inbred_F), max(hyb_F),
                            "true-inbred and true-hybrid estimated-F distributions overlap")


@unittest.skipUnless(_have_real_data() and _have_ckpt(),
                      "real data or diploid-indel-v3-overlay-affinity checkpoint not found")
class TestRealDataEndToEndPairAccuracy(unittest.TestCase):
    """Gate for using this checkpoint as a baseline: the model, given the
    (already-verified-correct) affinity signal and an affinity-derived
    homo_scale, must correctly resolve both true-homozygous (inbred) and
    true-heterozygous (hybrid) real individuals. A fixed homo_scale cannot
    pass this for both classes at once (penalty=3 -> ~1.5% inbred pair_acc;
    penalty=0 -> 0% hybrid pair_acc) -- if this test fails, the checkpoint
    (or its homo_scale handling) is the wrong baseline to build on."""

    @classmethod
    def setUpClass(cls):
        from python.crf.train_diploid_indel import GRITSCRFDiploidIndel
        from python.crf.crf_kernels import _dcrf_viterbi
        cls._dcrf_viterbi = staticmethod(_dcrf_viterbi)
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = GRITSCRFDiploidIndel.load_from_checkpoint(
            CKPT, map_location=cls.device).eval().to(cls.device)

    def _pair_acc(self, ds, ind, p1, p2):
        data = np.load(_sample_path(ds, ind))
        N, T, W = data.shape
        tern = data[:, :, :K].astype(np.float32)
        dist = data[:, :, K + 2:2 * K + 2].astype(np.float32)
        feats = torch.tensor(np.stack([tern, dist], axis=-1), dtype=torch.float32)
        M = (tern == 1).astype(np.float32)
        affinity = _founder_affinity(M)
        homo_scale = homo_scale_from_affinity(affinity[:, 0])
        ext_emb = torch.tensor(affinity, dtype=torch.float32).unsqueeze(0).expand(N, -1, -1)
        true_lo, true_hi = min(IDX[p1], IDX[p2]), max(IDX[p1], IDX[p2])

        preds_lo, preds_hi = [], []
        with torch.no_grad():
            for s in range(0, N, 64):
                xb = feats[s:s + 64].to(self.device)
                eb = ext_emb[s:s + 64].to(self.device)
                hs = torch.full((xb.shape[0],), homo_scale, device=self.device)
                emis_p, g, c = self.model(xb, ext_emb=eb, homo_scale=hs)
                pred = self._dcrf_viterbi(emis_p, c, self.model.nsw_pair, self.model.stay_bonus)
                preds_lo.append(self.model.pi[pred].cpu())
                preds_hi.append(self.model.pj[pred].cpu())
        pred_lo, pred_hi = torch.cat(preds_lo), torch.cat(preds_hi)
        return ((pred_lo == true_lo) & (pred_hi == true_hi)).float().mean().item()

    def test_real_inbred_pair_accuracy_with_adaptive_homo_scale(self):
        for ds, ind, (p1, p2) in SAMPLES:
            if p1 != p2:
                continue
            with self.subTest(sample=f"{ds}/{ind}"):
                acc = self._pair_acc(ds, ind, p1, p2)
                self.assertGreater(acc, 0.9, f"{ds}/{ind}: pair_acc={acc:.3f}, expected >0.9")

    def test_real_hybrid_pair_accuracy_with_adaptive_homo_scale(self):
        for ds, ind, (p1, p2) in SAMPLES:
            if p1 == p2:
                continue
            with self.subTest(sample=f"{ds}/{ind}"):
                acc = self._pair_acc(ds, ind, p1, p2)
                self.assertGreater(acc, 0.9, f"{ds}/{ind}: pair_acc={acc:.3f}, expected >0.9")


if __name__ == "__main__":
    unittest.main()
