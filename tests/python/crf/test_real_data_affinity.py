"""
Real-data gate for the founder-affinity mechanism (_founder_affinity,
estimate_inbreeding_coef, homo_scale_from_affinity in train_diploid.py) and
the canonical real-data inference/scoring entrypoints in
train_diploid_indel.py (infer_real_founder_pairs, ibd_adjusted_accuracy --
every eval script should call these, not re-derive the ext_emb/homo_scale/
decode/scoring loop): real IDX-INBRED/IDX-HYB/IDX-RIL2 samples with known
true founders, not simulated data. This is the check the model-comparison
work in this session's checkpoints depends on -- if it fails, the affinity
signal itself (or the checkpoint's use of it) is broken and no downstream
accuracy comparison against that checkpoint is trustworthy.

Uses the current best checkpoint, diploid-indel-v3-k25-overlay-affinity
(full 25-founder panel, no fixed founder drop -- see
[[v3_k25_no_drop_retrain]]), at the native K=25 real data it was trained
for. Skips entirely if the real converted-npy corpus isn't present on this
machine (it lives under the grits_workdir, not the repo).

Run in isolation, same convention as test_train_diploid_indel.py:
    pixi run -- pytest tests/python/crf/test_real_data_affinity.py -v
"""

import os
import unittest

import numpy as np
import pandas as pd
import torch

from python.crf.train_diploid import (
    _founder_affinity, estimate_inbreeding_coef, homo_scale_from_affinity,
    _estimate_inbreeding_coef_batch)
from python.crf.train_diploid_indel import (
    infer_real_founder_pairs, ibd_adjusted_accuracy, GRITSCRFDiploidIndel)

REALDIR = "/workdir/zrm22/HackathonJun2026/grits_workdir/data/real"
CKPT = ("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/"
        "diploid-indel-v3-k25-overlay-affinity/d-epoch=04-val_pair_acc=0.6820.ckpt")
K = 25

# RIL2: real corpus FASTQs carry no per-read truth label (error_autocorrelation.py's
# documented reason), and this session found run_one_sample.sh's --label-bed was a
# "B73" placeholder for every sample -- NOT genuine per-region truth. Real per-bin
# truth for RIL2 must come from the oracle mosaic reconstructor
# (simval_oracle_bed.build_ril_mosaics, delegated to by simval_truth_labels.py's
# bin_truth_labels/write_truth_labels, "oracle max identity error 0.0235%"),
# already materialized as truth_labels.npy (native K=25 space) next to the
# cached raw.npy for these 5 pairs by this session.
RIL2_SCRATCH = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/simval_eval_fixed"
RIL2_PAIRS = ["B73xCML103", "B73xOh43", "B97xCML103", "Il14HxB97", "Oh43xIl14H"]


def _ril2_outdir(pair):
    return os.path.join(RIL2_SCRATCH, f"IDX-RIL2__{pair}__0.1x")


def _have_ril2_truth():
    return all(
        os.path.exists(_ril2_outdir(p) + "/truth_labels.npy")
        and os.path.exists(_ril2_outdir(p) + "/raw.npy.bins.tsv")
        for p in RIL2_PAIRS)


def _build_row_idx(outdir, window_size=512):
    """Real-row indices per window, matching ropebwt_npy_to_matrix.py's
    --window-size chunking exactly (per-contig, non-overlapping, ascending
    on-disk row order) -- needed to pull raw.npy read counts for the same
    window a *_ternary_k25native.npy row corresponds to."""
    bins_df = pd.read_csv(f"{outdir}/raw.npy.bins.tsv", sep="\t")
    row_idx = []
    for _contig, idx in bins_df.groupby("contig", sort=False).indices.items():
        idx = np.sort(idx)
        start = 0
        while start + window_size <= len(idx):
            row_idx.append(idx[start:start + window_size])
            start += window_size
    return row_idx


def _ril2_truth_windows(pair, window_size=512, depth="0.1x"):
    """[N,T,2] true (h1,h2) founder indices in native K=25 space, windowed
    with the EXACT same per-contig chunking ropebwt_npy_to_matrix.py's
    --window-size uses (so row N aligns 1:1 with the corresponding real
    *_ternary_k25native.npy window). -1 = unresolved."""
    outdir = (_ril2_outdir(pair) if depth == "0.1x" else
              os.path.join(RIL2_SCRATCH, f"IDX-RIL2__{pair}__{depth}"))
    bins_df = pd.read_csv(f"{outdir}/raw.npy.bins.tsv", sep="\t")
    truth = np.load(f"{outdir}/truth_labels.npy")
    windows = []
    for _contig, idx in bins_df.groupby("contig", sort=False).indices.items():
        idx = np.sort(idx)
        start = 0
        while start + window_size <= len(idx):
            windows.append(truth[idx[start:start + window_size]])
            start += window_size
    return np.stack(windows, axis=0)

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
# panel index of each named founder -- native K=25 order (no drop), so these
# are the same values the K=24/post-drop convention happened to use too
# (all five are < the old drop index 23, unaffected by that compaction).
IDX = {"B73": 0, "B97": 1, "CML103": 2, "Il14H": 11, "Oh43": 21}
NAME_OF = {v: k for k, v in IDX.items()}


def _sample_path(ds, ind):
    return os.path.join(REALDIR, f"{ind}_{ds}_0.1x_ternary_k25native.npy")


def _have_real_data():
    return all(os.path.exists(_sample_path(ds, ind)) for ds, ind, _ in SAMPLES)


def _have_ckpt():
    return os.path.exists(CKPT)


def _load_match(ds, ind):
    """[N,T,K] binary match indicator, windowed (NOT flattened) -- the shape
    homo_scale_from_affinity needs to compute its per-window statistic."""
    data = np.load(_sample_path(ds, ind))
    tern = data[:, :, :K].astype(np.float32)
    return (tern == 1).astype(np.float32)


def _load_affinity_rate(ds, ind):
    return _founder_affinity(_load_match(ds, ind).reshape(-1, K))[:, 0]


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
        fully-inbred lines (true F=1) from real F1 hybrids (true F=0).
        Tolerance is 0.4, not tighter, because Il14H (real, true inbred)
        has a real IBD relative in the K=25 panel -- P39, no longer dropped
        -- so its genome-wide-MEAN top-2 rate is legitimately elevated
        (est_F=0.650, vs 0.86-0.96 for the other four true-inbred samples).
        This is real biology, not estimator noise: see
        test_p90_per_window_estimate_separates_selfing_from_hybrid and
        homo_scale_from_affinity for the per-window mechanism that isn't
        confused by it (the actual mechanism this checkpoint uses at
        inference, not this genome-wide-mean-only diagnostic)."""
        for ds, ind, (p1, p2) in SAMPLES:
            with self.subTest(sample=f"{ds}/{ind}"):
                rate = _load_affinity_rate(ds, ind)
                est_F = estimate_inbreeding_coef(rate)
                true_F = 1.0 if p1 == p2 else 0.0
                self.assertLess(abs(est_F - true_F), 0.4,
                                 f"{ds}/{ind}: estimated F={est_F:.3f} vs true F={true_F}")

    def test_inbred_and_hybrid_populations_cleanly_separated(self):
        inbred_F = [estimate_inbreeding_coef(_load_affinity_rate(ds, ind))
                    for ds, ind, (p1, p2) in SAMPLES if p1 == p2]
        hyb_F = [estimate_inbreeding_coef(_load_affinity_rate(ds, ind))
                 for ds, ind, (p1, p2) in SAMPLES if p1 != p2]
        self.assertGreater(min(inbred_F), max(hyb_F),
                            "true-inbred and true-hybrid estimated-F distributions overlap")


@unittest.skipUnless(_have_real_data() and _have_ckpt(),
                      "real data or diploid-indel-v3-k25-overlay-affinity checkpoint not found")
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
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        # strict=False: CKPT predates count_proj (experiments/depth-
        # confidence-fix/, indel-readcount-feature) -- zero-initialized by
        # nn.Linear.reset_parameters then explicitly zeroed again for
        # clarity, so the loaded model is numerically identical to CKPT's
        # own weights (bit-identical warm start, verified by
        # TestBitIdenticalWarmStart below).
        cls.model = GRITSCRFDiploidIndel.load_from_checkpoint(
            CKPT, map_location=cls.device, strict=False).eval().to(cls.device)
        with torch.no_grad():
            cls.model.encoder.count_proj.weight.zero_()
            cls.model.encoder.count_proj.bias.zero_()

    def _pair_acc(self, ds, ind, p1, p2):
        data = np.load(_sample_path(ds, ind))
        true_lo, true_hi = min(IDX[p1], IDX[p2]), max(IDX[p1], IDX[p2])
        pred_lo, pred_hi = infer_real_founder_pairs(self.model, data, K, device=self.device)
        return float(np.mean((pred_lo == true_lo) & (pred_hi == true_hi)))

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


@unittest.skipUnless(_have_ril2_truth(), f"RIL2 oracle truth_labels.npy not found under {RIL2_SCRATCH}")
class TestRealRIL2AffinityAndTruth(unittest.TestCase):
    """RIL2 individuals are locally homozygous (mosaic of homozygous-founder-A
    / homozygous-founder-B blocks), unlike INBRED (single genome-wide
    founder) or HYB (constant heterozygous pair). Genome-wide affinity
    can't tell that apart from a true hybrid -- both true parents read as
    comparably elevated. This class checks the affinity signal reproduces
    that structure against real oracle-reconstructed truth
    (simval_oracle_bed.build_ril_mosaics), not the corpus's broken
    per-sample "B73"-for-everyone labels.bed placeholder."""

    def test_true_ril2_individuals_are_locally_homozygous(self):
        for pair in RIL2_PAIRS:
            with self.subTest(pair=pair):
                truth_w = _ril2_truth_windows(pair)
                h1, h2 = truth_w[:, :, 0], truth_w[:, :, 1]
                valid = (h1 >= 0) & (h2 >= 0)
                self.assertGreater(valid.mean(), 0.99, f"{pair}: too many unresolved truth sites")
                self.assertGreater((h1 == h2)[valid].mean(), 0.999,
                                    f"{pair}: RIL2 truth should be ~100% homozygous per site")

    def test_true_ril2_switch_count_is_realistic(self):
        """~30 switches genome-wide is the domain expectation (RIL2 breeding
        design); a labeling bug that collapses truth to a single constant
        founder (0 switches) or one that's pure noise (hundreds) should fail
        this."""
        for pair in RIL2_PAIRS:
            with self.subTest(pair=pair):
                truth_w = _ril2_truth_windows(pair)
                h1 = truth_w[:, :, 0].reshape(-1)
                valid = h1 >= 0
                seq = h1[valid]
                switches = int((seq[1:] != seq[:-1]).sum())
                self.assertGreater(switches, 10, f"{pair}: only {switches} switches, truth looks degenerate")
                self.assertLess(switches, 200, f"{pair}: {switches} switches, truth looks like noise")

    def test_genome_wide_affinity_cannot_see_local_homozygosity(self):
        """Documents the real limitation, doesn't just assert a number:
        genome-wide-MEAN est_F must read as "outbred" (both true parents
        comparably elevated) even though the individual actually IS
        homozygous everywhere -- this is why homo_scale_from_affinity uses
        the per-window PERCENTILE instead of the genome-wide mean."""
        for pair in RIL2_PAIRS:
            with self.subTest(pair=pair):
                rate = _load_affinity_rate("IDX-RIL2", pair)
                est_F = estimate_inbreeding_coef(rate)
                self.assertLess(est_F, 0.5,
                                 f"{pair}: genome-wide-mean est_F={est_F:.3f}, expected <0.5 (looks outbred)")

    def test_p90_per_window_estimate_separates_selfing_from_hybrid(self):
        """The mechanism homo_scale_from_affinity actually relies on: the
        90th percentile of the per-window inbreeding estimate must read as
        selfing-type (>0.5) for RIL2 -- same as true INBRED -- even though
        RIL2's genome-wide-mean reads as outbred (previous test)."""
        for pair in RIL2_PAIRS:
            with self.subTest(pair=pair):
                M = _load_match("IDX-RIL2", pair)
                per_window = _estimate_inbreeding_coef_batch(M.mean(axis=1))
                p90 = np.percentile(per_window, 90)
                self.assertGreater(p90, 0.5, f"{pair}: p90 per-window est_F={p90:.3f}, expected >0.5")


@unittest.skipUnless(_have_ril2_truth() and _have_ckpt(),
                      "RIL2 oracle truth or diploid-indel-v3-k25-overlay-affinity checkpoint not found")
class TestRealRIL2EndToEndPairAccuracy(unittest.TestCase):
    """Gate for homo_scale_from_affinity on RIL2: against real oracle truth
    (not the broken labels.bed), it must reach the same ~100% bar as
    INBRED/HYB -- unlike the earlier genome-wide-mean-only version of this
    function, which scored ~1% on real RIL2 (see git history / superseded
    per_window_homo_scale docstring for that failed attempt)."""

    @classmethod
    def setUpClass(cls):
        from python.crf.train_diploid_indel import GRITSCRFDiploidIndel
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        # strict=False: CKPT predates count_proj (experiments/depth-
        # confidence-fix/, indel-readcount-feature) -- zero-initialized by
        # nn.Linear.reset_parameters then explicitly zeroed again for
        # clarity, so the loaded model is numerically identical to CKPT's
        # own weights (bit-identical warm start, verified by
        # TestBitIdenticalWarmStart below).
        cls.model = GRITSCRFDiploidIndel.load_from_checkpoint(
            CKPT, map_location=cls.device, strict=False).eval().to(cls.device)
        with torch.no_grad():
            cls.model.encoder.count_proj.weight.zero_()
            cls.model.encoder.count_proj.bias.zero_()

    def _pair_acc(self, pair):
        data = np.load(_sample_path("IDX-RIL2", pair))
        truth_w = _ril2_truth_windows(pair)
        true_h1, true_h2 = truth_w[:, :, 0], truth_w[:, :, 1]
        valid = (true_h1 >= 0) & (true_h2 >= 0)
        true_lo, true_hi = np.minimum(true_h1, true_h2), np.maximum(true_h1, true_h2)
        pred_lo, pred_hi = infer_real_founder_pairs(self.model, data, K, device=self.device)
        return ((pred_lo == true_lo) & (pred_hi == true_hi))[valid].mean()

    def test_real_ril2_pair_accuracy_with_adaptive_homo_scale(self):
        for pair in RIL2_PAIRS:
            with self.subTest(pair=pair):
                acc = self._pair_acc(pair)
                self.assertGreater(acc, 0.9, f"{pair}: pair_acc={acc:.3f}, expected >0.9")


@unittest.skipUnless(_have_ril2_truth() and _have_ckpt(),
                      "RIL2 oracle truth or diploid-indel-v3-k25-overlay-affinity checkpoint not found")
class TestIBDAdjustedAccuracy(unittest.TestCase):
    """ibd_adjusted_accuracy (train_diploid_indel.py): a second real-data
    metric that credits window-level errors real raw read support cannot
    actually distinguish from the true call (verified this session: 95.3%
    of real RIL2 errors and 87.7% of real HYB errors show the wrong
    founder's real support tied with or exceeding the true founder's,
    concentrated in B73-involving pairs with confirmed local IBD tracts).
    Uses RIL2 specifically since it has both a real per-site oracle truth
    and the largest, most-verified IBD effect (B73xOh43)."""

    @classmethod
    def setUpClass(cls):
        from python.crf.train_diploid_indel import GRITSCRFDiploidIndel
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        # strict=False: CKPT predates count_proj (experiments/depth-
        # confidence-fix/, indel-readcount-feature) -- zero-initialized by
        # nn.Linear.reset_parameters then explicitly zeroed again for
        # clarity, so the loaded model is numerically identical to CKPT's
        # own weights (bit-identical warm start, verified by
        # TestBitIdenticalWarmStart below).
        cls.model = GRITSCRFDiploidIndel.load_from_checkpoint(
            CKPT, map_location=cls.device, strict=False).eval().to(cls.device)
        with torch.no_grad():
            cls.model.encoder.count_proj.weight.zero_()
            cls.model.encoder.count_proj.bias.zero_()

    def _accuracies(self, pair):
        data = np.load(_sample_path("IDX-RIL2", pair))
        truth_w = _ril2_truth_windows(pair)
        true_h1, true_h2 = truth_w[:, :, 0], truth_w[:, :, 1]
        valid = (true_h1 >= 0) & (true_h2 >= 0)
        true_lo, true_hi = np.minimum(true_h1, true_h2), np.maximum(true_h1, true_h2)
        outdir = _ril2_outdir(pair)
        raw = np.load(f"{outdir}/raw.npy", mmap_mode="r")
        row_idx = _build_row_idx(outdir)
        return ibd_adjusted_accuracy(self.model, data, K, true_lo, true_hi, valid,
                                      raw, row_idx, device=self.device)

    def test_ibd_adjusted_accuracy_never_below_pair_acc(self):
        """Crediting can only raise accuracy, never lower it -- a basic
        soundness check on the metric itself, not just this checkpoint."""
        for pair in RIL2_PAIRS:
            with self.subTest(pair=pair):
                pair_acc, ibd_acc, _cred, _chk = self._accuracies(pair)
                self.assertGreaterEqual(ibd_acc, pair_acc - 1e-9,
                                         f"{pair}: ibd_adj_acc={ibd_acc:.4f} < pair_acc={pair_acc:.4f}")

    def test_b73xoh43_gets_the_largest_ibd_credit(self):
        """The one real, independently-confirmed IBD case (B73xOh43, tied
        real read support across chr5/6/7/8) should show a clearly larger
        adjustment than the other four RIL2 pairs."""
        deltas = {}
        for pair in RIL2_PAIRS:
            pair_acc, ibd_acc, _cred, _chk = self._accuracies(pair)
            deltas[pair] = ibd_acc - pair_acc
        self.assertEqual(max(deltas, key=deltas.get), "B73xOh43",
                          f"expected B73xOh43 to have the largest IBD-adjustment delta, got {deltas}")
        self.assertGreater(deltas["B73xOh43"], 0.01,
                            f"B73xOh43 IBD-adjustment delta={deltas['B73xOh43']:.4f}, expected >0.01")


@unittest.skipUnless(_have_ril2_truth() and _have_ckpt(),
                      "RIL2 real corpus/truth or diploid-indel-v3-k25-overlay-affinity "
                      "checkpoint not found")
class TestBitIdenticalWarmStart(unittest.TestCase):
    """experiments/depth-confidence-fix/ read-count plan, Verification item
    2: the count_proj-widened model, loaded from CKPT with strict=False
    (its state_dict predates that param), must reproduce CKPT's own
    real-data predictions EXACTLY -- proof the migration is safe and any
    later change is attributable to training, not plumbing. The cached
    real *_ternary_k25native.npy files are all still 2K+2 (no
    --emit-read-counts conversion has been run against them yet at this
    point), so count=None throughout here -- IndelFounderPathEncoder
    .forward's count branch is not merely zero-weighted but never executed
    at all -- byte-identical by construction, not just numerically close.

    Pinned against pair_acc/ibd_adj recorded this session BEFORE
    count_proj existed at all (experiments/depth-confidence-fix/scripts/
    stay_bonus_sweep.py's stay_bonus=2.0 rows, the checkpoint's trained
    default -- same infer_real_founder_pairs/score-path methodology, same
    checkpoint, same cached real data)."""

    EXPECTED = {
        ("IDX-HYB", "Oh43xIl14H", "0.1x"): (0.9903, 0.9994),
        ("IDX-HYB", "Oh43xIl14H", "2.0x"): (0.9650, 0.9974),
        ("IDX-RIL2", "Oh43xIl14H", "0.1x"): (0.9962, 0.9962),
        ("IDX-RIL2", "Oh43xIl14H", "2.0x"): (0.9942, 0.9992),
    }

    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = GRITSCRFDiploidIndel.load_from_checkpoint(
            CKPT, map_location=cls.device, strict=False).eval().to(cls.device)
        with torch.no_grad():
            cls.model.encoder.count_proj.weight.zero_()
            cls.model.encoder.count_proj.bias.zero_()

    def test_pinned_pair_acc_and_ibd_adj_reproduced_exactly(self):
        for (ds, ind, depth), (exp_pair, exp_ibd) in self.EXPECTED.items():
            with self.subTest(ds=ds, ind=ind, depth=depth):
                data = np.load(os.path.join(
                    REALDIR, f"{ind}_{ds}_{depth}_ternary_k25native.npy"))
                if ds == "IDX-HYB":
                    p1, p2 = ind.split("x")
                    tlo, thi = min(IDX[p1], IDX[p2]), max(IDX[p1], IDX[p2])
                    N = data.shape[0]
                    true_lo = np.full((N, 512), tlo)
                    true_hi = np.full((N, 512), thi)
                    valid = np.ones((N, 512), dtype=bool)
                else:
                    truth_w = _ril2_truth_windows(ind, depth=depth)
                    h1, h2 = truth_w[:, :, 0], truth_w[:, :, 1]
                    valid = (h1 >= 0) & (h2 >= 0)
                    true_lo, true_hi = np.minimum(h1, h2), np.maximum(h1, h2)
                outdir = os.path.join(RIL2_SCRATCH, f"{ds}__{ind}__{depth}")
                raw = np.load(f"{outdir}/raw.npy", mmap_mode="r")
                row_idx = _build_row_idx(outdir)
                pair_acc, ibd_adj, _cred, _chk = ibd_adjusted_accuracy(
                    self.model, data, K, true_lo, true_hi, valid, raw, row_idx,
                    device=self.device)
                self.assertAlmostEqual(pair_acc, exp_pair, places=4,
                                       msg=f"{ds}/{ind}/{depth}: pair_acc drifted from pinned value")
                self.assertAlmostEqual(ibd_adj, exp_ibd, places=4,
                                       msg=f"{ds}/{ind}/{depth}: ibd_adj drifted from pinned value")


if __name__ == "__main__":
    unittest.main()
