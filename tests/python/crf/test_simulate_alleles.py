"""Tests for the --simulate-indels indel-modeling additions to
simulate_alleles.py (experiments/simulator-indels/PLAN.md). Run in
isolation: `pixi run -- pytest tests/python/crf/test_simulate_alleles.py -v`
-- several pre-existing, unrelated test files in this repo already fail
at collection time, so the full-suite baseline is not green.
"""
import numpy as np
import pytest

from python.crf.simulate_alleles import (
    TERN_DEL, TERN_DIV, TERN_MATCH, TERN_PAD,
    DIST_PAD, DIST_SAT, LABEL_PAD, DIST_LOG_SCALE,
    _indel_lengths, _encode_dist, _anchor_distance, _gather_by_lineage,
    _indel_tracts, _draw_lineages, _coalescent_feats, _good_mask, simulate,
    _indel_suppressed_rate, _row_counts, _sample_rows, _indel_chunk,
)

# --- golden hashes: pre-change simulate() output on fixed args/seed, ------
# recorded BEFORE any --simulate-indels code was added (see this branch's
# commit history / experiments/simulator-indels/PLAN.md). Every new rng
# call this feature adds sits inside an `if simulate_indels:` (or
# `per_gamete=True`) branch, so the flag-off path's draw sequence -- and
# therefore this hash -- must never change.
_GOLDEN_COALESCENT_SHA256 = "382822cc04ab24849f39776285511e35b6bcab8a938bb3ad7c9d383410ebb2aa"
_GOLDEN_INDEPENDENT_SHA256 = "9f67f97fbb28fde38e83b07ecb2d443aaa7b626ff7ff8c9b1d703baa224c7911"


# --- _encode_dist -----------------------------------------------------

def test_encode_dist_shape_dtype_range():
    d = np.arange(0, 200_000)
    code = _encode_dist(d)
    assert code.dtype == np.int8
    assert code.shape == d.shape
    assert (code >= 0).all() and (code <= DIST_SAT - 1).all()


def test_encode_dist_zero_at_zero():
    assert _encode_dist(np.array([0]))[0] == 0


def test_encode_dist_monotone_nondecreasing():
    d = np.arange(0, 100_000)
    code = _encode_dist(d).astype(np.int64)
    assert (np.diff(code) >= 0).all()


def test_encode_dist_tiny_exact():
    # code = round(8*log2(1+d)): d=0->0, 1->8, 3->16, 7->24, 255->64
    got = _encode_dist(np.array([0, 1, 3, 7, 255]), scale=8.0)
    np.testing.assert_array_equal(got, np.array([0, 8, 16, 24, 64], dtype=np.int8))


# --- _anchor_distance ---------------------------------------------------

def test_anchor_distance_shape_dtype():
    del_mask = np.zeros((3, 5, 20), dtype=bool)
    dist = _anchor_distance(del_mask)
    assert dist.shape == del_mask.shape
    assert dist.dtype == np.int32
    assert (dist >= 0).all()


def test_anchor_distance_zero_outside_tracts():
    rng = np.random.default_rng(0)
    del_mask = rng.random((4, 6, 40)) < 0.3
    dist = _anchor_distance(del_mask)
    assert (dist[~del_mask] == 0).all()
    assert (dist[del_mask] >= 1).all()


def test_anchor_distance_interior_run_tiny_exact():
    # Real anchors flank both sides of every deleted run here, so this case
    # is unaffected by the no-fabricated-edge-anchor design (see docstring).
    d = np.array([False, True, True, True, False, False, True, False])
    got = _anchor_distance(d)
    np.testing.assert_array_equal(got, np.array([0, 1, 2, 1, 0, 0, 1, 0]))


def test_anchor_distance_left_edge_run_tiny_exact():
    # [T,T,F]: only real anchor is at index 2. site1 is 1 away (real anchor),
    # site0 is 2 away (real anchor) -- NOT 1, which would require fabricating
    # a virtual anchor at the array's own left edge (index -1). This is the
    # specific behavior this implementation deliberately avoids (see
    # _anchor_distance's docstring) -- verified independently before writing
    # the implementation, not copied from an unverified design note.
    d = np.array([True, True, False])
    got = _anchor_distance(d)
    np.testing.assert_array_equal(got, np.array([2, 1, 0]))


def test_anchor_distance_no_real_anchor_saturates():
    # Entirely-deleted row, R=4: no real anchor anywhere, so every position
    # must report a distance that exceeds any possible real (in-array)
    # distance (which is bounded by R-1=3) -- not a small fabricated value.
    d = np.array([True, True, True, True])
    got = _anchor_distance(d)
    assert (got > 3).all()


def test_anchor_distance_matches_bruteforce_oracle_fuzz():
    def bruteforce(del_mask):
        R = del_mask.shape[-1]
        flat = del_mask.reshape(-1, R)
        out = np.zeros(flat.shape, dtype=np.int64)
        has_no_anchor = np.zeros(flat.shape[0], dtype=bool)
        for i in range(flat.shape[0]):
            row = flat[i]
            anchors = np.flatnonzero(~row)
            if anchors.size == 0:
                has_no_anchor[i] = True
                continue
            for t in range(R):
                out[i, t] = 0 if not row[t] else int(np.min(np.abs(anchors - t)))
        return out.reshape(del_mask.shape), has_no_anchor.reshape(del_mask.shape[:-1])

    rng = np.random.default_rng(1)
    for _ in range(200):
        n, R = rng.integers(1, 4), rng.integers(1, 15)
        p = rng.uniform(0.0, 1.0)
        del_mask = rng.random((n, R)) < p
        expect, no_anchor = bruteforce(del_mask)
        got = _anchor_distance(del_mask)
        # rows with a real anchor somewhere: exact match
        keep = ~np.repeat(no_anchor[:, None], R, axis=1)
        np.testing.assert_array_equal(got[keep], expect[keep])
        # rows with no real anchor at all: every deleted position must
        # exceed the max possible real in-array distance (R-1)
        if no_anchor.any():
            assert (got[no_anchor] > R - 1).all()


# --- _gather_by_lineage ---------------------------------------------------

def test_gather_by_lineage_matches_take_along_axis():
    rng = np.random.default_rng(2)
    n, M, K, T = 3, 5, 4, 6
    a_lin = rng.integers(0, 100, size=(n, M, T))
    lineage = rng.integers(0, M, size=(n, K, T))
    got = _gather_by_lineage(a_lin, lineage)
    expect = np.take_along_axis(a_lin, lineage, axis=1)
    np.testing.assert_array_equal(got, expect)
    assert got.shape == (n, K, T)


def test_gather_by_lineage_ibd_founders_share_content():
    # Two founders assigned the same lineage at every site must gather
    # byte-identical content -- this is the mechanism that makes indel
    # presence IBD-coupled with SNP sharing (PLAN SS2.2).
    n, M, K, T = 1, 3, 4, 10
    a_lin = np.arange(n * M * T).reshape(n, M, T)
    lineage = np.zeros((n, K, T), dtype=np.int64)
    lineage[:, 0, :] = 1
    lineage[:, 2, :] = 1   # founders 0 and 2 share lineage 1 everywhere
    lineage[:, 1, :] = 0
    lineage[:, 3, :] = 2
    got = _gather_by_lineage(a_lin, lineage)
    np.testing.assert_array_equal(got[:, 0, :], got[:, 2, :])
    assert not np.array_equal(got[:, 0, :], got[:, 1, :])


# --- _indel_lengths ---------------------------------------------------

def test_indel_lengths_shape_dtype_range():
    rng = np.random.default_rng(3)
    L = _indel_lengths(rng, 1000, large_frac=0.03, small_alpha=1.7,
                        small_max=50, large_logmean=8.6, large_logsd=1.6,
                        max_len=65536)
    assert L.shape == (1000,)
    assert L.dtype == np.int64
    assert (L >= 1).all() and (L <= 65536).all()


def test_indel_lengths_zero_large_frac_is_all_small():
    rng = np.random.default_rng(4)
    L = _indel_lengths(rng, 500, large_frac=0.0, small_alpha=1.7,
                        small_max=50, large_logmean=8.6, large_logsd=1.6,
                        max_len=65536)
    assert (L <= 50).all()


def test_indel_lengths_one_large_frac_is_all_large():
    rng = np.random.default_rng(5)
    L = _indel_lengths(rng, 500, large_frac=1.0, small_alpha=1.7,
                        small_max=50, large_logmean=8.6, large_logsd=1.6,
                        max_len=65536)
    assert (L >= 51).all()


def test_indel_lengths_matches_independent_reference_fuzz():
    def reference(rng, size, large_frac, small_alpha, small_max,
                  large_logmean, large_logsd, max_len):
        # Independently reimplemented (Python loop + branching, not
        # vectorized where/clip) but fed the SAME rng draw sequence, per
        # this repo's test_accuracy.py:183 precedent for probabilistic code.
        u = rng.random(size)
        z = rng.zipf(small_alpha, size)
        ln = rng.lognormal(large_logmean, large_logsd, size)
        out = np.empty(size, dtype=np.int64)
        for i in range(size):
            if u[i] < large_frac:
                v = round(ln[i])
                v = max(small_max + 1, min(max_len, v))
            else:
                v = max(1, min(small_max, int(z[i])))
            out[i] = v
        return out

    param_rng = np.random.default_rng(6)
    for _ in range(200):
        large_frac = float(param_rng.uniform(0.0, 1.0))
        small_alpha = float(param_rng.uniform(1.05, 3.0))
        small_max = int(param_rng.integers(2, 200))
        max_len = int(param_rng.integers(small_max + 10, 1_000_000))
        large_logmean = float(param_rng.uniform(3.0, 12.0))
        large_logsd = float(param_rng.uniform(0.5, 2.5))
        size = int(param_rng.integers(1, 50))
        seed = int(param_rng.integers(0, 2**31 - 1))

        got = _indel_lengths(np.random.default_rng(seed), size, large_frac,
                              small_alpha, small_max, large_logmean,
                              large_logsd, max_len)
        expect = reference(np.random.default_rng(seed), size, large_frac,
                            small_alpha, small_max, large_logmean,
                            large_logsd, max_len)
        np.testing.assert_array_equal(got, expect)


# --- _indel_tracts ---------------------------------------------------

_TRACT_KW = dict(density=2.65e-3, ins_frac=0.5, large_frac=0.027,
                  small_alpha=1.7, small_max=50, large_logmean=8.6,
                  large_logsd=1.6, max_len=65536)


def test_indel_tracts_shape_dtype():
    rng = np.random.default_rng(20)
    del_mask, ins_bp = _indel_tracts(rng, n=2, M=3, R=500, **_TRACT_KW)
    assert del_mask.shape == (2, 3, 500)
    assert ins_bp.shape == (2, 3, 500)
    assert del_mask.dtype == bool
    assert ins_bp.dtype == np.int32
    assert (ins_bp >= 0).all()
    assert (ins_bp[del_mask] == 0).all()


def test_indel_tracts_zero_density_is_empty():
    rng = np.random.default_rng(21)
    del_mask, ins_bp = _indel_tracts(rng, n=2, M=3, R=500,
                                      density=0.0, ins_frac=0.5,
                                      large_frac=0.027, small_alpha=1.7,
                                      small_max=50, large_logmean=8.6,
                                      large_logsd=1.6, max_len=65536)
    assert not del_mask.any()
    assert ins_bp.sum() == 0


def test_indel_tracts_deterministic():
    a_del, a_ins = _indel_tracts(np.random.default_rng(22), n=3, M=4, R=2000, **_TRACT_KW)
    b_del, b_ins = _indel_tracts(np.random.default_rng(22), n=3, M=4, R=2000, **_TRACT_KW)
    np.testing.assert_array_equal(a_del, b_del)
    np.testing.assert_array_equal(a_ins, b_ins)


def test_indel_tracts_ibd_lineages_share_content():
    del_lin, ins_lin = _indel_tracts(np.random.default_rng(23), n=2, M=5,
                                      R=2000, **_TRACT_KW)
    K = 4
    lineage = np.zeros((2, K, 2000), dtype=np.int64)
    lineage[:, 0, :] = 1
    lineage[:, 2, :] = 1   # founders 0,2 share lineage 1 -> must be identical
    lineage[:, 1, :] = 0
    lineage[:, 3, :] = 3
    del_f = _gather_by_lineage(del_lin, lineage)
    ins_f = _gather_by_lineage(ins_lin, lineage)
    np.testing.assert_array_equal(del_f[:, 0, :], del_f[:, 2, :])
    np.testing.assert_array_equal(ins_f[:, 0, :], ins_f[:, 2, :])


def test_indel_tracts_matches_bruteforce_oracle_fuzz():
    def reference(rng, n, M, R, density, ins_frac, large_frac, small_alpha,
                  small_max, large_logmean, large_logsd, max_len):
        # Independently reimplemented as a plain per-event Python loop (not
        # the vectorized scatter+cumsum interval-union trick), fed the SAME
        # rng draw sequence -- this repo's test_accuracy.py:183 precedent
        # for probabilistic code.
        span = R + max_len
        lam = density * span
        cnt = rng.poisson(lam, size=n * M)
        E = int(cnt.sum())
        cell = np.repeat(np.arange(n * M), cnt)
        start = rng.integers(-max_len, R, E)
        L = _indel_lengths(rng, E, large_frac, small_alpha, small_max,
                            large_logmean, large_logsd, max_len)
        is_ins = rng.random(E) < ins_frac
        del_mask = np.zeros((n * M, R), dtype=bool)
        ins_bp = np.zeros((n * M, R), dtype=np.int64)
        for i in range(E):
            c = cell[i]
            if is_ins[i]:
                a = int(start[i])
                if 0 <= a < R:
                    ins_bp[c, a] += L[i]
            else:
                s, e = max(0, int(start[i])), min(R, int(start[i] + L[i]))
                if s < e:
                    del_mask[c, s:e] = True
        ins_bp[del_mask] = 0
        return del_mask.reshape(n, M, R), ins_bp.reshape(n, M, R)

    param_rng = np.random.default_rng(24)
    for _ in range(30):
        n = int(param_rng.integers(1, 3))
        M = int(param_rng.integers(1, 4))
        R = int(param_rng.integers(20, 200))
        density = float(param_rng.uniform(1e-4, 1e-2))
        ins_frac = float(param_rng.uniform(0.2, 0.8))
        large_frac = float(param_rng.uniform(0.0, 0.3))
        small_alpha = float(param_rng.uniform(1.2, 2.5))
        small_max = int(param_rng.integers(2, 20))
        max_len = int(param_rng.integers(small_max + 5, 300))
        large_logmean = float(param_rng.uniform(2.0, 5.0))
        large_logsd = float(param_rng.uniform(0.5, 1.5))
        seed = int(param_rng.integers(0, 2**31 - 1))

        got_del, got_ins = _indel_tracts(
            np.random.default_rng(seed), n, M, R, density, ins_frac,
            large_frac, small_alpha, small_max, large_logmean, large_logsd,
            max_len)
        exp_del, exp_ins = reference(
            np.random.default_rng(seed), n, M, R, density, ins_frac,
            large_frac, small_alpha, small_max, large_logmean, large_logsd,
            max_len)
        np.testing.assert_array_equal(got_del, exp_del)
        np.testing.assert_array_equal(got_ins, exp_ins.astype(np.int32))


# --- flag-off backward-compat regression ---------------------------------

def test_simulate_flag_off_matches_golden_hash_coalescent():
    import hashlib
    rng = np.random.default_rng(42)
    out, *_ = simulate(
        rng, windows=50, sites=64, founders=8, min_cross=2, max_cross=6,
        inbreeding=0.5, allele_sharing=0.2, bad_frac=0.05,
        sharing_model="coalescent", sharing_theta=4.0, ancestors=6,
        ancestor_crossovers=8, derived_sfs=0.3, read_snps=8,
        gamete_balance=0.5, chunk=1000)
    assert out.shape == (50, 64, 10)
    assert out.dtype == np.int8
    assert hashlib.sha256(out.tobytes()).hexdigest() == _GOLDEN_COALESCENT_SHA256


def test_simulate_flag_off_matches_golden_hash_independent():
    import hashlib
    rng = np.random.default_rng(7)
    out, *_ = simulate(
        rng, windows=30, sites=48, founders=6, min_cross=1, max_cross=4,
        inbreeding=1.0, allele_sharing=0.2, bad_frac=0.05,
        sharing_model="independent", chunk=1000)
    assert out.shape == (30, 48, 8)
    assert out.dtype == np.int8
    assert hashlib.sha256(out.tobytes()).hexdigest() == _GOLDEN_INDEPENDENT_SHA256


def test_simulate_flag_off_deterministic_same_seed():
    kw = dict(windows=20, sites=32, founders=6, min_cross=1, max_cross=3,
              inbreeding=0.5, allele_sharing=0.2, bad_frac=0.05,
              sharing_model="coalescent", sharing_theta=3.0, chunk=1000)
    a, *_ = simulate(np.random.default_rng(99), **kw)
    b, *_ = simulate(np.random.default_rng(99), **kw)
    np.testing.assert_array_equal(a, b)
    c, *_ = simulate(np.random.default_rng(100), **kw)
    assert not np.array_equal(a, c)


# --- _draw_lineages / _coalescent_feats(lineage=, per_gamete=) -----------

def test_draw_lineages_shape_and_range():
    rng = np.random.default_rng(30)
    n, T, K, A = 3, 40, 5, 4
    lineage, M = _draw_lineages(rng, n, T, K, A, anc_cx=3, rate=None,
                                 theta=None, max_lineages=None)
    assert lineage.shape == (n, K, T)
    assert M == A
    assert (lineage >= 0).all() and (lineage < M).all()


def test_draw_lineages_theta_mode_shape_and_range():
    rng = np.random.default_rng(31)
    n, T, K = 2, 40, 5
    lineage, M = _draw_lineages(rng, n, T, K, A=4, anc_cx=3, rate=None,
                                 theta=2.0, max_lineages=8)
    assert lineage.shape == (n, K, T)
    assert M == 8
    assert (lineage >= 0).all() and (lineage < M).all()


def test_coalescent_feats_precomputed_lineage_requires_M():
    rng = np.random.default_rng(32)
    n, T, K = 2, 20, 4
    lineage, M = _draw_lineages(rng, n, T, K, A=3, anc_cx=2, rate=None,
                                 theta=None, max_lineages=None)
    h1 = np.zeros((n, T), dtype=np.int64)
    good = _good_mask(rng, n, T, bad_frac=0.0, block=1.0)
    with pytest.raises(ValueError, match="lineage_M"):
        _coalescent_feats(rng, n, T, K, A=3, anc_cx=2, sfs_shape=0.3,
                           read_snps=4, h1=h1, h2=h1, rate=None, good=good,
                           gamete=good, lineage=lineage)


def test_coalescent_feats_per_gamete_shape_dtype():
    rng = np.random.default_rng(33)
    n, T, K = 3, 25, 5
    lineage, M = _draw_lineages(rng, n, T, K, A=4, anc_cx=2, rate=None,
                                 theta=None, max_lineages=None)
    h1 = rng.integers(0, K, size=(n, T))
    h2 = rng.integers(0, K, size=(n, T))
    good = _good_mask(rng, n, T, bad_frac=0.05, block=1.0)
    (match1, match2), lin_out, panel = _coalescent_feats(
        rng, n, T, K, A=4, anc_cx=2, sfs_shape=0.3, read_snps=4,
        h1=h1, h2=h2, rate=None, good=good, gamete=good,
        lineage=lineage, lineage_M=M, per_gamete=True)
    assert match1.shape == (n, T, K)
    assert match2.shape == (n, T, K)
    assert match1.dtype == np.int8 and match2.dtype == np.int8
    assert set(np.unique(match1)) <= {0, 1}
    assert set(np.unique(match2)) <= {0, 1}
    np.testing.assert_array_equal(lin_out, lineage)


def test_coalescent_feats_per_gamete_matches_at_good_sites_when_inbred():
    # h1 == h2 (fully inbred): at GOOD (uncorrupted) sites both gametes
    # observe the identical active founder's mini-haplotype, so their
    # match vectors must agree exactly there. Bad sites are excluded --
    # each gamete draws its OWN independent corruption founder (real
    # reads corrupt independently), so they may legitimately disagree
    # exactly at bad sites.
    rng = np.random.default_rng(34)
    n, T, K = 4, 60, 6
    lineage, M = _draw_lineages(rng, n, T, K, A=5, anc_cx=3, rate=None,
                                 theta=None, max_lineages=None)
    h1 = rng.integers(0, K, size=(n, T))
    h2 = h1.copy()
    good = _good_mask(rng, n, T, bad_frac=0.1, block=1.0)
    (match1, match2), _, _ = _coalescent_feats(
        rng, n, T, K, A=5, anc_cx=3, sfs_shape=0.3, read_snps=4,
        h1=h1, h2=h2, rate=None, good=good, gamete=good,
        lineage=lineage, lineage_M=M, per_gamete=True)
    good_bc = np.broadcast_to(good[:, :, None], match1.shape)
    np.testing.assert_array_equal(match1[good_bc], match2[good_bc])


# --- _indel_suppressed_rate -----------------------------------------------

def test_indel_suppressed_rate_no_variation_leaves_rate_unchanged():
    del_mask = np.zeros((2, 3, 50), dtype=bool)
    ins_bp = np.zeros((2, 3, 50), dtype=np.int32)
    rate = np.full((2, 50), 7.0)
    out = _indel_suppressed_rate(rate, del_mask, ins_bp, suppress=0.9, flank=5)
    np.testing.assert_allclose(out, rate)


def test_indel_suppressed_rate_none_rate_starts_from_ones():
    del_mask = np.zeros((1, 2, 20), dtype=bool)
    ins_bp = np.zeros((1, 2, 20), dtype=np.int32)
    out = _indel_suppressed_rate(None, del_mask, ins_bp, suppress=0.9, flank=0)
    np.testing.assert_allclose(out, np.ones((1, 20)))


def test_indel_suppressed_rate_never_hits_zero():
    rng = np.random.default_rng(40)
    del_mask = rng.random((3, 4, 200)) < 0.9   # heavily variable, near-worst-case
    ins_bp = np.zeros((3, 4, 200), dtype=np.int32)
    out = _indel_suppressed_rate(None, del_mask, ins_bp, suppress=0.9, flank=10)
    assert (out > 0).all()


def test_indel_suppressed_rate_lower_inside_and_near_tracts():
    del_mask = np.zeros((1, 4, 60), dtype=bool)
    del_mask[:, :, 20:30] = True   # every lineage deleted over [20,30)
    ins_bp = np.zeros((1, 4, 60), dtype=np.int32)
    out = _indel_suppressed_rate(None, del_mask, ins_bp, suppress=0.9, flank=3)
    # strictly suppressed well inside the tract and its flank
    assert (out[0, 20:30] < 0.2).all()
    assert (out[0, 17:20] < 1.0).all() and (out[0, 30:33] < 1.0).all()
    # unaffected far from the tract
    assert np.allclose(out[0, 0:15], 1.0)
    assert np.allclose(out[0, 40:60], 1.0)


def test_indel_suppressed_rate_dilation_tiny_exact():
    del_mask = np.array([[[False, False, True, True, False, False]]])
    ins_bp = np.zeros((1, 1, 6), dtype=np.int32)
    out = _indel_suppressed_rate(None, del_mask, ins_bp, suppress=0.9, flank=1)
    np.testing.assert_allclose(out, np.array([[1.0, 0.1, 0.1, 0.1, 0.1, 1.0]]))


# --- _row_counts ---------------------------------------------------------

def test_row_counts_shape_dtype():
    rng = np.random.default_rng(50)
    n, R = 3, 40
    pres1 = rng.random((n, R)) < 0.7
    pres2 = rng.random((n, R)) < 0.7
    ins1 = rng.integers(0, 5000, (n, R)) * pres1
    ins2 = rng.integers(0, 5000, (n, R)) * pres2
    on1, on2, c1, c2, cnt = _row_counts(rng, pres1, pres2, ins1, ins2,
                                         gamete_balance=0.5, coverage=2.0,
                                         ins_read_per_bp=2e-3, max_stack=64)
    assert on1.dtype == bool and on2.dtype == bool
    assert c1.dtype == np.int32 and c2.dtype == np.int32
    assert (c1 >= 0).all() and (c1 <= 64).all()
    assert (c2 >= 0).all() and (c2 <= 64).all()
    np.testing.assert_array_equal(cnt, on1.astype(np.int32) + on2.astype(np.int32) + c1 + c2)


def test_row_counts_absent_never_gets_a_colinear_read():
    rng = np.random.default_rng(51)
    n, R = 1, 200
    pres1 = np.zeros((n, R), dtype=bool)
    pres2 = np.zeros((n, R), dtype=bool)
    ins1 = np.zeros((n, R), dtype=np.int64)
    ins2 = np.zeros((n, R), dtype=np.int64)
    on1, on2, c1, c2, cnt = _row_counts(rng, pres1, pres2, ins1, ins2,
                                         gamete_balance=0.5, coverage=1e9,
                                         ins_read_per_bp=0.0, max_stack=64)
    assert not on1.any() and not on2.any()
    assert (cnt == 0).all()


def test_row_counts_hemizygous_only_surviving_homolog_reads_exact():
    # coverage huge -> colinear Bernoulli saturates to ~1 for any present
    # homolog; H2 absent everywhere -> every colinear read must come from
    # H1 only, exactly the "hemizygous" acceptance-criterion mechanism.
    rng = np.random.default_rng(52)
    n, R = 1, 500
    pres1 = np.ones((n, R), dtype=bool)
    pres2 = np.zeros((n, R), dtype=bool)
    ins1 = np.zeros((n, R), dtype=np.int64)
    ins2 = np.zeros((n, R), dtype=np.int64)
    on1, on2, c1, c2, cnt = _row_counts(rng, pres1, pres2, ins1, ins2,
                                         gamete_balance=0.5, coverage=1e9,
                                         ins_read_per_bp=0.0, max_stack=64)
    assert on1.all()
    assert not on2.any()


def test_row_counts_stack_capped_at_max_stack():
    rng = np.random.default_rng(53)
    n, R = 1, 10
    pres1 = np.ones((n, R), dtype=bool)
    pres2 = np.zeros((n, R), dtype=bool)
    ins1 = np.full((n, R), 1_000_000, dtype=np.int64)  # huge insertion
    ins2 = np.zeros((n, R), dtype=np.int64)
    _, _, c1, c2, _ = _row_counts(rng, pres1, pres2, ins1, ins2,
                                   gamete_balance=0.5, coverage=0.0,
                                   ins_read_per_bp=1.0, max_stack=64)
    assert (c1 == 64).all()
    assert (c2 == 0).all()


# --- _sample_rows ----------------------------------------------------

def test_sample_rows_tiny_exact():
    cnt = np.array([[0, 2, 0, 1, 1], [1, 0, 0, 0, 0]])
    w_of, t_of, row_of, o_of, short = _sample_rows(cnt, T=3)
    np.testing.assert_array_equal(w_of, [0, 0, 0, 1])
    np.testing.assert_array_equal(t_of, [1, 1, 3, 0])
    np.testing.assert_array_equal(row_of, [0, 1, 2, 0])
    # the two rows stacked at (window0, site1) get within-site ordinals
    # 0 and 1; every other kept row is alone at its site -> ordinal 0.
    np.testing.assert_array_equal(o_of, [0, 1, 0, 0])
    np.testing.assert_array_equal(short, [False, True])


def test_sample_rows_empty_cnt():
    cnt = np.zeros((3, 10), dtype=np.int64)
    w_of, t_of, row_of, o_of, short = _sample_rows(cnt, T=5)
    assert w_of.size == 0 and t_of.size == 0 and row_of.size == 0 and o_of.size == 0
    np.testing.assert_array_equal(short, [True, True, True])


def test_sample_rows_exactly_T_no_shortfall():
    cnt = np.zeros((2, 10), dtype=np.int64)
    cnt[0, [1, 4, 7]] = 1   # exactly T=3 rows
    cnt[1, [0, 1, 2, 3, 4]] = 1  # 5 rows, more than T=3
    w_of, t_of, row_of, o_of, short = _sample_rows(cnt, T=3)
    np.testing.assert_array_equal(short, [False, False])
    # each window contributes exactly T rows
    for w in (0, 1):
        assert (w_of == w).sum() == 3


def test_sample_rows_rows_within_window_are_reference_ordered():
    rng = np.random.default_rng(54)
    cnt = rng.integers(0, 4, size=(5, 60))
    w_of, t_of, row_of, o_of, short = _sample_rows(cnt, T=20)
    for w in range(5):
        sites = t_of[w_of == w]
        assert (np.diff(sites) >= 0).all()
        rows = row_of[w_of == w]
        np.testing.assert_array_equal(rows, np.arange(rows.size))


def test_sample_rows_o_of_within_site_ordinal_exact():
    # 3 rows stacked at (window0, site2): within-site ordinals 0,1,2.
    cnt = np.array([[0, 0, 3, 0]])
    w_of, t_of, row_of, o_of, short = _sample_rows(cnt, T=3)
    np.testing.assert_array_equal(t_of, [2, 2, 2])
    np.testing.assert_array_equal(o_of, [0, 1, 2])


def test_sample_rows_matches_pure_python_oracle_fuzz():
    def reference(cnt, T):
        n, R = cnt.shape
        w_out, t_out, r_out, o_out = [], [], [], []
        short = np.zeros(n, dtype=bool)
        for w in range(n):
            rows = []   # (site, within-site ordinal)
            for t in range(R):
                for o in range(int(cnt[w, t])):
                    rows.append((t, o))
            short[w] = len(rows) < T
            for r, (t, o) in enumerate(rows[:T]):
                w_out.append(w); t_out.append(t); r_out.append(r); o_out.append(o)
        return (np.array(w_out, dtype=np.int64), np.array(t_out, dtype=np.int64),
                np.array(r_out, dtype=np.int64), np.array(o_out, dtype=np.int64), short)

    rng = np.random.default_rng(55)
    for _ in range(100):
        n = int(rng.integers(1, 4))
        R = int(rng.integers(1, 30))
        T = int(rng.integers(1, 15))
        cnt = rng.integers(0, 4, size=(n, R))
        got = _sample_rows(cnt, T)
        exp = reference(cnt, T)
        for g, e in zip(got, exp):
            np.testing.assert_array_equal(g, e)


# --- _indel_chunk ----------------------------------------------------

def _chunk_fixture():
    """A fully deterministic (coverage=1e9, ins_read_per_bp=1.0) 2-founder,
    2-lineage scenario, hand-traced then numerically verified before being
    fixed as a test value: founder0 (lineage0) is deleted at site2;
    founder1 (lineage1) carries a 3000bp insertion anchored at site1;
    H1 is always founder0, H2 always founder1."""
    n, R, K = 1, 6, 2
    lineage = np.array([[[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]], dtype=np.int64)
    del_lin = np.array([[[False, False, True, False, False, False],
                          [False, False, False, False, False, False]]])
    ins_lin = np.zeros((1, 2, 6), dtype=np.int32)
    ins_lin[0, 1, 1] = 3000
    h1 = np.zeros((1, 6), dtype=np.int64)
    h2 = np.ones((1, 6), dtype=np.int64)
    match1 = np.zeros((1, 6, 2), dtype=np.int8); match1[0, :, 0] = 1
    match2 = np.zeros((1, 6, 2), dtype=np.int8); match2[0, :, 1] = 1
    return n, R, K, lineage, del_lin, ins_lin, h1, h2, match1, match2


def test_indel_chunk_tiny_exact():
    n, R, K, lineage, del_lin, ins_lin, h1, h2, match1, match2 = _chunk_fixture()
    T = 8
    rng = np.random.default_rng(0)
    tern, dist, lab1, lab2, refpos, short = _indel_chunk(
        rng, n, R, T, K, h1, h2, lineage, del_lin, ins_lin, match1, match2,
        gamete_balance=0.5, coverage=1e9, ins_read_per_bp=1.0, max_stack=2,
        anchor_thresh=0, ref_founder=-1, dist_scale=DIST_LOG_SCALE)

    expect_tern = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [0, 1],
                             [-1, 1], [1, 0]], dtype=np.int8)
    expect_dist = np.array([[0, 0]] * 6 + [[8, 0], [0, 0]], dtype=np.int8)
    np.testing.assert_array_equal(tern[0], expect_tern)
    np.testing.assert_array_equal(dist[0], expect_dist)
    np.testing.assert_array_equal(lab1[0], np.zeros(8, dtype=np.int8))
    np.testing.assert_array_equal(lab2[0], np.ones(8, dtype=np.int8))
    np.testing.assert_array_equal(refpos[0], [0, 0, 1, 1, 1, 1, 2, 3])
    np.testing.assert_array_equal(short, [False])


def test_indel_chunk_shape_dtype():
    n, R, K, lineage, del_lin, ins_lin, h1, h2, match1, match2 = _chunk_fixture()
    T = 5
    rng = np.random.default_rng(1)
    tern, dist, lab1, lab2, refpos, short = _indel_chunk(
        rng, n, R, T, K, h1, h2, lineage, del_lin, ins_lin, match1, match2,
        gamete_balance=0.5, coverage=2.0, ins_read_per_bp=2e-3, max_stack=64,
        anchor_thresh=0, ref_founder=-1, dist_scale=DIST_LOG_SCALE)
    assert tern.shape == (n, T, K) and tern.dtype == np.int8
    assert dist.shape == (n, T, K) and dist.dtype == np.int8
    assert lab1.shape == (n, T) and lab2.shape == (n, T)
    assert refpos.shape == (n, T) and refpos.dtype == np.int32
    assert short.shape == (n,)


def test_indel_chunk_rare_shortfall_pads_and_flags():
    # T far larger than any realistic row count over a tiny R with zero
    # coverage/insertions -> every window falls short and must be padded
    # with the module's sentinels, not silently truncated to fewer rows.
    n, R, K = 2, 4, 3
    lineage = np.zeros((n, K, R), dtype=np.int64)
    del_lin = np.zeros((n, K, R), dtype=bool)
    ins_lin = np.zeros((n, K, R), dtype=np.int32)
    h1 = np.zeros((n, R), dtype=np.int64)
    h2 = np.zeros((n, R), dtype=np.int64)
    match1 = np.zeros((n, R, K), dtype=np.int8)
    match2 = np.zeros((n, R, K), dtype=np.int8)
    T = 50
    rng = np.random.default_rng(2)
    tern, dist, lab1, lab2, refpos, short = _indel_chunk(
        rng, n, R, T, K, h1, h2, lineage, del_lin, ins_lin, match1, match2,
        gamete_balance=0.5, coverage=0.0, ins_read_per_bp=0.0, max_stack=1,
        anchor_thresh=0, ref_founder=-1, dist_scale=DIST_LOG_SCALE)
    assert short.all()
    assert (tern == TERN_PAD).all()
    assert (dist == DIST_PAD).all()
    assert (lab1 == LABEL_PAD).all() and (lab2 == LABEL_PAD).all()
    assert (refpos == -1).all()


def test_indel_chunk_ternary_never_one_where_deleted_fuzz():
    rng = np.random.default_rng(3)
    for _ in range(20):
        n, M, K, R, T = 2, 4, 3, 60, 40
        lineage = rng.integers(0, M, size=(n, K, R)).astype(np.int64)
        del_lin, ins_lin = _indel_tracts(rng, n, M, R, **_TRACT_KW)
        h1 = rng.integers(0, K, size=(n, R))
        h2 = rng.integers(0, K, size=(n, R))
        match1 = (rng.random((n, R, K)) < 0.3).astype(np.int8)
        match2 = (rng.random((n, R, K)) < 0.3).astype(np.int8)
        tern, dist, lab1, lab2, refpos, short = _indel_chunk(
            rng, n, R, T, K, h1, h2, lineage, del_lin, ins_lin, match1, match2,
            gamete_balance=0.5, coverage=2.0, ins_read_per_bp=2e-3,
            max_stack=64, anchor_thresh=0, ref_founder=-1,
            dist_scale=DIST_LOG_SCALE)
        real = tern != TERN_PAD
        # decode: dist code 0 means real distance 0 (colinear); any nonzero
        # code means a real deletion at that (row,founder) under thresh=0
        deleted = (dist > 0) & (dist != DIST_PAD)
        assert not ((tern == 1) & deleted & real).any()
        assert ((tern == -1) == (deleted & real))[real].all()


def test_indel_chunk_insertion_stacking_appears_with_enough_budget():
    n, R, K, lineage, del_lin, ins_lin, h1, h2, match1, match2 = _chunk_fixture()
    T = 8
    rng = np.random.default_rng(4)
    tern, dist, lab1, lab2, refpos, short = _indel_chunk(
        rng, n, R, T, K, h1, h2, lineage, del_lin, ins_lin, match1, match2,
        gamete_balance=0.5, coverage=1e9, ins_read_per_bp=1.0, max_stack=2,
        anchor_thresh=0, ref_founder=-1, dist_scale=DIST_LOG_SCALE)
    # site 1 (the insertion anchor) must appear more than twice (its 2
    # colinear reads) -- the extra occurrences are the stacked insertion
    # reads, all landing at the SAME reference position.
    assert (refpos[0] == 1).sum() > 2
