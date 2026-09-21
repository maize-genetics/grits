"""
Simulate shared-allele patterns for GRITS founder-path imputation.

Generates a well-posed synthetic imputation problem: each window is a piecewise
constant founder path (the recombination mosaic) observed only through a noisy
allele-sharing pattern.  The model must recover the path from temporal
continuity plus the fact that the true founder almost always shares the
sample's allele.

Per site the K-wide feature vector is a binary allele-match pattern:
  feature[f] = 1  if founder f carries the same allele as the sample, else 0.
The true founder is forced to 1 on "good" sites; on "bad" sites the pattern is
left fully random so the true founder need not match (genotyping error).
Average allele sharing p sets the rate at which *other* founders also match,
q = (p*K - 1) / (K - 1), so the expected fraction of matching founders is p.

Inbreeding coefficient F controls haplotype identity: with probability F the two
haplotypes (H1, H2) share one path (fully inbred when F=1); otherwise H2 is an
independent path.

Diploid read sampling: each of the T positions is ONE read drawn from a single
gamete — H1's chromosome with prob `--gamete-balance`, else H2's. The site's
feature is the allele-match pattern of that one gamete's founder only (not both).
So in an outbred window (F=0) consecutive reads alternate between two independent
founder paths, and the model must disentangle the interleaved sharing patterns
using continuity within each path. Hemizygosity/allele-dropout is intrinsic (only
one chromosome is seen per site); `--bad-frac` adds explicit corrupt/missing
reads. When F=1 (h1==h2) the active founder is identical for both gametes, so the
output is unchanged from the haploid layout (backward compatible).

Recombination-rate heterogeneity (E2): with --recomb-span > 1 the breakpoints are
placed proportional to a per-window, per-site recombination-rate map that is
piecewise-constant in `--recomb-tile`-site blocks and spans `recomb_span`-fold
(log-uniform hot/cold regions).  This rate map is HIDDEN — it is never a model input.
The encoder must *infer* the local recombination rate from the allele-match patterns
(LD structure) and feed it into the CRF transition cost c_t, because at inference the
true map is unknown.  The true rate is appended as a trailing column for EVALUATION
ONLY (to test whether the inferred c_t tracks it); the training dataset slices only
the K feature columns, so it is never seen during training.
With --recomb-span 1 (default) recombination is spatially uniform and no track is
emitted, reproducing the E1 layout.

Output  (matches LabeledDatasetDiploid):
    NPY int8 tensor [windows, sites, founders + 2 (+1)]
      cols 0:K   allele-match features
      col  K     H1 founder label  (0..K-1)
      col  K+1   H2 founder label  (0..K-1)
      col  K+2   per-site TRUE recomb rate  (eval-only diagnostic, never a model
                 input; present only when --recomb-span > 1)

Usage:
    pixi run -- python src/python/crf/simulate_alleles.py \
        --workdir /workdir/esb33 --windows 100000               # E1 (uniform)
    pixi run -- python src/python/crf/simulate_alleles.py \
        --workdir /workdir/esb33 --recomb-span 100 --out sim_alleles_e2.npy  # E2
"""

import argparse
from pathlib import Path

import numpy as np

# --- Indel-mode (--simulate-indels) sentinels and constants ---------------
# Matrix-1 ternary state, mirroring ropebwt3-phg's rb3_lift_ternary_state
# exactly: deletion / diverged / match relative to reference. TERN_PAD is
# reserved for the rare edge case where a window's generation region could
# not produce T real rows (see _sample_rows) -- never written on a normal
# window.
TERN_DEL, TERN_DIV, TERN_MATCH, TERN_PAD = -1, 0, 1, -2
# Matrix-2 (distance-to-anchor) padding and saturation sentinels, int8.
DIST_PAD = -1
DIST_SAT = 127
# H1/H2 label padding on the same rare-edge-case rows. Kept at -1 (not K),
# matching the existing real-data convention documented in
# ropebwt_npy_to_matrix.py: the producer emits -1 for "no label", the
# consumer remaps -1 -> K (the CRF's own unknown-founder state).
LABEL_PAD = -1
# Log-scale compression for the distance-to-anchor block, matching the
# existing recomb-rate track's clip-and-round precedent (:threshold near
# line 492 below): code = clip(round(scale*log2(1+d)), 0, DIST_SAT-1).
DIST_LOG_SCALE = 8.0


def _rate_map(rng, n, T, span, tile):
    """Hidden per-window recomb-rate map [n, T] in [1, span].

    Piecewise-constant in `tile`-site blocks, each block log-uniform over
    [1, span] so the rate varies ~span-fold between hot and cold regions.
    """
    n_tiles = int(np.ceil(T / tile))
    coarse = np.exp(rng.uniform(0.0, np.log(span), size=(n, n_tiles)))
    return np.repeat(coarse, tile, axis=1)[:, :T]


def _build_paths(rng, n, T, K, n_cross, rate=None):
    """Piecewise-constant founder paths [n, T] with exactly n_cross switches each.

    n_cross is a [n] array; windows are grouped by switch count so the whole
    thing stays vectorized.  Adjacent segments are guaranteed to differ.
    If `rate` [n, T] is given, breakpoints are placed proportional to the local
    rate (Gumbel-top-k sampling without replacement); else uniform.
    """
    path = np.empty((n, T), dtype=np.int64)
    for v in np.unique(n_cross):
        rows = np.flatnonzero(n_cross == v)
        nv = rows.size

        # v distinct breakpoints in [1, T-1] per window
        if rate is None:
            bp = np.argsort(rng.random((nv, T - 1)), axis=1)[:, :v] + 1
        else:
            # Gumbel-top-k: top-v of (log rate + Gumbel) samples v distinct
            # positions with prob proportional to the local recomb rate.
            gum = -np.log(-np.log(rng.random((nv, T - 1))))
            keys = np.log(rate[rows][:, 1:]) + gum
            bp = np.argsort(-keys, axis=1)[:, :v] + 1
        bp.sort(axis=1)

        # v+1 segment founders, adjacent guaranteed distinct
        seg = np.empty((nv, v + 1), dtype=np.int64)
        seg[:, 0] = rng.integers(0, K, nv)
        for s in range(1, v + 1):
            step = rng.integers(1, K, nv)          # 1..K-1 -> never repeats prev
            seg[:, s] = (seg[:, s - 1] + step) % K

        # segment index at each site = #breakpoints <= t
        sw = np.zeros((nv, T), dtype=np.int32)
        np.add.at(sw, (np.repeat(np.arange(nv), v), bp.ravel()), 1)
        seg_idx = np.cumsum(sw, axis=1)            # 0..v
        path[rows] = np.take_along_axis(seg, seg_idx, axis=1)
    return path


def _individual_assignment(rng, windows, G, K, kmin, kmax):
    """Per-window individual id + founder subset for E5 grouping.

    Every G consecutive windows form one individual that descends from a random
    subset of k ~ Uniform[kmin, kmax] of the K founders.  Returns:
      ind   [windows]    int   individual id (window // G)
      win_k [windows]    int   #founders available to that window's individual
      sub   [windows, K] int   subset[w, :win_k[w]] = the available founder ids,
                               the rest are -1 (padding).
    """
    n_ind = windows // G
    if n_ind * G != windows:
        raise ValueError(
            f"--windows {windows} not divisible by --windows-per-individual {G}")
    k_ind = rng.integers(kmin, kmax + 1, n_ind)
    sub_ind = np.full((n_ind, K), -1, dtype=np.int64)
    for i in range(n_ind):
        sub_ind[i, :k_ind[i]] = rng.permutation(K)[:k_ind[i]]
    ind = np.repeat(np.arange(n_ind), G)
    return ind, np.repeat(k_ind, G), np.repeat(sub_ind, G, axis=0)


def _breeding_pop_assignment(rng, windows, G, K, classes, inbred_frac,
                              constant_pair_frac=0.0, constant_inbred_frac=0.5):
    """E11 breeding-population grouping. Each individual draws a founder-count
    CLASS (discrete mixture) and is inbred (F=1) or het with a class-specific
    TARGET het-loci fraction. `classes` = list of (weight, kmin, kmax, het_target).
    Returns per-window: ind [W], win_k [W], sub [W,K] (founder ids, -1 pad),
    het_t [W] (target het fraction; 0 if inbred), cls [W] (class index),
    is_const [W] (bool), const_h1 [W], const_h2 [W] (founder ids, meaningful
    only where is_const).

    Tier2 (crf-relatedness): `constant_pair_frac` of individuals bypass the
    class/het-target mechanism entirely and instead get a FIXED founder
    identity, constant across every T site of every G window -- 0
    breakpoints, het EXACTLY 0 or 1, not a target. Neither the class mixture
    above nor --mixed-inbreeding can reach k=1 (hard-guarded >=2 elsewhere)
    or a true het=1 k=2 individual (the realized het ceiling for an
    independent-tract k=2 individual is structurally 1-1/k=0.5, confirmed
    empirically -- see docs/RESULTS.md's E11 smoke-gen table) -- these are
    the two real-world regimes (true inbred line, true F1 hybrid) neither
    mechanism can produce. `constant_inbred_frac` of these individuals are
    single-founder homozygous (k=1, het=0); the rest are two-founder
    constant pairs (k=2, het=1). Class id 3 marks a constant individual in
    the returned `cls` array (the 3 breeding classes are 0/1/2)."""
    n_ind = windows // G
    if n_ind * G != windows:
        raise ValueError(f"--windows {windows} not divisible by "
                         f"--windows-per-individual {G}")

    is_const_ind = rng.random(n_ind) < constant_pair_frac
    const_inbred_ind = rng.random(n_ind) < constant_inbred_frac
    const_h1_ind = rng.integers(0, K, n_ind)
    const_h2_ind = const_h1_ind.copy()
    het_rows = np.flatnonzero(is_const_ind & ~const_inbred_ind)
    if het_rows.size:
        step = rng.integers(1, K, het_rows.size)      # 1..K-1 -> never repeats h1
        const_h2_ind[het_rows] = (const_h1_ind[het_rows] + step) % K

    w = np.array([c[0] for c in classes], dtype=float)
    w = w / w.sum()
    cls_ind = rng.choice(len(classes), size=n_ind, p=w)
    inbred = rng.random(n_ind) < inbred_frac
    k_ind = np.empty(n_ind, dtype=np.int64)
    het_ind = np.zeros(n_ind, dtype=np.float32)
    sub_ind = np.full((n_ind, K), -1, dtype=np.int64)
    for i in range(n_ind):
        if is_const_ind[i]:
            k = 1 if const_inbred_ind[i] else 2
            k_ind[i] = k
            sub_ind[i, 0] = const_h1_ind[i]
            if k == 2:
                sub_ind[i, 1] = const_h2_ind[i]
            het_ind[i] = 0.0 if const_inbred_ind[i] else 1.0
            continue
        _, kmin, kmax, het = classes[cls_ind[i]]
        k = int(kmin) if kmin == kmax else int(rng.integers(kmin, kmax + 1))
        k_ind[i] = k
        sub_ind[i, :k] = rng.permutation(K)[:k]
        het_ind[i] = 0.0 if inbred[i] else float(het)
    cls_out = np.where(is_const_ind, 3, cls_ind)
    rep = lambda a: np.repeat(a, G, axis=0)
    return (rep(np.arange(n_ind)), rep(k_ind), rep(sub_ind),
            rep(het_ind), rep(cls_out.astype(np.int64)),
            rep(is_const_ind), rep(const_h1_ind), rep(const_h2_ind))


def _build_paths_subset(rng, sub, win_k, T, n_cross, rate=None):
    """Founder paths [m, T] restricted per-row to that row's founder subset.

    sub [m, K] holds each window's available founder ids (padded with -1);
    win_k [m] their counts.  Paths are built in local index space [0, k) via
    `_build_paths`, then remapped to the global founder ids — so each individual's
    windows only ever visit its own founders.
    """
    m = sub.shape[0]
    path = np.empty((m, T), dtype=np.int64)
    for k in np.unique(win_k):
        rows = np.flatnonzero(win_k == k)
        local = _build_paths(rng, rows.size, T, int(k), n_cross[rows],
                             None if rate is None else rate[rows])
        path[rows] = np.take_along_axis(sub[rows], local, axis=1)
    return path


def _segment_index(rng, n, T, n_cross, rate=None):
    """Piecewise-constant segment index [n, T] with n_cross[r] breakpoints/row.

    Breakpoint placement mirrors `_build_paths` (uniform, or Gumbel-top-k by the
    local rate); returns only the segment index (0..n_cross) per site, leaving
    the label assignment to the caller.
    """
    seg = np.zeros((n, T), dtype=np.int32)
    for v in np.unique(n_cross):
        if v == 0:
            continue
        rows = np.flatnonzero(n_cross == v)
        nv = rows.size
        if rate is None:
            bp = np.argsort(rng.random((nv, T - 1)), axis=1)[:, :v] + 1
        else:
            gum = -np.log(-np.log(rng.random((nv, T - 1))))
            keys = np.log(rate[rows][:, 1:]) + gum
            bp = np.argsort(-keys, axis=1)[:, :v] + 1
        bp.sort(axis=1)
        sw = np.zeros((nv, T), dtype=np.int32)
        np.add.at(sw, (np.repeat(np.arange(nv), v), bp.ravel()), 1)
        seg[rows] = np.cumsum(sw, axis=1)
    return seg


def _gem_lineages(rng, n, K, T, theta, M, anc_nc, rate_rep):
    """Ewens/GEM(theta) IBD lineage labels [n, K, T].

    Per window: stick-breaking lineage proportions p ~ GEM(theta), truncated to
    M lineages.  Each founder is a mosaic (segments from `anc_nc` ancestral
    breakpoints); each segment's lineage is drawn iid from p.  iid assignment
    from a GEM(theta) measure yields the Ewens partition, whose class-size
    spectrum is n_i ∝ theta/i — unfolded, singleton-dominated, with a real tail
    to K (the neutral-coalescent SFS).  Larger theta ⇒ more singletons.
    """
    beta = rng.beta(1.0, theta, size=(n, M))
    rem = np.cumprod(np.concatenate(
        [np.ones((n, 1)), 1.0 - beta[:, :-1]], axis=1), axis=1)
    p = beta * rem                                      # [n, M] stick-breaking weights
    p /= p.sum(axis=1, keepdims=True)                   # normalize truncated tail
    cdf = np.cumsum(p, axis=1)                          # [n, M]; no giant last class

    seg = _segment_index(rng, n * K, T, anc_nc, rate_rep).reshape(n, K, T)
    max_seg = int(seg.max()) + 1
    u = rng.random((n, K, max_seg, 1))
    lin_seg = (u > cdf[:, None, None, :]).sum(axis=-1)  # [n,K,max_seg] categorical
    lin_seg = np.clip(lin_seg, None, M - 1)
    return np.take_along_axis(lin_seg, seg, axis=2).astype(np.int32)


def _good_mask(rng, n, T, bad_frac, block):
    """Boolean [n, T] of 'good' (uncorrupted) sites.

    block <= 1  → independent per-site (P(bad)=bad_frac), identical to the
    original sim.  block > 1 → autocorrelated bad runs via a 2-state Markov
    chain with stationary P(bad)=bad_frac and mean bad-run length = block, so
    genotyping/nomination error arrives in tracts (SV blocks, mis-mapping).
    """
    if block <= 1.0:
        return rng.random((n, T)) >= bad_frac
    p_bg = 1.0 / block                                  # bad → good
    p_gb = bad_frac * p_bg / max(1e-9, 1.0 - bad_frac)  # good → bad (sets stationary)
    good = np.empty((n, T), dtype=bool)
    u = rng.random((n, T))
    good[:, 0] = u[:, 0] >= bad_frac
    for t in range(1, T):
        prev = good[:, t - 1]
        flip = np.where(prev, u[:, t] < p_gb, u[:, t] < p_bg)
        good[:, t] = np.where(flip, ~prev, prev)
    return good


# --- Indel modeling (--simulate-indels), PLAN.md experiments/simulator- ---
# indels/PLAN.md SS2 (Design) / SS4 (reusable vs. new). Indels are a
# property of the founder's LINEAGE (SS2.2), not the raw founder id or the
# individual, so the functions below operate on the same lineage axis
# `_gem_lineages` already produces (`M` lineages, gathered onto founders via
# `_gather_by_lineage`, mirroring how `_coalescent_feats` gathers
# lin_alleles -> G).

def _indel_lengths(rng, size, large_frac, small_alpha, small_max,
                    large_logmean, large_logsd, max_len):
    """Indel event lengths [size] in bp, two-component mixture (PLAN SS2.4).

    Small component (replication slippage at homopolymer/microsatellite
    tracts): a discrete power law P(L) ~ L^-small_alpha on [1, small_max]
    via `rng.zipf`, clipped into range -- 1bp-dominant, matching the
    measured ~41.5%-of-events-at-1bp / ~0.07%-of-bp shape (see
    experiments/simulator-indels/results/indel_biology_notes.md).
    Large component (LTR-retrotransposon insertion/removal): LogNormal in
    bp, clipped to [small_max+1, max_len] -- this component carries the
    large majority of indel bp on a small minority of events.

    Fully vectorized, no Python-level loop: both components are drawn at
    the full `size` and selected by one Bernoulli mask, so rng consumption
    is always exactly 3 draws (mask, small, large) regardless of the
    mixture weight -- required so a reference-oracle reimplementation can
    reproduce the exact draw sequence for testing.
    """
    is_large = rng.random(size) < large_frac
    small = np.clip(rng.zipf(small_alpha, size), 1, small_max)
    large = np.clip(np.rint(rng.lognormal(large_logmean, large_logsd, size)),
                     small_max + 1, max_len)
    return np.where(is_large, large, small).astype(np.int64)


def _encode_dist(d, scale=DIST_LOG_SCALE):
    """int (bp) distance -> int8 log-scale code.

    code = clip(round(scale * log2(1 + d)), 0, DIST_SAT - 1); exactly 0 at
    d=0 (colinear), monotone non-decreasing, naturally saturates toward
    DIST_SAT-1 for very large or "no real anchor found" distances (see
    `_anchor_distance`) -- same clip-and-round precedent as the existing
    recomb-rate track (:~510 below).
    """
    code = np.rint(scale * np.log2(1.0 + np.asarray(d, dtype=np.float64)))
    return np.clip(code, 0, DIST_SAT - 1).astype(np.int8)


def _anchor_distance(del_mask):
    """Distance [same leading shape as del_mask] to the nearest colinear
    (non-deleted) site along the last axis.

    0 wherever this lineage/founder is colinear (it IS an anchor); inside a
    deletion tract, the distance to the true NEAREST REAL anchor on either
    side -- matching rb3_lift_nearest_ref's nearest-anchor semantics.
    Deliberately does NOT fabricate a "virtual anchor" at the edge of the
    array for a tract that runs off either edge without a confirmed real
    anchor on that side: doing so would systematically UNDERESTIMATE the
    true distance for edge-touching tracts (verified numerically while
    designing this function -- a virtual-edge-anchor version gives
    distance 1 immediately at the array boundary regardless of how far the
    tract actually extends beyond what's simulated, which is a real,
    avoidable bias). Instead, a side with no real anchor in view uses a
    sentinel far outside the array's own index range, so `_encode_dist`
    naturally saturates it; only genuinely known anchors ever produce a
    small distance.

    Vectorized via the running max/min "nearest real anchor index seen so
    far" trick (`np.maximum.accumulate` / `np.minimum.accumulate`) instead
    of a per-tract Python loop; works on any leading shape (`[n,M,R]`
    lineage-space or `[n,K,R]` founder-space) since only the last axis is
    treated as the reference-site axis.
    """
    R = del_mask.shape[-1]
    idx = np.arange(R)
    neg_sentinel, pos_sentinel = -(R + 1), 2 * R
    last = np.maximum.accumulate(np.where(del_mask, neg_sentinel, idx), axis=-1)
    rev_del = del_mask[..., ::-1]
    rev_idx = idx[::-1]
    nxt = np.minimum.accumulate(
        np.where(rev_del, pos_sentinel, rev_idx), axis=-1)[..., ::-1]
    dist = np.minimum(idx - last, nxt - idx)
    return np.where(del_mask, dist, 0).astype(np.int32)


def _gather_by_lineage(a_lin, lineage):
    """[n,M,...] lineage-space array -> [n,K,...] founder-space, via the
    SAME `take_along_axis` gather `_coalescent_feats` uses for
    `lin_alleles -> G`, so IBD founders (those sharing a lineage) are
    guaranteed byte-identical rather than merely similar."""
    return np.take_along_axis(a_lin, lineage.astype(np.intp), axis=1)


def _indel_tracts(rng, n, M, R, density, ins_frac, large_frac, small_alpha,
                   small_max, large_logmean, large_logsd, max_len):
    """Per-LINEAGE indel structure over an R-site generation region
    (PLAN.md SS2.2, SS2.4): `(del_mask[n,M,R] bool, ins_bp[n,M,R] int32)`.

    Indels are a property of the LINEAGE, not the raw founder id -- this
    array is `[n,M,R]`, gathered onto founders via `_gather_by_lineage` the
    same way `_coalescent_feats` gathers `lin_alleles -> G`, which is what
    makes two IBD founders carry byte-identical indel content.

    Events are drawn as one flat bundle across all (window,lineage) cells
    (Poisson total count, then each event assigned a (window,lineage) cell
    and a start position), turned into an interval-union mask via the
    standard +1/-1 scatter + cumsum trick (`np.add.at`, no per-event Python
    loop). Insertions are point events (zero reference span): their length
    is scatter-ADDED into `ins_bp` at their single reference anchor, which
    is `start` itself -- the reference site immediately before the
    insertion begins (matching VCF/lift-file convention: an insertion has
    no reference span of its own, so its anchor is the last colinear
    reference base before it, not "nearest of either flank").

    Boundary correction: event start positions are drawn on the EXTENDED
    interval `[-max_len, R)`, not `[0, R)` -- a tract or a large insertion
    whose start falls before the region but whose body/anchor still
    overlaps it must still be represented, or long tracts are
    systematically undercounted near the region's left edge.
    """
    span = R + max_len
    lam = density * span
    cnt = rng.poisson(lam, size=n * M)
    E = int(cnt.sum())
    if E == 0:
        return (np.zeros((n, M, R), dtype=bool),
                np.zeros((n, M, R), dtype=np.int32))

    cell = np.repeat(np.arange(n * M), cnt)
    start = rng.integers(-max_len, R, E)
    L = _indel_lengths(rng, E, large_frac, small_alpha, small_max,
                        large_logmean, large_logsd, max_len)
    is_ins = rng.random(E) < ins_frac
    is_del = ~is_ins

    depth = np.zeros((n * M, R + 1), dtype=np.int16)
    if is_del.any():
        s = np.clip(start[is_del], 0, R)
        e = np.clip(start[is_del] + L[is_del], 0, R)
        np.add.at(depth, (cell[is_del], s), 1)
        np.add.at(depth, (cell[is_del], e), -1)
    del_mask = (np.cumsum(depth, axis=1)[:, :R] > 0).reshape(n, M, R)

    ins_bp = np.zeros((n * M, R), dtype=np.int32)
    if is_ins.any():
        a = start[is_ins]
        ok = (a >= 0) & (a < R)
        np.add.at(ins_bp, (cell[is_ins][ok], a[ok]), L[is_ins][ok])
    ins_bp = ins_bp.reshape(n, M, R)
    ins_bp[del_mask] = 0    # an insertion inside a deletion for the same
                             # lineage is not present in that haplotype
    return del_mask, ins_bp


def _lineage_indels(rng, n, K, M, R, lineage, frac, ins_frac, max_len):
    """v3 "lineage" `--indel-model`: derive indel structure directly from
    the EXISTING coalescent lineage mosaic instead of drawing a separate
    Poisson/length-mixture process (`_indel_tracts`'s "tracts" mode).

    For each (window, lineage m), a maximal contiguous run of sites where
    >=1 founder currently occupies lineage m is a candidate: with
    probability `frac` the WHOLE run is promoted to an indel (deletion,
    or with probability `ins_frac` an insertion anchored at the run's own
    start), using the run's OWN natural length -- no separate length-
    mixture knob, `max_len` only caps pathologically long runs for
    numeric/log-code sanity. Two founders sharing a lineage over the same
    interval are on the same run and so always get an identical decision,
    which is exactly how `_indel_tracts` + `_gather_by_lineage` already
    make IBD founders share indel content -- correlation here falls
    straight out of the SAME lineage structure that already drives SNP-
    level sharing, not a second stacked stochastic layer (contrast with
    `_indel_tracts`'s `--indel-shared-frac`/`--indel-cluster-theta`,
    which was added on top specifically to fake this).

    Implementation: `occ[n,m,r]` (any founder on lineage m at site r) is
    segmented into runs the same way `_anchor_distance` finds nearest
    anchors -- a "last run-start seen so far" / "next run-end seen from
    here" running accumulate along the site axis (`np.maximum.accumulate`
    / `np.minimum.accumulate`), not a per-run Python loop. One Bernoulli
    draw per (n,m,r) decides promotion/ins-vs-del; only the draw AT each
    run's own start position is used, gathered onto every other position
    in that run via the accumulated start index -- so only two full-size
    `[n,M,R]` draws are needed regardless of how many runs exist.

    Returns `(del_mask[n,M,R] bool, ins_bp[n,M,R] int32)`, the exact
    `_indel_tracts` output contract -- drop-in compatible with
    `_indel_suppressed_rate`/`_indel_chunk`, no other `simulate()` change
    needed for this mode's downstream handling.
    """
    occ = np.zeros((n, M, R), dtype=bool)
    ii = np.arange(n)[:, None]
    rr = np.arange(R)[None, :]
    for k in range(K):
        occ[ii, lineage[:, k, :], rr] = True

    pos = np.arange(R)[None, None, :]
    occ_prev = np.concatenate([np.zeros((n, M, 1), dtype=bool), occ[:, :, :-1]], axis=-1)
    occ_next = np.concatenate([occ[:, :, 1:], np.zeros((n, M, 1), dtype=bool)], axis=-1)
    run_start = occ & ~occ_prev
    run_end = occ & ~occ_next

    last_start_idx = np.maximum.accumulate(np.where(run_start, pos, -1), axis=-1)
    end_idx_at_pos = np.where(run_end, pos, R)
    next_end_idx = np.minimum.accumulate(end_idx_at_pos[:, :, ::-1], axis=-1)[:, :, ::-1]

    valid = occ & (last_start_idx >= 0)
    gather_idx = np.clip(last_start_idx, 0, R - 1)

    promote_draw = rng.random((n, M, R)) < frac
    ins_draw = rng.random((n, M, R)) < ins_frac
    promote = np.take_along_axis(promote_draw, gather_idx, axis=-1) & valid
    is_ins = np.take_along_axis(ins_draw, gather_idx, axis=-1)

    run_len_full = (next_end_idx - last_start_idx + 1).clip(1, max_len)
    run_len = np.take_along_axis(run_len_full, gather_idx, axis=-1)

    # A run longer than max_len must not delete more than max_len sites from
    # its own start -- without this, a single very-long-lived lineage (real
    # possibility under the Ewens/GEM SFS) can wipe out most of an
    # individual's coverage in one promoted run. Matches _indel_tracts's own
    # convention of capping an individual event's span at max_len.
    within_cap = (pos - gather_idx) < max_len
    del_mask = promote & ~is_ins & within_cap
    ins_bp = np.zeros((n, M, R), dtype=np.int32)
    ins_event = promote & is_ins & run_start
    ins_bp[ins_event] = run_len[ins_event]
    return del_mask, ins_bp


def _overlay_indels(rng, n, K, R, rate, mean_len, ins_frac, founder_freq, max_len):
    """v3 "overlay" `--indel-model`: indels fully decoupled from the
    coalescent lineage array -- each founder gets its own private
    process, not tied to lineage-sharing, so IBD founders do NOT
    automatically share indel content in this mode (an accepted,
    documented simplification: "reference-stable" is read here as
    "indel generation independent of the coalescent," while the
    coalescent still supplies the SNP-match signal so founder-affinity
    -- which only reads the SNP-match columns -- stays unaffected).

    Reuses `_indel_tracts` verbatim with M=K (each founder treated as its
    own private lineage slot) and a single LogNormal length scale
    (`large_frac=1.0`, median `mean_len`) instead of the two-component
    mixture -- one rate, one length scale, one founder-participation
    frequency; the caller must pass a per-founder-IDENTITY lineage array
    (`lineage[n,k,r]=k` for all r) to `_indel_suppressed_rate`/
    `_indel_chunk` instead of the real coalescent lineage array, which is
    what actually severs the tie to SNP-level IBD sharing (see
    `simulate()`'s `indel_model=="overlay"` branch).

    `founder_freq < 1` additionally gates entire founders out of indel
    generation (a fraction of founders carry no indels at all, on top of
    `rate` controlling event density among the founders that do).

    Returns `(del_mask[n,K,R] bool, ins_bp[n,K,R] int32)`.
    """
    del_mask, ins_bp = _indel_tracts(
        rng, n, K, R, rate, ins_frac,
        large_frac=1.0, small_alpha=1.7, small_max=50,
        large_logmean=float(np.log(max(mean_len, 1.0))), large_logsd=0.7,
        max_len=max_len)
    if founder_freq < 1.0:
        gate = (rng.random((n, K)) < founder_freq)[:, :, None]
        del_mask = del_mask & gate
        ins_bp = ins_bp * gate.astype(ins_bp.dtype)
    return del_mask, ins_bp


def _indel_suppressed_rate(rate, del_mask, ins_bp, suppress, flank):
    """Recomb-rate map [n, R] with crossover locally suppressed near
    indels (PLAN.md SS2.6), reusing the EXISTING `--recomb-span` rate-map
    channel rather than a new breakpoint mechanism -- the result is passed
    straight into `_build_paths` / `_build_paths_subset` /
    `_segment_index`'s existing `rate` argument, unchanged otherwise.

    Suppression is driven by the per-site FRACTION OF LINEAGES that are
    structurally variable, `f[n,R] = mean over M of (del_mask | ins_bp>0)`
    -- not by the individual's own two homologs, which don't exist yet at
    this point in the pipeline (that is exactly the circular dependency
    this design avoids: paths need the rate map; the rate map needs the
    tracts; the tracts are drawn independent of any individual's path).

    `f` is dilated by `flank` sites via an explicit sliding-window max (a
    small shift-and-max loop over `flank`, not scipy), then
    `rate *= clip(1 - suppress*f, floor, 1)`, floored at a small positive
    value so `_build_paths`'s `log(rate)` never sees exactly zero.

    `rate=None` (uniform recombination, `--recomb-span` 1) is handled by
    starting from an all-ones map, so suppression works even in the E1
    layout, not only when `--recomb-span > 1`.
    """
    n, M, R = del_mask.shape
    f = (del_mask | (ins_bp > 0)).astype(np.float64).mean(axis=1)  # [n,R]

    if flank > 0:
        padded = np.pad(f, ((0, 0), (flank, flank)), mode="edge")  # [n,R+2*flank]
        dil = padded[:, 0:R]
        for shift in range(1, 2 * flank + 1):
            dil = np.maximum(dil, padded[:, shift:shift + R])
        f = dil

    base = np.ones((n, R)) if rate is None else np.asarray(rate, dtype=np.float64)
    return base * np.clip(1.0 - suppress * f, 1e-3, 1.0)


def _read_fragment_coverage(rng, n, R, target_x, gamete_weight, read_len):
    """Boolean [n,R] "does a real read fragment cover this site" mask, for
    `coverage_model="reads"` -- the v4 fix for a gap `coverage_model=
    "poisson"` didn't close: matching the AGGREGATE per-site probability to
    a real depth target still leaves every site an INDEPENDENT Bernoulli
    trial, with no memory of its neighbors, so a 256bp bin gets 256
    independent chances to come up covered and ends up covered almost
    always -- real reads are literal `read_len`-bp CONTIGUOUS fragments
    with sparse, spatially clustered starts, so a real bin's coverage
    depends on whether ONE rare read happened to land nearby at all, not
    256 independent coin flips.

    Draws read START positions as a Poisson process along R at rate
    `target_x * gamete_weight / read_len` per bp (so the resulting mean
    per-site depth still equals the requested real target X), same
    "+1/-1 scatter, extended start range, cumsum interval union" trick
    `_indel_tracts` already uses for indel spans -- each start expands to
    a `read_len`-bp contiguous covered block, giving real reads' spatial
    clustering instead of `coverage_model="poisson"`'s per-site
    independence.
    """
    rate = target_x * gamete_weight / read_len
    lam = rate * R
    cnt = rng.poisson(lam, size=n)
    depth = np.zeros((n, R + 1), dtype=np.int32)
    E = int(cnt.sum())
    if E == 0:
        return np.zeros((n, R), dtype=bool)
    cell = np.repeat(np.arange(n), cnt)
    start = rng.integers(-read_len, R, E)   # extended range: a read starting just
    s = np.clip(start, 0, R)                # before the region can still cover its
    e = np.clip(start + read_len, 0, R)     # first few real sites
    np.add.at(depth, (cell, s), 1)
    np.add.at(depth, (cell, e), -1)
    return np.cumsum(depth, axis=1)[:, :R] > 0


def _row_counts(rng, pres1, pres2, ins1, ins2, gamete_balance, coverage,
                 ins_read_per_bp, max_stack, coverage_model="linear", read_len=150):
    """Reads emitted per (window, reference site) by source (PLAN.md
    SS2.5). Replaces today's "exactly ONE read per site" convention (this
    module's own docstring, and the `active = np.where(gamete, h1, h2)`
    line in `simulate()`) with a real coverage model:

      * colinear reads, per homolog: `Bernoulli(P) AND structurally
        present`, where `P` depends on `coverage_model`:
          - "linear" (default): `P = coverage * gamete_weight`.
            `gamete_weight` is `gamete_balance` for H1, `1 -
            gamete_balance` for H2, so `coverage=1` reproduces today's
            exactly-one-read density on average. At this module's own
            defaults (`coverage=2.0, gamete_balance=0.5`), `P=1.0` --
            every present site gets a GUARANTEED read, which is why
            T-row windows span far less reference bp than a real window
            of the same row count (v4 finding, see --indel-coverage-model
            help text).
          - "poisson" (v4): `P = 1 - exp(-coverage * gamete_weight)`, the
            standard Lander-Waterman coverage-depth relation for a real
            target sequencing depth `coverage` (e.g. 0.1 for 0.1x) --
            same Bernoulli structure, properly calibrated probability,
            but every site is still an INDEPENDENT trial: matches the
            real MEAN depth but not real reads' spatial clustering (a
            256bp bin gets 256 independent chances to come up covered,
            so it almost always does, unlike a real bin whose coverage
            depends on one sparse read happening to land nearby).
          - "reads" (v4): real `read_len`-bp CONTIGUOUS fragments via
            `_read_fragment_coverage`, Poisson-distributed START
            positions at rate `coverage*gamete_weight/read_len` per bp
            (same mean depth as "poisson", genuine spatial clustering) --
            the actual fix for the gap above.
        A hemizygous site (one homolog deleted) loses half its expected
        depth AND every read there comes from the surviving homolog --
        this IS the "hemizygous-looking read sharing" SS2.7 acceptance
        criterion, falling directly out of sampling rather than being
        special-cased.
      * insertion reads, per homolog: `Binomial(ins_bp, ins_read_per_bp)`
        capped at `max_stack`. Inserted sequence has no reference span, so
        all of these project to the SINGLE flanking reference anchor ->
        real row stacking at one position.
      * a nullizygous site (both homologs structurally absent) emits ZERO
        rows from either source.

    Returns `(on1, on2 [n,R] bool, c1, c2 [n,R] int32, cnt [n,R] int32)`
    where `cnt = on1 + on2 + c1 + c2`.
    """
    w1, w2 = gamete_balance, 1.0 - gamete_balance
    if coverage_model == "reads":
        n, R = pres1.shape
        on1 = _read_fragment_coverage(rng, n, R, coverage, w1, read_len) & pres1
        on2 = _read_fragment_coverage(rng, n, R, coverage, w2, read_len) & pres2
    else:
        if coverage_model == "linear":
            p1, p2 = coverage * w1, coverage * w2
        elif coverage_model == "poisson":
            p1, p2 = 1.0 - np.exp(-coverage * w1), 1.0 - np.exp(-coverage * w2)
        else:
            raise ValueError(f"coverage_model must be 'linear', 'poisson', or "
                              f"'reads', got {coverage_model!r}")
        on1 = (rng.random(pres1.shape) < p1) & pres1
        on2 = (rng.random(pres2.shape) < p2) & pres2
    p = float(np.clip(ins_read_per_bp, 0.0, 1.0))
    c1 = np.minimum(rng.binomial(ins1, p), max_stack).astype(np.int32)
    c2 = np.minimum(rng.binomial(ins2, p), max_stack).astype(np.int32)
    cnt = on1.astype(np.int32) + on2.astype(np.int32) + c1 + c2
    return on1, on2, c1, c2, cnt


def _sample_rows(cnt, T):
    """Flat row plan over the R-site generation region: which (window,
    site) each of the first `T` real rows (in reference-site order) comes
    from, per window -- a plain PREFIX-TAKE, not random thinning. Real
    ps4g windowing is exactly this operation: a window is "the next T rows
    in sorted order," not "rows covering a fixed reference-bp span" (this
    is the corrected design from this session -- see the plan's Context
    section for why the earlier padded-row-budget approach was dropped).

    `cnt[n,R]` gives the row count at each (window, reference site); rows
    are enumerated in non-decreasing site order within each window (the
    natural order of iterating sites 0..R-1), and only the first `T` per
    window are kept.

    Returns `(w_of, t_of, row_of, o_of [Rows] int64, short [n] bool)`. One
    entry per emitted row that's kept: `w_of`/`t_of` are its window and
    source reference site, `row_of` its destination row index `0..T-1`,
    `o_of` its WITHIN-SITE ordinal (0-based; distinguishes multiple rows
    stacked at the same site, in emission order -- needed by `_indel_chunk`
    to tell which underlying read/homolog a stacked row came from). `short`
    is True for windows whose TOTAL row count over the whole R-site region
    falls short of `T` -- the rare edge case; `_indel_chunk` pads those,
    see its docstring.
    """
    n, R = cnt.shape
    cnt64 = cnt.astype(np.int64)
    total = cnt64.sum(axis=1)
    short = total < T

    flat_cnt = cnt64.ravel()
    nz = np.flatnonzero(flat_cnt)
    if nz.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty, empty, short

    w_site = nz // R
    t_site = nz % R
    reps = flat_cnt[nz]
    w_of_all = np.repeat(w_site, reps)
    t_of_all = np.repeat(t_site, reps)

    order = np.arange(w_of_all.size)

    # Ordinal of each row WITHIN its (window, site) CELL (0-based, in
    # emission order): the standard "repeat with within-group index" trick
    # -- cumsum(reps)-reps is each cell's starting absolute-order index,
    # broadcast back out to its reps rows.
    cell_starts = np.repeat(np.cumsum(reps) - reps, reps)
    o_of_all = order - cell_starts

    # Ordinal of each row WITHIN its window (0-based, in emission order --
    # already non-decreasing site order since `nz` is sorted by flat index
    # window*R + site). `w_of_all` is sorted ascending by window, so the
    # first position of each window's block is a plain searchsorted.
    win_starts = np.searchsorted(w_of_all, np.arange(n))
    ordinal = order - win_starts[w_of_all]

    keep = ordinal < T
    return (w_of_all[keep], t_of_all[keep], ordinal[keep], o_of_all[keep],
            short)


def _draw_lineages(rng, n, T, K, A, anc_cx, rate, theta, max_lineages):
    """(lineage [n,K,T] int32, M int) -- the lineage draw hoisted out of
    `_coalescent_feats` (body identical to its original inline version,
    same rng call order) so indel tracts, which are a LINEAGE-level
    property (PLAN.md SS2.2), can be built BEFORE the founder paths are
    drawn -- required because the indel tracts feed the suppressed
    recombination rate map the paths are drawn on (SS2.6), which would
    otherwise be circular (paths need the rate map; the rate map needs the
    tracts; the tracts need the lineage assignment paths are drawn from).

    theta is None -> A fixed ancestors, each founder iid-uniform over them
    (legacy island model). theta set -> Ewens/GEM(theta) partition via
    `_gem_lineages`. M is the number of lineages actually used (A in the
    legacy case, max_lineages or K in the Ewens/GEM case).
    """
    anc_nc = rng.poisson(anc_cx, n * K).clip(0, T - 1).astype(np.int64)
    rate_rep = None if rate is None else np.repeat(rate, K, axis=0)
    if theta is None:
        M = A
        lineage = _build_paths(rng, n * K, T, A, anc_nc, rate_rep)
        lineage = lineage.reshape(n, K, T).astype(np.int32)
    else:
        M = max_lineages or K
        lineage = _gem_lineages(rng, n, K, T, theta, M, anc_nc, rate_rep)
    return lineage, M


def _lineage_substitutions(rng, n, M, T, L, rate, sfs_shape):
    """v4 "sparse" `--subst-model`: per-lineage mini-haplotype alleles with
    genuine POSITIONAL persistence, instead of `_coalescent_feats`'s default
    "dense" behavior of redrawing `f`/`lin_alleles` independently at every
    single site (`rng.beta(..., size=(n,1,T,L))` / `rng.random(...) < f` at
    `size=(n,M,T,L)`) -- a fresh coin-flip per site with no memory of where a
    substitution "happened," so two non-IBD lineages that coincide at one
    site have no more or less chance of coinciding at the very next site.
    Real substitutions occur once, at one genomic position, and are visible
    identically at every site downstream of it; this function reproduces
    that structure directly.

    Draws substitution SITE positions once per window as a Poisson process
    along T (a genomic position either has a variant or it doesn't -- shared
    context for every lineage there, not drawn per lineage), at `rate`
    events/site (calibrated to the measured ~1/55bp real Oh43/CML103-vs-B73
    SNP density, experiments/simulator-indels/results/... founder_divergence
    -- see --subst-rate's help text). At each substitution site, draws one
    `Beta(sfs_shape,1)` frequency and one `L`-SNP Bernoulli panel per
    lineage -- IDENTICAL arithmetic to `_coalescent_feats`'s existing dense
    draw, just restricted to the sparse site set rather than every site, so
    the already-calibrated within-event match-probability shape is reused
    unchanged. Every non-substitution site gets a constant (all-zero)
    allele: every lineage trivially agrees there, matching real biology (no
    polymorphism between substitution events) and giving `_coalescent_feats`
    real spatial structure in its match/diverged output for the first time.

    Returns `lin_alleles [n,M,T,L] int8`, the exact same shape/dtype
    `_coalescent_feats` already produces internally -- a drop-in replacement
    for its `f`/`lin_alleles` lines, so nothing downstream of that point
    (`G = lin_alleles[wi, lineage, ti]`, the exact-match logic, `per_gamete`
    handling) needs to change.
    """
    lin_alleles = np.zeros((n, M, T, L), dtype=np.int8)
    lam = rate * T
    cnt = rng.poisson(lam, size=n)
    E = int(cnt.sum())
    if E == 0:
        return lin_alleles

    cell = np.repeat(np.arange(n), cnt)
    site = rng.integers(0, T, E)
    f_event = rng.beta(sfs_shape, 1.0, size=(E, L))                 # [E,L]
    alleles_event = (rng.random((E, M, L)) < f_event[:, None, :]).astype(np.int8)  # [E,M,L]
    lin_alleles[cell, :, site, :] = alleles_event
    return lin_alleles


def _coalescent_feats(rng, n, T, K, A, anc_cx, sfs_shape, read_snps,
                      h1, h2, rate, good, gamete, theta=None, max_lineages=None,
                      emit_panel=False, lineage=None, lineage_M=None,
                      per_gamete=False, subst_model="dense", subst_rate=0.018):
    """Mini-haplotype match features [n, T, K] + IBD lineage labels [n, K, T].

    Each founder is a piecewise-constant mosaic over ancestral lineages (same
    rate map as the sample paths, so recombination hotspots shorten founder IBD
    tracts).  Two founders on the same lineage at a site are identical-by-descent
    → identical mini-haplotype.  Lineage model:
      * theta is None → A fixed ancestors, each founder iid-uniform over them
        (legacy island model; sharing peaks near K/A → bell-shaped SFS).
      * theta set     → Ewens/GEM(theta) partition: class sizes ~ theta/i, an
        unfolded singleton-dominated SFS with a tail to K (neutral-coalescent
        shape, derived from theory rather than matched to folded real data).

    Each position is a read spanning `read_snps` (L) biallelic SNPs whose per-SNP
    derived-allele frequencies ~ Beta(sfs_shape, 1).  A founder "matches" a read
    only if its mini-haplotype agrees across ALL L SNPs (RopeBWT exact match), so
    a full-read match is a strong IBD signal, not a common-allele coincidence
    (different lineages can still coincide → thin homoplasy).  The returned
    `lineage` is the per-site IBD ground truth for the ceiling analysis.

    `lineage`/`lineage_M` (both or neither): if given, skip the internal
    lineage draw and use this precomputed array instead -- used by
    --simulate-indels so indel tracts are drawn from the SAME partition
    (PLAN.md SS2.2). If both are None (default), drawn exactly as before
    via `_draw_lineages`, byte-identical rng order.

    `per_gamete`: if True, compute and return BOTH gametes' reads
    `(match1, match2)`, each `[n,T,K]`, instead of one `gamete`-selected
    read -- the indel read-sampling layer needs both homologs' support
    independently, since real coverage/stacking is sampled per homolog.
    `gamete` is unused in this mode. If False (default), behavior is
    bit-for-bit unchanged from before this addition.

    `subst_model`: "dense" (default) draws `f`/`lin_alleles` fresh at every
    site, exactly as before this parameter existed -- byte-identical output.
    "sparse" (v4) instead calls `_lineage_substitutions`, giving divergence
    real positional persistence (a substitution happens once, at one site,
    not a fresh coin-flip every site) -- see that function's docstring.
    `subst_rate` is `_lineage_substitutions`'s events/site Poisson rate,
    unused when `subst_model="dense"`.
    """
    L = read_snps
    if lineage is None:
        lineage, M = _draw_lineages(rng, n, T, K, A, anc_cx, rate, theta, max_lineages)
    else:
        if lineage_M is None:
            raise ValueError("lineage_M must be given alongside a precomputed lineage array")
        M = lineage_M

    if subst_model == "dense":
        f = rng.beta(sfs_shape, 1.0, size=(n, 1, T, L))     # per-SNP derived freq
        lin_alleles = (rng.random((n, M, T, L)) < f).astype(np.int8)  # [n,M,T,L]
    elif subst_model == "sparse":
        lin_alleles = _lineage_substitutions(rng, n, M, T, L, subst_rate, sfs_shape)
    else:
        raise ValueError(f"subst_model must be 'dense' or 'sparse', got {subst_model!r}")

    wi = np.arange(n)[:, None, None]
    ti = np.arange(T)[None, None, :]
    G = lin_alleles[wi, lineage, ti]                    # [n,K,T,L] mini-haplotypes
    ii = np.arange(n)[:, None]
    tt = np.arange(T)[None, :]
    bad = ~good
    # E6: founder × SNP allele panel for SNP-level imputation accuracy. G[w,k,t,l]
    # is founder k's allele at read t, SNP l — the genotype implied by decoding the
    # path to founder k. Returned as [n,T,K,L] for per-site indexing (eval only).
    panel = np.transpose(G, (0, 2, 1, 3)).astype(np.int8) if emit_panel else None

    def _match_for(active, rand_founder):
        # Bad sites: corrupt the read to a RANDOM founder's mini-haplotype (the
        # read mis-maps to a wrong founder) rather than an out-of-panel marginal
        # draw. The latter usually matched no founder's exact L-SNP haplotype,
        # leaving ~40% of bad sites all-zero; sourcing from a real founder
        # guarantees the read still matches that founder's lineage group, so
        # bad sites show random (wrong) matches instead of blanks.
        Sa = G[ii, active, tt]                          # active mini-hap read [n,T,L]
        Sa = np.where(bad[:, :, None], G[ii, rand_founder, tt], Sa)
        return (G == Sa[:, None]).all(-1)               # [n,K,T] exact full-read match

    if per_gamete:
        rf1 = rng.integers(0, K, size=(n, T))           # independent corruption
        rf2 = rng.integers(0, K, size=(n, T))            # draw per homolog
        match1 = np.transpose(_match_for(h1, rf1), (0, 2, 1)).astype(np.int8)
        match2 = np.transpose(_match_for(h2, rf2), (0, 2, 1)).astype(np.int8)
        return (match1, match2), lineage, panel

    # One read per site, sampled from the active gamete (H1 if gamete else H2).
    active = np.where(gamete, h1, h2)                   # [n,T]
    rand_founder = rng.integers(0, K, size=(n, T))      # [n,T] usually != active
    match = _match_for(active, rand_founder)
    return np.transpose(match, (0, 2, 1)).astype(np.int8), lineage, panel


def _indel_chunk(rng, n, R, T, K, h1, h2, lineage, del_lin, ins_lin,
                  match1, match2, gamete_balance, coverage, ins_read_per_bp,
                  max_stack, anchor_thresh, ref_founder, dist_scale,
                  coverage_model="linear", read_len=150):
    """Assemble one chunk's indel-mode output:
    `(tern, dist, count [n,T,K] int8, lab1, lab2 [n,T] int8, refpos [n,T]
    int32, short [n] bool, n_either int, n_hemi int, n_null int)`. `count`
    is always computed (cheap, deterministic) -- callers decide whether to
    write it into a widened output array (experiments/depth-confidence-fix/,
    --emit-read-counts).

    `h1`/`h2`/`lineage`/`del_lin`/`ins_lin`/`match1`/`match2` are all
    indexed over the R-site GENERATION region; the output arrays are
    indexed over the T-row OUTPUT -- a plain prefix of the R-site region's
    real rows, in reference order (PLAN.md's row-assembly note; see
    `_sample_rows`).

    Ternary derivation mirrors `rb3_lift_ternary_state` exactly: `tern=1`
    if this row's read matched founder k (from `match1`/`match2`,
    unchanged SNP-identity logic); elif founder k's nearest-anchor
    distance exceeds `anchor_thresh` -> `tern=-1` (deletion); else `0`
    (diverged -- structurally present, no read here). Distance is
    reference-site-resolved: it does NOT depend on which row/read is
    being assembled at a stacked site, only on the reference site.

    Insertion-stacked rows (no SNP-identity read exists for them -- they
    represent a read sampled from the inserted sequence itself) match
    exactly the founders sharing the emitting lineage: a pure IBD signal
    this round, documented simplification (PLAN.md's adopted-conventions
    note; no additional SNP-level homoplasy noise modeled here).

    Rare edge case: a window whose R-site region produces fewer than `T`
    real rows is padded with the module's sentinel constants
    (`TERN_PAD`/`DIST_PAD`/`LABEL_PAD`) -- flagged via the returned
    `short` array for the caller's QC reporting, not silently absorbed.

    Also returns `(n_either, n_hemi, n_null)` -- int counts, over the
    FULL `[n,R]` generation region (not the T-row output), of sites
    where h1's or h2's own true founder is structurally present/absent.
    This is the metric PLAN.md SS2.7's real-data "either true founder
    has a read" target actually refers to. It is DELIBERATELY not
    derived from the emitted T-row output: a site where BOTH homologs'
    founders are absent can never produce a row (there is no DNA left
    to sample a colinear read from), so any row-conditioned measurement
    -- e.g. dedup-and-check over `refpos`/output rows -- structurally
    excludes exactly the sites this metric needs to count, and reads a
    misleadingly high ~97% regardless of the true rate (confirmed
    2026-09-14 during a calibration sweep: that row-conditioned number
    stayed flat near 97% even as `--indel-density` tripled the realized
    indel-affected fraction).

    Also returns `(n_deleted, n_founder_sites)` -- the population
    marginal deletion rate over ALL K founders x ALL R sites (from
    `dist_kt`, already fully materialized here). This is the correct,
    unconditioned "indel-affected reference bp" measurement; the
    `_print_indel_summary` QC line of that name is instead computed
    from emitted rows (dedup by site THIS individual's h1/h2 happened
    to cover), a smaller instance of the same row-conditioning bias.
    """
    dist_lin = _anchor_distance(del_lin)                     # [n,M,R]
    dist_kt = _gather_by_lineage(dist_lin, lineage)           # [n,K,R]
    if ref_founder >= 0:
        dist_kt = dist_kt.copy()
        dist_kt[:, ref_founder, :] = 0    # reference has no anchors vs itself

    ii = np.arange(n)[:, None]
    tt = np.arange(R)[None, :]
    m1 = lineage[ii, h1, tt]              # [n,R] active lineage, H1
    m2 = lineage[ii, h2, tt]
    pres1 = dist_kt[ii, h1, tt] <= anchor_thresh
    pres2 = dist_kt[ii, h2, tt] <= anchor_thresh
    n_either = int((pres1 | pres2).sum())
    n_hemi = int((pres1 != pres2).sum())
    # Population marginal deletion rate, over ALL K founders x ALL R sites
    # (dist_kt is already fully materialized here) -- unconditioned on any
    # one individual's own coverage, unlike the "indel-affected ref bp" QC
    # line derived from emitted rows (that one dedups by site THIS
    # individual's h1/h2 happened to cover, a smaller version of the same
    # row-conditioning bias n_either/n_hemi/n_null were added to fix).
    n_deleted = int((dist_kt > anchor_thresh).sum())
    n_founder_sites = dist_kt.size
    n_null = int((~pres1 & ~pres2).sum())
    ins1 = ins_lin[ii, m1, tt]
    ins2 = ins_lin[ii, m2, tt]

    on1, on2, c1, c2, cnt = _row_counts(rng, pres1, pres2, ins1, ins2,
                                         gamete_balance, coverage,
                                         ins_read_per_bp, max_stack,
                                         coverage_model, read_len)
    w, t, r, o, short = _sample_rows(cnt, T)

    tern_out = np.full((n, T, K), TERN_PAD, dtype=np.int8)
    dist_out = np.full((n, T, K), DIST_PAD, dtype=np.int8)
    count_out = np.full((n, T, K), 0, dtype=np.int8)   # 0 = no PAD sentinel needed, itself informative
    lab1_out = np.full((n, T), LABEL_PAD, dtype=np.int8)
    lab2_out = np.full((n, T), LABEL_PAD, dtype=np.int8)
    refpos_out = np.full((n, T), -1, dtype=np.int32)

    if w.size:
        b1 = on1[w, t].astype(np.int64)
        b2 = b1 + on2[w, t].astype(np.int64)
        b3 = b2 + c1[w, t].astype(np.int64)
        kind = np.where(o < b1, 0, np.where(o < b2, 1, np.where(o < b3, 2, 3)))

        tern_rows = np.zeros((w.size, K), dtype=np.int8)
        for k_id, mt in ((0, match1), (1, match2)):
            sel = kind == k_id
            if sel.any():
                tern_rows[sel] = mt[w[sel], t[sel]]
        for k_id, ml in ((2, m1), (3, m2)):
            sel = kind == k_id
            if sel.any():
                tern_rows[sel] = (lineage[w[sel], :, t[sel]] ==
                                   ml[w[sel], t[sel]][:, None]).astype(np.int8)

        dist_row = dist_kt[w, :, t]                           # [Rows,K]
        deleted = dist_row > anchor_thresh
        tern_rows = np.where(deleted, TERN_DEL, tern_rows).astype(np.int8)

        tern_out[w, r] = tern_rows
        dist_out[w, r] = _encode_dist(dist_row, dist_scale)
        lab1_out[w, r] = h1[w, t].astype(np.int8)
        lab2_out[w, r] = h2[w, t].astype(np.int8)
        refpos_out[w, r] = t

        # count_out: per-row read-support -- experiments/depth-confidence-fix/.
        # SUPERSEDES an earlier per-founder-MATCH-tally-pooled-across-kinds
        # design (git history): that design broadcast a site-wide count to
        # EVERY row at a site regardless of the row's own pattern (so e.g.
        # an on1-kind row and an on2-kind row at the same site, which CAN
        # show genuinely different ternary vectors -- verified against
        # test_indel_chunk_tiny_exact's fixture -- got the same count),
        # and was always 0 for non-MATCH cells even when many reads agreed
        # on a DIVERGED/deleted call. A full-scale retrain with that design
        # broke real-data accuracy broadly (not just for the affected
        # founder) -- see depth_confidence_fix_2026-09-18.md's "RESOLVED"
        # section. This is the corrected design: group rows by (window,
        # site, their own EXACT K-length ternary vector) and use each row's
        # own group size as its count -- "how many reads showed this exact
        # observed pattern here," which is always >0 (a row is always a
        # member of its own group) and meaningful for DEL/DIVERGED patterns
        # too, not just MATCH. The scalar is broadcast across all K columns
        # of its own row (not other rows at the site) purely to keep the
        # on-disk width/model wiring (K-wide, per-cell count_proj) unchanged
        # -- semantically this is now a per-ROW quantity, not per-founder.
        # Deterministic from already-drawn tern_rows -- no new RNG draws.
        site_key = w.astype(np.int64) * R + t.astype(np.int64)
        group_key = np.concatenate([site_key[:, None], tern_rows.astype(np.int64)], axis=1)
        _, group_id, group_sizes = np.unique(group_key, axis=0, return_inverse=True,
                                              return_counts=True)
        row_group_size = group_sizes[group_id]                # [Rows]
        count_rows = np.repeat(row_group_size[:, None], K, axis=1)  # [Rows,K]
        count_out[w, r] = _encode_dist(count_rows, dist_scale)

    return (tern_out, dist_out, count_out, lab1_out, lab2_out, refpos_out, short,
            n_either, n_hemi, n_null, n_deleted, n_founder_sites)


def simulate(rng, windows, sites, founders, min_cross, max_cross,
             inbreeding, allele_sharing, bad_frac, recomb_span=1.0,
             recomb_tile=64, sharing_model="independent", ancestors=6,
             ancestor_crossovers=8, derived_sfs=0.3, read_snps=8,
             error_block=1.0, gamete_balance=0.5, sharing_theta=None,
             windows_per_individual=0, min_founders=2, max_founders=24,
             emit_snp_panel=False, inbreeding_per_window=None,
             breeding_classes=None, class_inbred_frac=0.5,
             constant_pair_frac=0.0, constant_inbred_frac=0.5, chunk=1000,
             simulate_indels=False, indel_model="tracts",
             indel_density=2.3e-3, indel_ins_frac=0.5,
             indel_large_frac=0.027, indel_small_alpha=1.7, indel_small_max=50,
             indel_large_logmean=8.6, indel_large_logsd=1.6,
             indel_max_len=65536, indel_coverage=2.0,
             indel_ins_read_per_bp=2e-3, indel_max_stack=64,
             indel_region_mult=4, indel_recomb_suppress=0.9,
             indel_recomb_flank=32, indel_anchor_thresh=0,
             indel_ref_founder=-1, indel_lineage_frac=0.15,
             indel_overlay_rate=2.3e-3, indel_overlay_mean_len=300.0,
             indel_overlay_founder_freq=1.0, subst_model="dense",
             subst_rate=0.018, coverage_model="linear", read_len=150,
             emit_read_counts=False):
    """... (see module docstring / experiments/simulator-indels/PLAN.md
    for the full --simulate-indels design). All `simulate_indels=False`
    (default) behavior, including rng draw order, is byte-for-byte
    unchanged from before this parameter existed -- every new rng call
    this feature adds sits inside an `if simulate_indels:` branch.

    simulate_indels: emit the indel-aware ternary+distance layout
    (2K+2 columns: [ternary(K) | H1 | H2 | distance(K)]) instead of the
    binary K+2 (or K+3 with the eval-only recomb-rate column) layout.
    `emit_read_counts` (experiments/depth-confidence-fix/) widens this to
    3K+2 by appending a 4th per-founder block: [ternary(K) | H1 | H2 |
    distance(K) | count(K)], count = the site's total real read count
    wherever that founder's ternary state is MATCH, else 0, log-coded via
    the same `_encode_dist` scheme as distance (no new RNG draws). Off by
    default (`ncol` stays 2K+2, byte-identical to before this param
    existed).
    Windows still have exactly `sites` OUTPUT rows (PLAN.md's row-
    assembly note: a window is "the next T real rows in reference
    order," not "rows covering a fixed reference-bp span" -- no padding
    in the routine case); generation itself runs over a larger
    `indel_region_mult * sites`-site region internally, purely as a
    buffer to produce enough real rows. Requires `sharing_model=
    'coalescent'` (indel content is drawn through the lineage array,
    which only exists there). `emit_snp_panel` and the eval-only
    recomb-rate column are dropped in this mode (both are reference-
    site-indexed at `sites` granularity; the row axis no longer aligns
    1:1 with reference sites) -- use the `.refpos.npy`-equivalent
    return value (`refpos_out`) to join rows back to reference
    coordinates instead.

    Final return value `true_cov_out` is `None` unless `simulate_indels`,
    in which case it is `(n_either, n_hemi, n_null, n_total, n_deleted,
    n_founder_sites)` int counts, summed over the FULL R-site generation
    region across every window -- the correct genome-wide "either true
    founder has support" and "indel-affected reference bp" measures
    PLAN.md SS2.7 targets, deliberately NOT derivable from the T-row
    output alone (see `_indel_chunk`'s docstring: a site with both
    homologs' founders absent emits zero rows by construction, so it is
    invisible to any row-conditioned measurement).
    """
    K = founders
    T = sites
    coalescent = sharing_model == "coalescent"
    if simulate_indels and not coalescent:
        raise ValueError(
            "simulate_indels=True requires sharing_model='coalescent' "
            "(indel content is drawn through the lineage array, which "
            "only exists in coalescent mode)")
    if simulate_indels and emit_snp_panel:
        raise ValueError(
            "emit_snp_panel is not supported with simulate_indels (the "
            "panel is reference-site indexed; this mode's row axis is "
            "not)")
    if emit_read_counts and not simulate_indels:
        raise ValueError("emit_read_counts requires simulate_indels=True "
                          "(the count block only exists in that mode)")
    if indel_model not in ("tracts", "lineage", "overlay"):
        raise ValueError(f"indel_model must be one of tracts/lineage/overlay, "
                          f"got {indel_model!r}")
    if subst_model not in ("dense", "sparse"):
        raise ValueError(f"subst_model must be 'dense' or 'sparse', got {subst_model!r}")
    if coverage_model not in ("linear", "poisson", "reads"):
        raise ValueError(f"coverage_model must be 'linear', 'poisson', or "
                          f"'reads', got {coverage_model!r}")
    if not coalescent:
        q = (allele_sharing * K - 1.0) / (K - 1)   # match rate for non-true founders
        if q < 0:
            raise ValueError(
                f"allele_sharing={allele_sharing} too low for K={K}; "
                f"minimum is {1.0 / K:.4f}")

    # E5/E11: assign every G windows to one individual drawing from a founder subset.
    grouped = windows_per_individual > 0
    breeding = breeding_classes is not None
    het_tw = cls_w = is_const = const_h1 = const_h2 = None
    if breeding:
        (ind, win_k, sub, het_tw, cls_w,
         is_const, const_h1, const_h2) = _breeding_pop_assignment(
            rng, windows, windows_per_individual, K, breeding_classes,
            class_inbred_frac, constant_pair_frac, constant_inbred_frac)
    elif grouped:
        if min_founders < 2:
            raise ValueError("--min-founders must be >= 2")
        ind, win_k, sub = _individual_assignment(
            rng, windows, windows_per_individual, K, min_founders, max_founders)
    ind_out = ind if (grouped or breeding) else None

    refpos_out = short_out = None
    true_either_sum = true_hemi_sum = true_null_sum = true_total_sum = 0
    true_deleted_sum = true_founder_sites_sum = 0
    if simulate_indels:
        R = indel_region_mult * T
        ncol = (3 if emit_read_counts else 2) * K + 2
        out = np.empty((windows, T, ncol), dtype=np.int8)
        # Unlike the base case, ibd is R-site indexed here (the same
        # generation-region axis as the indel tracts), not T-site indexed
        # -- it is genuinely more informative to keep the full region's
        # lineage ground truth than to force it onto the row axis.
        ibd = np.empty((windows, R, K), dtype=np.int8)
        panel = None
        track = False
        refpos_out = np.empty((windows, T), dtype=np.int32)
        short_out = np.empty(windows, dtype=bool)
        # Tract arrays scale as n*M*R; keep the per-chunk memory bounded
        # (~4M site-slots/chunk) the same way regardless of how large R is.
        chunk = min(chunk, max(16, int(2 ** 22 // max(1, R))))
    else:
        track = recomb_span > 1.0                       # emit hidden true-rate column
        ncol = K + 2 + (1 if track else 0)
        out = np.empty((windows, T, ncol), dtype=np.int8)
        ibd = np.empty((windows, T, K), dtype=np.int8) if coalescent else None
        panel_on = emit_snp_panel and coalescent
        panel = np.empty((windows, T, K, read_snps), dtype=np.int8) if panel_on else None

    for start in range(0, windows, chunk):
        n = min(chunk, windows - start)
        sl = slice(start, start + n)

        if simulate_indels:
            # ---- indel-mode chunk body: everything below runs over the
            # R-site generation region, then _indel_chunk collapses it to
            # the T-row output. Tracts/lineages must be drawn BEFORE paths
            # (they feed the suppressed rate map paths are drawn on) --
            # the one real reordering relative to the base case below.
            rmap_R = (_rate_map(rng, n, R, recomb_span, recomb_tile)
                      if recomb_span > 1.0 else None)
            max_lin = (None if sharing_theta is None
                       else min(64, max(K, int(round(4 * sharing_theta)))))
            lineage, M = _draw_lineages(rng, n, R, K, ancestors,
                                         ancestor_crossovers, rmap_R,
                                         sharing_theta, max_lin)
            if indel_model == "tracts":
                del_lin, ins_lin = _indel_tracts(
                    rng, n, M, R, indel_density, indel_ins_frac,
                    indel_large_frac, indel_small_alpha, indel_small_max,
                    indel_large_logmean, indel_large_logsd, indel_max_len)
                lineage_indel, M_indel = lineage, M
            elif indel_model == "lineage":
                del_lin, ins_lin = _lineage_indels(
                    rng, n, K, M, R, lineage, indel_lineage_frac,
                    indel_ins_frac, indel_max_len)
                lineage_indel, M_indel = lineage, M
            else:  # "overlay" -- decoupled from the coalescent lineage array
                del_lin, ins_lin = _overlay_indels(
                    rng, n, K, R, indel_overlay_rate, indel_overlay_mean_len,
                    indel_ins_frac, indel_overlay_founder_freq, indel_max_len)
                lineage_indel = np.broadcast_to(
                    np.arange(K, dtype=np.int32)[None, :, None], (n, K, R))
                M_indel = K
            rmap_path = _indel_suppressed_rate(
                rmap_R, del_lin, ins_lin, indel_recomb_suppress,
                indel_recomb_flank)

            n_cross = rng.integers(min_cross, max_cross + 1, n)
            if is_const is not None:
                n_cross = np.where(is_const[sl], 0, n_cross)
            if grouped or breeding:
                h1 = _build_paths_subset(rng, sub[sl], win_k[sl], R, n_cross, rmap_path)
            else:
                h1 = _build_paths(rng, n, R, K, n_cross, rmap_path)

            if breeding:
                het_t = het_tw[sl]
                k_w = np.maximum(win_k[sl], 2)
                p_ind = np.clip(het_t / (1.0 - 1.0 / k_w), 0.0, 1.0)
                nc2 = rng.integers(min_cross, max_cross + 1, n)
                if is_const is not None:
                    nc2 = np.where(is_const[sl], 0, nc2)
                h2_ind = _build_paths_subset(rng, sub[sl], win_k[sl], R, nc2, rmap_path)
                n_share = rng.integers(min_cross, max_cross + 1, n)
                seg = _segment_index(rng, n, R, n_share, rmap_path)
                max_seg = int(seg.max()) + 1
                seg_indep = rng.random((n, max_seg)) < p_ind[:, None]
                independent = np.take_along_axis(seg_indep, seg, axis=1)
                h2 = np.where(independent, h2_ind, h1)
                if is_const is not None:
                    m = is_const[sl]
                    if m.any():
                        h1[m] = const_h1[sl][m][:, None]
                        h2[m] = const_h2[sl][m][:, None]
            else:
                F = inbreeding if inbreeding_per_window is None else inbreeding_per_window[sl]
                inbred = rng.random(n) < F
                h2 = h1.copy()
                outbred = np.flatnonzero(~inbred)
                if outbred.size:
                    nc2 = rng.integers(min_cross, max_cross + 1, outbred.size)
                    r2 = None if rmap_path is None else rmap_path[outbred]
                    if grouped:
                        h2[outbred] = _build_paths_subset(
                            rng, sub[sl][outbred], win_k[sl][outbred], R, nc2, r2)
                    else:
                        h2[outbred] = _build_paths(rng, outbred.size, R, K, nc2, r2)

            good = _good_mask(rng, n, R, bad_frac, error_block)
            (match1, match2), lineage, _ = _coalescent_feats(
                rng, n, R, K, ancestors, ancestor_crossovers, derived_sfs,
                read_snps, h1, h2, None, good, gamete=None,
                theta=sharing_theta, max_lineages=max_lin,
                lineage=lineage, lineage_M=M, per_gamete=True,
                subst_model=subst_model, subst_rate=subst_rate)

            (tern, dist, count, lab1, lab2, refpos, short, n_either, n_hemi, n_null,
             n_deleted, n_founder_sites) = _indel_chunk(
                rng, n, R, T, K, h1, h2, lineage_indel, del_lin, ins_lin,
                match1, match2, gamete_balance, indel_coverage,
                indel_ins_read_per_bp, indel_max_stack, indel_anchor_thresh,
                indel_ref_founder, DIST_LOG_SCALE, coverage_model, read_len)

            out[sl, :, :K] = tern
            out[sl, :, K] = lab1
            out[sl, :, K + 1] = lab2
            out[sl, :, K + 2:2 * K + 2] = dist
            if emit_read_counts:
                out[sl, :, 2 * K + 2:3 * K + 2] = count
            ibd[sl] = np.transpose(lineage, (0, 2, 1)).astype(np.int8)
            refpos_out[sl] = refpos
            short_out[sl] = short
            true_either_sum += n_either
            true_hemi_sum += n_hemi
            true_null_sum += n_null
            true_total_sum += n * R
            true_deleted_sum += n_deleted
            true_founder_sites_sum += n_founder_sites
            continue

        # ---- non-indel chunk body: UNCHANGED from before this feature ----
        rmap = _rate_map(rng, n, T, recomb_span, recomb_tile) if track else None

        n_cross = rng.integers(min_cross, max_cross + 1, n)
        if is_const is not None:
            # Tier2: constant individuals get 0 explicit crossovers here --
            # also what avoids a real crash, since _build_paths would raise
            # (rng.integers(1,K,nv) with K=1) if a k=1 constant-inbred row
            # requested >=1 crossover. Their h1/h2 are overridden below
            # regardless, once the breeding het-tract logic (which also
            # needs a crash-safe n_cross for its own K=1 rows) has run.
            n_cross = np.where(is_const[sl], 0, n_cross)
        if grouped or breeding:
            h1 = _build_paths_subset(rng, sub[sl], win_k[sl], T, n_cross, rmap)
        else:
            h1 = _build_paths(rng, n, T, K, n_cross, rmap)

        if breeding:
            # E11 selfing-aware het: H2 shares H1 over IBD tracts, independent
            # elsewhere. The independent-tract fraction p_ind hits the target
            # het-loci fraction given k founders (within an independent tract the
            # het rate ≈ 1 − 1/k). p_ind=0 (inbred) → H2≡H1.
            het_t = het_tw[sl]
            k_w = np.maximum(win_k[sl], 2)
            p_ind = np.clip(het_t / (1.0 - 1.0 / k_w), 0.0, 1.0)        # [n]
            nc2 = rng.integers(min_cross, max_cross + 1, n)
            if is_const is not None:
                nc2 = np.where(is_const[sl], 0, nc2)          # same K=1 crash guard as n_cross
            h2_ind = _build_paths_subset(rng, sub[sl], win_k[sl], T, nc2, rmap)
            n_share = rng.integers(min_cross, max_cross + 1, n)
            seg = _segment_index(rng, n, T, n_share, rmap)             # [n,T] tracts
            max_seg = int(seg.max()) + 1
            seg_indep = rng.random((n, max_seg)) < p_ind[:, None]      # per-tract coin
            independent = np.take_along_axis(seg_indep, seg, axis=1)   # [n,T]
            h2 = np.where(independent, h2_ind, h1)

            if is_const is not None:
                # Tier2: force the FIXED founder identity for constant
                # individuals, constant across every site of this window.
                # Not redundant with the n_cross=0 clamp above: that alone
                # only makes each window internally constant, not constant
                # ACROSS an individual's G windows (each window's starting
                # founder is still an independent draw in _build_paths), and
                # for k=2 doesn't guarantee both designated founders are the
                # ones that appear at all (h1/h2 could coincidentally match).
                m = is_const[sl]
                if m.any():
                    h1[m] = const_h1[sl][m][:, None]
                    h2[m] = const_h2[sl][m][:, None]
        else:
            # Second haplotype: identical with prob F (inbred), else independent.
            # E7: a per-window F (constant within an individual) mixes inbreeding.
            F = inbreeding if inbreeding_per_window is None else inbreeding_per_window[sl]
            inbred = rng.random(n) < F
            h2 = h1.copy()
            outbred = np.flatnonzero(~inbred)
            if outbred.size:
                nc2 = rng.integers(min_cross, max_cross + 1, outbred.size)
                r2 = None if rmap is None else rmap[outbred]
                if grouped:
                    h2[outbred] = _build_paths_subset(
                        rng, sub[sl][outbred], win_k[sl][outbred], T, nc2, r2)
                else:
                    h2[outbred] = _build_paths(rng, outbred.size, T, K, nc2, r2)

        good = _good_mask(rng, n, T, bad_frac, error_block)

        # Per-site read origin: True => sampled from H1's chromosome, else H2's.
        # gamete_balance = P(H1). One read per site (diploid low-coverage sampling).
        gamete = rng.random((n, T)) < gamete_balance
        active = np.where(gamete, h1, h2)               # observed founder per site

        if coalescent:
            max_lin = (None if sharing_theta is None
                       else min(64, max(K, int(round(4 * sharing_theta)))))
            feats, lineage, panel_chunk = _coalescent_feats(
                rng, n, T, K, ancestors, ancestor_crossovers,
                derived_sfs, read_snps, h1, h2, rmap, good, gamete,
                theta=sharing_theta, max_lineages=max_lin, emit_panel=panel_on)
            ibd[start:start + n] = np.transpose(lineage, (0, 2, 1)).astype(np.int8)
            if panel_on:
                panel[start:start + n] = panel_chunk
        else:
            # Independent background; force only the ACTIVE gamete's founder to
            # match on good sites (one read, from one chromosome).
            feats = (rng.random((n, T, K)) < q).astype(np.int8)
            ii = np.arange(n)[:, None]
            tt = np.arange(T)[None, :]
            feats[ii, tt, active] = np.where(good, 1, feats[ii, tt, active])

        out[start:start + n, :, :K] = feats
        out[start:start + n, :, K] = h1.astype(np.int8)
        out[start:start + n, :, K + 1] = h2.astype(np.int8)
        if track:
            out[start:start + n, :, K + 2] = np.clip(
                np.rint(rmap), 1, 127).astype(np.int8)

    true_cov_out = (None if not simulate_indels else
                    (true_either_sum, true_hemi_sum, true_null_sum, true_total_sum,
                     true_deleted_sum, true_founder_sites_sum))
    return out, ibd, ind_out, panel, het_tw, cls_w, refpos_out, short_out, true_cov_out


def parse_args():
    p = argparse.ArgumentParser(description="Simulate shared-allele patterns")
    p.add_argument("--workdir", default="/workdir/esb33")
    p.add_argument("--out", default="sim_alleles.npy",
                   help="Filename written under <workdir>/data/training/")
    p.add_argument("--founders", type=int, default=24)
    p.add_argument("--sites", type=int, default=512, help="Site window length")
    p.add_argument("--windows", type=int, default=100000, help="Number of windows")
    p.add_argument("--min-crossovers", type=int, default=2)
    p.add_argument("--max-crossovers", type=int, default=10)
    p.add_argument("--inbreeding", type=float, default=1.0,
                   help="Inbreeding coefficient F in [0,1]; P(H1==H2 path). "
                        "F=0 → fully outbred diploid (interleaved single-gamete reads)")
    p.add_argument("--gamete-balance", type=float, default=0.5,
                   help="P(a site's read is sampled from H1's chromosome); 0.5 = even. "
                        "Skew simulates one chromosome dominating coverage.")
    p.add_argument("--allele-sharing", type=float, default=0.2,
                   help="Average fraction of founders sharing the sample allele")
    p.add_argument("--sharing-model", choices=["independent", "coalescent"],
                   default="independent",
                   help="independent: per-site random sharing (default). coalescent: "
                        "mosaic-of-ancestors panel → LD tracts + founder relatedness.")
    p.add_argument("--ancestors", type=int, default=6,
                   help="Coalescent mode: number of ancestral lineages A (legacy "
                        "island model, used only when --sharing-theta is unset)")
    p.add_argument("--sharing-theta", type=float, default=None,
                   help="Coalescent mode: Ewens/GEM concentration. Set to use the "
                        "theory-based unfolded SFS (class sizes ~ theta/i, "
                        "singleton-dominated, tail to K) instead of A fixed "
                        "ancestors. Larger ⇒ more singletons / less sharing. "
                        "Also writes <out>.ibd.npy (per-site IBD lineage labels).")
    p.add_argument("--ancestor-crossovers", type=int, default=8,
                   help="Coalescent mode: mean ancestor switches per founder per window")
    p.add_argument("--derived-sfs", type=float, default=0.3,
                   help="Coalescent mode: Beta(shape,1) site-frequency-spectrum shape; "
                        "<1 ⇒ rare derived alleles / many shared common alleles")
    p.add_argument("--read-snps", type=int, default=8,
                   help="Coalescent mode: SNPs per mini-haplotype read (exact-match "
                        "length L); more SNPs ⇒ rarer, more-specific full-read matches")
    p.add_argument("--bad-frac", type=float, default=0.05,
                   help="Proportion of sites with corrupted (random) patterns")
    p.add_argument("--error-block", type=float, default=1.0,
                   help="Mean length (sites) of correlated error runs; 1 = independent")
    p.add_argument("--recomb-span", type=float, default=1.0,
                   help="Hot/cold recomb-rate ratio (E2). >1 places breakpoints by a "
                        "hidden variable rate map + appends the true rate as an "
                        "eval-only column. 1 = uniform (E1 layout, no track).")
    p.add_argument("--recomb-tile", type=int, default=64,
                   help="Block size (sites) of constant recomb rate")
    p.add_argument("--windows-per-individual", type=int, default=0,
                   help="E5: group every G consecutive windows into one individual "
                        "whose paths draw only from a per-individual founder subset. "
                        "0 = off (legacy: each window samples all founders). Also "
                        "writes <out>.ind.npy (per-window individual id).")
    p.add_argument("--min-founders", type=int, default=2,
                   help="E5: min founders contributing to an individual")
    p.add_argument("--max-founders", type=int, default=24,
                   help="E5: max founders contributing to an individual")
    p.add_argument("--emit-snp-panel", action="store_true",
                   help="E6: also write <out>.panel.npy [N,T,K,L] = each founder's "
                        "per-SNP allele (coalescent mode only), for SNP-level "
                        "imputation-accuracy eval. Large; use a modest --windows.")
    p.add_argument("--mixed-inbreeding", action="store_true",
                   help="E7: draw a per-INDIVIDUAL inbreeding coefficient F (a spike "
                        "at F=1 for inbred lines + Uniform[0,1] for the rest) instead "
                        "of the single --inbreeding scalar; needs "
                        "--windows-per-individual. Writes <out>.finb.npy (per-window F).")
    p.add_argument("--inbred-line-frac", type=float, default=0.5,
                   help="E7: fraction of individuals that are fully inbred (F=1).")
    p.add_argument("--breeding-pop", action="store_true",
                   help="E11: discrete founder-count mixture (25%% k=2 F2, 25%% k=8 S1, "
                        "50%% k in [12,24] outbred); each class split inbred vs het with a "
                        "per-class target het level. Needs --windows-per-individual. "
                        "Writes <out>.ind.npy, <out>.finb.npy (per-window het target) and "
                        "<out>.cls.npy (per-window class id 0/1/2).")
    p.add_argument("--het-by-class", type=str, default="0.5,0.25,0.9",
                   help="E11: comma-separated target het-loci fraction for the three "
                        "classes (k=2, k=8, k in [12,24]).")
    p.add_argument("--class-inbred-frac", type=float, default=0.5,
                   help="E11: fraction of individuals in each class that are fully inbred.")
    p.add_argument("--constant-pair-frac", type=float, default=0.0,
                   help="crf-relatedness Tier2: fraction of --breeding-pop individuals "
                        "given a FIXED founder identity constant across every site of "
                        "every window (0 breakpoints, het exactly 0 or 1) -- the true "
                        "inbred-line and true-F1-hybrid regimes the class mixture and "
                        "--mixed-inbreeding cannot reach (k=1 is hard-guarded elsewhere; "
                        "the realized het ceiling for an independent-tract k=2 "
                        "individual is 1-1/k=0.5, never 1.0). Requires --breeding-pop.")
    p.add_argument("--constant-inbred-frac", type=float, default=0.5,
                   help="crf-relatedness Tier2: of the --constant-pair-frac individuals, "
                        "fraction that are single-founder homozygous (k=1, het=0) "
                        "rather than a constant two-founder pair (k=2, het=1).")

    # --simulate-indels: experiments/simulator-indels/PLAN.md SS2. Realistic
    # indels + the ternary/distance two-matrix layout. Default off = today's
    # exact output, byte for byte (pinned by tests/python/crf/test_simulate_alleles.py's
    # golden-hash regression test).
    p.add_argument("--simulate-indels", action="store_true",
                   help="Emit the indel-aware two-matrix layout "
                        "[ternary(K) | H1 | H2 | distance(K)] = 2K+2 columns, instead "
                        "of today's binary K+2 (or K+3 with --recomb-span) layout. "
                        "Requires --sharing-model coalescent (indels are a lineage "
                        "property, SS2.2). Off = today's exact output, byte for byte. "
                        "--sites stays small-window-friendly by default; indel "
                        "structure (mean event ~530bp, bp-dominant class 4-64kb) needs "
                        "a much larger --sites (8192+) to be meaningfully represented "
                        "in a single window.")
    p.add_argument("--indel-model", choices=["tracts", "lineage", "overlay"],
                   default="tracts",
                   help="How indel structure is generated (v3, "
                        "experiments/simulator-indels/PLAN.md v3 section). "
                        "'tracts' (default): today's independent Poisson/length-"
                        "mixture process (--indel-density etc, below) -- unchanged. "
                        "'lineage': derive indels directly from the EXISTING "
                        "coalescent lineage mosaic -- a contiguous run where a "
                        "lineage is shared becomes a candidate indel with "
                        "probability --indel-lineage-frac, using the run's own "
                        "length; IBD correlation falls out of the same lineage "
                        "structure that already drives SNP sharing, no separate "
                        "stacked process. 'overlay': indels fully decoupled from "
                        "the coalescent lineage array -- each founder gets its own "
                        "private process (--indel-overlay-*), so IBD founders do "
                        "NOT automatically share indel content in this mode.")
    p.add_argument("--indel-lineage-frac", type=float, default=0.15,
                   help="'lineage' --indel-model only: probability that a "
                        "contiguous lineage-occupancy run is promoted to an indel.")
    p.add_argument("--indel-overlay-rate", type=float, default=2.3e-3,
                   help="'overlay' --indel-model only: indel events per reference "
                        "bp per founder (among participating founders).")
    p.add_argument("--indel-overlay-mean-len", type=float, default=300.0,
                   help="'overlay' --indel-model only: median event length (bp), "
                        "one LogNormal scale (no two-component mixture).")
    p.add_argument("--indel-overlay-founder-freq", type=float, default=1.0,
                   help="'overlay' --indel-model only: fraction of founders that "
                        "participate in indel generation at all (1.0 = all K).")
    p.add_argument("--subst-model", choices=["dense", "sparse"], default="dense",
                   help="v4: how founder divergence (match vs diverged) is generated. "
                        "'dense' (default): today's exact behavior -- match probability "
                        "is redrawn independently at EVERY site, no positional memory. "
                        "'sparse': substitutions happen at sparse Poisson-selected SITE "
                        "positions (--subst-rate), giving divergence genuine positional "
                        "persistence -- a substitution is visible at the same site for "
                        "every lineage, not re-rolled every site. Same within-event "
                        "match-probability arithmetic as 'dense' (--derived-sfs, "
                        "--read-snps), just restricted to the sparse site set.")
    p.add_argument("--subst-rate", type=float, default=0.018,
                   help="'sparse' --subst-model only: substitution events per reference "
                        "bp. Default ~1/55bp, from founder_divergence_from_b73.py's "
                        "measured Oh43/CML103-vs-B73 SNP density (16.5-18.8 SNPs/kb, "
                        "experiments/ril2-error-regions/results/ril2_error_regions/"
                        "DIAGNOSTICS.md) -- only two founders measured, re-tune if a "
                        "panel-wide number becomes available.")
    p.add_argument("--indel-coverage-model", choices=["linear", "poisson", "reads"],
                   default="linear",
                   help="v4: how --indel-coverage maps to read presence in _row_counts. "
                        "'linear' (default): today's exact behavior, "
                        "P=coverage*gamete_weight -- at the default coverage=2.0/"
                        "gamete-balance=0.5 this is P=1.0, a GUARANTEED read at every "
                        "present site, which is why simulated T-row windows span far "
                        "less reference bp than a real window of equal row count. "
                        "'poisson': P=1-exp(-coverage*gamete_weight), the standard "
                        "Lander-Waterman depth relation for a REAL target coverage (e.g. "
                        "0.1 for 0.1x) -- matches the real MEAN depth, but every site is "
                        "still an independent trial, so it doesn't reproduce real reads' "
                        "spatial clustering (a 256bp bin gets 256 independent chances to "
                        "come up covered, so it almost always does). 'reads': real "
                        "--indel-read-len-bp CONTIGUOUS fragments with Poisson-distributed "
                        "start positions -- same mean depth as 'poisson', genuine spatial "
                        "clustering, the actual real-coverage model. Pass a REAL target "
                        "coverage (e.g. 0.1 for 0.1x) via --indel-coverage for 'poisson' "
                        "or 'reads'.")
    p.add_argument("--indel-read-len-bp", type=int, default=150,
                   help="'reads' --indel-coverage-model only: read fragment length in bp, "
                        "matching this project's established wgsim convention "
                        "(config.py READ_LEN=150).")
    p.add_argument("--indel-density", type=float, default=2.3e-3,
                   help="Indel events per reference bp per lineage (ins+del combined). "
                        "Calibrated against BOTH real maize acceptance targets jointly "
                        "(indel-affected-reference-bp ~39%%, either-founder-covered "
                        "~71.6%% for inbred/NAM-founder-like individuals -- these two "
                        "targets pull the fit in opposite directions as density changes, "
                        "so this value is a deliberate balance, not an exact hit on "
                        "either one -- see "
                        "experiments/simulator-indels/results/calibration_sweep_2026-09-14.md "
                        "and results/indel_biology_notes.md for the maize numbers. "
                        "Re-tune if --indel-ins-frac or the length-mixture params change, "
                        "or set independently for a different organism (e.g. cassava's "
                        "own ~32.7%%/~72.1%% targets).")
    p.add_argument("--indel-ins-frac", type=float, default=0.5,
                   help="Fraction of indel events that are insertions relative to "
                        "reference; 0.5 reproduces the measured ~0.99:1 ins:del symmetry.")
    p.add_argument("--indel-large-frac", type=float, default=0.027,
                   help="Weight of the LARGE (LTR-retrotransposon) length-mixture "
                        "component. Measured: ~2.7%% of events but ~94.8%% of indel bp "
                        "-- calibrate against the bp-weighted table, never the raw "
                        "event histogram (indel_biology_notes.md explains why).")
    p.add_argument("--indel-small-alpha", type=float, default=1.7,
                   help="Small (replication-slippage) component: discrete power-law "
                        "exponent, P(L) ~ L^-alpha on [1,--indel-small-max]. 1.7 puts "
                        "~40-50%% of small events at exactly 1bp, matching the measured "
                        "41.5%%.")
    p.add_argument("--indel-small-max", type=int, default=50,
                   help="Max length (bp) of the small/slippage mixture component.")
    p.add_argument("--indel-large-logmean", type=float, default=8.6,
                   help="Large component: mean of log(length in bp). 8.6 => median "
                        "~5.4kb, spanning the measured 4-64kb bp-dominant classes.")
    p.add_argument("--indel-large-logsd", type=float, default=1.6,
                   help="Large component: sd of log(length in bp).")
    p.add_argument("--indel-max-len", type=int, default=65536,
                   help="Hard clip on event length (bp); also the left-overhang span "
                        "on which tract start positions are drawn, so a long tract "
                        "overlapping the generation region from outside is still "
                        "represented rather than undercounted at the region's edge.")
    p.add_argument("--indel-coverage", type=float, default=2.0,
                   help="Expected colinear reads per reference bp when BOTH homologs "
                        "are present. 1.0 reproduces today's exactly-one-read-per-site "
                        "density; higher is needed to approach the measured ~70-75%% "
                        "either-true-founder-covered acceptance target (PLAN.md SS2.7).")
    p.add_argument("--indel-ins-read-per-bp", type=float, default=2e-3,
                   help="Reads sampled per bp of INSERTED founder sequence; all of "
                        "them project to the single reference site immediately before "
                        "the insertion (real row stacking, not a count value -- PLAN.md "
                        "SS2.3/SS2.5). 2e-3 ~= one read per 500bp of inserted sequence.")
    p.add_argument("--indel-max-stack", type=int, default=64,
                   help="Cap on insertion-derived rows per homolog per reference site.")
    p.add_argument("--indel-region-mult", type=int, default=4,
                   help="Generation-side buffer only, NOT an output-shape parameter: "
                        "indel/path/coverage structure is drawn over "
                        "(mult * --sites) reference sites, then the first --sites REAL "
                        "rows (in reference order) become the output -- a plain "
                        "prefix-take, not padding to a budget. A window whose region "
                        "genuinely can't produce --sites rows is the rare-edge-case "
                        "fallback (padded, counted in the printed QC line) -- raise "
                        "this or --indel-coverage if that rate is not ~0%%.")
    p.add_argument("--indel-recomb-suppress", type=float, default=0.9,
                   help="Fractional reduction of the local recombination rate at "
                        "indel-variable sites (0=off, 0.9=10x colder), reusing the "
                        "existing --recomb-span rate-map channel. Soft suppression "
                        "only this round -- crossovers are NOT hard-forbidden from "
                        "landing inside an indel tract (PLAN.md SS2.6).")
    p.add_argument("--indel-recomb-flank", type=int, default=32,
                   help="Sites of dilation applied to the recombination-suppression "
                        "footprint around each indel-variable region.")
    p.add_argument("--indel-anchor-thresh", type=int, default=0,
                   help="Nearest-anchor distance (bp) at or below which a founder "
                        "with no read still scores 0 (diverged) rather than -1 "
                        "(deletion) -- the `thresh` argument of the real "
                        "rb3_lift_ternary_state. 0 = any site inside a deletion tract "
                        "is -1.")
    p.add_argument("--indel-ref-founder", type=int, default=-1,
                   help="Founder index treated as the reference (e.g. B73): never "
                        "given deletions, distance pinned to 0. -1 = no reference "
                        "founder in this panel (all K founders carry indels).")
    p.add_argument("--emit-read-counts", action="store_true",
                   help="--simulate-indels only: widen the output from 2K+2 to 3K+2 "
                        "by appending a 4th per-founder block -- per-cell real "
                        "read-support count (the site's total real read count "
                        "wherever that founder's ternary state is MATCH, else 0), "
                        "log-coded via _encode_dist. Deterministic from "
                        "already-drawn cnt/tern -- no new RNG draws. Off by "
                        "default. See experiments/depth-confidence-fix/.")

    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _print_indel_summary(args, data, refpos, short, out_path, refpos_path, true_cov):
    """--simulate-indels verification summary: the real empirical
    validation of PLAN.md SS2.7's acceptance criteria, not just "the code
    ran". Most stats are computed per UNIQUE (window, reference site) --
    deduplicated across any stacked rows sharing a site -- via a single
    combined-key trick, not a per-window Python loop, so this stays fast
    even at large --windows. `true_cov` (from `simulate()`'s return) is
    the one exception: a genuinely correct "either true founder covered"
    reading needs the FULL R-site generation region, not just sites that
    produced an output row -- see `_indel_chunk`'s docstring.
    """
    K = args.founders
    tern = data[:, :, :K]
    lab1 = data[:, :, K].astype(np.int64)
    lab2 = data[:, :, K + 1].astype(np.int64)
    dist_code = data[:, :, K + 2:2 * K + 2]
    count_code = data[:, :, 2 * K + 2:3 * K + 2] if data.shape[-1] >= 3 * K + 2 else None

    print(f"\nWrote {out_path}")
    print(f"  shape={data.shape}  dtype={data.dtype}  "
          f"size={data.nbytes / 1e9:.2f} GB  "
          f"({'3K+2 ternary+distance+count' if count_code is not None else '2K+2 ternary+distance'} layout)")
    print(f"  windows needing padding: {short.mean()*100:.2f}%  "
          f"(should be ~0% at sane --indel-coverage/--indel-density; if not, "
          f"raise --indel-region-mult or --indel-coverage)")

    valid = lab1 != LABEL_PAD
    if not valid.any():
        print("  WARNING: every row was padding -- no real output to summarize.")
        return

    w_idx = np.repeat(np.arange(data.shape[0])[:, None], data.shape[1], axis=1)
    # combined (window, reference-site) key; refpos is bounded by the
    # generation region size, always well under 1e6 at any sane --sites.
    key = w_idx[valid].astype(np.int64) * 1_000_000 + refpos[valid].astype(np.int64)
    uniq_keys, first_idx, inv, counts = np.unique(
        key, return_index=True, return_inverse=True, return_counts=True)

    flat_tern = tern[valid]                             # [n_valid_rows, K]
    flat_lab1, flat_lab2 = lab1[valid], lab2[valid]

    # Deletion state is a pure function of (window, site, founder) -- ANY
    # row at a shared site carries the same deletion truth -- so a plain
    # first-occurrence dedup is exact, not an approximation. NOTE: still
    # conditioned on THIS individual's h1/h2 having covered the site at
    # all (a smaller instance of the same row-conditioning bias the old
    # either-covered stat had) -- see the TRUE genome-wide line below.
    site_tern = flat_tern[first_idx]
    indel_affected = float((site_tern == TERN_DEL).mean())
    print(f"  indel-affected ref bp (rows only, see TRUE line below): "
          f"{indel_affected*100:.2f}%")

    if true_cov is not None:
        n_either, n_hemi, n_null, n_total, n_deleted, n_founder_sites = true_cov
        if n_founder_sites > 0:
            print(f"  indel-affected ref bp TRUE (genome-wide, all K founders "
                  f"x all {n_total:,} R-site positions): "
                  f"{n_deleted / n_founder_sites * 100:.2f}%  "
                  f"(measured maize target ~39.3%, cassava ~32.7% -- "
                  f"experiments/simulator-indels/results/indel_biology_notes.md)")
        if n_total > 0:
            print(f"  either-founder TRUE covered (genome-wide, all "
                  f"{n_total:,} R-site positions): {n_either / n_total * 100:.2f}%  "
                  f"(measured target ~70-75% for outbred individuals -- "
                  f"PLAN.md SS2.7; this is the metric that target refers to, "
                  f"see 'either-founder covered (rows only)' below for why it "
                  f"differs from that number)")
            print(f"  hemizygous / nullizygous ref sites (genome-wide, TRUE): "
                  f"{n_hemi / n_total * 100:.2f}% / {n_null / n_total * 100:.2f}%  "
                  f"(one vs. both true homologs structurally absent; hemi is "
                  f"exactly 0 whenever --inbreeding=1.0 since h1==h2 then by "
                  f"construction)")

    # "Either true founder covered (rows only)": per UNIQUE OUTPUT-ROW
    # site, was any row there (stacked or not) a real match (ternary==1)
    # to h1's or h2's founder -- a max-reduction via np.maximum.at. This
    # is DELIBERATELY not the SS2.7 target metric -- it's conditioned on
    # a row existing at all, and a site where both homologs are absent
    # can never produce a row, so it structurally cannot see that case
    # and reads a near-tautological ~97% regardless of true coverage
    # (confirmed 2026-09-14: flat near 97% across a 4x --indel-density
    # sweep that tripled the true indel-affected fraction). Kept as a
    # secondary diagnostic -- "of the reads we did sample, how often did
    # genotyping-error noise corrupt which founder they matched" -- not
    # as the SS2.7 acceptance check; see the TRUE genome-wide line above
    # for that.
    row_ids = np.arange(flat_lab1.size)
    m1v = flat_tern[row_ids, flat_lab1] == TERN_MATCH
    m2v = flat_tern[row_ids, flat_lab2] == TERN_MATCH
    covered_row = (m1v | m2v).astype(np.int8)
    site_covered = np.zeros(uniq_keys.size, dtype=np.int8)
    np.maximum.at(site_covered, inv, covered_row)
    either_covered = float(site_covered.mean())
    print(f"  either-founder covered (rows only, NOT the SS2.7 metric -- "
          f"see TRUE line above): {either_covered*100:.2f}%")

    R = args.sites * args.indel_region_mult
    coverage_rate = float(uniq_keys.size / (data.shape[0] * R))
    print(f"  genome-wide site coverage: {coverage_rate*100:.2f}% of the "
          f"{data.shape[0]}x{R}-site generation region produced >=1 output "
          f"row (the rest is either truly uncovered -- including "
          f"nullizygous sites -- or beyond the first-{data.shape[1]}-rows "
          f"prefix-take)")

    # Coverage dispersion (index of dispersion, var/mean) over sites that
    # received >=1 row; Poisson/uniform sampling gives ~1, real blotchy
    # coverage should be markedly higher -- no numeric precedent yet, this
    # run establishes a first real baseline (PLAN.md SS2.7). Printed with
    # extra precision -- this is often a small-but-nonzero number, not
    # literally 0, and `.2f` was rounding it away to a misleading "0.00".
    disp = float(counts.var() / counts.mean()) if counts.mean() > 0 else float("nan")
    print(f"  coverage dispersion (var/mean rows-per-site): {disp:.4f}  "
          f"(Poisson~=1; want >>1 -- qualitative target, no numeric precedent yet)")
    print(f"  insertion stacking: max {int(counts.max())} rows at one site, "
          f"mean {counts[counts > 1].mean() if (counts > 1).any() else 0:.2f} "
          f"among stacked sites ({(counts > 1).mean()*100:.2f}% of covered sites)")

    # Mean gap (in skipped reference sites) between consecutive covered
    # sites within a window -- a cheap proxy for zero-coverage run length.
    refpos_valid = refpos[valid]
    w_flat = w_idx[valid]
    order = np.lexsort((refpos_valid, w_flat))
    rp_sorted, w_sorted = refpos_valid[order], w_flat[order]
    same_window = w_sorted[1:] == w_sorted[:-1]
    gaps = (rp_sorted[1:] - rp_sorted[:-1] - 1)[same_window]
    gaps = gaps[gaps >= 0]
    if gaps.size:
        print(f"  zero-coverage gap (sites) between covered positions: "
              f"mean {gaps.mean():.2f}  p95 {np.percentile(gaps, 95):.1f}")

    tern_vals, tern_counts = np.unique(flat_tern, return_counts=True)
    tot = flat_tern.size
    hist = {int(v): c / tot for v, c in zip(tern_vals, tern_counts)}
    print(f"  ternary histogram (all real rows): "
          f"deletion(-1)={hist.get(TERN_DEL, 0)*100:.2f}%  "
          f"diverged(0)={hist.get(TERN_DIV, 0)*100:.2f}%  "
          f"match(1)={hist.get(TERN_MATCH, 0)*100:.2f}%")

    dist_valid = dist_code[valid]
    real_dist = dist_valid[dist_valid != DIST_PAD]
    if real_dist.size:
        print(f"  distance code (all real rows): mean {real_dist.mean():.2f}  "
              f"p99 {np.percentile(real_dist, 99):.1f}  "
              f"saturated(={DIST_SAT-1}) frac {(real_dist == DIST_SAT-1).mean()*100:.2f}%")
    if count_code is not None:
        match_mask = tern[valid] == TERN_MATCH
        real_count = count_code[valid][match_mask]
        if real_count.size:
            print(f"  read-support count (MATCH cells only): mean {real_count.mean():.2f}  "
                  f"p99 {np.percentile(real_count, 99):.1f}  "
                  f"frac==0 {(real_count == 0).mean()*100:.2f}%")
    if refpos_path is not None:
        print(f"  reference positions → {refpos_path}  shape={refpos.shape}")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # E7: per-individual inbreeding coefficient F → per-window F (constant within
    # an individual). Spike at F=1 (inbred lines) plus Uniform[0,1] for the rest.
    finb = None
    if args.mixed_inbreeding:
        G = args.windows_per_individual
        if G <= 0:
            raise SystemExit("--mixed-inbreeding requires --windows-per-individual")
        n_ind = args.windows // G
        f_ind = np.where(rng.random(n_ind) < args.inbred_line_frac, 1.0,
                         rng.random(n_ind))
        finb = np.repeat(f_ind, G).astype(np.float32)

    # E11: discrete founder-count breeding-population mixture. Each individual is
    # one of three classes (k=2 F2, k=8 S1, k in [12,24] outbred), split inbred vs
    # het with a per-class target het-loci fraction.
    breeding_classes = None
    if args.breeding_pop:
        if args.windows_per_individual <= 0:
            raise SystemExit("--breeding-pop requires --windows-per-individual")
        het_by_class = [float(x) for x in args.het_by_class.split(",")]
        if len(het_by_class) != 3:
            raise SystemExit("--het-by-class needs 3 comma-separated values")
        # (weight, kmin, kmax, target_het)
        breeding_classes = [
            (0.25, 2, 2, het_by_class[0]),
            (0.25, 8, 8, het_by_class[1]),
            (0.50, 12, 24, het_by_class[2]),
        ]
    if args.constant_pair_frac > 0 and not args.breeding_pop:
        raise SystemExit("--constant-pair-frac requires --breeding-pop")

    if args.simulate_indels:
        if args.sharing_model != "coalescent":
            raise SystemExit(
                "--simulate-indels requires --sharing-model coalescent "
                "(indel content is drawn through the lineage array, which "
                "only exists in coalescent mode)")
        if args.sharing_theta is None:
            print("  WARNING: --simulate-indels without --sharing-theta falls back "
                  f"to the legacy island model, so only --ancestors={args.ancestors} "
                  "distinct indel haplotypes exist panel-wide; --sharing-theta is "
                  "strongly recommended.")
        if args.emit_snp_panel:
            raise SystemExit(
                "--emit-snp-panel is not supported with --simulate-indels (the "
                "panel is reference-site indexed; this mode's row axis is not)")
        if args.sites < 4096:
            print(f"  WARNING: --simulate-indels with --sites={args.sites} is small "
                  "relative to real indel structure (mean event ~530bp, bp-dominant "
                  "class 4-64kb) -- consider --sites 8192 or larger so a window can "
                  "represent more than one indel-affected region.")

    data, ibd, ind, panel, het_tgt, cls, refpos, short, true_cov = simulate(
        rng, args.windows, args.sites, args.founders,
        args.min_crossovers, args.max_crossovers,
        args.inbreeding, args.allele_sharing, args.bad_frac,
        args.recomb_span, args.recomb_tile,
        args.sharing_model, args.ancestors, args.ancestor_crossovers,
        args.derived_sfs, args.read_snps, args.error_block, args.gamete_balance,
        args.sharing_theta,
        args.windows_per_individual, args.min_founders, args.max_founders,
        args.emit_snp_panel, inbreeding_per_window=finb,
        breeding_classes=breeding_classes,
        class_inbred_frac=args.class_inbred_frac,
        constant_pair_frac=args.constant_pair_frac,
        constant_inbred_frac=args.constant_inbred_frac,
        simulate_indels=args.simulate_indels, indel_model=args.indel_model,
        indel_lineage_frac=args.indel_lineage_frac,
        indel_overlay_rate=args.indel_overlay_rate,
        indel_overlay_mean_len=args.indel_overlay_mean_len,
        indel_overlay_founder_freq=args.indel_overlay_founder_freq,
        subst_model=args.subst_model, subst_rate=args.subst_rate,
        coverage_model=args.indel_coverage_model, read_len=args.indel_read_len_bp,
        emit_read_counts=args.emit_read_counts,
        indel_density=args.indel_density, indel_ins_frac=args.indel_ins_frac,
        indel_large_frac=args.indel_large_frac,
        indel_small_alpha=args.indel_small_alpha,
        indel_small_max=args.indel_small_max,
        indel_large_logmean=args.indel_large_logmean,
        indel_large_logsd=args.indel_large_logsd,
        indel_max_len=args.indel_max_len, indel_coverage=args.indel_coverage,
        indel_ins_read_per_bp=args.indel_ins_read_per_bp,
        indel_max_stack=args.indel_max_stack,
        indel_region_mult=args.indel_region_mult,
        indel_recomb_suppress=args.indel_recomb_suppress,
        indel_recomb_flank=args.indel_recomb_flank,
        indel_anchor_thresh=args.indel_anchor_thresh,
        indel_ref_founder=args.indel_ref_founder)

    # E11: per-window het target plays the role of F (.finb) for eval-by-F tooling.
    if het_tgt is not None and finb is None:
        finb = het_tgt.astype(np.float32)

    out_dir = Path(args.workdir) / "data" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out
    np.save(out_path, data)
    if ibd is not None:
        ibd_path = out_dir / (Path(args.out).stem + ".ibd.npy")
        np.save(ibd_path, ibd)
    if ind is not None:
        ind_path = out_dir / (Path(args.out).stem + ".ind.npy")
        np.save(ind_path, ind.astype(np.int32))
    if panel is not None:
        panel_path = out_dir / (Path(args.out).stem + ".panel.npy")
        np.save(panel_path, panel)
    refpos_path = None
    if refpos is not None:
        refpos_path = out_dir / (Path(args.out).stem + ".refpos.npy")
        np.save(refpos_path, refpos)
    if finb is not None:
        finb_path = out_dir / (Path(args.out).stem + ".finb.npy")
        np.save(finb_path, finb)
        if args.breeding_pop:
            print(f"  het target (.finb)  → {finb_path}  "
                  f"inbred windows {(finb == 0).mean()*100:.0f}%, "
                  f"mean target het {finb.mean():.3f}")
        else:
            print(f"  mixed inbreeding    → {finb_path}  "
                  f"F: {(finb == 1).mean()*100:.0f}% inbred lines, "
                  f"mean {finb.mean():.2f}, het-individuals {(finb < 1).mean()*100:.0f}%")
    if cls is not None:
        cls_path = out_dir / (Path(args.out).stem + ".cls.npy")
        np.save(cls_path, cls.astype(np.int8))
        # class id 3 = Tier2 "constant" individuals (out-of-band, not one of
        # the 3 breeding classes) -- minlength=4 so bincount doesn't choke
        # on it even when --constant-pair-frac=0 (frac[3] is just 0 then).
        frac = np.bincount(cls, minlength=4) / len(cls)
        print(f"  class id (.cls)     → {cls_path}  "
              f"mix k2/k8/outbred/constant = "
              f"{frac[0]*100:.0f}/{frac[1]*100:.0f}/{frac[2]*100:.0f}/{frac[3]*100:.0f}%")

    if args.simulate_indels:
        _print_indel_summary(args, data, refpos, short, out_path, refpos_path, true_cov)
        return

    # Verification summary
    K = args.founders
    feats = data[:, :, :K]
    h1 = data[:, :, K].astype(np.int64)
    h2 = data[:, :, K + 1].astype(np.int64)
    ii = np.arange(data.shape[0])[:, None]
    tt = np.arange(args.sites)[None, :]
    m1 = feats[ii, tt, h1]                       # H1 founder match indicator
    m2 = feats[ii, tt, h2]                       # H2 founder match indicator
    either = np.maximum(m1, m2)                  # active gamete is one of the two
    het = (h1 != h2)                             # heterozygous sites
    switches = (h1[:, 1:] != h1[:, :-1]).sum(1)

    print(f"\nWrote {out_path}")
    print(f"  shape={data.shape}  dtype={data.dtype}  "
          f"size={data.nbytes / 1e9:.2f} GB")
    print(f"  mean allele sharing : {feats.mean():.4f}  (target {args.allele_sharing})")
    share = (feats != 0).sum(axis=2).ravel()     # # founders matching, per site
    nz = share[share > 0]
    if nz.size:
        sing = float((nz == 1).mean())
        print(f"  sharing SFS         : singleton {sing*100:.1f}%  "
              f"median {int(np.median(nz))}  mean {nz.mean():.2f}  max {int(nz.max())}  "
              f"(theta={args.sharing_theta}; want singleton-dominated, tail→K)")
    if ibd is not None:
        print(f"  IBD truth           → {ibd_path}  shape={ibd.shape}")
    if panel is not None:
        print(f"  SNP panel           → {panel_path}  shape={panel.shape}  "
              f"size={panel.nbytes / 1e9:.2f} GB")
    if ind is not None:
        G = args.windows_per_individual
        n_ind = len(ind) // G
        kpi = np.array([len(np.unique(h1[ind == i])) for i in range(min(n_ind, 200))])
        print(f"  individuals         → {ind_path}  n={n_ind}  windows/ind={G}  "
              f"founders/ind: mean {kpi.mean():.1f} range [{kpi.min()},{kpi.max()}]")
    print(f"  either-founder match: {either.mean():.4f}  "
          f"(active gamete forced on good sites; expect ~{1 - args.bad_frac:.3f}+)")
    print(f"  H1 / H2 match (het) : {m1[het].mean():.4f} / {m2[het].mean():.4f}  "
          f"(gamete-balance {args.gamete_balance}; only the sampled gamete is forced)")
    print(f"  het-site fraction   : {het.mean():.4f}  "
          f"(F={args.inbreeding}; 0 ⇒ mostly het, interleaved reads)")
    print(f"  crossovers/window   : mean={switches.mean():.2f}  "
          f"min={switches.min()}  max={switches.max()}")
    print(f"  H1==H2 fraction     : {(h1 == h2).all(1).mean():.4f}  "
          f"(target {args.inbreeding})")
    print(f"  label range         : [{data[:, :, K].min()}, {data[:, :, K + 1].max()}]")

    if data.shape[-1] == K + 3:
        rate = data[:, :, K + 2].astype(np.float64)
        sw = np.zeros_like(rate)
        sw[:, 1:] = (h1[:, 1:] != h1[:, :-1])      # switch entering site t
        r = np.corrcoef(rate.ravel(), sw.ravel())[0, 1]
        print(f"  recomb rate (hidden): min={rate.min():.0f}  max={rate.max():.0f}  "
              f"mean={rate.mean():.1f}  (span target {args.recomb_span:.0f})")
        print(f"  corr(rate, switch)  : {r:.4f}  "
              f"(positive ⇒ breakpoints follow the hidden rate; eval-only column)")

    # Feature LD: lag-1 autocorrelation of the per-founder match indicator.
    # Independent sharing ≈ 0; coalescent > 0 (shared IBD tracts).  When a rate
    # track is present, LD should be weaker in hotspots (faster tract breakdown).
    def _lag1(fa):
        a = fa[:, 1:, :].astype(np.float64)
        b = fa[:, :-1, :].astype(np.float64)
        sa, sb = a.std(), b.std()
        if sa < 1e-9 or sb < 1e-9:
            return 0.0
        return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))

    print(f"  match LD (lag-1 ac) : {_lag1(feats):.4f}  "
          f"(model={args.sharing_model}; >0 ⇒ allele-sharing tracts)")
    if data.shape[-1] == K + 3:
        rate_t = data[:, :, K + 2]
        hot = rate_t[:, :-1] >= np.percentile(rate_t, 90)
        cold = rate_t[:, :-1] <= np.percentile(rate_t, 10)
        same = (feats[:, 1:, :] == feats[:, :-1, :])
        ac_hot = same[hot].mean() if hot.any() else float("nan")
        ac_cold = same[cold].mean() if cold.any() else float("nan")
        print(f"  match persist hot/cold: {ac_hot:.4f} / {ac_cold:.4f}  "
              f"(cold > hot ⇒ recombination is observable in the features)")


if __name__ == "__main__":
    main()
