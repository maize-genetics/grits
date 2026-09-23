"""
Synthetic leave-one-out held-out training augmentation (Option B from the
region-buffering-fix follow-up discussion, experiments/depth-confidence-fix/).

Motivation: real held-out (OUT-panel) founder-decode accuracy is far below
in-panel accuracy (measured this session: OUT-INBRED error 23-40% vs IDX's
~0%), and the root cause isn't primarily a coverage or capacity problem --
it's that training has NEVER exposed the model to the switch RATE held-out
tracking actually needs. Every simulated training individual's true path is
built from REAL panel founders with realistic meiotic recombination, so the
model learns a strong "stay" prior calibrated to ~0.01 switches/T512-window
(measured directly from Branch B's own training data: 99.02% of windows
show zero switches). Held-out prediction needs to track ancestral
coalescent/IBD relatedness structure instead, which switches at whatever
rate the founders' own lineage-sharing tracts imply -- a completely
different, much higher-frequency signal nothing in the current recipe ever
demonstrates.

This module doesn't touch simulate_alleles.py's core generation at all --
it's a pure post-processing step. Key insight: `_coalescent_feats`'s
match1/match2 emission signal is already LINEAGE-based, not founder-identity
-based (`(G == Sa[:,None])` where G comes from the founder's mini-haplotype,
itself indexed by lineage) -- so any VISIBLE founder sharing a HIDDEN
founder's lineage at a site will show the exact same genuine match evidence
a real held-out founder's own reads would show there. The only thing that
needs to change is which founder's columns are exposed as candidates, and
what the true label says.

Recipe: generate normally with founders=K+1 (one extra "virtual" founder,
index K, hidden from the model) and --sharing-theta set (which makes
simulate_alleles.py additionally write a <out>.ibd.npy sidecar -- the full
per-(individual, site, founder) ancestral-lineage-assignment array, already
computed internally, just not normally exposed). This script then:
  1. Wherever the true label (lab1/lab2) points to the hidden founder K,
     relabels it to a VISIBLE founder (0..K-1) sharing K's lineage
     assignment at that exact site (from the .ibd.npy sidecar, looked up
     via the .refpos.npy sidecar since ibd is R-indexed, not T-indexed).
     Multiple candidates -> resolved via a seeded RNG (uniform choice, not
     always the lowest index, to avoid a systematic per-founder bias in the
     augmented labels).
  2. Rows with ZERO visible lineage-mates (measured 0.00% at
     --sharing-theta=4.0 --ancestors=6 on a real 10-individual/2000-site
     test run, but handled defensively regardless) get LABEL_PAD=-1 --
     REUSES the training pipeline's own existing, already-tested
     "unlabeled position" convention (IndelDiploidDataset already remaps
     LABEL_PAD to the null-founder index K for real data's own missing-label
     case; see train_diploid_indel.py's docstring), not a new code path.
  3. Drops the hidden founder's own ternary/distance/count columns from the
     output entirely, so the model can never see it as a candidate --
     producing a standard [n, T, 3*K+2] array, same on-disk contract as
     everything else, indistinguishable from a non-augmented file except by
     the caller's own bookkeeping of which file is which.

Usage:
    python scripts/heldout_augment.py \\
        --in-npy <out>.npy --ibd-npy <out>.ibd.npy --refpos-npy <out>.refpos.npy \\
        --num-parents 25 --out <augmented>.npy --seed 0
    (num-parents is K, the VISIBLE count -- the input file must have been
    generated with --founders K+1.)
"""
import argparse
import sys

sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/grits_workdir/indel-v3-simulator-wt/src")
import numpy as np

from python.crf.simulate_alleles import LABEL_PAD


def relabel_for_heldout(data, ibd, refpos, K, hidden_idx=None, rng=None):
    """data: [n,T,3*(K+1)+2] int8 (emit_read_counts=True output, K+1 founders
    simulated). ibd: [n,R,K+1] int8 (.ibd.npy sidecar, R-indexed). refpos:
    [n,T] int32 (.refpos.npy sidecar, maps each output row to its site in
    the R-region; -1 = PAD row). K: number of VISIBLE founders in the
    output (K+1 were simulated). hidden_idx: which of the K+1 simulated
    founders to hide (default K, the last one -- matches the recommended
    `--founders K+1` generation convention of appending one extra founder).
    rng: np.random.Generator for tie-breaking among multiple lineage-mates
    (required whenever any hidden-label row exists).

    Returns (out [n,T,3*K+2] int8, n_hidden_labels int, n_unlabelable int).
    n_hidden_labels counts every (row, haplotype) instance where the true
    label pointed to the hidden founder (relabeled or not); n_unlabelable
    is how many of those had zero visible lineage-mates and got LABEL_PAD.
    """
    if hidden_idx is None:
        hidden_idx = K
    Ktot = K + 1
    n, T, ncol = data.shape
    assert ncol == 3 * Ktot + 2, f"expected 3*{Ktot}+2={3*Ktot+2} cols, got {ncol}"
    assert ibd.shape[0] == n and ibd.shape[2] == Ktot, \
        f"ibd shape {ibd.shape} doesn't match data (n={n}, Ktot={Ktot})"
    assert refpos.shape == (n, T)

    tern = data[:, :, :Ktot]
    lab1 = data[:, :, Ktot].copy()
    lab2 = data[:, :, Ktot + 1].copy()
    dist = data[:, :, Ktot + 2:2 * Ktot + 2]
    count = data[:, :, 2 * Ktot + 2:3 * Ktot + 2]

    visible_cols = [c for c in range(Ktot) if c != hidden_idx]

    n_hidden_labels = 0
    n_unlabelable = 0

    for i in range(n):
        rp = refpos[i]
        valid = rp >= 0
        if not valid.any():
            continue
        rows = np.flatnonzero(valid)
        sites = rp[rows]
        ibd_i = ibd[i]  # [R, Ktot]

        for lab in (lab1, lab2):
            hit = lab[i, rows] == hidden_idx
            if not hit.any():
                continue
            hit_rows = rows[hit]
            hit_sites = sites[hit]
            n_hidden_labels += hit_rows.size

            hidden_lineage = ibd_i[hit_sites, hidden_idx]          # [H]
            visible_lineage = ibd_i[hit_sites][:, visible_cols]    # [H, K]
            match = visible_lineage == hidden_lineage[:, None]     # [H, K]
            n_cand = match.sum(axis=1)

            resolved = np.full(hit_rows.size, LABEL_PAD, dtype=np.int64)
            has_cand = n_cand > 0
            n_unlabelable += int((~has_cand).sum())
            if has_cand.any():
                # Uniform choice among candidates per row, via Gumbel-top-1
                # over the boolean match mask (vectorized, no per-row loop).
                gum = rng.random(match.shape)
                gum[~match] = -1.0  # never select a non-candidate
                choice = np.argmax(gum, axis=1)
                resolved[has_cand] = np.array(visible_cols)[choice[has_cand]]
            lab[i, hit_rows] = resolved

    # Remap visible-founder labels (still 0..Ktot-1 minus hidden_idx) down to
    # a dense 0..K-1 range once hidden_idx's column is dropped.
    remap = np.full(Ktot, LABEL_PAD, dtype=np.int64)
    remap[visible_cols] = np.arange(K)

    def apply_remap(lab):
        out = lab.copy()
        real = lab != LABEL_PAD
        out[real] = remap[lab[real]]
        return out

    lab1 = apply_remap(lab1)
    lab2 = apply_remap(lab2)

    tern_out = tern[:, :, visible_cols]
    dist_out = dist[:, :, visible_cols]
    count_out = count[:, :, visible_cols]

    out = np.concatenate(
        [tern_out, lab1[:, :, None], lab2[:, :, None], dist_out, count_out],
        axis=2).astype(np.int8)
    return out, n_hidden_labels, n_unlabelable


def permute_founders_per_individual(data, ibd, rng):
    """Founders are exchangeable in simulate_alleles.py's generative model
    (no founder index carries any special identity -- lineage/crossover
    draws are i.i.d. across the founder axis), so relabel_for_heldout's
    convention of always hiding the LAST column (index K, i.e. the (K+1)-th
    simulated founder) would, left alone, make the model see column K
    excluded from every single augmented training example -- a systematic
    "this column position is always suppressed" artifact that has nothing
    to do with the actual held-out-tracking skill being taught, and would
    bias inference on real data (where every column position is used
    normally). Fix: apply an INDEPENDENT random permutation of the Ktot
    founder columns per individual, consistently across `data`'s
    ternary/distance/count blocks (leaving lab1/lab2 correctly repointed
    through the same permutation) and `ibd`, before calling
    relabel_for_heldout -- so which REAL founder ends up hidden (column
    index Ktot-1 after permuting) varies per individual.

    data: [n,T,3*Ktot+2] int8 (Ktot founders, pre-relabel). ibd: [n,R,Ktot]
    int8. Returns (data_permuted, ibd_permuted) -- ready to feed into
    relabel_for_heldout unchanged (hidden_idx still defaults to Ktot-1).
    """
    n, T, ncol = data.shape
    Ktot = ibd.shape[2]
    assert ncol == 3 * Ktot + 2

    tern = data[:, :, :Ktot]
    lab1 = data[:, :, Ktot]
    lab2 = data[:, :, Ktot + 1]
    dist = data[:, :, Ktot + 2:2 * Ktot + 2]
    count = data[:, :, 2 * Ktot + 2:3 * Ktot + 2]

    tern_p = np.empty_like(tern)
    dist_p = np.empty_like(dist)
    count_p = np.empty_like(count)
    lab1_p = np.empty_like(lab1)
    lab2_p = np.empty_like(lab2)
    ibd_p = np.empty_like(ibd)

    for i in range(n):
        perm = rng.permutation(Ktot)          # perm[new_col] = old_col
        inv = np.empty(Ktot, dtype=np.int64)  # inv[old_col] = new_col
        inv[perm] = np.arange(Ktot)

        tern_p[i] = tern[i][:, perm]
        dist_p[i] = dist[i][:, perm]
        count_p[i] = count[i][:, perm]
        ibd_p[i] = ibd[i][:, perm]

        l1 = lab1[i]
        l2 = lab2[i]
        real1 = l1 != LABEL_PAD
        real2 = l2 != LABEL_PAD
        out1 = l1.copy()
        out2 = l2.copy()
        out1[real1] = inv[l1[real1]]
        out2[real2] = inv[l2[real2]]
        lab1_p[i] = out1
        lab2_p[i] = out2

    out = np.concatenate(
        [tern_p, lab1_p[:, :, None], lab2_p[:, :, None], dist_p, count_p],
        axis=2).astype(np.int8)
    return out, ibd_p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-npy", required=True)
    ap.add_argument("--ibd-npy", required=True)
    ap.add_argument("--refpos-npy", required=True)
    ap.add_argument("--num-parents", type=int, required=True,
                     help="K, the number of VISIBLE founders in the output "
                          "(the input file must have been generated with "
                          "--founders K+1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-permute", action="store_true",
                     help="skip the per-individual founder-column permutation "
                          "(default: on -- avoids the fixed-hidden-column artifact)")
    args = ap.parse_args()

    data = np.load(args.in_npy)
    ibd = np.load(args.ibd_npy)
    refpos = np.load(args.refpos_npy)
    rng = np.random.default_rng(args.seed)

    if not args.no_permute:
        data, ibd = permute_founders_per_individual(data, ibd, rng)

    out, n_hidden, n_unlabelable = relabel_for_heldout(
        data, ibd, refpos, args.num_parents, rng=rng)

    print(f"input shape: {data.shape}  output shape: {out.shape}")
    print(f"hidden-founder label instances: {n_hidden}")
    if n_hidden:
        print(f"unlabelable (0 visible lineage-mates): {n_unlabelable} "
              f"({100 * n_unlabelable / n_hidden:.2f}%)")
    np.save(args.out, out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
