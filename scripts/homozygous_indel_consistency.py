"""Homozygous indels-on/off consistency check (v3 acceptance test).

User's framing: "turn the indels on or off, and the pattern should be
similar for the homozygous lines. It gets tricky with hets, but that's
an overlay problem" -- i.e. for h1==h2 individuals there's no haplotype-
disambiguation work for indels to do, so the per-site "which founder is
supported" signal should point to the same true founder with roughly the
same reliability regardless of whether --simulate-indels is on, and
regardless of --indel-model.

This checks the raw simulator SIGNAL (not a trained model): per site,
each founder's "support" = count of MATCH(=1) rows for that founder
within the site's window (this definition is format-agnostic: plain
binary features and indel-mode ternary both use 1 for "match"). argmax
support vs. the true (h1==h2) label gives a per-site correctness call,
aggregated to a top-1 accuracy per configuration. Compares:
  - indels OFF (today's plain K+2 binary format)
  - indels ON, --indel-model tracts / lineage / overlay

A configuration whose accuracy differs sharply from the OFF baseline is
the failure mode this test exists to catch.

Usage: pixi run -- python scripts/homozygous_indel_consistency.py
(PYTHONPATH=src is set automatically by pixi; run from the repo root.)
"""
import numpy as np

from python.crf.simulate_alleles import simulate

K = 24
WINDOWS = 400
T = 512
COMMON = dict(
    # min/max_cross=0: ONE constant true founder per window (h1==h2, no
    # breakpoints) -- makes "true label" well-defined for the whole
    # window; this diagnostic is about raw signal quality with zero
    # haplotype/breakpoint ambiguity, not about decoding recombination.
    windows=WINDOWS, sites=T, founders=K, min_cross=0, max_cross=0,
    inbreeding=1.0, allele_sharing=0.6, bad_frac=0.02,
    sharing_model="coalescent", ancestors=K, sharing_theta=4.0,
    gamete_balance=0.5,
)


def top1_accuracy_plain(out, K):
    feats = out[:, :, :K]          # [windows,T,K] binary match
    lab1 = out[:, :, K]
    support = feats.sum(axis=1)    # [windows,K]
    pred = support.argmax(axis=1)
    true = lab1[:, 0]              # inbred: constant per window in this sim design? verify below
    return pred, true


def top1_accuracy_indel(out, K):
    tern = out[:, :, :K]           # [windows,T,K] ternary
    lab1 = out[:, :, K]
    support = (tern == 1).sum(axis=1)
    pred = support.argmax(axis=1)
    true = lab1[:, 0]
    return pred, true


results = {}

rng = np.random.default_rng(123)
out_off = simulate(rng, simulate_indels=False, **COMMON)[0]
pred, true = top1_accuracy_plain(out_off, K)
valid = true >= 0
acc = (pred[valid] == true[valid]).mean()
results["off"] = (acc, valid.sum(), WINDOWS)

for mode in ["tracts", "lineage", "overlay"]:
    rng = np.random.default_rng(123)
    out_on = simulate(rng, simulate_indels=True, indel_model=mode,
                       indel_region_mult=4, indel_coverage=2.0, **COMMON)[0]
    pred, true = top1_accuracy_indel(out_on, K)
    valid = true >= 0
    acc = (pred[valid] == true[valid]).mean()
    results[mode] = (acc, valid.sum(), WINDOWS)

print(f"{'config':<10} {'top1_acc':>10} {'valid/total windows':>22}")
for name, (acc, v, tot) in results.items():
    print(f"{name:<10} {acc*100:>9.2f}% {v:>10}/{tot}")

off_acc = results["off"][0]
print("\ndelta vs indels-off (pp):")
for name, (acc, v, tot) in results.items():
    if name == "off":
        continue
    print(f"  {name:<10} {100*(acc-off_acc):+.2f}pp")
