#!/usr/bin/env python
"""Shareable diagram of the v6 fit B decode on real reads, with the CRF numbers fit B learned
(checkpoints/v6-stage2B/last.ckpt) next to fit A's. Writes results/v6_fitB.{png,svg}."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "results"
C = {"sim": "#E8EAED", "enc": "#CFE2FF", "crf": "#FFE5CC", "out": "#D4EDDA", "edge": "#3C4043", "val": "#B06000",
     "off": "#80868B"}

fig, ax = plt.subplots(figsize=(16, 10.4))
ax.set_xlim(0, 16); ax.set_ylim(-0.9, 9.3); ax.axis("off")


def box(x, y, w, h, color, title, lines=(), fs=13, lfs=11, ls="-", tcol="#202124"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.15", fc=color, ec=C["edge"],
                                lw=1.5, linestyle=ls))
    ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top", fontsize=fs, weight="bold", color=tcol)
    for i, s in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.7 - 0.34 * i, s, ha="center", va="top", fontsize=lfs, color="#3C4043")


def arrow(x0, y0, x1, y1, color=C["edge"], ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15, lw=1.6, color=color,
                                 linestyle=ls))


ax.text(8, 9.05, "v6 fit B: decoding a real sample", ha="center", fontsize=21, weight="bold")
ax.text(8, 8.62, "encoder frozen after stage 1; the orange numbers are what fit B learned on simulated data (fit A in brackets)",
        ha="center", fontsize=12.5, color="#5F6368")

# inputs and encoder
box(0.2, 5.1, 2.4, 2.9, C["sim"], "Real reads", ["refmap ternary", "per founder", "(match / diverged / −1)", "",
                                                 "+ sample's genome-wide", "match rate"], fs=13, lfs=10.5)
arrow(2.65, 6.55, 3.15, 6.55)
box(3.15, 5.1, 2.3, 2.9, C["enc"], "Encoder", ["transformer over", "a 512-row window"], fs=13)
heads = [("gate", "trust in each row"), ("crossovers", "switches / window"), ("deletion dosage", "0 / 1 / 2 deleted"),
         ("out-of-panel", "window score"), ("het", "not used")]
for i, (h, d) in enumerate(heads):
    y = 7.45 - 0.55 * i
    off = h == "het"
    ax.add_patch(FancyBboxPatch((5.95, y), 2.55, 0.42, boxstyle="round,pad=0.03,rounding_size=0.1", fc=C["enc"],
                                ec=C["edge"], lw=1.2, linestyle="--" if off else "-"))
    ax.text(6.07, y + 0.21, h, va="center", fontsize=11.5, weight="bold", color=C["off"] if off else "#202124")
    ax.text(8.62, y + 0.21, d, va="center", fontsize=10, color=C["off"] if off else "#5F6368")
    arrow(5.5, 6.55, 5.95, y + 0.21)

# CRF terms
ax.text(0.2, 4.55, "CRF: what scores each founder pair along the window", fontsize=14.5, weight="bold", color="#E8710A")
terms = [
    ("Read evidence (per row)", "gate × mixture of the pair's read scores",
     "match +0.39 (+1.29) · diverged −3.67 (−4.36) · −1 −4.24 (−6.15)"),
    ("Deletion dosage (per row)", "gate × weight × log P(dosage = how many of the pair read −1)", "weight 0.15 (0)"),
    ("Switch cost (per transition)", "from crossovers × scale × exp(w × out-of-panel score)",
     "scale 0.34 (1) · w −1.95 (0): fewer switches in out-of-panel windows"),
    ("Founder prior (on entering a founder)", "weight × log(founder's share of the sample match rate)",
     "weight 4.03 (1) · out-of-panel fade −0.08 (1): effectively off"),
    ("Het prior", "head still trained; its prior is not used in decoding", "off in both fits"),
]
for i, (name, how, val) in enumerate(terms):
    y = 3.9 - 0.92 * i
    off = name == "Het prior"
    ax.add_patch(FancyBboxPatch((0.2, y - 0.38), 15.6, 0.78, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="#F8F9FA" if off else C["crf"], ec=C["edge"], lw=1.0, linestyle="--" if off else "-"))
    ax.text(0.4, y + 0.15, name, va="center", fontsize=11.5, weight="bold", color=C["off"] if off else "#202124")
    ax.text(0.4, y - 0.18, how, va="center", fontsize=10, color="#5F6368")
    ax.text(15.6, y, val, va="center", ha="right", fontsize=10.5, color=C["off"] if off else C["val"], weight="bold")

# output
box(10.9, 5.4, 2.2, 2.3, C["crf"], "Viterbi decode", ["best founder pair", "per row"], fs=13)
box(13.6, 5.4, 2.2, 2.3, C["out"], "Genotype calls", ["pair → alleles", "→ BED / VCF"], fs=13)
arrow(10.4, 6.55, 10.9, 6.55); arrow(13.15, 6.55, 13.6, 6.55)
ax.plot([10.15, 10.4], [6.55, 6.55], color=C["edge"], lw=1.6)
arrow(12.0, 4.4, 12.0, 5.38, color="#E8710A")
ax.text(12.15, 4.85, "CRF terms below", fontsize=10, color="#E8710A")

ax.text(0.2, -0.75, "What fit B changed vs A: softer read scores and a much stronger founder prior, with switching kept "
        "moderate (scale 0.34). It turned the out-of-panel affinity fade off and used the out-of-panel score to switch less, "
        "not more.", fontsize=10.5, color="#5F6368", style="italic", wrap=True)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"v6_fitB.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
print(OUT / "v6_fitB.png")
