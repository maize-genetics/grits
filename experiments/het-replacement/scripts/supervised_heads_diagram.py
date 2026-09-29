#!/usr/bin/env python
"""Shareable diagram: encoder trained on simulator-truth targets (gate, switch,
heterozygosity, affinity), then plugged into the likelihood-emission CRF.
Writes results/supervised_heads_design.{png,svg}."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "results"
C = {"sim": "#E8EAED", "truth": "#FFF4C2", "enc": "#CFE2FF", "lik": "#D4EDDA", "crf": "#FFE5CC", "edge": "#3C4043"}

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")


def box(x, y, w, h, color, title, lines=(), fs=14, lfs=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.15",
                                fc=color, ec=C["edge"], lw=1.5))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=fs, weight="bold", color="#202124")
    for i, s in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.78 - 0.36 * i, s, ha="center", va="top", fontsize=lfs, color="#3C4043")


def arrow(x0, y0, x1, y1, ls="-", color=C["edge"]):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.7,
                                 color=color, linestyle=ls))


ax.text(8, 8.7, "Encoder trained on the truth, then plugged into the CRF", ha="center", fontsize=21, weight="bold")
ax.text(8, 8.25, "each head learns a known quantity from simulated data; the CRF combines them with the read likelihood",
        ha="center", fontsize=13, color="#5F6368")

# panel titles
ax.text(0.3, 7.65, "1  Training — simulated data only", fontsize=15, weight="bold", color="#1A73E8")
ax.text(0.3, 3.75, "2  Decoding — real reads", fontsize=15, weight="bold", color="#E8710A")
ax.plot([0.3, 15.7], [4.05, 4.05], color="#DADCE0", lw=1.2)

# ---- training panel
box(0.3, 4.4, 2.8, 3.0, C["sim"], "Simulated reads", ["ternary × distance", "× read count", "4 depths (0.1–2x)"])
box(4.0, 4.4, 3.2, 3.0, C["enc"], "Encoder", ["transformer over", "the window"])
arrow(3.15, 5.9, 4.0, 5.9)
heads = [("gate", "row trustworthy?"), ("switch", "crossover here?"), ("het", "parents distinct?"), ("affinity", "founders in sample")]
for i, (h, q) in enumerate(heads):
    y = 6.7 - 0.72 * i
    for x, w, col, txt, bold in ((8.0, 2.4, C["enc"], h, True), (11.4, 4.3, C["truth"], q, False)):
        ax.add_patch(FancyBboxPatch((x, y), w, 0.5, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=col, ec=C["edge"], lw=1.3))
        ax.text(x + w / 2, y + 0.25, txt, ha="center", va="center", fontsize=12.5, weight="bold" if bold else "normal")
    arrow(7.25, 5.9, 8.0, y + 0.25)
    ax.add_patch(FancyArrowPatch((11.38, y + 0.25), (10.45, y + 0.25), arrowstyle="-|>", mutation_scale=14, lw=1.4,
                                 color="#B06000", linestyle="--"))
ax.text(13.55, 7.4, "simulator truth (known exactly)", ha="center", fontsize=12.5, weight="bold", color="#B06000")
ax.text(11.85, 4.3, "loss = each head vs its truth", ha="center", fontsize=11.5, style="italic", color="#5F6368")

# ---- decoding panel
box(0.3, 0.35, 2.8, 3.0, C["sim"], "Real reads", ["refmap → rows", "ternary × distance"])
box(4.0, 1.95, 3.2, 1.4, C["enc"], "Trained encoder", ["frozen"], fs=13)
arrow(3.15, 2.65, 4.0, 2.65)
box(4.0, 0.35, 3.2, 1.35, C["lik"], "Read likelihood ℓ", ["shared by all founders"], fs=13)
arrow(3.15, 1.05, 4.0, 1.05)
box(7.95, 1.95, 3.1, 1.4, C["enc"], "", [], fs=1)
for i, t in enumerate(("gate → row weight", "switch → change cost", "het → homozygous penalty", "affinity → founder prior")):
    ax.text(9.5, 3.12 - 0.3 * i, t, ha="center", va="center", fontsize=11.5)
arrow(7.25, 2.65, 7.95, 2.65)
arrow(11.1, 2.65, 11.8, 2.65)
arrow(7.25, 1.05, 11.8, 1.05)
box(11.8, 0.35, 3.9, 3.0, C["crf"], "CRF decode", ["emission = gate × ℓ + affinity", "mixture over haplotypes", "switch cost, het penalty", "→ founder-pair path"], fs=14)

for i, (lab, c) in enumerate((("data", C["sim"]), ("encoder (learned)", C["enc"]), ("simulator truth", C["truth"]),
                              ("likelihood", C["lik"]), ("CRF", C["crf"]))):
    ax.add_patch(Rectangle((0.35 + i * 3.1, 0.02), 0.28, 0.2, fc=c, ec=C["edge"], lw=0.8))
    ax.text(0.72 + i * 3.1, 0.12, lab, va="center", fontsize=11)

for ext in ("png", "svg"):
    fig.savefig(OUT / f"supervised_heads_design.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
print(OUT / "supervised_heads_design.png")
