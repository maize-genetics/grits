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


ax.text(8, 8.72, "Encoder trained on the truth, then plugged into the CRF", ha="center", fontsize=21, weight="bold")
ax.text(8, 8.3, "everything is learned or set on simulated data; real reads are only decoded and scored",
        ha="center", fontsize=13, color="#5F6368")

# ---- 1 train the encoder heads
ax.text(0.3, 7.8, "1  Train the encoder — simulated data", fontsize=14.5, weight="bold", color="#1A73E8")
box(0.3, 5.45, 2.8, 2.1, C["sim"], "Simulated reads", ["4 depths (0.1–2x)"], fs=13)
box(4.0, 5.45, 3.0, 2.1, C["enc"], "Encoder", ["transformer over", "the window"], fs=13)
arrow(3.15, 6.5, 4.0, 6.5)
heads = [("gate", "row trustworthy?"), ("switch", "crossover here?"), ("het", "parents distinct?"), ("affinity", "founders in sample")]
for i, (h, q) in enumerate(heads):
    y = 7.1 - 0.52 * i
    for x, w, col, txt, bold in ((7.8, 2.3, C["enc"], h, True), (11.2, 4.5, C["truth"], q, False)):
        ax.add_patch(FancyBboxPatch((x, y), w, 0.38, boxstyle="round,pad=0.03,rounding_size=0.1",
                                    fc=col, ec=C["edge"], lw=1.2))
        ax.text(x + w / 2, y + 0.19, txt, ha="center", va="center", fontsize=12, weight="bold" if bold else "normal")
    arrow(7.05, 6.5, 7.8, y + 0.19)
    ax.add_patch(FancyArrowPatch((11.18, y + 0.19), (10.15, y + 0.19), arrowstyle="-|>", mutation_scale=13, lw=1.3,
                                 color="#B06000", linestyle="--"))
ax.text(13.45, 7.62, "simulator truth", ha="center", fontsize=12, weight="bold", color="#B06000")
ax.plot([0.3, 15.7], [5.2, 5.2], color="#DADCE0", lw=1.2)

# ---- 2 fit the CRF's few numbers on the encoder's outputs
ax.text(0.3, 4.85, "2  Fit the emission and CRF numbers — simulated data, encoder frozen", fontsize=14.5, weight="bold", color="#188038")
box(0.3, 3.0, 2.3, 1.6, C["sim"], "Simulated reads", [], fs=12.5)
box(3.05, 3.0, 2.6, 1.6, C["enc"], "Frozen encoder", ["gate, switch,", "het, affinity"], fs=12.5, lfs=11)
box(6.1, 3.0, 3.6, 1.6, C["crf"], "CRF", ["read likelihood ℓ  ← fitted", "stay, penalty, prior weights ← fitted"], fs=12.5, lfs=10.5)
box(10.15, 3.0, 2.3, 1.6, C["crf"], "Decoded path", [], fs=12.5)
box(12.9, 3.0, 2.8, 1.6, C["truth"], "True path", ["loss → update", "the fitted numbers"], fs=12.5, lfs=11)
arrow(2.65, 3.8, 3.05, 3.8); arrow(5.7, 3.8, 6.1, 3.8); arrow(9.75, 3.8, 10.15, 3.8)
ax.add_patch(FancyArrowPatch((12.88, 3.8), (12.5, 3.8), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#B06000", linestyle="--"))
ax.add_patch(FancyArrowPatch((14.3, 3.0), (7.9, 2.98), connectionstyle="arc3,rad=-0.12", arrowstyle="-|>", mutation_scale=14,
                             lw=1.3, color="#B06000", linestyle="--"))
ax.plot([0.3, 15.7], [2.7, 2.7], color="#DADCE0", lw=1.2)

# ---- 3 decode real reads
ax.text(0.3, 2.35, "3  Decode real reads — nothing tuned here", fontsize=14.5, weight="bold", color="#E8710A")
box(0.3, 0.45, 2.8, 1.65, C["sim"], "Real reads", ["refmap → rows"], fs=13)
for (y, col, title, sub) in ((1.35, C["enc"], "Encoder", "gate, switch, het, affinity"),
                             (0.45, C["lik"], "Read likelihood ℓ", "fitted in step 2")):
    ax.add_patch(FancyBboxPatch((4.0, y), 5.0, 0.72, boxstyle="round,pad=0.03,rounding_size=0.12",
                                fc=col, ec=C["edge"], lw=1.4))
    ax.text(4.25, y + 0.36, title, ha="left", va="center", fontsize=13, weight="bold")
    ax.text(8.8, y + 0.36, sub, ha="right", va="center", fontsize=11.5, color="#3C4043")
    arrow(3.15, 1.27, 4.0, y + 0.36)
    arrow(9.05, y + 0.36, 9.8, 1.27)
box(9.8, 0.45, 5.9, 1.65, C["crf"], "CRF decode", ["emission = gate × ℓ + affinity, mixture of haplotypes",
                                                   "switch cost, het penalty → founder-pair path"], fs=13, lfs=11)

for i, (lab, c) in enumerate((("data", C["sim"]), ("encoder (learned)", C["enc"]), ("simulator truth", C["truth"]),
                              ("likelihood", C["lik"]), ("CRF", C["crf"]))):
    ax.add_patch(Rectangle((0.35 + i * 3.1, 0.0), 0.28, 0.2, fc=c, ec=C["edge"], lw=0.8))
    ax.text(0.72 + i * 3.1, 0.1, lab, va="center", fontsize=11)

for ext in ("png", "svg"):
    fig.savefig(OUT / f"supervised_heads_design.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
print(OUT / "supervised_heads_design.png")
