#!/usr/bin/env python
"""Shareable diagram of v6cfullfe + homozygous restriction (same style as v6_fitB_diagram.py):
1 train the encoder on closest-founder labels, 2 fit B with the read table frozen and no null state,
3 decode real reads, with the router's inbred/hybrid call limiting which founder pairs are allowed.
Tags mark what changed since v6. Writes results/v6cfullfe_homo.{png,svg}."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

OUT = Path(__file__).resolve().parents[1] / "results"
C = {"sim": "#E8EAED", "truth": "#FFF4C2", "enc": "#CFE2FF", "lik": "#D4EDDA", "crf": "#FFE5CC", "edge": "#3C4043",
     "new": "#7B1FA2", "route": "#F3E5F5", "off": "#80868B"}

fig, ax = plt.subplots(figsize=(16, 10.2))
ax.set_xlim(0, 16); ax.set_ylim(-0.6, 9.7); ax.axis("off")


def box(x, y, w, h, color, title, lines=(), fs=14, lfs=11.5, tag=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.15",
                                fc=color, ec=C["new"] if tag else C["edge"], lw=2.4 if tag else 1.5))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=fs, weight="bold", color="#202124")
    for i, s in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.78 - 0.36 * i, s, ha="center", va="top", fontsize=lfs, color="#3C4043")
    if tag:
        ax.text(x + w - 0.08, y + h + 0.02, tag, ha="right", va="bottom", fontsize=10.5, weight="bold",
                color="white", bbox=dict(boxstyle="round,pad=0.25", fc=C["new"], ec="none"))


def arrow(x0, y0, x1, y1, ls="-", color=C["edge"], lw=1.7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=lw,
                                 color=color, linestyle=ls))


def pair_grid(x, y, s, homo_only, n=8):
    """n x n founder-pair grid (lower triangle = unordered pairs); allowed cells filled."""
    c = s / n
    for i in range(n):
        for j in range(i + 1):
            ok = (i == j) or not homo_only
            ax.add_patch(Rectangle((x + j * c, y + (n - 1 - i) * c), c, c, fc=C["new"] if ok else "white",
                                   ec="#BDC1C6", lw=0.6, alpha=0.85 if ok else 1))


ax.text(8, 9.45, "v6cfullfe + homozygous restriction", ha="center", fontsize=19, weight="bold")
ax.text(8, 9.03, "trained and fitted on simulated data only; real reads are only decoded and scored",
        ha="center", fontsize=13, color="#5F6368")

# ---- 1 train the encoder
ax.text(0.3, 8.55, "1  Train the encoder — simulated data", fontsize=14.5, weight="bold", color="#1A73E8")
box(0.3, 6.55, 3.0, 1.7, C["sim"], "Simulated reads", ["30% with hidden founders"], fs=13)
box(4.2, 6.55, 3.6, 1.7, C["truth"], "Closest-founder labels", ["no more \"null\" label"], fs=13, tag="NEW")
box(8.7, 6.55, 3.2, 1.7, C["enc"], "Encoder", ["retrained"], fs=13, tag="CHANGED")
box(12.8, 6.55, 2.9, 1.7, C["enc"], "Heads", ["gate, crossovers,", "dosage, out-of-panel"], fs=13, lfs=11)
arrow(3.35, 7.4, 4.2, 7.4); arrow(7.85, 7.4, 8.7, 7.4); arrow(11.95, 7.4, 12.8, 7.4)
ax.plot([0.3, 15.7], [6.2, 6.2], color="#DADCE0", lw=1.2)

# ---- 2 fit B
ax.text(0.3, 5.85, "2  Fit B — simulated data, encoder frozen", fontsize=14.5, weight="bold", color="#188038")
box(0.3, 3.95, 4.6, 1.6, C["lik"], "Read likelihood table", ["frozen at v6's values"], fs=13, tag="CHANGED")
box(5.6, 3.95, 4.6, 1.6, C["crf"], "CRF states", ["25 founders, null removed"], fs=13, tag="NEW")
box(10.9, 3.95, 4.8, 1.6, C["crf"], "Fitted", ["switch rate, founder prior,", "out-of-panel weights"], fs=13, lfs=11)
ax.plot([0.3, 15.7], [3.6, 3.6], color="#DADCE0", lw=1.2)

# ---- 3 decode real reads
ax.text(0.3, 3.25, "3  Decode real reads", fontsize=14.5, weight="bold", color="#E8710A")
box(0.3, 0.2, 2.4, 2.7, C["sim"], "Real reads", ["refmap → rows"], fs=13)
box(3.3, 0.2, 2.6, 2.7, C["route"], "Router", ["whole sample:", "hybrid?"], fs=13, tag="NOW USED")
arrow(2.75, 1.55, 3.3, 1.55)
# allowed-pair cartoons
ax.text(7.65, 2.75, "allowed founder pairs", ha="center", fontsize=12, weight="bold", color=C["new"])
pair_grid(6.55, 1.25, 1.15, homo_only=False)
pair_grid(7.85, 1.25, 1.15, homo_only=True)
ax.text(7.12, 1.0, "hybrid", ha="center", va="top", fontsize=11.5, weight="bold")
ax.text(7.12, 0.66, "all pairs", ha="center", va="top", fontsize=10.5, color="#3C4043")
ax.text(8.42, 1.0, "inbred / RIL", ha="center", va="top", fontsize=11.5, weight="bold")
ax.text(8.42, 0.66, "A/A only", ha="center", va="top", fontsize=10.5, color="#3C4043")
arrow(5.95, 1.85, 6.5, 1.85)
box(9.6, 0.2, 3.4, 2.7, C["crf"], "CRF decode", ["encoder heads", "+ read likelihood", "→ founder pair per row"],
    fs=13, lfs=11)
arrow(9.05, 1.55, 9.6, 1.55)
box(13.5, 0.2, 2.2, 2.7, C["lik"], "Genotype", ["calls, ~99%", "of genome"], fs=13, lfs=11)
arrow(13.05, 1.55, 13.5, 1.55)
ax.annotate("", xy=(11.3, 2.95), xytext=(1.5, 2.95), arrowprops=dict(arrowstyle="-", color="#DADCE0", lw=0))

for i, (lab, c) in enumerate((("data", C["sim"]), ("encoder", C["enc"]), ("labels", C["truth"]),
                              ("likelihood / output", C["lik"]), ("CRF", C["crf"]), ("router", C["route"]))):
    ax.add_patch(Rectangle((0.35 + i * 2.35, -0.45), 0.28, 0.2, fc=c, ec=C["edge"], lw=0.8))
    ax.text(0.72 + i * 2.35, -0.35, lab, va="center", fontsize=11)
ax.add_patch(Rectangle((14.4, -0.45), 0.28, 0.2, fc="white", ec=C["new"], lw=2.2))
ax.text(14.77, -0.35, "changed vs v6", va="center", fontsize=11, color=C["new"], weight="bold")

for ext in ("png", "svg"):
    fig.savefig(OUT / f"v6cfullfe_homo.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
print(OUT / "v6cfullfe_homo.png")
