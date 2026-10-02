#!/usr/bin/env python
"""Slide diagram of the likelihood-emission encoder+CRF (--emission likelihood[_dist]
--pair-emission mixture). Writes results/architecture_likelihood_crf.{png,svg}."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "results"
C = {"input": "#E8EAED", "enc": "#CFE2FF", "lik": "#D4EDDA", "crf": "#FFE5CC", "edge": "#3C4043",
     "match": "#2E7D32", "div": "#BDBDBD", "del": "#C62828"}

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")


def box(x, y, w, h, color, title, lines=(), fs=15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.18",
                                fc=color, ec=C["edge"], lw=1.6))
    ax.text(x + w / 2, y + h - 0.38, title, ha="center", va="top", fontsize=fs, weight="bold", color="#202124")
    for i, s in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.95 - 0.42 * i, s, ha="center", va="top", fontsize=12, color="#3C4043")


def arrow(x0, y0, x1, y1, label=None, dy=0.18, color=C["edge"]):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, lw=1.8, color=color))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", va="bottom", fontsize=12, style="italic")


ax.text(8, 8.65, "Encoder + CRF with a likelihood emission", ha="center", fontsize=22, weight="bold")
ax.text(8, 8.18, "the encoder only gates rows and sets the switch cost; the CRF decides the founders",
        ha="center", fontsize=13.5, color="#5F6368")

# input: rows x founders mini-grid
box(0.3, 1.2, 3.0, 6.1, C["input"], "Reads → rows", ["512 rows × 25 founders"])
import numpy as np
rng = np.random.default_rng(3)
states = rng.choice([1, 0, -1], size=(9, 8), p=[.4, .4, .2])
col = {1: C["match"], 0: C["div"], -1: C["del"]}
for r in range(9):
    for f in range(8):
        ax.add_patch(Rectangle((0.72 + f * 0.27, 5.75 - r * 0.27), 0.25, 0.25, fc=col[states[r, f]], ec="white", lw=0.5))
for i, (k, lab) in enumerate(((1, "match"), (0, "present, no match"), (-1, "deleted"))):
    ax.add_patch(Rectangle((0.7, 2.95 - 0.38 * i), 0.22, 0.22, fc=col[k], ec="none"))
    ax.text(1.02, 3.06 - 0.38 * i, lab, va="center", fontsize=11.5)
ax.text(1.8, 1.55, "+ anchor distance, read count", ha="center", fontsize=10.5, color="#5F6368")

# encoder and its two outputs
box(4.2, 5.4, 3.4, 1.9, C["enc"], "Encoder", ["transformer over the window", "no founder scores"])
arrow(3.35, 6.35, 4.2, 6.35)
box(8.6, 6.05, 2.6, 1.05, C["enc"], "switch cost cₜ", [], fs=14)
arrow(7.65, 6.6, 8.6, 6.6)
box(5.05, 4.0, 1.7, 0.95, C["enc"], "gate gₜ", [], fs=14)
arrow(5.9, 5.4, 5.9, 5.0)

# likelihood table
box(4.2, 1.2, 3.4, 2.1, C["lik"], "Read likelihood ℓ", ["per ternary state", "(× distance band)", "shared by all founders"])
arrow(3.35, 2.25, 4.2, 2.25)

# per-founder emission and pair mixture
box(8.6, 1.2, 2.6, 1.85, C["lik"], "Founder emission", ["gₜ · ℓ[stateₜ,f]"])
arrow(7.65, 2.1, 8.6, 2.1)
arrow(6.8, 4.3, 8.6, 2.75)
box(8.6, 3.55, 2.6, 1.85, C["crf"], "Pair emission", ["read from either", "haplotype (mixture)"])
arrow(9.9, 3.1, 9.9, 3.55)

# CRF and output
box(12.1, 3.55, 3.6, 3.55, C["crf"], "CRF decode", ["−cₜ per founder change", "+ stay bonus", "homozygous penalty", "(inbred 0 / hybrid 1)"])
arrow(11.25, 4.45, 12.1, 4.45)
arrow(11.25, 6.6, 12.1, 6.6)
box(12.1, 1.2, 3.6, 1.85, C["input"], "Founder-pair path", ["per row → genotypes"])
arrow(13.9, 3.55, 13.9, 3.1)

legend = [("input / output", C["input"]), ("encoder (learned)", C["enc"]), ("likelihood (few learned values)", C["lik"]),
          ("CRF", C["crf"])]
for i, (lab, c) in enumerate(legend):
    ax.add_patch(Rectangle((0.35 + i * 3.9, 0.12), 0.3, 0.22, fc=c, ec=C["edge"], lw=0.8))
    ax.text(0.75 + i * 3.9, 0.23, lab, va="center", fontsize=11.5)

for ext in ("png", "svg"):
    fig.savefig(OUT / f"architecture_likelihood_crf.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
print(OUT / "architecture_likelihood_crf.png")
