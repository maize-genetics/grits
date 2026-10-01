#!/usr/bin/env python
"""Shareable 3-step diagram of v6 fit B (same layout as supervised_heads_diagram.py / v6_model):
1 train the encoder heads, 2 fit B's CRF numbers (learned values; fit A in brackets),
3 decode real reads with those numbers. Values from checkpoints/v6-stage2B/last.ckpt.
Writes results/v6_fitB.{png,svg}."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "results"
C = {"sim": "#E8EAED", "truth": "#FFF4C2", "enc": "#CFE2FF", "lik": "#D4EDDA", "crf": "#FFE5CC", "edge": "#3C4043",
     "val": "#B06000", "off": "#80868B"}

fig, ax = plt.subplots(figsize=(16, 10.2))
ax.set_xlim(0, 16); ax.set_ylim(-0.5, 9.7); ax.axis("off")


def box(x, y, w, h, color, title, lines=(), fs=14, lfs=11.5, lcol="#3C4043"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.15",
                                fc=color, ec=C["edge"], lw=1.5))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=fs, weight="bold", color="#202124")
    for i, s in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.78 - 0.36 * i, s, ha="center", va="top", fontsize=lfs, color=lcol)


def arrow(x0, y0, x1, y1, ls="-", color=C["edge"], lw=1.7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=lw,
                                 color=color, linestyle=ls))


ax.text(8, 9.45, "v6 fit B: encoder trained on the truth, CRF numbers fitted, real reads decoded", ha="center",
        fontsize=19, weight="bold")
ax.text(8, 9.03, "everything is learned or set on simulated data; real reads are only decoded and scored",
        ha="center", fontsize=13, color="#5F6368")

# ---- 1 train the encoder heads
ax.text(0.3, 8.55, "1  Train the encoder — simulated data (v6 set, 30% held-out individuals)", fontsize=14.5,
        weight="bold", color="#1A73E8")
box(0.3, 5.6, 2.8, 2.7, C["sim"], "Simulated reads", ["ternary rows", "4 depths (0.1–2x)"], fs=13)
box(4.0, 5.6, 3.0, 2.7, C["enc"], "Encoder", ["transformer over", "the window", "+ sample match rate"], fs=13)
arrow(3.15, 6.95, 4.0, 6.95)
heads = [("gate", "row matches its true founder?"), ("crossovers", "switches in this window"),
         ("deletion dosage", "0 / 1 / 2 haplotypes deleted"), ("out-of-panel", "true founder hidden?"),
         ("het", "parents distinct?  (not used)")]
for i, (h, q) in enumerate(heads):
    y = 7.85 - 0.5 * i
    off = h == "het"
    for x, w, col, txt, bold in ((7.8, 2.6, C["enc"], h, True), (11.2, 4.5, C["truth"], q, False)):
        ax.add_patch(FancyBboxPatch((x, y), w, 0.38, boxstyle="round,pad=0.03,rounding_size=0.1", fc=col,
                                    ec=C["edge"], lw=1.2, linestyle="--" if off else "-"))
        ax.text(x + w / 2, y + 0.19, txt, ha="center", va="center", fontsize=12, weight="bold" if bold else "normal",
                color=C["off"] if off else "#202124")
    arrow(7.05, 6.95, 7.8, y + 0.19, lw=1.4)
    ax.add_patch(FancyArrowPatch((11.18, y + 0.19), (10.45, y + 0.19), arrowstyle="-|>", mutation_scale=13, lw=1.3,
                                 color="#B06000", linestyle="--"))
ax.text(13.45, 8.33, "simulator truth", ha="center", fontsize=12, weight="bold", color="#B06000")
ax.plot([0.3, 15.7], [5.35, 5.35], color="#DADCE0", lw=1.2)

# ---- 2 fit B
ax.text(0.3, 5.0, "2  Fit B: tune the CRF numbers — simulated data, encoder frozen", fontsize=14.5, weight="bold",
        color="#188038")
box(0.3, 2.95, 2.3, 1.8, C["sim"], "Simulated reads", [], fs=12.5)
box(3.05, 2.95, 2.6, 1.8, C["enc"], "Frozen encoder", ["head outputs"], fs=12.5, lfs=11)
box(6.1, 2.95, 3.6, 1.8, C["crf"], "CRF", ["numbers start at fit A's,", "then tuned (step 3)"], fs=12.5, lfs=10.5)
box(10.15, 2.95, 2.3, 1.8, C["crf"], "Decoded path", [], fs=12.5)
box(12.9, 2.95, 2.8, 1.8, C["truth"], "True path", ["loss → update", "the CRF numbers"], fs=12.5, lfs=11)
arrow(2.65, 3.85, 3.05, 3.85); arrow(5.7, 3.85, 6.1, 3.85); arrow(9.75, 3.85, 10.15, 3.85)
ax.add_patch(FancyArrowPatch((12.88, 3.85), (12.5, 3.85), arrowstyle="-|>", mutation_scale=14, lw=1.4,
                             color="#B06000", linestyle="--"))
ax.add_patch(FancyArrowPatch((14.3, 2.95), (7.9, 2.93), connectionstyle="arc3,rad=-0.12", arrowstyle="-|>",
                             mutation_scale=14, lw=1.3, color="#B06000", linestyle="--"))
ax.plot([0.3, 15.7], [2.45, 2.45], color="#DADCE0", lw=1.2)

# ---- 3 decode real reads with fit B's numbers
ax.text(0.3, 2.1, "3  Decode real reads with fit B's numbers (fit A's in brackets) — nothing tuned here", fontsize=14.5,
        weight="bold", color="#E8710A")
rows = [("read evidence", "gate × read scores", "match +0.39 (+1.29) · diverged −3.67 (−4.36) · −1 −4.24 (−6.15)"),
        ("deletion dosage", "gate × weight × log P(dosage)", "weight 0.15 (0)"),
        ("switch cost", "crossovers × scale × exp(w × out-of-panel)", "scale 0.34 (1) · w −1.95 (0)"),
        ("founder prior", "weight × log(match-rate share)", "weight 4.03 (1) · out-of-panel fade −0.08 (1) = off"),
        ("het prior", "", "off in both fits")]
for i, (what, how, val) in enumerate(rows):
    y = 1.65 - 0.37 * i
    off = what == "het prior"
    ax.text(0.45, y, what, fontsize=11.5, weight="bold", va="center", color=C["off"] if off else "#202124")
    ax.text(3.0, y, how, fontsize=11, va="center", color="#5F6368")
    ax.text(15.6, y, val, fontsize=11, va="center", ha="right", weight="bold", color=C["off"] if off else C["val"])
ax.text(0.45, -0.25, "→ Viterbi: best founder pair per row → genotype calls", fontsize=11.5, color="#3C4043")

for i, (lab, c) in enumerate((("data", C["sim"]), ("encoder (learned)", C["enc"]), ("simulator truth", C["truth"]),
                              ("CRF", C["crf"]))):
    ax.add_patch(Rectangle((6.9 + i * 2.25 + (0.35 if i > 1 else 0), -0.35), 0.25, 0.18, fc=c, ec=C["edge"], lw=0.8))
    ax.text(7.22 + i * 2.25 + (0.35 if i > 1 else 0), -0.26, lab, va="center", fontsize=10.5)

for ext in ("png", "svg"):
    fig.savefig(OUT / f"v6_fitB.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
print(OUT / "v6_fitB.png")
