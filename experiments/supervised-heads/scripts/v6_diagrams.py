#!/usr/bin/env python
"""Shareable diagrams for v6, in the style of supervised_heads_diagram.py:
  results/v6_model.{png,svg}          encoder heads -> CRF, what changed since sh3/sh5 marked
  results/v6_training_data.{png,svg}  how the simulated training set is built"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "results"
C = {"sim": "#E8EAED", "truth": "#FFF4C2", "enc": "#CFE2FF", "lik": "#D4EDDA", "crf": "#FFE5CC",
     "edge": "#3C4043", "gone": "#F1F3F4"}
TAG = {"new": "#188038", "changed": "#C5221F", "off": "#80868B"}


def setup(w=16, h=9):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, color, title, lines=(), fs=14, lfs=11.5, tag=None, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.15",
                                fc=color, ec=TAG.get(tag, C["edge"]) if tag else C["edge"],
                                lw=2.4 if tag in ("new", "changed") else 1.5, linestyle=ls))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=fs, weight="bold",
            color="#80868B" if tag == "off" else "#202124")
    for i, s in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.78 - 0.36 * i, s, ha="center", va="top", fontsize=lfs, color="#3C4043")
    if tag:
        badge(ax, x + w - 0.05, y + h + 0.02, tag)


def badge(ax, x, y, tag):
    word = {"new": "NEW", "changed": "CHANGED", "off": "NOT USED"}[tag]
    ax.text(x, y, word, ha="right", va="center", fontsize=8.5, weight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.22", fc=TAG[tag], ec="none"))


def arrow(ax, x0, y0, x1, y1, ls="-", color=C["edge"], lw=1.7, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=lw,
                                 color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def legend(ax, items, y=0.0):
    for i, (lab, c) in enumerate(items):
        ax.add_patch(Rectangle((0.35 + i * 3.1, y), 0.28, 0.2, fc=c, ec=C["edge"], lw=0.8))
        ax.text(0.72 + i * 3.1, y + 0.1, lab, va="center", fontsize=11)


def pill(ax, x, y, w, txt, color, bold=False, tag=None, fs=12, h=0.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1", fc=color,
                                ec=TAG.get(tag, C["edge"]) if tag in ("new", "changed") else C["edge"],
                                lw=2.2 if tag in ("new", "changed") else 1.2,
                                linestyle="--" if tag == "off" else "-"))
    ax.text(x + 0.15, y + h / 2, txt, ha="left", va="center", fontsize=fs, weight="bold" if bold else "normal",
            color="#80868B" if tag == "off" else "#202124")
    if tag:
        badge(ax, x + w - 0.1, y + h / 2, tag)


# ------------------------------------------------------------------------------------------ model
fig, ax = setup(16, 9.8)
ax.set_ylim(-0.9, 9.4)
ax.text(8, 9.12, "v6: the supervised-heads encoder and CRF", ha="center", fontsize=21, weight="bold")
ax.text(8, 8.7, "changes since sh3/sh5 marked; everything is learned or set on simulated data, real reads are only decoded",
        ha="center", fontsize=12.5, color="#5F6368")

ax.text(0.3, 8.2, "1  Train the encoder heads — simulated data", fontsize=14.5, weight="bold", color="#1A73E8")
box(ax, 0.3, 5.05, 2.8, 2.85, C["sim"], "Simulated reads", ["ternary rows only", "(distance channel off)"], fs=13, lfs=11, tag="changed")
box(ax, 3.9, 5.05, 2.9, 2.85, C["enc"], "Encoder", ["transformer over", "the window", "+ sample match rate"], fs=13, lfs=11)
arrow(ax, 3.15, 6.5, 3.9, 6.5)
heads = [("gate", "row matches its true founder?", "changed"),
         ("crossovers", "switches in this window", None),
         ("het", "parents distinct?", "off"),
         ("deletion dosage", "0 / 1 / 2 haplotypes deleted", "new"),
         ("out-of-panel", "true founder hidden?", "new")]
for i, (h, q, tag) in enumerate(heads):
    y = 7.45 - 0.58 * i
    pill(ax, 7.6, y, 3.1, h, C["enc"], bold=True, tag=tag)
    pill(ax, 11.25, y, 4.45, q, C["truth"])
    arrow(ax, 6.85, 6.5, 7.6, y + 0.2, lw=1.4)
    ax.add_patch(FancyArrowPatch((11.23, y + 0.2), (10.72, y + 0.2), arrowstyle="-|>", mutation_scale=13, lw=1.3,
                                 color="#B06000", linestyle="--"))
ax.text(13.47, 7.98, "simulator truth", ha="center", fontsize=12, weight="bold", color="#B06000")
ax.text(15.7, 4.88, "affinity head removed (CRF prior: the sample's read match rate)", ha="right",
        fontsize=10.5, color=TAG["changed"], style="italic")
ax.plot([0.3, 15.7], [4.75, 4.75], color="#DADCE0", lw=1.2)

ax.text(0.3, 4.4, "2  Fit the CRF numbers — simulated data, encoder frozen", fontsize=14.5, weight="bold", color="#188038")
box(ax, 0.3, 2.55, 2.3, 1.6, C["sim"], "Simulated reads", [], fs=12.5)
box(ax, 3.05, 2.55, 2.6, 1.6, C["enc"], "Frozen encoder", ["head outputs"], fs=12.5, lfs=11)
box(ax, 6.1, 2.55, 3.6, 1.6, C["crf"], "CRF", ["read likelihood, switch scale,", "affinity + new-head weights"], fs=12.5,
    lfs=10.5)
box(ax, 10.15, 2.55, 2.3, 1.6, C["crf"], "Decoded path", [], fs=12.5)
box(ax, 12.9, 2.55, 2.8, 1.6, C["truth"], "True path", ["loss → update", "the CRF numbers"], fs=12.5, lfs=11)
arrow(ax, 2.65, 3.35, 3.05, 3.35); arrow(ax, 5.7, 3.35, 6.1, 3.35); arrow(ax, 9.75, 3.35, 10.15, 3.35)
ax.add_patch(FancyArrowPatch((12.88, 3.35), (12.5, 3.35), arrowstyle="-|>", mutation_scale=14, lw=1.4,
                             color="#B06000", linestyle="--"))
ax.plot([0.3, 15.7], [2.25, 2.25], color="#DADCE0", lw=1.2)

ax.text(0.3, 1.9, "3  Decode real reads — how the CRF uses each output", fontsize=14.5, weight="bold", color="#E8710A")
uses = [("gate", "trust each row's read likelihood", None),
        ("read likelihood", "from ALL simulated rows (mismatch no longer ~impossible)", "changed"),
        ("crossovers × out-of-panel", "switch rate, raised in out-of-panel windows", "new"),
        ("match rate × (1 − out-of-panel)", "founder prior, faded in out-of-panel windows", "new"),
        ("deletion dosage", "scores how many of the pair read −1", "new")]
for i, (what, how, tag) in enumerate(uses):
    y = 1.45 - 0.37 * i
    ax.text(0.45, y, what, fontsize=11.5, weight="bold", va="center")
    ax.text(5.1, y, how, fontsize=11.2, va="center", color="#3C4043")
    if tag:
        ax.text(14.55, y, {"new": "NEW", "changed": "CHANGED"}[tag], fontsize=8.5, weight="bold", color="white", va="center",
                bbox=dict(boxstyle="round,pad=0.22", fc=TAG[tag], ec="none"))
ax.text(0.45, -0.45, "het head: trained, but its prior is off in decoding (miscalibrated on real reads)", fontsize=10.5,
        color=TAG["off"], style="italic", va="center")
legend(ax, (("data", C["sim"]), ("encoder (learned)", C["enc"]), ("simulator truth", C["truth"]), ("CRF", C["crf"])), y=-0.85)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"v6_model.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ------------------------------------------------------------------------------------------ data
fig, ax = setup(16, 8.2)
ax.text(8, 7.95, "v6 training data: how the simulated set is built", ha="center", fontsize=21, weight="bold")
ax.text(8, 7.53, "each realism fix calibrated on the calibration corpus (never the evaluation lines)",
        ha="center", fontsize=12.5, color="#5F6368")

box(ax, 0.3, 4.55, 2.6, 2.5, C["sim"], "Simulator", ["27 founders", "4 depths (0.1–2x)"], fs=13.5, lfs=11.5, tag="changed")
arrow(ax, 2.95, 5.8, 3.55, 5.8)
fixes = [("each read covers its own SNPs", "no identical neighbour rows", "new"),
         ("per-read true-founder misses", "1.23% of reads, like real", "new"),
         ("shared anchor distances", "real within-row structure", "new"),
         ("more founder sharing", "hybrids overlap like real", "new")]
for i, (a, b, tag) in enumerate(fixes):
    y = 6.5 - 0.6 * i
    pill(ax, 3.55, y, 4.6, a, C["lik"], bold=True, tag=tag, fs=11.5, h=0.45)
    ax.text(8.3, y + 0.22, b, fontsize=11, va="center", color="#3C4043")
ax.text(5.85, 7.1, "realism fixes", ha="center", fontsize=12, weight="bold", color="#188038")
arrow(ax, 10.75, 5.8, 11.35, 5.8)
box(ax, 11.35, 4.55, 4.35, 2.5, C["enc"], "Held-out augment", ["hide 2 of 27 founders", "30%: its OWN founders hidden",
                                                            "→ out-of-panel individuals"], fs=13.5, lfs=11.2, tag="new")

ax.plot([0.3, 15.7], [4.1, 4.1], color="#DADCE0", lw=1.2)
ax.text(0.3, 3.75, "What the model sees vs. what it is trained on", fontsize=14.5, weight="bold", color="#1A73E8")
box(ax, 0.3, 1.45, 4.2, 2.0, C["sim"], "Input (25 founders)", ["ternary rows", "sample match rate"], fs=13, lfs=11.5)
targets = [("true pair", "any lineage-mate of a hidden founder counts", "changed"),
           ("gate", "row matches its true founder", "changed"),
           ("deletion dosage", "true founders deleted at the site", "new"),
           ("out-of-panel", "row's true founder hidden", "new")]
for i, (a, b, tag) in enumerate(targets):
    y = 3.0 - 0.55 * i
    pill(ax, 5.3, y, 3.6, a, C["truth"], bold=True, tag=tag, fs=11.5, h=0.42)
    ax.text(9.1, y + 0.21, b, fontsize=11, va="center", color="#3C4043")
ax.text(7.1, 3.52, "targets", ha="center", fontsize=12, weight="bold", color="#B06000")
ax.text(15.7, 0.55, "tie-aware loss: picking any lineage-mate is not label noise (the first held-out attempt's problem)",
        ha="right", fontsize=10.5, color="#5F6368", style="italic")
legend(ax, (("data", C["sim"]), ("simulator change", C["lik"]), ("held-out step", C["enc"]), ("training target", C["truth"])),
       y=0.0)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"v6_training_data.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
print(OUT / "v6_model.png", OUT / "v6_training_data.png")
