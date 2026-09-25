"""Workflow diagram for experiments/het-replacement/PLAN.md: calibration data,
simulator, encoder+CRF pair emission, training and evaluation stages, colored
reused / new / modified, capped by the §6 acceptance gate. Regenerate after
any change to the PLAN's stage list:
    python experiments/het-replacement/scripts/het_replacement_workflow_diagram.py
"""
import subprocess
from pathlib import Path

W, H = 1920, 1080
INK = "#1c2430"
INK_DIM = "#535f6e"
LINE = "#c7d0d8"
LINE_STRONG = "#9fb0bd"

REUSED = "#5b6b7c"; REUSED_BG = "#eef1f3"; REUSED_BORDER = "#c7d0d8"
NEW = "#1f7a4c"; NEW_BG = "#e1f3e8"; NEW_BORDER = "#1f7a4c"
MOD = "#6b4c9a"; MOD_BG = "#ece5f5"; MOD_BORDER = "#6b4c9a"
GATE = "#8a5a12"; GATE_BG = "#f7ecd9"; GATE_BORDER = "#c98a1f"

SANS = "Liberation Sans, Arial, sans-serif"
MONO = "Liberation Mono, Consolas, monospace"

parts = []

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def text(x, y, s, size=18, weight="normal", color=INK, family=SANS, anchor="start"):
    parts.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
                  f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{esc(s)}</text>')

def tspans(x, y, lines, size=15, lh=22, color=INK_DIM, family=SANS, weight="normal"):
    spans = "".join(f'<tspan x="{x}" dy="{0 if i==0 else lh}">{esc(l)}</tspan>' for i, l in enumerate(lines))
    parts.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
                  f'font-weight="{weight}" fill="{color}">{spans}</text>')

def rect(x, y, w, h, fill="#fff", stroke=LINE, sw=2.5, rx=14):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

def hline(x1, x2, y, color=LINE, sw=1.5):
    parts.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="{sw}"/>')

def right_arrow(x1, x2, y, color=LINE_STRONG, sw=4):
    parts.append(f'<line x1="{x1}" y1="{y}" x2="{x2-14}" y2="{y}" stroke="{color}" stroke-width="{sw}"/>')
    parts.append(f'<polygon points="{x2-16},{y-9} {x2},{y} {x2-16},{y+9}" fill="{color}"/>')

def down_arrow(x, y1, y2, color=LINE_STRONG, sw=4):
    parts.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2-14}" stroke="{color}" stroke-width="{sw}"/>')
    parts.append(f'<polygon points="{x-9},{y2-16} {x},{y2} {x+9},{y2-16}" fill="{color}"/>')

def numbered_circle(cx, cy, n, color):
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="17" fill="{color}"/>')
    text(cx, cy + 6, str(n), size=17, weight="bold", color="#fff", anchor="middle")

def tag(x, y, label, fg, bg):
    w = 16 + 8.2 * len(label)
    rect(x, y, w, 28, fill=bg, stroke="none", rx=14)
    text(x + w/2, y + 19, label, size=13, weight="bold", color=fg, family=MONO, anchor="middle")
    return w


rect(0, 0, W, H, fill="#ffffff", stroke="none", rx=0)
text(60, 62, "Heterozygous-Replacement Plan: Workflow", size=38, weight="bold", color=INK)
text(60, 94, "experiments/het-replacement/PLAN.md -- encoder + CRF, trained on simulated data only",
     size=18, color=INK_DIM, family=MONO)
lgx, lgy = 60, 128
for label, fg, bg in [("REUSED, unchanged", REUSED, REUSED_BG), ("+ NEW", NEW, NEW_BG), ("MODIFIED", MOD, MOD_BG)]:
    w = tag(lgx, lgy, label, fg, bg)
    lgx += w + 20

box_w, box_h = 420, 250
gap = 40
xs = [60 + i * (box_w + gap) for i in range(4)]
row1_y, row2_y = 200, 560
KIND = {"new": (NEW, NEW_BG, NEW_BORDER, "+ NEW"), "mod": (MOD, MOD_BG, MOD_BORDER, "MODIFIED"),
        "reused": (REUSED, REUSED_BG, REUSED_BORDER, "REUSED")}
row1 = [
    (1, "Calibration corpus", "new", ["5 new OUT + 5 new IDX lines", "0.1x + 1x reads, AnchorWave truth"], "§2.1"),
    (2, "Measure real stats", "new", ["replacement blocks, read-type mix", "PLACED shift, feature dists"], "§2.2"),
    (3, "Simulator", "mod", ["simulate_alleles.py", "het replacements, PLACED-like reads"], "§2.3"),
    (4, "Synthetic dosage check", "new", ["0 / 1 / 2 deleted copies", "held-out simulated individuals"], "§2.5"),
]
row2 = [
    (5, "Pair emission", "mod", ["train_diploid_indel.py:320", "presence-aware mixture"], "§2.4"),
    (6, "Train", "reused", ["simulated data only", "flag guard + shuffled split"], "§2.6"),
    (7, "Evaluate", "reused", ["fast_eval_ckpt.py", "existing simval corpus"], "§2.7"),
    (8, "Report", "new", ["dosage confusion + events", "vs baselines (HMM = comparison)"], "§2.7"),
]

def box(x, y, n, title, kind, lines, sec):
    fg, bg, border, lab = KIND[kind]
    rect(x, y, box_w, box_h, fill=bg, stroke=border, sw=3)
    numbered_circle(x + 36, y + 42, n, fg)
    text(x + 66, y + 50, title, size=26, weight="bold", color=INK)
    tspans(x + 30, y + 112, lines, size=19, lh=30, color=INK_DIM)
    tag(x + 30, y + box_h - 52, lab, fg, "#ffffff")
    text(x + box_w - 30, y + box_h - 30, sec, size=17, color=INK_DIM, family=MONO, anchor="end")

for (n, t, k, l, s), x in zip(row1, xs):
    box(x, row1_y, n, t, k, l, s)
for (n, t, k, l, s), x in zip(row2, xs):
    box(x, row2_y, n, t, k, l, s)
for i in range(3):
    right_arrow(xs[i] + box_w + 4, xs[i + 1] - 4, row1_y + box_h // 2)
    right_arrow(xs[i] + box_w + 4, xs[i + 1] - 4, row2_y + box_h // 2)
# elbow from stage 4 down and back to stage 5
ex, ey = xs[3] + box_w // 2, row1_y + box_h
mid = (row1_y + box_h + row2_y) // 2
parts.append(f'<polyline points="{ex},{ey + 4} {ex},{mid} {xs[0] + box_w // 2},{mid} {xs[0] + box_w // 2},{row2_y - 16}" '
             f'fill="none" stroke="{LINE_STRONG}" stroke-width="4"/>')
down_arrow(xs[0] + box_w // 2, row2_y - 20, row2_y - 2)

gy = row2_y + box_h + 40
rect(60, gy, W - 120, 140, fill=GATE_BG, stroke=GATE_BORDER, sw=3)
text(90, gy + 46, "§6 Acceptance gate", size=26, weight="bold", color=GATE)
tspans(90, gy + 84, ["OUT-HYB het-deletion calls up from 41% (and >= the comparison HMM's 60%);  IDX stays < 1%;",
                     "OUT improves on all-sites, SNP+RC and deletion events;  nothing tuned on the eval set"],
       size=19, lh=28, color=INK)

svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">' + "".join(parts) + "</svg>"
out = Path(__file__).resolve().parent.parent / "results" / "het_replacement_workflow"
out.with_suffix(".svg").write_text(svg)
subprocess.run(["rsvg-convert", "-o", str(out.with_suffix(".png")), str(out.with_suffix(".svg"))], check=True)
print("wrote", out.with_suffix(".png"))
