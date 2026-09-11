"""Workflow diagram: simulate_alleles.py's pipeline, reused vs. new vs.
modified stages, per experiments/simulator-indels/PLAN.md §2 (Design) and
§4 (Reusable vs. genuinely new). Meant to live alongside the plan doc."""

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

# ---- background ----
rect(0, 0, W, H, fill="#ffffff", stroke="none", rx=0)

# ---- header ----
text(60, 62, "Simulator Pipeline: What's Reused, What's New", size=38, weight="bold", color=INK)
text(60, 94, "simulate_alleles.py -- experiments/simulator-indels/PLAN.md §2, §4, §5 Phase 1",
     size=18, color=INK_DIM, family=MONO)

# legend
lgx, lgy = 60, 128
for label, fg, bg in [("REUSED, unchanged", REUSED, REUSED_BG), ("+ NEW", NEW, NEW_BG), ("MODIFIED", MOD, MOD_BG)]:
    w = tag(lgx, lgy, label, fg, bg)
    lgx += w + 20

# ---- box grid ----
box_w, box_h = 420, 260
gap = 40
xs = [60 + i * (box_w + gap) for i in range(4)]
row1_y = 210
row2_y = 580

stages_row1 = [
    (1, "Founder Paths", REUSED, REUSED_BG, REUSED_BORDER,
     "_build_paths, individual /", "breeding-pop assignment", ["who's where, unchanged"]),
    (2, "Lineage Assignment", REUSED, REUSED_BG, REUSED_BORDER,
     "_gem_lineages", "Ewens / GEM(theta)", ["now also drives indels →"]),
    (3, "Indel Tracts", NEW, NEW_BG, NEW_BORDER,
     "two-component mixture", "(§2.4)", ["via _segment_index /", "_good_mask (reused)"]),
    (4, "Breakpoint Snapping", NEW, NEW_BG, NEW_BORDER,
     "keep crossovers out of", "indel tracts (§2.6)", ["reuses --recomb-span", "rate map (E2)"]),
]
stages_row2 = [
    (5, "SNP Match Features", REUSED, REUSED_BG, REUSED_BORDER,
     "_coalescent_feats", "",
     ["engine unchanged; output", "now feeds a new encoder"]),
    (6, "Ternary + Distance", NEW, NEW_BG, NEW_BORDER,
     "replaces binary matrix 1", "(§2.1, §2.3)",
     ["deletion / diverged / match", "+ distance-to-anchor"]),
    (7, "Read Sampling & Coverage", NEW, NEW_BG, NEW_BORDER,
     "doesn't exist today", "(§2.5)",
     ["1 read/site by construction", "→ produces stacking"]),
    (8, "Write Matrix", MOD, MOD_BG, MOD_BORDER,
     "K+2 → 2K+2 layout", "(§2.3)",
     ["labels stay at fixed", "offsets, unchanged"]),
]

def draw_stage(x, y, n, title, fg, bg, border, l1, l2, extra):
    rect(x, y, box_w, box_h, fill="#ffffff", stroke=border, sw=2.5)
    numbered_circle(x + 26, y + 28, n, fg)
    text(x + 52, y + 34, title, size=21, weight="bold", color=INK)
    hline(x + 24, x + box_w - 24, y + 54, LINE, 1.5)
    fn_lines = [l1] + ([l2] if l2 else [])
    tspans(x + 24, y + 90, fn_lines, size=17, lh=25, color=INK, family=MONO)
    ytag = y + 90 + 25 * len(fn_lines) + 14
    tw = tag(x + 24, ytag, ("REUSED" if fg == REUSED else ("+ NEW" if fg == NEW else "MODIFIED")), fg, bg)
    tspans(x + 24, ytag + 52, extra, size=14, lh=20, color=INK_DIM)

for (x, s) in zip(xs, stages_row1):
    draw_stage(x, row1_y, *s)
for (x, s) in zip(xs, stages_row2):
    draw_stage(x, row2_y, *s)

# arrows within each row
row1_cy = row1_y + box_h/2
row2_cy = row2_y + box_h/2
for i in range(3):
    right_arrow(xs[i] + box_w, xs[i+1], row1_cy)
    right_arrow(xs[i] + box_w, xs[i+1], row2_cy)

# elbow connector row1 box4 -> row2 box5
b4_bottom_x = xs[3] + box_w/2
b5_top_x = xs[0] + box_w/2
stem_y = row1_y + box_h + 34
parts.append(f'<line x1="{b4_bottom_x}" y1="{row1_y+box_h}" x2="{b4_bottom_x}" y2="{stem_y}" '
              f'stroke="{LINE_STRONG}" stroke-width="4"/>')
parts.append(f'<line x1="{b4_bottom_x}" y1="{stem_y}" x2="{b5_top_x}" y2="{stem_y}" '
              f'stroke="{LINE_STRONG}" stroke-width="4"/>')
down_arrow(b5_top_x, stem_y, row2_y)

# ---- acceptance-criteria gate ----
gy = row2_y + box_h + 40
rect(60, gy, 1800, 100, fill=GATE_BG, stroke=GATE_BORDER, sw=2.5, rx=16)
text(84, gy + 38, "Done when (PLAN.md §2.7) --", size=20, weight="bold", color=GATE)
text(84, gy + 74,
     "~40% of reference bp indel-affected   *   ~72% either-founder-covered for het individuals (not ~96%)   *   coverage dispersion >> 1",
     size=18, color=INK)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}">' + "".join(parts) + "</svg>")

with open("/home/zrm22/.claude/jobs/0e12e67f/tmp/simulator_workflow.svg", "w") as f:
    f.write(svg)
print("wrote svg,", len(svg), "bytes; gate bottom =", gy + 100)
