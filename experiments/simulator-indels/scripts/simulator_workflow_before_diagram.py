"""Workflow diagram: simulate_alleles.py's pipeline AS IT EXISTS TODAY (before
any indel-modeling work), grounded directly in simulate()'s actual body
(src/python/crf/simulate_alleles.py:354-497). Same visual system as
simulator_workflow.png (the planned "after" diagram) so the two compare
directly -- same box size/grid/fonts, no reused/new/modified tags here since
this is a plain snapshot, not a diff."""

W, H = 1920, 1080
INK = "#1c2430"
INK_DIM = "#535f6e"
LINE = "#c7d0d8"
LINE_STRONG = "#9fb0bd"
TODAY = "#5b6b7c"; TODAY_BG = "#eef1f3"; TODAY_BORDER = "#c7d0d8"
GAP = "#9a3b3b"; GAP_BG = "#f6e3e1"; GAP_BORDER = "#c65a5a"

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
text(60, 62, "Simulator Pipeline: Today (Before Indel Modeling)", size=38, weight="bold", color=INK)
text(60, 94, "simulate_alleles.py, simulate() -- src/python/crf/simulate_alleles.py:354-497",
     size=18, color=INK_DIM, family=MONO)
tagw = tag(60, 122, "SNAPSHOT, not a diff -- compare against simulator_workflow.png", TODAY, TODAY_BG)

# ---- box grid: same size/position constants as simulator_workflow.png ----
box_w, box_h = 420, 260
gap = 40
xs = [60 + i * (box_w + gap) for i in range(4)]
row1_y = 210

stages = [
    (1, "Founder Paths", "_build_paths /", "_build_paths_subset",
     ["H1 + H2: inbreeding coin-flip", "or E11 breeding het-tracts"]),
    (2, "Genotyping-Error Mask", "_good_mask", "",
     ["2-state Markov 'bad site'", "tract generator"]),
    (3, '"One Read" Per Site', "inline in simulate() --", "no separate function",
     ["pick H1 or H2 per site", "(gamete_balance); that IS", "the whole read model"]),
    (4, "SNP Match Features", "_coalescent_feats  or", "legacy independent q-rate",
     ["branches on --sharing-model;", "coalescent path uses", "_gem_lineages (Ewens/GEM)"]),
]

def draw_stage(x, y, n, title, l1, l2, extra):
    rect(x, y, box_w, box_h, fill="#ffffff", stroke=TODAY_BORDER, sw=2.5)
    numbered_circle(x + 26, y + 28, n, TODAY)
    text(x + 52, y + 34, title, size=21, weight="bold", color=INK)
    hline(x + 24, x + box_w - 24, y + 54, LINE, 1.5)
    fn_lines = [l1] + ([l2] if l2 else [])
    tspans(x + 24, y + 90, fn_lines, size=17, lh=25, color=INK, family=MONO)
    ytag = y + 90 + 25 * len(fn_lines) + 20
    tspans(x + 24, ytag, extra, size=14, lh=20, color=INK_DIM)

for (x, s) in zip(xs, stages):
    draw_stage(x, row1_y, *s)

row1_cy = row1_y + box_h/2
for i in range(3):
    right_arrow(xs[i] + box_w, xs[i+1], row1_cy)

# ---- stage 5: write matrix, own row (matches "after" diagram's row-2 start x) ----
row2_y = 580
b4_bottom_x = xs[3] + box_w/2
b5_top_x = xs[0] + box_w/2
stem_y = row1_y + box_h + 34
parts.append(f'<line x1="{b4_bottom_x}" y1="{row1_y+box_h}" x2="{b4_bottom_x}" y2="{stem_y}" '
              f'stroke="{LINE_STRONG}" stroke-width="4"/>')
parts.append(f'<line x1="{b4_bottom_x}" y1="{stem_y}" x2="{b5_top_x}" y2="{stem_y}" '
              f'stroke="{LINE_STRONG}" stroke-width="4"/>')
down_arrow(b5_top_x, stem_y, row2_y)

x5 = xs[0]
rect(x5, row2_y, box_w, box_h, fill="#ffffff", stroke=TODAY_BORDER, sw=2.5)
numbered_circle(x5 + 26, row2_y + 28, 5, TODAY)
text(x5 + 52, row2_y + 34, "Write Matrix", size=21, weight="bold", color=INK)
hline(x5 + 24, x5 + box_w - 24, row2_y + 54, LINE, 1.5)
tspans(x5 + 24, row2_y + 90, ["K+2 layout", "(K+3 with --recomb-span", "diagnostic track)"],
       size=17, lh=25, color=INK, family=MONO)
tspans(x5 + 24, row2_y + 90 + 25*3 + 20, ["binary features only --", "no ternary, no distance"],
       size=14, lh=20, color=INK_DIM)

# ---- what's missing, right where the "after" diagram's boxes 5-8 would be ----
gx = xs[1]
gy = row2_y
gw = xs[3] + box_w - gx
rect(gx, gy, gw, box_h, fill=GAP_BG, stroke=GAP_BORDER, sw=2.5)
text(gx + 24, gy + 42, "Not present today", size=22, weight="bold", color=GAP)
hline(gx + 24, gx + gw - 24, gy + 60, GAP_BORDER, 1.5)
tspans(gx + 24, gy + 96, [
    "No indel tracts, no breakpoint snapping, no ternary/distance encoding,",
    "and no read-sampling or coverage layer at all -- stage 3's single H1-or-H2",
    "pick per site IS the entire read model. Nothing can stack; nothing can go",
    "uncovered. See simulator_workflow.png for what's being added and why.",
], size=17, lh=28, color=INK)

# ---- bottom-line comparison ----
by = row2_y + box_h + 40
rect(60, by, 1800, 100, fill=TODAY_BG, stroke=TODAY_BORDER, sw=2.5, rx=16)
text(84, by + 38, "Today vs. planned --", size=20, weight="bold", color=TODAY)
text(84, by + 74,
     "5 stages, binary K+2 output   *   planned: 8 stages + acceptance gate, ternary+distance 2K+2 output (simulator_workflow.png)",
     size=18, color=INK)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}">' + "".join(parts) + "</svg>")

with open("/home/zrm22/.claude/jobs/0e12e67f/tmp/simulator_workflow_before.svg", "w") as f:
    f.write(svg)
print("wrote svg,", len(svg), "bytes; bottom bar bottom =", by + 100)
