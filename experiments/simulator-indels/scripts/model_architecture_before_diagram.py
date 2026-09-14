"""Model architecture diagram: the diploid CRF model AS IT EXISTS TODAY (before
the indel-aware model), grounded directly in FounderPathEncoder (train_crf.py:
159-277) and GRITSCRFDiploid (train_diploid.py:354-549). Same visual system as
model_architecture_after.png so the two compare directly -- same box size/grid/
fonts, no reused/new/modified tags here since this is a plain snapshot."""

W, H = 1920, 1080
INK = "#1c2430"
INK_DIM = "#535f6e"
LINE = "#c7d0d8"
LINE_STRONG = "#9fb0bd"
TODAY = "#5b6b7c"; TODAY_BG = "#eef1f3"; TODAY_BORDER = "#c7d0d8"

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
text(60, 62, "Diploid CRF Model: Today (Before Indel-Aware Model)", size=38, weight="bold", color=INK)
text(60, 94, "FounderPathEncoder -- train_crf.py:159-277  /  GRITSCRFDiploid -- train_diploid.py:354-549",
     size=18, color=INK_DIM, family=MONO)
tag(60, 122, "SNAPSHOT, not a diff -- compare against model_architecture_after.png", TODAY, TODAY_BG)

# ---- box grid ----
box_w, box_h = 420, 260
gap = 40
xs = [60 + i * (box_w + gap) for i in range(4)]
row1_y = 210
row2_y = 580

stages_row1 = [
    (1, "Input .npy", "int8 [N,512,26]", "K+2, binary {0,1}",
     ["K=24 founders, T=512 sites,", "cols K/K+1 = H1/H2 labels"]),
    (2, "Dataset", "PreWindowedDiploidDataset", "(+Individual / +Affinity)",
     ["input_embeds [B,512,24] f32", "+ h1,h2 [B,512] i64"]),
    (3, "Cell Embed", "Linear(1,256)->GELU", "->Linear(256,256)",
     ["per (site,founder) cell:", "cell(log1p(X))"]),
    (4, "Founder Pool", "fpool: MultiheadAttention", "(256,8), 1 query / site",
     ["collapses 25 founders", "-> h [B,512,256]"]),
]
stages_row2 = [
    (5, "Site Transformer", "pos_encoder: TransformerEncoder", "x6 (ff=1024, GELU)",
     ["+ fixed sinusoidal PE", "-> H [B,512,256]"]),
    (6, "Emission + Gate", "einsum(H,cells)*scale", "x sigmoid(gate_head(H))",
     ["+ ext_bias(ext_emb),", "zero-init  -> emis_f [B,512,25]"]),
    (7, "Recomb Head", "Linear(259,1)->softplus", "over [H|depth|entropy|log1p(dbp)]",
     ["depth=log1p(X.sum(-1))", "-> c [B,512]"]),
    (8, "Pair CRF", "emis_p=emis_f[pi]+emis_f[pj]", "tr_t = -c*nsw + stay_bonus*I",
     ["_dcrf_nll / _dcrf_viterbi", "in train_diploid.py"]),
]

def draw_stage(x, y, n, title, l1, l2, extra):
    rect(x, y, box_w, box_h, fill="#ffffff", stroke=TODAY_BORDER, sw=2.5)
    numbered_circle(x + 26, y + 28, n, TODAY)
    text(x + 52, y + 34, title, size=21, weight="bold", color=INK)
    hline(x + 24, x + box_w - 24, y + 54, LINE, 1.5)
    fn_lines = [l1] + ([l2] if l2 else [])
    tspans(x + 24, y + 90, fn_lines, size=16, lh=24, color=INK, family=MONO)
    ytag = y + 90 + 24 * len(fn_lines) + 20
    tspans(x + 24, ytag, extra, size=14, lh=20, color=INK_DIM)

for (x, s) in zip(xs, stages_row1):
    draw_stage(x, row1_y, *s)
for (x, s) in zip(xs, stages_row2):
    draw_stage(x, row2_y, *s)

row1_cy = row1_y + box_h/2
row2_cy = row2_y + box_h/2
for i in range(3):
    right_arrow(xs[i] + box_w, xs[i+1], row1_cy)
    right_arrow(xs[i] + box_w, xs[i+1], row2_cy)

b4_bottom_x = xs[3] + box_w/2
b5_top_x = xs[0] + box_w/2
stem_y = row1_y + box_h + 34
parts.append(f'<line x1="{b4_bottom_x}" y1="{row1_y+box_h}" x2="{b4_bottom_x}" y2="{stem_y}" '
              f'stroke="{LINE_STRONG}" stroke-width="4"/>')
parts.append(f'<line x1="{b4_bottom_x}" y1="{stem_y}" x2="{b5_top_x}" y2="{stem_y}" '
              f'stroke="{LINE_STRONG}" stroke-width="4"/>')
down_arrow(b5_top_x, stem_y, row2_y)

# ---- bottom-line summary ----
gy = row2_y + box_h + 40
rect(60, gy, 1800, 100, fill=TODAY_BG, stroke=TODAY_BORDER, sw=2.5, rx=16)
text(84, gy + 38, "What the model cannot see today --", size=20, weight="bold", color=TODAY)
text(84, gy + 74,
     "binary read-sharing only: a founder with no matching read reads identically whether it's diverged or genuinely absent (deleted) at that locus",
     size=17, color=INK)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}">' + "".join(parts) + "</svg>")

with open("/home/zrm22/.claude/jobs/d44f819b/tmp/diagrams/model_architecture_before.svg", "w") as f:
    f.write(svg)
print("wrote svg,", len(svg), "bytes; bottom bar bottom =", gy + 100)
