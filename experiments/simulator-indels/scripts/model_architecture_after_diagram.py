"""Model architecture diagram: the planned indel-aware diploid CRF model,
color-coded reused / new / modified against today's FounderPathEncoder /
GRITSCRFDiploid (see model_architecture_before.png). Grounded in
experiments/simulator-indels/TRAINING_PLAN.md and this session's plan
(streamed-mixing-aurora). Same box grid/size as the "before" diagram so the
two compare directly. Regenerate whenever the IndelFounderPathEncoder /
train_diploid_indel.py design changes, so the picture doesn't drift from code."""

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
text(60, 62, "Diploid CRF Model: Indel-Aware (Planned)", size=38, weight="bold", color=INK)
text(60, 94, "IndelFounderPathEncoder (train_crf.py) -- GRITSCRFDiploidIndel (train_diploid_indel.py)",
     size=18, color=INK_DIM, family=MONO)

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
    (1, "Input .npy", MOD, MOD_BG, MOD_BORDER,
     "int8 [N,512,50]", "2K+2, ternary+distance",
     ["{-2,-1,0,1} match state +", "int8 log-coded distance"]),
    (2, "Dataset", NEW, NEW_BG, NEW_BORDER,
     "IndelDiploidDataset", "(+Affinity)",
     ["labels: (<0) -> K, not clip;", "M=(tern==1) drives affinity"]),
    (3, "Cell Embed", MOD, MOD_BG, MOD_BORDER,
     "Linear(5,256)->GELU", "->Linear(256,256)",
     ["onehot4(tern) | dist/127", "+ 516-row exact fast path"]),
    (4, "Founder Pool", REUSED, REUSED_BG, REUSED_BORDER,
     "fpool: MultiheadAttention", "(256,8), 1 query / site",
     ["identical to today", "-> h [B,512,256]"]),
]
stages_row2 = [
    (5, "Site Transformer", REUSED, REUSED_BG, REUSED_BORDER,
     "pos_encoder: TransformerEncoder", "x6 (ff=1024, GELU)",
     ["identical to today", "-> H [B,512,256]"]),
    (6, "Emission + Gate", REUSED, REUSED_BG, REUSED_BORDER,
     "einsum(H,cells)*scale", "x sigmoid(gate_head(H))",
     ["+ ext_bias(ext_emb),", "identical to today"]),
    (7, "Recomb Head", MOD, MOD_BG, MOD_BORDER,
     "Linear(260,1)->softplus", "+ del_frac input",
     ["depth=log1p((tern==1).sum())", "del_frac=(tern==-1).mean()"]),
    (8, "Pair CRF", REUSED, REUSED_BG, REUSED_BORDER,
     "crf_kernels.py (extracted)", "emis_p, tr_t unchanged",
     ["same _dcrf_nll/_dcrf_viterbi,", "now a shared module"]),
]

def draw_stage(x, y, n, title, fg, bg, border, l1, l2, extra):
    rect(x, y, box_w, box_h, fill="#ffffff", stroke=border, sw=2.5)
    numbered_circle(x + 26, y + 28, n, fg)
    text(x + 52, y + 34, title, size=21, weight="bold", color=INK)
    hline(x + 24, x + box_w - 24, y + 54, LINE, 1.5)
    fn_lines = [l1] + ([l2] if l2 else [])
    tspans(x + 24, y + 90, fn_lines, size=16, lh=24, color=INK, family=MONO)
    ytag = y + 90 + 24 * len(fn_lines) + 14
    tag(x + 24, ytag, ("REUSED" if fg == REUSED else ("+ NEW" if fg == NEW else "MODIFIED")), fg, bg)
    tspans(x + 24, ytag + 52, extra, size=14, lh=20, color=INK_DIM)

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

# ---- acceptance / comparison gate ----
gy = row2_y + box_h + 40
rect(60, gy, 1800, 100, fill=GATE_BG, stroke=GATE_BORDER, sw=2.5, rx=16)
text(84, gy + 38, "New checkpoint lineage -- comparison target, not a checkpoint to load --", size=20, weight="bold", color=GATE)
text(84, gy + 74,
     "diploid-affinity-sim512-h3 (val_pair_acc 0.6179) stays deployed unchanged; beat it on genome-wide 0.0711% + chr5 IBD-zone 1.181% SNP-class error",
     size=17, color=INK)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}">' + "".join(parts) + "</svg>")

with open("/home/zrm22/.claude/jobs/d44f819b/tmp/diagrams/model_architecture_after.svg", "w") as f:
    f.write(svg)
print("wrote svg,", len(svg), "bytes; gate bottom =", gy + 100)
