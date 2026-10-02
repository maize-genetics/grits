"""Plot head_windows_dump.py output: one figure per window, ternary matrix vs every head."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

z = np.load(sys.argv[1]); outp = sys.argv[2]
TC = ListedColormap(["#d7301f", "#e8e8e8", "#08306b"])          # -1 del / 0 div / 1 match
PK = {"collinear": "#9ecae1", "insertion": "#c7e9c0", "replacement": "#fdd0a2",
      "repl OFF-site": "#e6550d", "bad site": "#756bb1", "pad": "#ffffff"}


def prov_class(p):
    c = np.full(p.shape, "collinear", object)
    k = p & 7
    c[(k == 2) | (k == 3)] = "insertion"
    c[(k == 4) | (k == 5)] = "replacement"
    c[(p >= 0) & (p & 8 > 0)] = "repl OFF-site"
    c[(p >= 0) & (p & 16 > 0)] = "bad site"
    c[p < 0] = "pad"
    return c


for i in range(int(z["n"])):
    g = lambda k: z[f"{i}/{k}"]
    tern, h1, h2 = g("tern"), g("h1"), g("h2")
    T, K = tern.shape
    x = np.arange(T)
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(6, 2, width_ratios=[7, 1.3], height_ratios=[5, .35, 1.3, 1.3, 1.3, .1],
                          hspace=.35, wspace=.04)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(tern.T, aspect="auto", cmap=TC, vmin=-1, vmax=1, interpolation="nearest")
    ok = h1 < K
    ax.scatter(x[ok], h1[ok], s=5, c="#ffd92f", marker="s", lw=0, label="true hap1 founder")
    ok = h2 < K
    ax.scatter(x[ok], h2[ok] + .3, s=5, c="#1a9850", marker="s", lw=0, label="true hap2 founder")
    ax.set_yticks(range(K)); ax.set_yticklabels([f"F{k}" for k in range(K)], fontsize=6)
    ax.set_ylabel("founder"); ax.set_xticks([]); ax.set_xlim(-.5, T - .5)
    ax.set_title(f"{g('label')}   —   ternary input (rows = reads/sites along window)", loc="left", fontsize=11)
    ax.legend(handles=[Patch(color="#08306b", label="match (1)"), Patch(color="#e8e8e8", label="diverge (0)"),
                       Patch(color="#d7301f", label="deleted / no anchor (-1)"),
                       *ax.get_legend_handles_labels()[0]],
              loc="upper left", bbox_to_anchor=(0, -.01), ncol=5, fontsize=8, frameon=False, markerscale=3)

    aa = fig.add_subplot(gs[0, 1], sharey=ax)
    yy = np.arange(K)
    aa.barh(yy - .2, g("aff_true"), .4, color="#636363", label="true")
    aa.barh(yy + .2, g("aff_ind"), .4, color="#e6550d", label="head (ind. mean)")
    aa.set_xlim(0, 1.05); aa.tick_params(labelleft=False); aa.set_title("affinity", fontsize=10)
    aa.legend(fontsize=7, loc="lower right")

    pa = fig.add_subplot(gs[1, 0])
    pc = prov_class(g("prov"))
    cols = np.array([matplotlib.colors.to_rgb(PK[c]) for c in pc])
    pa.imshow(cols[None], aspect="auto", interpolation="nearest")
    pa.set_yticks([]); pa.set_xticks([]); pa.set_ylabel("row\nsource", rotation=0, ha="right", va="center", fontsize=8)
    la = fig.add_subplot(gs[1, 1]); la.axis("off")
    la.legend(handles=[Patch(color=v, label=k) for k, v in PK.items() if k in set(pc)], fontsize=6,
              loc="center left", frameon=False, ncol=1)

    def track(r, title):
        a = fig.add_subplot(gs[r, 0], sharex=ax)
        a.set_ylim(-.05, 1.05); a.set_xlim(-.5, T - .5)
        a.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=9)
        a.tick_params(labelbottom=False, labelsize=7)
        return a

    gm = g("gate_m")
    a = track(2, "gate\nP(row clean)")
    a.fill_between(x, 0, np.where(gm, g("gate_y"), np.nan), step="mid", color="#bdbdbd", alpha=.6, label="true clean")
    a.plot(x, g("gate_p"), color="#3182bd", lw=.9, label="head")
    y = g("gate_y")[gm].astype(bool); p = g("gate_p")[gm]
    a.set_title(f"clean rows {y.mean():.0%}   head mean on clean {p[y].mean() if y.any() else np.nan:.2f}, "
                f"on unclean {p[~y].mean() if (~y).any() else np.nan:.2f}", fontsize=8, loc="right")
    a.legend(fontsize=7, loc="lower left", ncol=2)

    a = track(3, "switch\nP(t-1→t)")
    sw = g("sw_p")[1:]
    a.plot(x[1:], sw, color="#e6550d", lw=.9, label="row switch prob sigmoid(-c)")
    ts = np.flatnonzero(g("sw_y") & g("sw_m")) + 1
    for t in ts:
        a.axvline(t, color="k", ls="--", lw=1.2)
    a.set_ylim(0, max(.05, sw.max() * 1.2))
    a.set_title(f"window crossovers: head λ = {float(g('xo_lam')):.2f}   true = {int(g('xo_true'))}   "
                f"(Σ row p = {sw[g('sw_m')].sum():.2f})   dashed = true switch", fontsize=8, loc="right")
    a.legend(fontsize=7, loc="upper left")

    hm = g("het_m")
    a = track(4, "het\nP(het)")
    a.fill_between(x, 0, np.where(hm, g("het_y"), np.nan), step="mid", color="#bdbdbd", alpha=.6, label="true het")
    a.plot(x, g("het_p"), color="#31a354", lw=.9, label="head")
    a.tick_params(labelbottom=True); a.set_xlabel("row in window")
    a.legend(fontsize=7, loc="lower left", ncol=2)
    fig.savefig(f"{outp}_{i}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"{outp}_{i}.png")
