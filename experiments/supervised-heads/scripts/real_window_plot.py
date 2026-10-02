"""Plot real_window_dump.py output: one figure per real window -- what the encoder+CRF sees,
what the heads say, what each --het-prior mode decodes, and where truth disagrees."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

z = np.load(sys.argv[1]); outp = sys.argv[2]
F = [str(f) for f in z["founders"]]
TC = ListedColormap(["#d7301f", "#e8e8e8", "#08306b"])
MODE_STYLE = {"row": ("#e41a1c", "x", "het prior on every row"),
              "segment": ("#ff7f00", "o", "het prior once per segment"),
              "off": ("#4daf4a", ".", "no het prior")}

for i in range(int(z["n"])):
    g = lambda k: z[f"{i}/{k}"]
    tern = g("tern"); T, K = tern.shape
    x = np.arange(T)
    comp = g("comp")                                             # [2,T,K]
    fig = plt.figure(figsize=(17, 15))
    gs = fig.add_gridspec(8, 2, width_ratios=[7, 1.2], height_ratios=[4.2, 4.2, .9, .9, .9, .9, 1.2, .1],
                          hspace=.42, wspace=.03)

    def overlay(ax):
        for md, (col, mk, _l) in MODE_STYLE.items():
            p1, p2 = g(f"p1_{md}"), g(f"p2_{md}")
            off = {"row": -.25, "segment": 0, "off": .25}[md]
            ok1, ok2 = p1 < K, p2 < K
            ax.scatter(x[ok1], p1[ok1] + off, s=7 if mk != "." else 10, c=col, marker=mk, lw=.7)
            ax.scatter(x[ok2], p2[ok2] + off, s=7 if mk != "." else 10, c=col, marker=mk, lw=.7)
        ax.set_yticks(range(K)); ax.set_yticklabels(F, fontsize=6); ax.set_xlim(-.5, T - .5); ax.set_xticks([])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(tern.T, aspect="auto", cmap=TC, vmin=-1, vmax=1, interpolation="nearest")
    overlay(ax)
    ax.set_title(f"{g('label')}\nINPUT: ternary matrix (rows = reads along the window), with each decode's two founders marked",
                 loc="left", fontsize=10)
    ax.legend(handles=[Patch(color="#08306b", label="match"), Patch(color="#e8e8e8", label="diverged"),
                       Patch(color="#d7301f", label="-1 (no anchor/deleted)")] +
              [Line2D([], [], color=c, marker=m, ls="", label=f"decode: {l}") for c, m, l in MODE_STYLE.values()],
              loc="upper left", bbox_to_anchor=(0, -.005), ncol=6, fontsize=7, frameon=False)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    best = np.fmax(comp[0], comp[1])
    ax2.imshow(best.T, aspect="auto", cmap="Greys", vmin=.5, vmax=1, interpolation="nearest")
    overlay(ax2)
    ax2.set_title("TRUTH (from gVCF): how well each founder carries this sample's alleles near the row "
                  "(black = identical to a true haplotype, white <= 50%)", loc="left", fontsize=9)

    aa = fig.add_subplot(gs[0, 1], sharey=ax)
    aa.barh(np.arange(K), g("aff"), color="#e6550d"); aa.set_xlim(0, 1.05)
    aa.tick_params(labelleft=False, labelsize=7); aa.set_title("affinity head\n(whole sample)", fontsize=8)
    ab = fig.add_subplot(gs[1, 1], sharey=ax2)
    ab.barh(np.arange(K) - .2, np.nanmean(comp[0], 0), .4, color="#fdae61", label="hap1")
    ab.barh(np.arange(K) + .2, np.nanmean(comp[1], 0), .4, color="#2c7bb6", label="hap2")
    ab.set_xlim(.5, 1.0); ab.tick_params(labelleft=False, labelsize=7)
    ab.set_title("window-mean truth match", fontsize=8); ab.legend(fontsize=6, loc="lower right")

    def track(r, name):
        a = fig.add_subplot(gs[r, 0], sharex=ax)
        a.set_xlim(-.5, T - .5); a.tick_params(labelbottom=False, labelsize=7)
        a.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
        return a

    a = track(2, "GATE head\nP(row trusted)")
    a.plot(x, g("gate"), color="#3182bd", lw=.8); a.set_ylim(-.05, 1.05)
    a = track(3, "HET head\nP(het)")
    a.plot(x, g("het"), color="#31a354", lw=.8); a.set_ylim(-.05, 1.05)
    truth_het = np.nanmax(np.abs(comp[0] - comp[1]), 1) > .2
    a.fill_between(x, 0, truth_het.astype(float), step="mid", color="#bdbdbd", alpha=.5, lw=0)
    a.text(1.002, .5, "grey = haplotypes\ndiffer (truth)", transform=a.transAxes, fontsize=6, va="center")
    a = track(4, "SWITCH head\nP(row t-1->t)")
    a.plot(x[1:], g("sw")[1:], color="#e6550d", lw=.8)
    a.set_title(f"window crossovers predicted: {float(g('lam')):.2f}", fontsize=7, loc="right")
    a = track(5, "decoded\nhomozygous?")
    for k, md in enumerate(MODE_STYLE):
        hom = (g(f"p1_{md}") == g(f"p2_{md}")).astype(float)
        a.fill_between(x, k, k + hom * .8, step="mid", color=MODE_STYLE[md][0], lw=0)
    a.set_ylim(-.1, 3); a.set_yticks([.4, 1.4, 2.4]); a.set_yticklabels(["row", "segment", "off"], fontsize=6)
    a = track(6, "SNP+RefCall\nerror near row")
    for md, (col, _m, lab) in MODE_STYLE.items():
        e = g(f"err_{md}")
        a.plot(x, e, color=col, lw=1, label=f"{lab}: mean {np.nanmean(e) * 100:.1f}%")
    a.set_ylim(-.02, 1.02); a.legend(fontsize=7, loc="upper right", ncol=3)
    a.tick_params(labelbottom=True); a.set_xlabel("row in window")
    fig.savefig(f"{outp}_{i}.png", dpi=105, bbox_inches="tight")
    plt.close(fig)
    print(f"{outp}_{i}.png")
