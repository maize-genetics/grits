"""Plot xo_regression_dump.py output: crossover head (predicted crossovers per window) vs truth."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

z = np.load(sys.argv[1]); out = sys.argv[2]; title = sys.argv[3] if len(sys.argv) > 3 else ""
P, Y, D, M = z["sim_pred"], z["sim_true"], z["sim_depth"], z["sim_mode"]
DEP = ["0.1x", "0.5x", "1x", "2x"]
MODES = {0: ("in-panel", "#1A73E8"), 1: ("one own founder hidden", "#E8710A"), 2: ("both founders hidden", "#C5221F")}
fig = plt.figure(figsize=(17, 10.5))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.38, wspace=0.28)
rng = np.random.default_rng(0)
lines = []
for di in range(4):
    ax = fig.add_subplot(gs[0, di])
    top = 0
    for mode, (lab, col) in MODES.items():
        s = (D == di) & (M == mode)
        if not s.any():
            continue
        y, p = Y[s], P[s]
        ax.scatter(y + rng.uniform(-0.18, 0.18, y.size), p, s=4, alpha=0.18, color=col, lw=0)
        ks = np.unique(y)
        mu = [p[y == k].mean() for k in ks]
        ax.plot(ks, mu, "-o", color=col, ms=4, lw=1.8, label=f"{lab}: mean {p.mean():.2f} vs true {y.mean():.2f}")
        top = max(top, y.max(), np.percentile(p, 99.5))
        r = np.corrcoef(y, p)[0, 1] if y.std() > 0 else np.nan
        lines.append(f"{DEP[di]:<5} {lab:<22} windows {s.sum():>5}  mean pred {p.mean():6.3f}  mean true {y.mean():6.3f}  r {r:5.2f}")
    top = max(1.0, top)
    ax.plot([0, top], [0, top], "--", color="#80868B", lw=1)
    ax.set_xlim(-0.5, top + 0.5); ax.set_ylim(-0.1, top + 0.3)
    ax.set_title(f"simulated validation, {DEP[di]}", fontsize=12)
    ax.set_xlabel("true switches in window"); ax.set_ylabel("predicted crossovers (head)")
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
# calibration: bin by prediction, all depths/modes
ax = fig.add_subplot(gs[1, :2])
edges = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8, 1e9])
for mode, (lab, col) in MODES.items():
    s = M == mode
    b = np.digitize(P[s], edges) - 1
    xs, ys, ns = [], [], []
    for i in range(len(edges) - 1):
        k = b == i
        if k.sum() >= 20:
            xs.append(P[s][k].mean()); ys.append(Y[s][k].mean()); ns.append(k.sum())
    ax.plot(xs, ys, "-o", color=col, label=lab)
lim = [0.005, 12]
ax.plot(lim, lim, "--", color="#80868B", lw=1, label="perfect calibration")
ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=0.01)
ax.set_xlim(lim); ax.set_ylim(0, 1.2); ax.set_xlabel("predicted crossovers per window (binned)"); ax.set_ylabel("mean true switches in those windows")
ax.set_title("calibration, all depths (bins with >= 20 windows)", fontsize=12); ax.legend(fontsize=9, frameon=False)
# real samples
ax = fig.add_subplot(gs[1, 2:])
classes = ["IDX-INBRED", "IDX-HYB", "IDX-RIL2", "MIX-HYB", "MIX-RIL2", "OUT-INBRED", "OUT-HYB", "OUT-RIL2"]
for j, dep in enumerate(("0.1", "1.0")):
    for ci, c in enumerate(classes):
        ks = [k for k in z.files if k.startswith(f"real_{dep}__{c}__")]
        vals = [z[k].mean() for k in ks]
        xpos = ci + (-0.17 if j == 0 else 0.17)
        ax.scatter(np.full(len(vals), xpos), vals, s=22, color=("#1A73E8" if j == 0 else "#E8710A"),
                   label=("0.1x" if j == 0 else "1x") if ci == 0 else None, zorder=3)
ax.axhline(0, color="#80868B", lw=0.8)
ax.plot([-0.4, 1.4], [0, 0], color="#188038", lw=3, alpha=0.6, label="truth: 0 (IDX inbred / hybrid)")
ax.plot([1.6, 2.4], [0.015, 0.015], color="#188038", lw=3, alpha=0.6)
ax.set_xticks(range(len(classes))); ax.set_xticklabels([c.replace("-", "\n") for c in classes], fontsize=9)
ax.set_ylabel("mean predicted crossovers per window"); ax.set_title("real samples (one dot per sample; truth known only for IDX)", fontsize=12)
ax.legend(fontsize=9, frameon=False, loc="upper left")
fig.suptitle(f"Crossover head vs truth {title}", fontsize=16, weight="bold")
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("\n".join(lines))
real = {c: [float(z[k].mean()) for k in z.files if k.startswith(f"real_0.1__{c}__")] for c in classes}
print("real 0.1x mean per class:", {c: round(float(np.mean(v)), 3) for c, v in real.items() if v})
