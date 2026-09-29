"""Plot head_tracks.py output: one PNG per tracks_<sample>.npz (needs only numpy + matplotlib)."""
import glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = sys.argv[1]
for f in sorted(glob.glob(d + "/tracks_*.npz")):
    z = np.load(f)
    models = sorted({k.split("/")[0] for k in z.files if "/" in k})
    fig, ax = plt.subplots(5, 1, figsize=(16, 9), sharex=True)
    for m in models:
        x = np.arange(len(z[m + "/gate"]))
        ax[0].plot(x, z[m + "/gate"], lw=.7, label=m)
        ax[1].semilogy(x, z[m + "/p_switch_row"], lw=.7)
        ax[2].plot(x, z[m + "/het"], lw=.7)
        ax[3].plot(x, z[m + "/switches"], lw=.7)
        ax[4].plot(x, z[m + "/snprc_err_pct"], lw=.7)
    for a_, lab in zip(ax, ["gate g", "p(switch) per row\n(row head)", "het h", "decoded switches", "SNP+RC err %"]):
        a_.set_ylabel(lab, fontsize=8)
        for o in z["chrom_offsets_mb"]:
            a_.axvline(o, color="0.85", lw=.5)
    ax[0].legend(fontsize=8); ax[0].set_title(str(z["title"]) + " - supervised heads, 1 Mb bins")
    ax[4].set_xlabel("Mb along concatenated autosomes")
    fig.tight_layout(); fig.savefig(f.replace(".npz", ".png"), dpi=110); plt.close(fig)
    print("wrote", f.replace(".npz", ".png"))
