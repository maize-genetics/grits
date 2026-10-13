#!/usr/bin/env python3
"""Pool feature_errors.py counts into per-feature tables + plots (run with /usr/bin/python3 for matplotlib).

Pooling: errors and sites are summed over samples in a group (pooled ratio, not mean of rates).
Groups: each class, OUT (3 OUT classes), MIX (2), IDX (3). Enrichment = feature SNP+RC error rate /
genome-wide SNP+RC error rate of the same model+group (= error share / site share).
"""
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

OUT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/genomic_feature_errors")
MODELS = ["v6cfullfe_homo", "diploid_affinity", "ternary_baseline", "old_v6", "ceiling"]
GROUPS = {"OUT": ["OUT-INBRED", "OUT-HYB", "OUT-RIL2"], "MIX": ["MIX-HYB", "MIX-RIL2"],
          "IDX": ["IDX-INBRED", "IDX-HYB", "IDX-RIL2"]}
for c in sum(GROUPS.values(), []):
    GROUPS[c] = [c]
COLS = ["snprc_sites", "snprc_err", "all_sites", "all_err", "elig", "uncalled"]


def load():
    agg = defaultdict(lambda: [0] * 6)
    samples = defaultdict(set)
    for f in glob.glob(str(OUT / "counts_p*.tsv")):
        for r in csv.DictReader(open(f), delimiter="\t"):
            for g, cl in GROUPS.items():
                if r["dataset"] in cl:
                    a = agg[(r["depth"], g, r["model"], r["key"])]
                    for i, c in enumerate(COLS):
                        a[i] += int(r[c])
                    samples[(r["depth"], g, r["model"])].add(r["individual"])
    return agg, samples


def pct(e, s):
    return 100.0 * e / s if s else float("nan")


def main():
    agg, samples = load()
    meta = json.loads((OUT / "feature_annotation_meta.json").read_text())
    keys = ["all"] + [f"primary:{p}" for p in meta["PRIMARY"]] + [f"flag:{f}" for f in meta["FLAG_NAMES"]] + \
           [f"arm:{a}" for a in meta["ARM_NAMES"]]
    # long table
    with open(OUT / "feature_errors_long.tsv", "w") as fh:
        fh.write("depth\tgroup\tmodel\tn_samples\tkey\tsnprc_sites\tsnprc_err_pct\tsnprc_err_share\t"
                 "snprc_site_share\tsnprc_enrichment\tall_sites\tall_err_pct\tall_enrichment\tuncalled_pct\n")
        for (dep, g, m, k), v in sorted(agg.items()):
            tot = agg[(dep, g, m, "all")]
            s, e, as_, ae, el, un = v
            rate, trate = pct(e, s), pct(tot[1], tot[0])
            fh.write(f"{dep}\t{g}\t{m}\t{len(samples[(dep, g, m)])}\t{k}\t{s}\t{rate:.4f}\t"
                     f"{e / tot[1] if tot[1] else float('nan'):.5f}\t{s / tot[0] if tot[0] else float('nan'):.5f}\t"
                     f"{rate / trate if trate else float('nan'):.3f}\t{as_}\t{pct(ae, as_):.4f}\t"
                     f"{pct(ae, as_) / pct(tot[3], tot[2]) if tot[3] else float('nan'):.3f}\t{pct(un, el):.3f}\n")
    # wide markdown-ish tables per depth x group (primary + arm + selected flags)
    lines = []
    for dep in ("0.1", "1.0"):
        for g in ("OUT", "MIX", "IDX"):
            lines.append(f"\n### {g} pooled, {dep}x  (SNP+RC err% [enrichment]; sites = SNP+RC compared, millions, "
                         f"v6cfullfe_homo; all-sites err% second block; old_v6 uncalled% of truth-REF/SNP records)\n")
            hdr = "| feature | share of truth-REF/SNP records | SNP+RC sites (M, summed over samples, v6cfh) | " + " | ".join(m for m in MODELS) + \
                  " | all-sites: v6cfh / DA / tern / ceil | old_v6 uncalled% |"
            lines.append(hdr)
            lines.append("|" + "---|" * (hdr.count("|") - 1))
            for k in keys:
                if k.startswith("flag:") and k.split(":")[1] in ("CDS", "UTR", "intron", "LTR_Gypsy", "LTR_Copia",
                                                                 "LTR_unknown", "Helitron", "LINE", "knob",
                                                                 "centromeric_CentC_CRM", "rDNA_spacer", "subtelomere"):
                    continue
                cells = []
                for m in MODELS:
                    v = agg.get((dep, g, m, k)); t = agg.get((dep, g, m, "all"))
                    if not v or not t or not v[0]:
                        cells.append("-")
                        continue
                    r = pct(v[1], v[0]); tr = pct(t[1], t[0])
                    cells.append(f"{r:.2f} [{r / tr:.2f}]" if tr else f"{r:.2f} [-]")
                allc = []
                for m in ("v6cfullfe_homo", "diploid_affinity", "ternary_baseline", "ceiling"):
                    v = agg.get((dep, g, m, k))
                    allc.append(f"{pct(v[3], v[2]):.2f}" if v and v[2] else "-")
                v6 = agg.get((dep, g, "old_v6", k))
                ref = agg.get((dep, g, "v6cfullfe_homo", k))
                allrec = agg.get((dep, g, "v6cfullfe_homo", "all"))
                share = f"{ref[4] / allrec[4]:.3f}" if ref and allrec and allrec[4] else "-"
                lines.append(f"| {k} | {share} | {ref[0] / 1e6 if ref else 0:.2f} | " + " | ".join(cells) +
                             f" | {' / '.join(allc)} | {pct(v6[5], v6[4]) if v6 else float('nan'):.1f} |")
    (OUT / "feature_tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plots(agg, meta)
    mb_corr()


def plots(agg, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"v6cfullfe_homo": "#2a78d6", "diploid_affinity": "#eb6834", "ternary_baseline": "#1baf7a",
           "ceiling": "#8a8984"}
    cats = [p for p in meta["PRIMARY"]]
    for dep in ("0.1", "1.0"):
        fig, axes = plt.subplots(1, 3, figsize=(17, 7.5), sharey=True)
        for ax, g in zip(axes, ("OUT", "MIX", "IDX")):
            ms = list(col)
            h = 0.8 / len(ms)
            for j, m in enumerate(ms):
                ys, xs = [], []
                for i, c in enumerate(cats):
                    v = agg.get((dep, g, m, f"primary:{c}"))
                    if v and v[0] >= 1000:
                        xs.append(pct(v[1], v[0])); ys.append(i + (j - (len(ms) - 1) / 2) * h)
                ax.barh(ys, xs, height=h * 0.9, color=col[m], label=m)
            ax.set_yticks(range(len(cats)))
            ax.set_yticklabels([f"{c}" for c in cats], fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f"{g} pooled, {dep}x")
            ax.set_xlabel("SNP+RC error % (pooled)")
            ax.grid(axis="x", color="#e4e3df", lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0].legend(fontsize=8, frameon=False, loc="lower right")
        fig.suptitle("SNP+RC genotype error by genomic feature (primary category; categories with <1000 sites omitted)")
        fig.tight_layout()
        fig.savefig(OUT / f"feature_error_bars_{dep}x.png", dpi=130)
        plt.close(fig)


def mb_corr():
    """1-Mb bins: pooled OUT SNP+RC error rate per bin vs gene / TE density; Spearman rho."""
    import numpy as np
    dens = {}
    for r in csv.DictReader(open(OUT / "mb_bin_density.tsv"), delimiter="\t"):
        ci = int(r["chrom"][3:]) - 1
        dens[ci * 1000 + int(r["mb"])] = (int(r["gene_bp"]) / int(r["bin_bp"]), int(r["te_bp"]) / int(r["bin_bp"]))
    acc = defaultdict(lambda: [0, 0])
    for f in glob.glob(str(OUT / "mbbins_p*.tsv")):
        for r in csv.DictReader(open(f), delimiter="\t"):
            g = r["dataset"][:3]
            a = acc[(r["depth"], g, r["model"], int(r["mb_id"]))]
            a[0] += int(r["snprc_sites"]); a[1] += int(r["snprc_err"])
    def rank(x):
        o = np.argsort(x); rk = np.empty(len(x)); rk[o] = np.arange(len(x)); return rk
    rows = []
    with open(OUT / "mb_bin_errors.tsv", "w") as fh:
        fh.write("depth\tgroup\tmodel\tmb_id\tsnprc_sites\tsnprc_err\tgene_frac\tte_frac\n")
        for (dep, g, m, b), (s, e) in sorted(acc.items()):
            gd, td = dens.get(b, (float("nan"), float("nan")))
            fh.write(f"{dep}\t{g}\t{m}\t{b}\t{s}\t{e}\t{gd:.5f}\t{td:.5f}\n")
    out = ["depth\tgroup\tmodel\tn_bins\tspearman_err_vs_gene_frac\tspearman_err_vs_te_frac"]
    keys = sorted({k[:3] for k in acc})
    for dep, g, m in keys:
        bs = [b for (d2, g2, m2, b), (s, e) in acc.items() if (d2, g2, m2) == (dep, g, m) and s >= 2000 and b in dens]
        er = np.array([acc[(dep, g, m, b)][1] / acc[(dep, g, m, b)][0] for b in bs])
        gd = np.array([dens[b][0] for b in bs]); td = np.array([dens[b][1] for b in bs])
        rg = np.corrcoef(rank(er), rank(gd))[0, 1]; rt = np.corrcoef(rank(er), rank(td))[0, 1]
        out.append(f"{dep}\t{g}\t{m}\t{len(bs)}\t{rg:.3f}\t{rt:.3f}")
    (OUT / "mb_bin_correlation.tsv").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    # plot: OUT 0.1x error along chromosomes for v6cfullfe_homo + ceiling, with gene density
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meta = json.loads((OUT / "feature_annotation_meta.json").read_text())
    for dep in ("0.1", "1.0"):
        fig, axes = plt.subplots(10, 1, figsize=(14, 20), sharex=True)
        for ci in range(10):
            ax = axes[ci]
            for m, c in (("v6cfullfe_homo", "#2a78d6"), ("diploid_affinity", "#eb6834"), ("ceiling", "#8a8984")):
                bs = sorted(b for (d2, g2, m2, b) in acc if (d2, g2, m2) == (dep, "OUT", m) and b // 1000 == ci)
                x = [b % 1000 for b in bs]
                y = [100 * acc[(dep, "OUT", m, b)][1] / acc[(dep, "OUT", m, b)][0] if acc[(dep, "OUT", m, b)][0] >= 2000
                     else float("nan") for b in bs]
                ax.plot(x, y, color=c, lw=1.2, label=m)
            ax2 = ax.twinx() if False else None  # single axis rule: gene density drawn as shaded band below
            bsd = sorted(b for b in dens if b // 1000 == ci)
            gmax = max(dens[b][0] for b in dens)
            ylim = ax.get_ylim()[1]
            ax.fill_between([b % 1000 for b in bsd], 0, [dens[b][0] / gmax * ylim * 0.35 for b in bsd],
                            color="#cfcdc6", lw=0, label="gene density (scaled, shading)")
            cen = meta["centromere_centre"][f"chr{ci + 1}"] / 1e6
            ax.axvline(cen, color="#52514e", ls=":", lw=1)
            ax.set_ylabel(f"chr{ci + 1}\nerr %", fontsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0].legend(fontsize=8, frameon=False, ncol=4, loc="upper right")
        axes[-1].set_xlabel("B73 v5 position (Mb); dotted = estimated centromere")
        fig.suptitle(f"OUT pooled SNP+RC error per 1-Mb bin, {dep}x (grey shading = gene density, scaled)")
        fig.tight_layout()
        fig.savefig(OUT / f"mb_error_along_chrom_OUT_{dep}x.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


def crosstab_summary():
    """Arm-position-standardized enrichment from feature_arm_crosstab.py output (0.1x):
    observed feature errors / expected errors if, within each arm bin, the feature had the
    genome-wide error rate of that bin (indirect standardization). Removes the distance-from-
    centromere (recombination-proxy) confound from the per-feature enrichment."""
    groups = {"genic_CDS+UTR": ["CDS", "UTR"], "intron": ["intron"], "gene_flank_2kb": ["gene_flank_2kb"],
              "intergenic_nonTE": ["intergenic_nonTE"], "LTR_Gypsy": ["LTR_Gypsy"], "LTR_Copia": ["LTR_Copia"],
              "LTR_unknown": ["LTR_unknown"],
              "TIR_all": ["TIR_DTA_hAT", "TIR_DTC_CACTA", "TIR_DTH_PIF", "TIR_DTM_Mutator", "TIR_DTT_Tc1"],
              "Helitron": ["Helitron"], "knob": ["knob"], "centromeric_CentC_CRM": ["centromeric_CentC_CRM"]}
    inv = {c: g for g, cs in groups.items() for c in cs}
    acc = defaultdict(lambda: [0, 0])
    for f in glob.glob(str(OUT / "crosstab_c*.tsv")):
        for r in csv.DictReader(open(f), delimiter="\t"):
            G = r["dataset"][:3]
            for key in ((G, r["model"], inv.get(r["primary"], "other"), r["arm"]),
                        (G, r["model"], "ALL", r["arm"])):
                a = acc[key]; a[0] += int(r["snprc_sites"]); a[1] += int(r["snprc_err"])
    meta = json.loads((OUT / "feature_annotation_meta.json").read_text())
    arms = meta["ARM_NAMES"]
    lines = ["group\tmodel\tfeature\tsnprc_sites\tsnprc_err_pct\tcrude_enrichment\tarm_standardized_enrichment\t"
             + "\t".join(f"err_pct_{a}" for a in arms)]
    for G in ("OUT", "MIX", "IDX"):
        for m in ("v6cfullfe_homo", "diploid_affinity", "ceiling"):
            tot_s = sum(acc[(G, m, "ALL", a)][0] for a in arms); tot_e = sum(acc[(G, m, "ALL", a)][1] for a in arms)
            if not tot_s:
                continue
            for feat in list(groups) + ["other"]:
                s = sum(acc[(G, m, feat, a)][0] for a in arms); e = sum(acc[(G, m, feat, a)][1] for a in arms)
                if not s:
                    continue
                exp = sum(acc[(G, m, feat, a)][0] * (acc[(G, m, "ALL", a)][1] / acc[(G, m, "ALL", a)][0])
                          for a in arms if acc[(G, m, "ALL", a)][0])
                crude = (e / s) / (tot_e / tot_s) if tot_e else float("nan")
                std = e / exp if exp else float("nan")
                per = "\t".join(f"{pct(acc[(G, m, feat, a)][1], acc[(G, m, feat, a)][0]):.2f}" for a in arms)
                lines.append(f"{G}\t{m}\t{feat}\t{s}\t{pct(e, s):.3f}\t{crude:.2f}\t{std:.2f}\t{per}")
    (OUT / "arm_standardized_enrichment_0.1x.tsv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "crosstab":
    crosstab_summary()
