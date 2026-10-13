#!/usr/bin/env python
"""Assign every autosomal panel record (fast_snprc_score Panel order) to B73 v5 genomic features.

Inputs (B73 v5 / Zm-B73-REFERENCE-NAM-5.0 coordinates, chr1..chr10):
  genes : /workdir/shared_files/maize/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3  (identical md5 to
          https://download.maizegdb.org/Zm-B73-REFERENCE-NAM-5.0/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3.gz)
  TEs   : https://download.maizegdb.org/Zm-B73-REFERENCE-NAM-5.0/Zm-B73-REFERENCE-NAM-5.0.TE.gff3.gz
          (panEDTA; includes knob180/TR-1, CentC, CRM, rDNA spacer, subtelomere)
  CNS   : https://download.maizegdb.org/Zm-B73-REFERENCE-NAM-5.0/conservatoryV10.ZmB73NAM5.gff

Output npz (one entry per panel record):
  primary  int8   mutually exclusive category, precedence
                  CDS > UTR > intron > TE (innermost = shortest overlapping TE feature wins)
                  > gene flank (<=2 kb, non-TE) > intergenic non-TE
  flags    uint32 non-exclusive membership bitmask (FLAG_NAMES)
  armbin   int8   relative distance from the (estimated) centromere, binned (ARM_BINS)
  mb       int32  global 1-Mb bin id (chrom_index*1000 + pos//1e6)
Plus per-1Mb-bin gene bp / TE bp density table (TSV) and the centromere estimates.

Centromeres are NOT taken from a published table (none on disk): centre = CentC+CRM-bp-weighted
mean position within +-1.5 Mb of the densest 500-kb CentC+CRM bin; core = centre +- 1 Mb.
"""
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fast_snprc_score as fss  # noqa: E402

GENE_GFF = "/workdir/shared_files/maize/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3"
ANN = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/annotation")
TE_GFF = ANN / "Zm-B73-REFERENCE-NAM-5.0.TE.gff3.gz"
CNS_GFF = ANN / "conservatoryV10.ZmB73NAM5.gff"
OUT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/genomic_feature_errors")
FLANK = 2000

PRIMARY = ["intergenic_nonTE", "gene_flank_2kb", "CDS", "UTR", "intron",
           "LTR_Gypsy", "LTR_Copia", "LTR_unknown", "TIR_DTA_hAT", "TIR_DTC_CACTA",
           "TIR_DTH_PIF", "TIR_DTM_Mutator", "TIR_DTT_Tc1", "Helitron", "LINE",
           "knob", "centromeric_CentC_CRM", "rDNA_spacer", "subtelomere", "other_repeat"]
P = {n: i for i, n in enumerate(PRIMARY)}
FLAG_NAMES = ["gene_any", "CDS", "UTR", "intron", "flank_2kb", "TE_any", "LTR_Gypsy", "LTR_Copia",
              "LTR_unknown", "TIR_any", "MITE", "Helitron", "LINE", "knob", "centromeric_CentC_CRM",
              "rDNA_spacer", "subtelomere", "CNS", "centromere_core_2Mb", "TE_in_gene"]
FL = {n: 1 << i for i, n in enumerate(FLAG_NAMES)}
ARM_BINS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0001]
ARM_NAMES = ["arm_0-10%", "arm_10-25%", "arm_25-50%", "arm_50-75%", "arm_75-100%"]


def te_group(cls):
    fam = cls.split("/")[-1]
    if cls == "LTR/Gypsy":
        return "LTR_Gypsy"
    if cls == "LTR/Copia":
        return "LTR_Copia"
    if cls == "LTR/unknown":
        return "LTR_unknown"
    if cls == "LTR/CRM" or cls == "Cent/CentC":
        return "centromeric_CentC_CRM"
    if cls.startswith(("DNA/DT", "MITE/DT")):
        return {"DTA": "TIR_DTA_hAT", "DTC": "TIR_DTC_CACTA", "DTH": "TIR_DTH_PIF",
                "DTM": "TIR_DTM_Mutator", "DTT": "TIR_DTT_Tc1"}[fam]
    if cls == "DNA/Helitron":
        return "Helitron"
    if cls.startswith("LINE/"):
        return "LINE"
    if cls.startswith("knob/"):
        return "knob"
    if cls.startswith("rDNA/"):
        return "rDNA_spacer"
    if cls.startswith("subtelomere/"):
        return "subtelomere"
    return "other_repeat"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = fss.Panel()
    chroms = fss.AUTOSOMES
    clen = {}
    genes = {c: [] for c in chroms}
    feats = {c: {"CDS": [], "UTR": [], "exon": [], "mRNA": []} for c in chroms}
    for line in open(GENE_GFF):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if f[0] not in genes:
            continue
        s, e = int(f[3]), int(f[4])
        if f[2] == "chromosome":
            clen[f[0]] = e
        elif f[2] == "gene":
            genes[f[0]].append((s, e))
        elif f[2] == "CDS":
            feats[f[0]]["CDS"].append((s, e))
        elif f[2] in ("five_prime_UTR", "three_prime_UTR"):
            feats[f[0]]["UTR"].append((s, e))
        elif f[2] == "exon":
            feats[f[0]]["exon"].append((s, e))
        elif f[2] == "mRNA":
            feats[f[0]]["mRNA"].append((s, e))
    tes = {c: [] for c in chroms}
    rx = re.compile(r"Classification=([^;]+)")
    for line in gzip.open(TE_GFF, "rt"):
        if line.startswith("#"):
            continue
        f = line.split("\t", 9)
        if f[0] not in tes:
            continue
        m = rx.search(f[8])
        if not m:
            continue
        cls = m.group(1)
        tes[f[0]].append((int(f[3]), int(f[4]), te_group(cls), cls.startswith("MITE/")))
    cns = {c: [] for c in chroms}
    for line in open(CNS_GFF):
        if line.startswith("#"):
            continue
        f = line.split("\t", 6)
        if f[0] in cns:
            cns[f[0]].append((int(f[3]), int(f[4])))

    n = panel.pos.size
    primary = np.zeros(n, np.int8)
    flags = np.zeros(n, np.uint32)
    armbin = np.zeros(n, np.int8)
    mb = np.zeros(n, np.int32)
    cen = {}
    dens_rows = []
    for ci, c in enumerate(chroms):
        L = clen[c] + 2
        prim = np.zeros(L, np.int8)
        fl = np.zeros(L, np.uint32)

        def paint(arr, ivs, val, op="set"):
            for s, e in ivs:
                if op == "set":
                    arr[s:e + 1] = val
                else:
                    arr[s:e + 1] |= val

        # TEs: longest first so the innermost (shortest, nested) element wins the primary label
        te = sorted(tes[c], key=lambda t: -(t[1] - t[0]))
        te_bp = np.zeros(L, bool)
        for s, e, g, mite in te:
            prim[s:e + 1] = P[g]
            te_bp[s:e + 1] = True
            fl[s:e + 1] |= FL["TE_any"] | (FL[g] if g in FL else 0) | \
                (FL["TIR_any"] if g.startswith("TIR_") else 0) | (FL["MITE"] if mite else 0)
        # gene flank (non-TE only for primary)
        flank = np.zeros(L, bool)
        for s, e in genes[c]:
            flank[max(1, s - FLANK):s] = True
            flank[e + 1:min(L, e + 1 + FLANK)] = True
        gene_bp = np.zeros(L, bool)
        for s, e in genes[c]:
            gene_bp[s:e + 1] = True
        flank &= ~gene_bp
        fl[flank] |= FL["flank_2kb"]
        prim[flank & ~te_bp] = P["gene_flank_2kb"]
        # genic: intron (mRNA span) < UTR / nc exon < CDS
        intr = np.zeros(L, bool)
        paint(intr, feats[c]["mRNA"], True)
        exon = np.zeros(L, bool)
        paint(exon, feats[c]["exon"], True)
        intr &= ~exon
        utr = np.zeros(L, bool)
        paint(utr, feats[c]["UTR"], True)
        cds = np.zeros(L, bool)
        paint(cds, feats[c]["CDS"], True)
        utr |= exon & ~cds        # exon parts not annotated CDS/UTR (rare) count as UTR
        utr &= ~cds
        intr &= ~utr & ~cds
        prim[intr] = P["intron"]
        prim[utr] = P["UTR"]
        prim[cds] = P["CDS"]
        fl[gene_bp] |= FL["gene_any"]
        fl[intr] |= FL["intron"]
        fl[utr] |= FL["UTR"]
        fl[cds] |= FL["CDS"]
        fl[gene_bp & te_bp] |= FL["TE_in_gene"]
        cb = np.zeros(L, bool)
        paint(cb, cns[c], True)
        fl[cb] |= FL["CNS"]
        # centromere estimate from CentC + CRM bp density
        cc = np.zeros(L, bool)
        for s, e, g, _ in tes[c]:
            if g == "centromeric_CentC_CRM":
                cc[s:e + 1] = True
        b500 = np.add.reduceat(cc, np.arange(0, L, 500_000))
        pk = int(np.argmax(b500)) * 500_000 + 250_000
        lo, hi = max(0, pk - 1_500_000), min(L, pk + 1_500_000)
        w = np.nonzero(cc[lo:hi])[0] + lo
        centre = int(w.mean())
        cen[c] = centre
        fl[max(0, centre - 1_000_000):centre + 1_000_000] |= FL["centromere_core_2Mb"]
        # per-1Mb density table
        for b in range(0, L, 1_000_000):
            dens_rows.append((c, b // 1_000_000, int(gene_bp[b:b + 1_000_000].sum()),
                              int(cds[b:b + 1_000_000].sum()), int(te_bp[b:b + 1_000_000].sum()),
                              min(L, b + 1_000_000) - b))
        # records
        lo_r, hi_r = panel.bounds[ci]
        p = panel.pos[lo_r:hi_r]
        primary[lo_r:hi_r] = prim[p]
        flags[lo_r:hi_r] = fl[p]
        rel = np.where(p < centre, (centre - p) / centre, (p - centre) / (clen[c] - centre))
        armbin[lo_r:hi_r] = np.clip(np.digitize(rel, ARM_BINS) - 1, 0, len(ARM_NAMES) - 1)
        mb[lo_r:hi_r] = ci * 1000 + p // 1_000_000
        print(c, "centre", centre, "records", hi_r - lo_r,
              {k: int((prim[p] == P[k]).sum()) for k in ("CDS", "intron", "LTR_Gypsy", "intergenic_nonTE")},
              flush=True)
    np.savez(OUT / "panel_feature_annotation.npz", primary=primary, flags=flags, armbin=armbin, mb=mb)
    (OUT / "feature_annotation_meta.json").write_text(json.dumps(
        {"PRIMARY": PRIMARY, "FLAG_NAMES": FLAG_NAMES, "ARM_NAMES": ARM_NAMES, "centromere_centre": cen,
         "sources": {"genes": GENE_GFF, "TE": str(TE_GFF), "CNS": str(CNS_GFF)}}, indent=1))
    with open(OUT / "mb_bin_density.tsv", "w") as fh:
        fh.write("chrom\tmb\tgene_bp\tcds_bp\tte_bp\tbin_bp\n")
        for r in dens_rows:
            fh.write("\t".join(map(str, r)) + "\n")
    # summary of record counts per primary category
    with open(OUT / "panel_records_per_category.tsv", "w") as fh:
        fh.write("category\tpanel_records\tfrac\n")
        cnt = np.bincount(primary, minlength=len(PRIMARY))
        for i, nme in enumerate(PRIMARY):
            fh.write(f"{nme}\t{cnt[i]}\t{cnt[i] / n:.5f}\n")
    print("done")


if __name__ == "__main__":
    main()
