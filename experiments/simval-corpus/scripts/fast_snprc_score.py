#!/usr/bin/env python
"""
Fast, exact re-implementation of the simval SNP+RefCall genotype score
(simval_eval_indel.py do_score: `sample bed-to-vcf` -> autosome filter ->
compare_gvcf_truth_diploid.compare_diploid(partial_credit=True,
snp_refcall_metrics=True)) that replaces the ~45 min/row bed-to-vcf + VCF
comparison with array lookups against one-time caches.

Why it can be exact: in compare_diploid the TRUTH side of every site
(TruthCursor.resolve at the panel record's position, then
gt_to_allele_multiset against the panel REF) never depends on the
prediction, and the IMPUTED side is fully determined by BedToVcf.kt's rule:
every panel record is emitted; a record gets GT (allele of parent1, allele of
parent2) -- each founder's first panel allele -- iff its POS lies in a BED
interval [start+1, end] (closed, 1-based), else it has no call and is not
compared. So:

  build-panel   once: per autosomal panel record, chrom/pos, allele class of
                each allele index (REF/SNP/INS/DEL), and every founder's
                allele index (-1 = no call).
  build-truth   once per truth gVCF (haplotype): per panel record, the truth
                allele resolved with the comparator's own TruthCursor +
                gt_to_allele_multiset -- coded as a panel allele index, or a
                novel allele (hash + class), or no-info / missing GT.
  score         per row, seconds: BED intervals -> predicted founder pair per
                record -> match / partial credit / SNP+RefCall class, with
                the same multiset and classify_alleles rules.

`score` is validated against existing full-pipeline score_result.json rows
(`validate`), which must agree to floating-point precision before any
number from this tool is trusted.
"""
import argparse
import glob
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS))
import compare_gvcf_truth_diploid as cgtd  # noqa: E402  (also puts vcf_eval on sys.path)
from compare_gvcf_truth_diploid import cgt, gt_to_allele_multiset  # noqa: E402

AUTOSOMES = [f"chr{i}" for i in range(1, 11)]
PANEL_VCF = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/maize_v2_rebuild/panel/panel_25founders_v2.vcf"
CACHE = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/fast_score_cache")

# allele classes, matching compare_gvcf_truth.classify_alleles
C_REF, C_SNP, C_INS, C_DEL = 0, 1, 2, 3
# truth codes (>= 0 are panel allele indices)
T_NOINFO, T_MISSING, T_NOVEL = -1, -3, -2


def allele_class(a, ref):
    if a == ref:
        return C_REF
    if a == cgt.SYMBOLIC_INS:
        return C_INS
    if a == cgt.SYMBOLIC_DEL:
        return C_DEL
    if len(a) > len(ref):
        return C_INS
    if len(a) < len(ref):
        return C_DEL
    return C_SNP


def build_panel(args):
    import pandas as pd
    CACHE.mkdir(parents=True, exist_ok=True)
    header = subprocess.run(["bcftools", "query", "-l", PANEL_VCF], capture_output=True,
                            text=True, check=True).stdout.split()
    K = len(header)
    MAXA = 8
    fmt = r"%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n"
    proc = subprocess.Popen(["bcftools", "query", "-f", fmt, PANEL_VCF], stdout=subprocess.PIPE)
    ci = {c: i for i, c in enumerate(AUTOSOMES)}
    chrom_p, pos_p, fa_p, cls_p = [], [], [], []
    t0 = time.time()
    n = 0
    with open(CACHE / "panel_refalt.tsv", "w") as refalt:
        for df in pd.read_csv(proc.stdout, sep="\t", header=None, dtype=str,
                              chunksize=5_000_000, keep_default_na=False):
            df = df[df[0].isin(ci)]
            if df.empty:
                continue
            ref, alt = df[2], df[3]
            alts = alt.where(alt != ".", "").str.split(",", expand=True)
            if alts.shape[1] + 1 > MAXA:
                raise SystemExit(f"a record has {alts.shape[1] + 1} alleles > {MAXA}")
            rlen = ref.str.len().to_numpy()
            cls = np.zeros((len(df), MAXA), np.int8)             # index 0 = REF
            for j in range(alts.shape[1]):
                a = alts[j].fillna("")
                al = a.str.len().to_numpy()
                c = np.where(al > rlen, C_INS, np.where(al < rlen, C_DEL, C_SNP))
                c = np.where(a == cgt.SYMBOLIC_INS, C_INS, np.where(a == cgt.SYMBOLIC_DEL, C_DEL, c))
                c = np.where((a == ref).to_numpy(), C_REF, c)
                c = np.where((a == "").to_numpy(), C_REF, c)       # padding (no such allele)
                cls[:, j + 1] = c
            g = df.iloc[:, 4:4 + K].to_numpy()
            fa = np.where(g == ".", "-1", g).astype(np.int8)
            chrom_p.append(df[0].map(ci).to_numpy(np.int8)); pos_p.append(df[1].to_numpy(np.int64))
            fa_p.append(fa); cls_p.append(cls)
            df[[0, 1, 2, 3]].to_csv(refalt, sep="\t", header=False, index=False)
            n += len(df)
            print(f"  {n:,} autosomal records ({time.time() - t0:.0f}s)", flush=True)
    proc.wait()
    np.save(CACHE / "panel_chrom.npy", np.concatenate(chrom_p))
    np.save(CACHE / "panel_pos.npy", np.concatenate(pos_p))
    np.save(CACHE / "panel_founder_allele.npy", np.concatenate(fa_p))
    np.save(CACHE / "panel_allele_class.npy", np.concatenate(cls_p))
    (CACHE / "panel_founders.json").write_text(json.dumps(header))
    print(f"panel: {n:,} autosomal records, {K} founders, {time.time() - t0:.0f}s")


def truth_key(truth_path):
    p = Path(truth_path)
    return f"{p.parent.name}__{p.name.replace('.g.vcf.gz', '')}"


def build_truth(args):
    """One haplotype truth gVCF -> per-panel-record truth allele code, using the
    comparator's own TruthCursor.resolve + gt_to_allele_multiset."""
    out = CACHE / f"truth_{truth_key(args.truth)}.npz"
    if out.exists():
        print(f"exists: {out}")
        return
    import tempfile
    t0 = time.time()
    n = sum(1 for _ in open(CACHE / "panel_refalt.tsv"))
    code = np.full(n, T_NOINFO, np.int8)
    ncls = np.zeros(n, np.int8)
    nhash = np.zeros(n, np.int64)
    with tempfile.TemporaryDirectory(dir=str(CACHE)) as td:
        contig_files = cgt.partition_truth_by_contig(str(args.truth), Path(td))
        cursors = {}
        for i, line in enumerate(open(CACHE / "panel_refalt.tsv")):
            chrom, pos, ref, alt = line.rstrip("\n").split("\t")
            r = cgtd._get_cursor(chrom, contig_files, cursors).resolve(chrom, int(pos), 1)
            if r is None:
                continue
            a = gt_to_allele_multiset(ref, r[0], r[1])
            if a is None:
                code[i] = T_MISSING
                continue
            allele = a[0]
            alleles = [ref] + ([] if alt in (".", "") else alt.split(","))
            if allele in alleles:
                code[i] = alleles.index(allele)
            else:
                code[i] = T_NOVEL
                ncls[i] = allele_class(allele, ref)
                nhash[i] = int.from_bytes(hashlib.blake2b(allele.encode(), digest_size=8).digest(),
                                          "little", signed=True)
            if i % 20_000_000 == 0 and i:
                print(f"  {truth_key(args.truth)}: {i:,}/{n:,} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(out, code=code, ncls=ncls, nhash=nhash)
    print(f"truth {truth_key(args.truth)}: {time.time() - t0:.0f}s -> {out}")


class Panel:
    def __init__(self):
        self.chrom = np.load(CACHE / "panel_chrom.npy")
        self.pos = np.load(CACHE / "panel_pos.npy")
        self.fa = np.load(CACHE / "panel_founder_allele.npy", mmap_mode="r")
        self.acls = np.load(CACHE / "panel_allele_class.npy", mmap_mode="r")
        self.founders = json.loads((CACHE / "panel_founders.json").read_text())
        self.fidx = {f: i for i, f in enumerate(self.founders)}
        self.bounds = {c: (np.searchsorted(self.chrom, c, "left"),
                           np.searchsorted(self.chrom, c, "right")) for c in range(len(AUTOSOMES))}


def pair_from_bed(panel, bed_dir):
    """Per panel record: predicted (founder1, founder2) indices, -1 where no
    BED interval covers the record (BedToVcf: no genotype)."""
    n = panel.pos.size
    f1 = np.full(n, -1, np.int16)
    f2 = np.full(n, -1, np.int16)
    for bed in glob.glob(str(Path(bed_dir) / "*.bed")):
        rows = [l.rstrip("\n").split("\t") for l in open(bed)
                if l.strip() and not l.startswith("chrom\tstart\tend\tparent1")]
        for r in rows:
            c = AUTOSOMES.index(r[0]) if r[0] in AUTOSOMES else None
            if c is None:
                continue
            lo, hi = panel.bounds[c]
            a = lo + np.searchsorted(panel.pos[lo:hi], int(r[1]) + 1, "left")
            b = lo + np.searchsorted(panel.pos[lo:hi], int(r[2]), "right")
            if b > a:
                g1 = r[3]
                g2 = r[4] if len(r) >= 5 else g1
                f1[a:b] = panel.fidx.get(g1, -2)
                f2[a:b] = panel.fidx.get(g2, -2)
    return f1, f2


def classify_pair(c1, c2):
    """classify_alleles over a 2-allele multiset given per-allele classes:
    0 HOMREF, 1 SNP, 2 INS, 3 DEL, 4 HET_MIXED."""
    nr1, nr2 = c1 != C_REF, c2 != C_REF
    out = np.zeros(c1.shape, np.int8)
    only1 = nr1 & ~nr2
    only2 = nr2 & ~nr1
    both = nr1 & nr2
    out[only1] = c1[only1]
    out[only2] = c2[only2]
    out[both] = np.where(c1[both] == c2[both], c1[both], 4)
    return out


def score_arrays(panel, t1, t2, f1, f2):
    n = panel.pos.size
    idx = np.arange(n)
    has_pred = (f1 >= 0) & (f2 >= 0)
    i1 = np.full(n, -1, np.int16)
    i2 = np.full(n, -1, np.int16)
    i1[has_pred] = panel.fa[idx[has_pred], f1[has_pred]]
    i2[has_pred] = panel.fa[idx[has_pred], f2[has_pred]]
    imputed_ok = has_pred & (i1 >= 0) & (i2 >= 0)
    truth_ok = (t1["code"] != T_NOINFO) & (t2["code"] != T_NOINFO) & \
               (t1["code"] != T_MISSING) & (t2["code"] != T_MISSING)
    m = truth_ok & imputed_ok
    # comparable allele identities: panel index >= 0, novel -> unique negative id per hash
    def ident(t, sel):
        c = t["code"][sel].astype(np.int64)
        return np.where(c == T_NOVEL, -(np.abs(t["nhash"][sel]) % (1 << 60)) - 10, c)
    a = ident(t1, m); b = ident(t2, m)
    x = i1[m].astype(np.int64); y = i2[m].astype(np.int64)
    ts_lo, ts_hi = np.minimum(a, b), np.maximum(a, b)
    is_lo, is_hi = np.minimum(x, y), np.maximum(x, y)
    match = (ts_lo == is_lo) & (ts_hi == is_hi)
    same_t = a == b
    inter = np.where(same_t, (x == a).astype(np.int64) + (y == a),
                     ((x == a) | (y == a)).astype(np.int64) + ((x == b) | (y == b)))
    partial = inter / 2.0
    acls = panel.acls
    rows = idx[m]
    def tcls(t, sel_rows):
        c = t["code"][sel_rows]
        return np.where(c == T_NOVEL, t["ncls"][sel_rows], acls[sel_rows, np.clip(c, 0, None)])
    t_cls = classify_pair(tcls(t1, rows), tcls(t2, rows))
    i_cls = classify_pair(acls[rows, i1[m]], acls[rows, i2[m]])
    snprc = (t_cls <= 1) & (i_cls <= 1)
    compared = int(m.sum())
    # compare_diploid's class_breakdown: classes by the TRUTH multiset
    classes = {}
    for ci, cname in enumerate(("HOMREF", "SNP", "INS", "DEL", "HET_MIXED")):
        sel = t_cls == ci
        classes[f"class_total_{cname}"] = int(sel.sum())
        classes[f"class_mismatch_{cname}"] = int((sel & ~match).sum())
    # Event view of deletion spans (per-site scoring weights a 50kb replacement
    # block ~thousands of times): each maximal run of consecutive compared sites
    # whose TRUTH class is DEL is one event; likewise each run where the IMPUTED
    # class is DEL but truth is not is one false-deletion event.
    chrom = panel.chrom[rows]

    def runs(flag):
        start = flag & ~np.concatenate([[False], flag[:-1] & (chrom[1:] == chrom[:-1])])
        rid = np.cumsum(start) - 1
        return rid[flag], int(start.sum())

    tdel = t_cls == 3
    rid, n_del = runs(tdel)
    if n_del:
        tot = np.bincount(rid, minlength=n_del)
        ok = np.bincount(rid, weights=match[tdel].astype(float), minlength=n_del)
        del_strict, del_frac = float((ok == tot).mean()), float((ok / tot).mean())
    else:
        del_strict = del_frac = None
    _, n_false_del = runs((i_cls == 3) & ~tdel)
    nd = ~tdel
    events = {"del_events": n_del, "del_event_strict_acc": del_strict,
              "del_event_mean_frac_correct": del_frac, "false_del_events": n_false_del,
              "error_rate_outside_truth_del": 1 - match[nd].sum() / nd.sum() if nd.any() else None}
    return {
        **classes, **events,
        "compared_sites": compared,
        "gt_allele_matches": int(match.sum()),
        "partial_credit_sum": float(partial.sum()),
        "snprc_compared_sites": int(snprc.sum()),
        "snprc_gt_allele_matches": int(match[snprc].sum()),
        "snprc_partial_credit_sum": float(partial[snprc].sum()),
        "error_rate": 1 - match.sum() / compared if compared else None,
        "partial_error_rate": 1 - partial.sum() / compared if compared else None,
        "snprc_error_rate": 1 - match[snprc].sum() / snprc.sum() if snprc.sum() else None,
    }


def load_truth(path):
    z = np.load(CACHE / f"truth_{truth_key(path)}.npz")
    return {k: z[k] for k in z.files}


def score(args):
    panel = Panel()
    t0 = time.time()
    f1, f2 = pair_from_bed(panel, args.bed_dir)
    res = score_arrays(panel, load_truth(args.truth_h1), load_truth(args.truth_h2), f1, f2)
    res["fast_score_s"] = round(time.time() - t0, 1)
    print(json.dumps(res, indent=1))
    return res


def validate(args):
    """Re-score existing full-pipeline rows and compare to their score_result.json."""
    manifest = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv"
    import csv
    truth = {}
    for r in csv.DictReader(open(manifest), delimiter="\t"):
        truth[(r["dataset_id"], r["individual"], r["coverage"])] = (r["truth_h1"], r["truth_h2"])
    panel = Panel()
    worst = 0.0
    for d in args.row_dirs:
        d = Path(d)
        ref = json.loads((d / "score_result.json").read_text())
        _tag, ds, ind, cov = d.name.split("__")
        h1, h2 = truth[(ds, ind, cov.rstrip("x"))]
        t0 = time.time()
        f1, f2 = pair_from_bed(panel, d / "bed")
        res = score_arrays(panel, load_truth(h1), load_truth(h2), f1, f2)
        diffs = {k: (res[k], ref.get(k)) for k in
                 ("compared_sites", "error_rate", "partial_error_rate",
                  "snprc_compared_sites", "snprc_error_rate")}
        dmax = max(abs(a - b) for a, b in diffs.values() if a is not None and b is not None)
        worst = max(worst, dmax)
        print(f"{d.name:<62} {time.time() - t0:5.1f}s  max|diff|={dmax:.2e}  "
              f"err fast={res['error_rate']:.6f} full={ref['error_rate']:.6f}  "
              f"sites fast={res['compared_sites']:,} full={ref['compared_sites']:,}", flush=True)
    print(f"WORST max|diff| over {len(args.row_dirs)} rows: {worst:.2e}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-panel")
    bt = sub.add_parser("build-truth"); bt.add_argument("--truth", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--bed-dir", required=True)
    sc.add_argument("--truth-h1", required=True)
    sc.add_argument("--truth-h2", required=True)
    va = sub.add_parser("validate"); va.add_argument("row_dirs", nargs="+")
    args = ap.parse_args()
    {"build-panel": build_panel, "build-truth": build_truth, "score": score,
     "validate": validate}[args.cmd](args)


if __name__ == "__main__":
    main()
