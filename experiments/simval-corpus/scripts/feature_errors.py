#!/usr/bin/env python
"""Per-genomic-feature SNP+RefCall (and all-sites) error counts for several models + the panel ceiling.

Uses per_record(), an exact per-record re-expression of fast_snprc_score.score_arrays (same masks,
same allele identities, same classify_pair rule), and validates it against score_arrays on the
first --validate rows of each process (sums must match exactly, else abort).

Feature labels come from feature_annotate.py (panel_feature_annotation.npz). For every
(depth, dataset, individual, model) it writes counts per feature key:
  primary:<cat>   mutually exclusive category (CDS > UTR > intron > TE > flank > intergenic)
  flag:<name>     non-exclusive membership
  arm:<bin>       relative distance from centromere
  all             whole genome
columns: snprc_sites, snprc_err, all_sites, all_err, elig (truth comparable and truth class
REF/SNP), uncalled (elig records with no predicted genotype).
Plus a per-1Mb-bin table (SNP+RC sites/errors) for the density correlation.

Ceiling: panel_ceiling.best_path(lam=50) per truth haplotype, cached as .npy under
scratch/annotation/ceiling_paths_lam50/.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fast_snprc_score as fss  # noqa: E402
from panel_ceiling import best_path  # noqa: E402

MANIFEST = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv"
SCR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch")
OUT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/genomic_feature_errors")
CEIL = SCR / "annotation" / "ceiling_paths_lam50"
TAGS = {"0.1": {"v6cfullfe_homo": "v6cfullfe_homo", "diploid_affinity": "diploid_affinity_s200",
                "ternary_baseline": "s200_ternary_routed", "old_v6": "v6_B"},
        "1.0": {"v6cfullfe_homo": "v6cfullfe_homo_1x", "diploid_affinity": "diploid_affinity_s200_1x",
                "ternary_baseline": "tern_1x", "old_v6": "v6_B_1x"}}
SCOPE = {"IDX": "idx", "OUT": "out", "MIX": "mix"}


def per_record(panel, t1, t2, f1, f2):
    """Exact per-record version of fss.score_arrays. Returns
    rows   : indices of compared records (truth_ok & imputed_ok)
    match  : bool per compared record (genotype multiset equal)
    snprc  : bool per compared record (truth and imputed classes both HOMREF/SNP)"""
    n = panel.pos.size
    idx = np.arange(n)
    has_pred = (f1 >= 0) & (f2 >= 0)
    i1 = np.full(n, -1, np.int16)
    i2 = np.full(n, -1, np.int16)
    i1[has_pred] = panel.fa[idx[has_pred], f1[has_pred]]
    i2[has_pred] = panel.fa[idx[has_pred], f2[has_pred]]
    imputed_ok = has_pred & (i1 >= 0) & (i2 >= 0)
    truth_ok = (t1["code"] != fss.T_NOINFO) & (t2["code"] != fss.T_NOINFO) & \
               (t1["code"] != fss.T_MISSING) & (t2["code"] != fss.T_MISSING)
    m = truth_ok & imputed_ok

    def ident(t, sel):
        c = t["code"][sel].astype(np.int64)
        return np.where(c == fss.T_NOVEL, -(np.abs(t["nhash"][sel]) % (1 << 60)) - 10, c)
    a = ident(t1, m); b = ident(t2, m)
    x = i1[m].astype(np.int64); y = i2[m].astype(np.int64)
    match = (np.minimum(a, b) == np.minimum(x, y)) & (np.maximum(a, b) == np.maximum(x, y))
    rows = idx[m]
    acls = panel.acls

    def tcls(t, r):
        c = t["code"][r]
        return np.where(c == fss.T_NOVEL, t["ncls"][r], acls[r, np.clip(c, 0, None)])
    t_cls = fss.classify_pair(tcls(t1, rows), tcls(t2, rows))
    i_cls = fss.classify_pair(acls[rows, i1[m]], acls[rows, i2[m]])
    snprc = (t_cls <= 1) & (i_cls <= 1)
    return rows, match, snprc, has_pred & (i1 >= 0) & (i2 >= 0)


def truth_elig(panel, t1, t2):
    """Records where truth is comparable and truth class is HOMREF/SNP (would enter SNP+RC if the
    imputed genotype is also REF/SNP) -- denominator for the 'uncalled' fraction."""
    ok = (t1["code"] != fss.T_NOINFO) & (t2["code"] != fss.T_NOINFO) & \
         (t1["code"] != fss.T_MISSING) & (t2["code"] != fss.T_MISSING)
    rows = np.nonzero(ok)[0]
    acls = panel.acls

    def tcls(t, r):
        c = t["code"][r]
        return np.where(c == fss.T_NOVEL, t["ncls"][r], acls[r, np.clip(c, 0, None)])
    tc = fss.classify_pair(tcls(t1, rows), tcls(t2, rows))
    e = np.zeros(panel.pos.size, bool)
    e[rows[tc <= 1]] = True
    return e


def ceiling_path(panel, tp, tcode):
    CEIL.mkdir(parents=True, exist_ok=True)
    f = CEIL / f"{fss.truth_key(tp)}.npy"
    if f.exists():
        return np.load(f)
    p, sw = best_path(panel, tcode, 50.0)
    tmp = CEIL / f".{fss.truth_key(tp)}.{time.time_ns()}.npy"
    np.save(tmp, p)
    tmp.rename(f)
    (CEIL / f"{fss.truth_key(tp)}.switches").write_text(str(sw))
    return p


def bed_dir(tag, ds, ind, dep):
    d = SCR / f"{tag}_{SCOPE[ds[:3]]}_fast" / f"{tag}__{ds}__{ind}__{dep}x" / "bed"
    return d if d.is_dir() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--part", required=True)
    ap.add_argument("--validate", type=int, default=2, help="validate per_record on first N model rows")
    a = ap.parse_args()
    ann = np.load(OUT / "panel_feature_annotation.npz")
    meta = json.loads((OUT / "feature_annotation_meta.json").read_text())
    primary, flags, armbin, mbid = ann["primary"], ann["flags"], ann["armbin"], ann["mb"]
    PRIM, FLAGS, ARMS = meta["PRIMARY"], meta["FLAG_NAMES"], meta["ARM_NAMES"]
    nP, nA = len(PRIM), len(ARMS)
    flagbits = [((flags >> np.uint32(i)) & np.uint32(1)).astype(bool) for i in range(len(FLAGS))]
    mb_u, mb_inv = np.unique(mbid, return_inverse=True)

    panel = fss.Panel()
    panel.fa = np.asarray(panel.fa)
    panel.acls = np.asarray(panel.acls)
    rows_m = [r for r in csv.DictReader(open(MANIFEST), delimiter="\t")
              if r["coverage"] in TAGS and r["dataset_id"] in a.classes]
    rows_m.sort(key=lambda r: (a.classes.index(r["dataset_id"]), r["individual"], r["coverage"]))
    out_f = OUT / f"counts_{a.part}.tsv"
    mb_f = OUT / f"mbbins_{a.part}.tsv"
    val_f = OUT / f"validation_{a.part}.tsv"
    done = set()
    if out_f.exists():
        for r in csv.DictReader(open(out_f), delimiter="\t"):
            done.add((r["depth"], r["dataset"], r["individual"], r["model"]))
    fo = open(out_f, "a")
    fm = open(mb_f, "a")
    fv = open(val_f, "a")
    if not done:
        fo.write("depth\tdataset\tindividual\tmodel\tkey\tsnprc_sites\tsnprc_err\tall_sites\tall_err\telig\tuncalled\n")
        fm.write("depth\tdataset\tindividual\tmodel\tmb_id\tsnprc_sites\tsnprc_err\n")
        fv.write("depth\tdataset\tindividual\tmodel\tsnprc_sites_perrec\tsnprc_sites_score_arrays\t"
                 "snprc_err_perrec\tsnprc_err_score_arrays\tall_sites_perrec\tall_sites_score_arrays\t"
                 "all_err_perrec\tall_err_score_arrays\n")
    nval = 0
    cur_truth = None
    for r in rows_m:
        dep, ds, ind = r["coverage"], r["dataset_id"], r["individual"]
        models = {m: bed_dir(t, ds, ind, dep) for m, t in TAGS[dep].items()}
        if not all(models.values()):
            print("SKIP missing", dep, ds, ind, [m for m, d in models.items() if not d], flush=True)
            continue
        if all((dep, ds, ind, m) in done for m in list(models) + ["ceiling"]):
            continue
        t0 = time.time()
        if cur_truth != (r["truth_h1"], r["truth_h2"]):
            t1, t2 = fss.load_truth(r["truth_h1"]), fss.load_truth(r["truth_h2"])
            elig = truth_elig(panel, t1, t2)
            cur_truth = (r["truth_h1"], r["truth_h2"])
            cpair = None
        preds = {m: (lambda d=d: fss.pair_from_bed(panel, d)) for m, d in models.items()}

        def ceil_pair():
            return (ceiling_path(panel, r["truth_h1"], t1["code"]),
                    ceiling_path(panel, r["truth_h2"], t2["code"]))
        preds["ceiling"] = ceil_pair
        for m, getp in preds.items():
            if (dep, ds, ind, m) in done:
                continue
            if m == "ceiling":
                if cpair is None:
                    cpair = getp()
                f1, f2 = cpair
            else:
                f1, f2 = getp()
            rows, match, snprc, called = per_record(panel, t1, t2, f1, f2)
            err = ~match
            if nval < a.validate:
                ref = fss.score_arrays(panel, t1, t2, f1, f2)
                v = (int(snprc.sum()), ref["snprc_compared_sites"],
                     1 - match[snprc].sum() / snprc.sum(), ref["snprc_error_rate"],
                     int(rows.size), ref["compared_sites"], 1 - match.sum() / rows.size, ref["error_rate"])
                fv.write(f"{dep}\t{ds}\t{ind}\t{m}\t" + "\t".join(map(str, v)) + "\n"); fv.flush()
                if v[0] != v[1] or v[2] != v[3] or v[4] != v[5] or v[6] != v[7]:
                    raise SystemExit(f"VALIDATION FAILED {dep} {ds} {ind} {m}: {v}")
                print("validated", dep, ds, ind, m, v, flush=True)
                nval += 1
            # SNP+RC
            sr = rows[snprc]; se = err[snprc]
            out = []

            def counts(lab, nlab, rr, ee):
                return (np.bincount(lab[rr], minlength=nlab), np.bincount(lab[rr], weights=ee, minlength=nlab))
            ps_s, ps_e = counts(primary, nP, sr, se)
            pa_s, pa_e = counts(primary, nP, rows, err)
            el = elig
            unc = elig & ~called
            pel = np.bincount(primary[el], minlength=nP)
            pun = np.bincount(primary[unc], minlength=nP)
            for i, nm in enumerate(PRIM):
                out.append((f"primary:{nm}", ps_s[i], ps_e[i], pa_s[i], pa_e[i], pel[i], pun[i]))
            as_s, as_e = counts(armbin, nA, sr, se)
            aa_s, aa_e = counts(armbin, nA, rows, err)
            ael = np.bincount(armbin[el], minlength=nA)
            aun = np.bincount(armbin[unc], minlength=nA)
            for i, nm in enumerate(ARMS):
                out.append((f"arm:{nm}", as_s[i], as_e[i], aa_s[i], aa_e[i], ael[i], aun[i]))
            for i, nm in enumerate(FLAGS):
                fb = flagbits[i]
                s_in = fb[sr]; a_in = fb[rows]
                out.append((f"flag:{nm}", int(s_in.sum()), int(se[s_in].sum()), int(a_in.sum()),
                            int(err[a_in].sum()), int((fb & el).sum()), int((fb & unc).sum())))
            out.append(("all", sr.size, int(se.sum()), rows.size, int(err.sum()), int(el.sum()), int(unc.sum())))
            for o in out:
                fo.write(f"{dep}\t{ds}\t{ind}\t{m}\t{o[0]}\t" + "\t".join(str(int(x)) for x in o[1:]) + "\n")
            ms = np.bincount(mb_inv[sr], minlength=mb_u.size)
            me = np.bincount(mb_inv[sr], weights=se, minlength=mb_u.size)
            for j in np.nonzero(ms)[0]:
                fm.write(f"{dep}\t{ds}\t{ind}\t{m}\t{mb_u[j]}\t{ms[j]}\t{int(me[j])}\n")
            fo.flush(); fm.flush()
        print(f"done {dep} {ds} {ind} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
