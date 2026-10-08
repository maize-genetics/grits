#!/usr/bin/env python
"""Imputation report card: one set of numbers per model, testing different parts of the problem.

For every sample of the simulated-validation corpus (8 classes x 5 samples, default depths 0.1x and
1x) it reads the model's imputed BEDs and result JSONs (no re-decoding) and measures:

  accuracy      SNP+RefCall error (lead), all-sites error, error on truly het vs homozygous sites
  coverage      % genome inside BED intervals, % of reference SNP+RC sites called, uncalled stretches
  zygosity      predicted vs true het fraction, per-site het precision / recall, router call
  recombination founder-pair changes vs the reference path, breakpoint precision / recall within
                10 kb and 100 kb, median segment length
  frequency     founder share across the genome vs the reference (total variation distance),
                error by panel allele frequency (singleton / rare / low / common)
  affinity      read-based founder affinity (the models' prior) vs reference founder share
                (Spearman), true parents in the affinity top-2 (in-panel parents only)
  headroom      SNP+RC error minus the panel ceiling's
  distance      error by distance to the nearest reference breakpoint
  cost          decode wall-clock seconds
  genewindow    (optional, minutes) VCF-free: imputed haplotype stitched over 1000 gene windows (gene
                +-5 kb), aligned to the truth assembly with minimap2; errors per kb (mismatches + indels +
                unaligned bases), the best-single-founder panel ceiling, % exact windows, trimmed (t50)
                errors, and the founder-choice decomposition. See eval_genewindow.py.

Reference path: the panel ceiling (panel_ceiling.best_path, exact haploid Viterbi per truth haplotype,
mismatch cost 1, switch cost --lam, default 50, ~7 switches per 512-row window out of panel). In-panel
it is the true founder path (0% error); out of panel no true path exists, so it stands in as the
best-possible panel path. Paths are cached under results/eval_suite/_reference/.

Sections: `run --sections core` (default; every section above except genewindow, ~1-2 min) and/or
`genewindow` (~5-15 min per model at <= 8 threads; model-independent pieces cached under
_reference/genewindow/, see eval_genewindow.py).

Usage:
  eval_suite.py run --model NAME --tags 0.1=TAG 1.0=TAG [--sections core genewindow] [--workers 4]
  eval_suite.py run --model NAME --sections genewindow      # tags taken from the registry
  eval_suite.py summary            # class tables for every registered model
  eval_suite.py dashboard          # results/eval_suite/dashboard.html (primary view) + report.html
"""
import argparse
import csv
import glob
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fast_snprc_score as fss  # noqa: E402
from panel_ceiling import best_path  # noqa: E402

WORK = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
MANIFEST = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv"
OUT = WORK / "results/eval_suite"
REF = OUT / "_reference"
CLASSES = ["IDX-INBRED", "IDX-HYB", "IDX-RIL2", "MIX-HYB", "MIX-RIL2", "OUT-INBRED", "OUT-HYB", "OUT-RIL2"]
SCOPE = {"IDX": "idx", "MIX": "mix", "OUT": "out"}
GENOME_BP = 2.130e9
DIST_BINS = [(0, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, np.inf)]
AF_BINS = [("singleton", 1, 1), ("rare_2-3", 2, 3), ("low_4-6", 4, 6), ("common_7+", 7, 99)]

_P = None


def panel():
    global _P
    if _P is None:
        _P = fss.Panel()
        _P.fa = np.asarray(_P.fa)
        _P.acls = np.asarray(_P.acls)
        nonref = (_P.fa > 0).sum(1)
        _P.nalt = nonref.astype(np.int16)            # founders carrying a non-reference allele
        _P.offset = np.zeros(_P.pos.size, np.int64)  # genome-wide coordinate for cross-chrom sorting
        for c, (lo, hi) in _P.bounds.items():
            _P.offset[lo:hi] = c * 10 ** 10
    return _P


# ---------------------------------------------------------------- per-record scoring
def per_record(P, t1, t2, f1, f2):
    """score_arrays' comparison, returned per record: rows (compared records), match, snprc mask,
    truth_het (truth haplotype alleles differ). Same rules as fast_snprc_score.score_arrays."""
    n = P.pos.size
    idx = np.arange(n)
    has = (f1 >= 0) & (f2 >= 0)
    i1 = np.full(n, -1, np.int16); i2 = np.full(n, -1, np.int16)
    i1[has] = P.fa[idx[has], f1[has]]; i2[has] = P.fa[idx[has], f2[has]]
    ok = (t1["code"] != fss.T_NOINFO) & (t2["code"] != fss.T_NOINFO) & \
         (t1["code"] != fss.T_MISSING) & (t2["code"] != fss.T_MISSING)
    m = ok & has & (i1 >= 0) & (i2 >= 0)

    def ident(t):
        c = t["code"][m].astype(np.int64)
        return np.where(c == fss.T_NOVEL, -(np.abs(t["nhash"][m]) % (1 << 60)) - 10, c)
    a, b = ident(t1), ident(t2)
    x, y = i1[m].astype(np.int64), i2[m].astype(np.int64)
    match = (np.minimum(a, b) == np.minimum(x, y)) & (np.maximum(a, b) == np.maximum(x, y))
    rows = idx[m]

    def tcls(t):
        c = t["code"][rows]
        return np.where(c == fss.T_NOVEL, t["ncls"][rows], P.acls[rows, np.clip(c, 0, None)])
    t_cls = fss.classify_pair(tcls(t1), tcls(t2))
    i_cls = fss.classify_pair(P.acls[rows, i1[m]], P.acls[rows, i2[m]])
    return dict(rows=rows, match=match, snprc=(t_cls <= 1) & (i_cls <= 1), truth_het=a != b, pred_het=x != y)


def rate(match, sel):
    n = int(sel.sum())
    return (1 - match[sel].sum() / n) if n else None, n


# ---------------------------------------------------------------- reference paths
def ref_path(P, truth_path, lam):
    REF.mkdir(parents=True, exist_ok=True)
    f = REF / f"ceiling_lam{lam:g}_{fss.truth_key(truth_path)}.npz"
    if f.exists():
        z = np.load(f)
        return z["path"].astype(np.int16), int(z["switches"])
    t = fss.load_truth(truth_path)
    p, s = best_path(P, t["code"], lam)
    np.savez(f, path=p.astype(np.int8), switches=s)
    return p, s


def pair_changes(P, f1, f2, called):
    """Records (indices) where the unordered founder pair differs from the previous called record
    on the same chromosome."""
    ci = np.flatnonzero(called)
    lo, hi = np.minimum(f1[ci], f2[ci]), np.maximum(f1[ci], f2[ci])
    same = P.chrom[ci[1:]] == P.chrom[ci[:-1]]
    ch = same & ((lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1]))
    return ci[1:][ch]


def bp_match(P, a_idx, b_idx, d):
    """Fraction of breakpoints in a within d bp of a breakpoint in b (same chromosome)."""
    if len(a_idx) == 0:
        return None
    if len(b_idx) == 0:
        return 0.0
    ga = P.offset[a_idx] + P.pos[a_idx]
    gb = np.sort(P.offset[b_idx] + P.pos[b_idx])
    j = np.searchsorted(gb, ga)
    left = np.abs(ga - gb[np.clip(j - 1, 0, len(gb) - 1)])
    right = np.abs(gb[np.clip(j, 0, len(gb) - 1)] - ga)
    return float(np.mean(np.minimum(left, right) <= d))


def median_segment_mb(P, br_idx, called):
    ci = np.flatnonzero(called)
    if len(ci) == 0:
        return None
    lens = []
    for c, (lo, hi) in P.bounds.items():
        cc = ci[(ci >= lo) & (ci < hi)]
        if len(cc) == 0:
            continue
        b = br_idx[(br_idx >= lo) & (br_idx < hi)]
        edges = np.concatenate([[P.pos[cc[0]]], P.pos[b], [P.pos[cc[-1]]]])
        lens.extend(np.diff(edges))
    return float(np.median(lens)) / 1e6 if lens else None


def founder_share(f1, f2, called, K):
    v = np.concatenate([f1[called], f2[called]])
    v = v[v >= 0]
    return np.bincount(v, minlength=K)[:K] / max(len(v), 1)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def bed_span(bed_dir):
    t = 0
    for f in glob.glob(str(Path(bed_dir) / "*.bed")):
        for l in open(f):
            if not l.startswith("chrom"):
                p = l.split("\t")
                t += int(p[2]) - int(p[1])
    return t / GENOME_BP


def gaps(P, called):
    """Uncalled stretches: number >= 100 kb and longest (Mb), measured between called records."""
    n100, longest = 0, 0
    for c, (lo, hi) in P.bounds.items():
        cc = np.flatnonzero(called[lo:hi])
        if len(cc) < 2:
            n100 += 1
            longest = max(longest, int(P.pos[hi - 1] - P.pos[lo]))
            continue
        g = np.diff(P.pos[lo:hi][cc])
        n100 += int((g >= 1e5).sum())
        longest = max(longest, int(g.max()))
    return n100, longest / 1e6


# ---------------------------------------------------------------- one sample
def sample_dirs(tag, ds, ind, dep):
    sc = SCOPE[ds.split("-")[0]]
    key = f"{tag}__{ds}__{ind}__{dep}x"
    return (WORK / "scratch" / f"{tag}_{sc}_fast" / key / "bed", WORK / "results" / f"{tag}_{sc}_fast" / f"{key}.json")


def eval_sample(r, tag, lam, validate=False):
    P = panel()
    ds, ind, dep = r["dataset_id"], r["individual"], r["coverage"]
    bed, js = sample_dirs(tag, ds, ind, dep)
    if not bed.is_dir() or not js.exists():
        return None
    t0 = time.time()
    res_json = json.loads(js.read_text())
    t1, t2 = fss.load_truth(r["truth_h1"]), fss.load_truth(r["truth_h2"])
    f1, f2 = fss.pair_from_bed(P, bed)
    K = len(P.founders)
    called = (f1 >= 0) & (f2 >= 0)
    pr = per_record(P, t1, t2, f1, f2)
    if validate:
        ref = fss.score_arrays(P, t1, t2, f1, f2)
        mine = 1 - pr["match"][pr["snprc"]].sum() / pr["snprc"].sum()
        assert pr["snprc"].sum() == ref["snprc_compared_sites"] and abs(mine - ref["snprc_error_rate"]) < 1e-12, \
            (pr["snprc"].sum(), ref["snprc_compared_sites"], mine, ref["snprc_error_rate"])
    (g1, s1), (g2, s2) = ref_path(P, r["truth_h1"], lam), ref_path(P, r["truth_h2"], lam)
    allc = np.ones(P.pos.size, bool)
    rr = per_record(P, t1, t2, g1, g2)
    m, sn, th = pr["match"], pr["snprc"], pr["truth_het"]
    rows = pr["rows"]
    out = dict(dataset=ds, individual=ind, depth=dep, tag=tag)
    # accuracy
    out["snprc_err"], out["snprc_sites"] = rate(m, sn)
    out["all_err"], _ = rate(m, np.ones_like(m))
    out["snprc_err_truth_het"], out["snprc_sites_truth_het"] = rate(m, sn & th)
    out["snprc_err_truth_hom"], _ = rate(m, sn & ~th)
    # coverage
    out["genome_covered"] = bed_span(bed)
    out["snprc_called_frac"] = out["snprc_sites"] / max(int(rr["snprc"].sum()), 1)
    out["gaps_100kb"], out["longest_gap_mb"] = gaps(P, called)
    # zygosity. Allele level, on compared records: does the call carry two different alleles, and
    # does the truth? Founder level: two different founders, vs the reference path's founder pair.
    ph = pr["pred_het"]
    out["pred_het_frac"] = float(ph.mean()) if len(ph) else None
    out["truth_het_frac"] = float(th.mean()) if len(th) else None
    out["het_precision"] = float(th[ph].mean()) if ph.any() else None
    out["het_recall"] = float(ph[th].mean()) if th.any() else None
    out["pred_founder_het_frac"] = float((f1[called] != f2[called]).mean()) if called.any() else None
    out["ref_founder_het_frac"] = float((g1 != g2).mean())
    hyb = re.search(r"hybrid=(True|False)", res_json.get("homo_scale_log") or "")
    out["router_hybrid"] = None if hyb is None else hyb.group(1) == "True"
    out["router_correct"] = None if hyb is None else out["router_hybrid"] == ds.endswith("-HYB")
    # recombination
    bp_p, bp_r = pair_changes(P, f1, f2, called), pair_changes(P, g1, g2, allc)
    out["pair_changes"], out["ref_pair_changes"] = len(bp_p), len(bp_r)
    out["ref_switches_per_hap"] = (s1 + s2) / 2
    for d, nm in ((1e4, "10kb"), (1e5, "100kb")):
        out[f"bp_precision_{nm}"] = bp_match(P, bp_p, bp_r, d)
        out[f"bp_recall_{nm}"] = bp_match(P, bp_r, bp_p, d)
    out["median_segment_mb"] = median_segment_mb(P, bp_p, called)
    out["ref_median_segment_mb"] = median_segment_mb(P, bp_r, allc)
    # frequency
    sp, sr = founder_share(f1, f2, called, K), founder_share(g1, g2, allc, K)
    out["founder_share_tvd"] = float(0.5 * np.abs(sp - sr).sum())
    na = P.nalt[rows]
    for nm, lo, hi in AF_BINS:
        out[f"snprc_err_af_{nm}"], out[f"snprc_sites_af_{nm}"] = rate(m, sn & (na >= lo) & (na <= hi))
    # affinity (read match rate from the model input, mapped to panel founders)
    src = Path(res_json.get("source_input") or "")
    out["affinity_spearman"] = out["parents_in_aff_top2"] = None
    if (src / "windowed_k25native_wcount.npy").exists():
        X = np.load(src / "windowed_k25native_wcount.npy", mmap_mode="r")
        gam = [l.rstrip("\n").split("\t") for l in open(src / "raw.npy.gametes.tsv")][1:]
        names = [g[-1] for g in gam if g and g[-1]]
        Kin = min(len(names), X.shape[2])
        aff = (np.asarray(X[:, :, :Kin]) == 1).reshape(-1, Kin).mean(0)
        a_full = np.zeros(K)
        for i, n in enumerate(names[:Kin]):
            if n in P.fidx:
                a_full[P.fidx[n]] = aff[i]
        out["affinity_spearman"] = spearman(a_full, sr)
        parents = [p for p in re.split(r"x", ind) if p in P.fidx] if "x" in ind else ([ind] if ind in P.fidx else [])
        if parents:
            top2 = {P.founders[i] for i in np.argsort(-a_full)[:2]}
            out["parents_in_aff_top2"] = float(np.mean([p in top2 for p in parents]))
        out["pred_parents_in_share_top2"] = (float(np.mean([p in {P.founders[i] for i in np.argsort(-sp)[:2]}
                                                            for p in parents])) if parents else None)
    # headroom
    out["ceiling_snprc_err"], _ = rate(rr["match"], rr["snprc"])
    out["excess_over_ceiling"] = (out["snprc_err"] - out["ceiling_snprc_err"]) if out["snprc_err"] is not None else None
    # distance to nearest reference breakpoint
    if len(bp_r):
        gb = np.sort(P.offset[bp_r] + P.pos[bp_r])
        gr = P.offset[rows] + P.pos[rows]
        j = np.searchsorted(gb, gr)
        dist = np.minimum(np.abs(gr - gb[np.clip(j - 1, 0, len(gb) - 1)]), np.abs(gb[np.clip(j, 0, len(gb) - 1)] - gr))
    else:
        dist = np.full(len(rows), np.inf)
    for lo, hi in DIST_BINS:
        nm = f"{int(lo / 1e3)}-{'inf' if hi == np.inf else int(hi / 1e3)}kb"
        out[f"snprc_err_dist_{nm}"], out[f"snprc_sites_dist_{nm}"] = rate(m, sn & (dist >= lo) & (dist < hi))
    # cost
    out["decode_seconds"] = res_json.get("wall_seconds")
    out["suite_seconds"] = round(time.time() - t0, 1)
    return out


def _work(args):
    rows, tag, lam, validate = args
    res = []
    for i, r in enumerate(rows):
        o = eval_sample(r, tag, lam, validate=validate and i == 0)
        if o is not None:
            res.append(o)
            print(f"  {tag} {o['dataset']} {o['individual']} {o['depth']}x snprc={100 * (o['snprc_err'] or 0):.2f}% "
                  f"covered={100 * o['genome_covered']:.0f}% ({o['suite_seconds']}s)", flush=True)
    return res


# ---------------------------------------------------------------- commands
def registry():
    f = OUT / "registry.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _ref_one(args):
    tp, lam = args
    t0 = time.time()
    _, s = ref_path(panel(), tp, lam)
    return tp, s, round(time.time() - t0)


def cmd_reference(a):
    """Compute and cache the reference (panel-ceiling) path of every truth haplotype once."""
    man = [r for r in csv.DictReader(open(MANIFEST), delimiter="\t") if r["dataset_id"] in a.classes]
    tps = sorted({r[k] for r in man for k in ("truth_h1", "truth_h2")})
    todo = [tp for tp in tps if not (REF / f"ceiling_lam{a.lam:g}_{fss.truth_key(tp)}.npz").exists()]
    print(f"{len(tps)} truth haplotypes, {len(todo)} to compute", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for tp, s, sec in ex.map(_ref_one, [(tp, a.lam) for tp in todo]):
            print(f"  {fss.truth_key(tp)}: {s:,} switches ({sec}s)", flush=True)


def cmd_run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    import fcntl
    with open(OUT / "registry.lock", "w") as lk:      # pipelines may finish at the same time
        fcntl.flock(lk, fcntl.LOCK_EX)
        reg = registry()
        if a.tags:
            tags = dict(t.split("=") for t in a.tags)
        elif a.model in reg:
            tags = reg[a.model]["tags"]
        else:
            sys.exit(f"--tags required: {a.model} is not registered")
        reg[a.model] = {**reg.get(a.model, {}), "tags": tags, "lam": a.lam,
                        "note": a.note or reg.get(a.model, {}).get("note", "")}
        (OUT / "registry.json").write_text(json.dumps(reg, indent=1))
    if "core" in a.sections:
        run_core(a, tags)
    if "genewindow" in a.sections:
        import eval_genewindow
        g = eval_genewindow.run_model(a.model, tags, a.classes, threads=a.gw_threads, jobs=a.gw_jobs)
        if g is not None:
            cols = ["depth", "level", "group"] + eval_genewindow.HEADLINE
            print(g[cols].round(3).to_string(index=False))


def run_core(a, tags):
    if a.reference:
        cmd_reference(a)
    man = [r for r in csv.DictReader(open(MANIFEST), delimiter="\t") if r["dataset_id"] in a.classes]
    jobs = []
    for dep, tag in tags.items():
        rows = [r for r in man if r["coverage"] == dep]
        for k in range(a.workers):
            jobs.append((rows[k::a.workers], tag, a.lam, k == 0))
    res = []
    with ProcessPoolExecutor(a.workers) as ex:
        for part in ex.map(_work, jobs):
            res.extend(part)
    d = OUT / a.model
    d.mkdir(parents=True, exist_ok=True)
    keys = list(res[0].keys())
    with open(d / "per_sample.tsv", "w") as f:
        f.write("\t".join(keys) + "\n")
        for o in sorted(res, key=lambda o: (o["depth"], CLASSES.index(o["dataset"]), o["individual"])):
            f.write("\t".join("" if o[k] is None else str(o[k]) for k in keys) + "\n")
    print(f"{len(res)} samples -> {d / 'per_sample.tsv'}")
    summarize(a.model)


def summarize(model):
    import pandas as pd
    df = pd.read_csv(OUT / model / "per_sample.tsv", sep="\t")
    for c in ("router_hybrid", "router_correct"):
        if c in df.columns:
            df[c] = df[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    num = [c for c in df.columns if c not in ("dataset", "individual", "depth", "tag") and df[c].dtype != object]
    g = df.groupby(["depth", "dataset"])[num].mean()
    g = g.reindex([(d, c) for d in sorted(df.depth.unique()) for c in CLASSES if (d, c) in g.index])
    g.to_csv(OUT / model / "class_summary.tsv", sep="\t", float_format="%.5g")
    return g


def cmd_summary(a):
    for model in registry():
        if (OUT / model / "per_sample.tsv").exists():
            g = summarize(model)
            print(f"\n== {model}")
            cols = ["snprc_err", "genome_covered", "excess_over_ceiling", "pred_het_frac", "truth_het_frac",
                    "pair_changes", "ref_pair_changes", "bp_recall_100kb", "founder_share_tvd", "affinity_spearman"]
            print(g[[c for c in cols if c in g.columns]].round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--tags", nargs="+", help="DEPTH=TAG, e.g. 0.1=v6cfullfe_homo 1.0=v6cfullfe_homo_1x "
                   "(default: the model's registered tags)")
    r.add_argument("--sections", nargs="+", choices=["core", "genewindow"], default=["core"],
                   help="core = every SNP+RC-based section; genewindow = minimap2 gene-window eval (minutes)")
    r.add_argument("--gw-threads", type=int, default=8, help="genewindow: total threads (projection, minimap2)")
    r.add_argument("--gw-jobs", type=int, default=2, help="genewindow: truth lines aligned at once")
    r.add_argument("--note", default=None)
    r.add_argument("--classes", nargs="+", default=CLASSES)
    r.add_argument("--lam", type=float, default=50.0)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--reference", action="store_true", help="compute missing reference paths first")
    rf = sub.add_parser("reference")
    rf.add_argument("--classes", nargs="+", default=CLASSES)
    rf.add_argument("--lam", type=float, default=50.0)
    rf.add_argument("--workers", type=int, default=4)
    sub.add_parser("summary")
    sd = sub.add_parser("genewindow-seed", help="create the gene-window cache from a finished gw run dir")
    sd.add_argument("--from", dest="src", default=None)
    db = sub.add_parser("dashboard", help="write dashboard.html (and the detailed report.html)")
    db.add_argument("--baseline", default="v6cfullfe_homo")
    db.add_argument("--out", default=str(OUT / "dashboard.html"))
    db.add_argument("--no-report", action="store_true", help="skip rebuilding report.html")
    a = ap.parse_args()
    {"run": cmd_run, "reference": cmd_reference, "summary": cmd_summary, "genewindow-seed": cmd_seed,
     "dashboard": cmd_dashboard}[a.cmd](a)


def cmd_seed(a):
    import eval_genewindow
    eval_genewindow.seed(a.src or eval_genewindow.SEED)


def cmd_dashboard(a):
    import eval_suite_dashboard
    print(eval_suite_dashboard.build(registry(), a.baseline, Path(a.out)))
    if not a.no_report:
        import eval_suite_report
        eval_suite_report.main()


if __name__ == "__main__":
    main()
