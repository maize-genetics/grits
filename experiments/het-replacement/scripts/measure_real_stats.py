#!/usr/bin/env python
"""
Stage 2.2 of experiments/het-replacement/PLAN.md: measure real statistics from
the CALIBRATION corpus and turn them into simulate_alleles.py `--indel-model
replacement` parameters.

Inputs per calibration sample (row of the calibration manifest):
  * truth gVCF(s)                       -> replacement blocks (DEL/INS bp, lengths, spacing)
  * aligned read outputs in --align-root (simval_eval_indel.py --stage align):
      raw.tsv  (per-read status + founder hits + placed position),
      windowed_k25native_wcount.npy + raw.npy.bins.tsv (model input)
  * fast_snprc_score truth caches        -> per-site deletion dosage
  * panel cache                          -> founder deletion state / sharing spectrum

Guard: refuses any evaluation-corpus line (the simulated_validation corpus is
the evaluation set) unless --allow-eval-lines-for-code-testing is passed, which
is only for checking code paths; numbers from such runs must never be used.

Output: one JSON with raw statistics and suggested simulator parameters.
"""
import argparse
import csv
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/regionfix-out-snprc-wt/experiments/simval-corpus/scripts")
sys.path.insert(0, str(SCRIPTS))
import fast_snprc_score as fss  # noqa: E402

K, T = 25, 512
EVAL_LINES = {"B73", "Oh43", "Il14H", "B97", "CML103", "Tx303", "A188", "EP1", "CML459", "Ia453"}
CALIB_MANIFEST = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_calibration/manifest.tsv"
AUTOSOMES = [f"chr{i}" for i in range(1, 11)]


def lines_of(ind):
    return set(ind.split("x"))


# ---------------------------------------------------------------- truth gVCF blocks
def gvcf_blocks(gvcf):
    """Per line: B73-autosomal bp by record type, DEL length quantiles, DEL spacing."""
    awk = r'''!/^#/ && $1 ~ /^chr([1-9]|10)$/ {
      end=$2; if (match($8,/END=[0-9]+/)) end=substr($8,RSTART+4,RLENGTH-4)+0; span=end-$2+1;
      split($10,g,":"); a=g[1];
      if (a=="." || a=="./." || a==".|.") { print "M\t" span "\t" $1 "\t" $2; next }
      if ($5=="<NON_REF>" || a=="0") { print "R\t" span "\t" $1 "\t" $2; next }
      split($5,alts,","); al=alts[a+0];
      if (al=="<DEL>") print "D\t" span "\t" $1 "\t" $2;
      else if (length(al)<length($4)) print "D\t" length($4)-length(al) "\t" $1 "\t" $2;
      else if (length(al)>length($4)) print "I\t" length(al)-length($4) "\t" $1 "\t" $2;
      else print "S\t" length(al) "\t" $1 "\t" $2 }'''
    out = subprocess.run(f"zcat {gvcf} | awk -F'\\t' '{awk}'", shell=True, capture_output=True,
                         text=True, check=True).stdout
    df = pd.read_csv(pd.io.common.StringIO(out), sep="\t", header=None, names=["t", "len", "chr", "pos"])
    bp = df.groupby("t")["len"].sum()
    tot = bp.drop("I", errors="ignore").sum()
    d = df[df.t == "D"]
    gaps = d.groupby("chr")["pos"].diff().dropna()
    q = lambda s, ps: {f"p{p}": float(np.percentile(s, p)) for p in ps} if len(s) else {}
    return {
        "bp_frac": {k: float(v / tot) for k, v in bp.items() if k != "I"},
        "del_bp": float(bp.get("D", 0)), "ins_bp": float(bp.get("I", 0)),
        "n_del": int(len(d)),
        "del_len": q(d["len"], [10, 25, 50, 75, 90, 99]),
        "del_len_bp_weighted_median": float(np.sort(d["len"])[np.searchsorted(np.cumsum(np.sort(d["len"])), d["len"].sum() / 2)]) if len(d) else None,
        "del_spacing": q(gaps, [10, 25, 50, 75, 90]),
    }


# ---------------------------------------------------------------- panel
def panel_state():
    P = fss.Panel()
    fa = np.asarray(P.fa)
    ac = np.asarray(P.acls)
    idx = np.arange(fa.shape[0])
    fdel = np.zeros(fa.shape, bool)
    for f in range(K):
        a = fa[:, f]
        fdel[:, f] = (a >= 0) & (ac[idx, np.clip(a, 0, None)] == 3)
    return P, ac, idx, fdel


def truth_del(t, ac, idx):
    c = t["code"]
    ok = (c != fss.T_NOINFO) & (c != fss.T_MISSING)
    cls = np.where(c >= 0, ac[idx, np.clip(c, 0, None)], np.where(c == fss.T_NOVEL, t["ncls"], -1))
    return cls == 3, ok


# ---------------------------------------------------------------- reads by dosage
def read_mix(raw_tsv, P, fdel, dosage):
    names = P.founders
    fi = {n: i for i, n in enumerate(names)}
    ci = {c: i for i, c in enumerate(AUTOSOMES)}
    r = pd.read_csv(raw_tsv, sep="\t", header=None, usecols=[0, 2, 4, 5, 7],
                    names=["name", "st", "hits", "pc", "pp"])
    r = r[r.st.isin(["EXACT", "PLACED"])].copy()
    r["pp"] = pd.to_numeric(r.pp, errors="coerce")
    r["pc"] = r.pc.str.replace("B73_", "", regex=False)
    r = r[r.pc.isin(ci) & r.pp.notna()]
    rec = np.empty(len(r), np.int64)
    for c, g in r.groupby("pc"):
        lo, hi = P.bounds[ci[c]]
        rec[r.index.get_indexer(g.index)] = lo + np.clip(np.searchsorted(P.pos[lo:hi], g.pp.to_numpy()), 0, hi - lo - 1)
    r["dos"] = dosage[rec]
    frac_del, frac_pres = [], []
    for h, j in zip(r.hits, rec):
        if not isinstance(h, str) or h == ".":
            frac_del.append(np.nan); frac_pres.append(np.nan); continue
        fs = [fi[x.split("_")[0]] for x in h.split(",") if x.split("_")[0] in fi]
        if not fs:
            frac_del.append(np.nan); frac_pres.append(np.nan); continue
        named = np.zeros(K, bool); named[fs] = True
        frac_del.append(fdel[j, named].mean())
        pres = ~fdel[j]
        frac_pres.append((named & pres).sum() / max(pres.sum(), 1))
    r["frac_del"], r["frac_pres"] = frac_del, frac_pres
    share = {d: float((dosage == d).mean()) for d in (0, 1, 2)}
    out = {}
    base = None
    for d in (0, 1, 2):
        g = r[r.dos == d]
        if not len(g) or not share[d]:
            continue
        dens = len(g) / share[d]
        base = base or dens
        pl = g[g.st == "PLACED"]
        out[str(d)] = {"site_share": share[d], "rel_read_density": float(dens / base),
                       "exact_share": float((g.st == "EXACT").mean()),
                       "placed_share": float((g.st == "PLACED").mean()),
                       "placed_named_deleted_frac": float(pl.frac_del.mean()),
                       "placed_named_present_per_present_founder": float(pl.frac_pres.mean())}
    return out, r


# ---------------------------------------------------------------- placement shift (inbred samples)
def placement_shift(reads, gvcf):
    awk = r'''!/^#/ && $1 ~ /^chr([1-9]|10)$/ { e=$2; ac=""; as=""; ae="";
      n=split($8,kv,";"); for(i=1;i<=n;i++){ split(kv[i],p,"="); if(p[1]=="END") e=p[2];
      else if(p[1]=="ASM_Chr") ac=p[2]; else if(p[1]=="ASM_Start") as=p[2]; else if(p[1]=="ASM_End") ae=p[2] }
      if (ac!="" && as!="") print $1 "\t" $2 "\t" e "\t" ac "\t" as "\t" ae }'''
    out = subprocess.run(f"zcat {gvcf} | awk -F'\\t' '{awk}'", shell=True, capture_output=True,
                         text=True, check=True).stdout
    m = pd.read_csv(pd.io.common.StringIO(out), sep="\t", header=None,
                    names=["bc", "bs", "be", "ac", "as", "ae"], dtype={"bc": str, "ac": str})
    m["ac"] = m["ac"].str.lower()
    m = m.sort_values(["ac", "as"])
    rd = reads.copy()
    nm = rd.name.str.split("_", expand=True)
    rd["oc"] = nm[0].str.lower()
    rd["mid"] = (pd.to_numeric(nm[1], errors="coerce") + pd.to_numeric(nm[2], errors="coerce")) // 2
    rd = rd.dropna(subset=["mid"])
    for c, g in rd.groupby("oc"):
        mm = m[m.ac == c]
        if mm.empty:
            continue
        a = mm["as"].to_numpy()
        i = np.clip(np.searchsorted(a, g.mid.to_numpy(), "right") - 1, 0, len(a) - 1)
        bs, be, asv, aev = mm.bs.to_numpy()[i], mm.be.to_numpy()[i], a[i], mm.ae.to_numpy()[i]
        rd.loc[g.index, "ec"] = mm.bc.to_numpy()[i]
        rd.loc[g.index, "ep"] = np.clip(bs + (g.mid.to_numpy() - asv), bs, be)
        rd.loc[g.index, "repl_seq"] = ((aev - asv + 1) >= 50) & ((be - bs + 1) <= 2)
    rd = rd.dropna(subset=["ec"])
    rd["repl_seq"] = rd["repl_seq"].astype(bool)
    same = rd.pc == rd.ec
    rd["shift"] = np.where(same, np.abs(rd.pp - rd.ep), np.nan)
    out = {}
    for st in ("EXACT", "PLACED"):
        for rs, lab in ((False, "collinear"), (True, "replacement")):
            g = rd[(rd.st == st) & (rd.repl_seq == rs)]
            if not len(g):
                continue
            sh = g["shift"].dropna()
            out[f"{st}_{lab}"] = {"n": int(len(g)), "wrong_chrom": float(1 - (g.pc == g.ec).mean()),
                                  "within_10kb": float((sh < 1e4).mean()) if len(sh) else None,
                                  "shift_median_bp": float(sh.median()) if len(sh) else None,
                                  "shift_p75_bp": float(sh.quantile(.75)) if len(sh) else None}
    return out


# ---------------------------------------------------------------- model-input features
def feature_stats(windowed):
    x = np.load(windowed, mmap_mode="r")
    sel = np.linspace(0, x.shape[0] - 1, min(400, x.shape[0])).astype(int)
    x = np.asarray(x[sel])
    t, d, c = x[..., :K], x[..., K + 2:2 * K + 2].astype(float), x[..., 2 * K + 2:3 * K + 2]
    nm = (t == 1).sum(-1)
    out = {"tern_frac": {s: float((t == v).mean()) for s, v in (("match", 1), ("div", 0), ("del", -1))},
           "n_match_per_row_mean": float(nm.mean()), "rows_match_ge20": float((nm >= 20).mean()),
           "count_zero_frac": float((c == 0).mean())}
    for s, v in (("match", 1), ("div", 0), ("del", -1)):
        m = t == v
        if m.any():
            out[f"dist_code_{s}"] = {f"p{p}": float(np.percentile(d[m], p)) for p in (10, 50, 90)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=CALIB_MANIFEST)
    ap.add_argument("--align-root", required=True, help="dir holding <tag>__<DS>__<IND>__<cov>x/ align outputs")
    ap.add_argument("--coverage", default="0.1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim-sites-per-window", type=float, default=None,
                    help="simulated sites spanned by a 512-row window (for bp->site conversion); "
                         "measured from a small simulate() run if omitted")
    ap.add_argument("--datasets", nargs="*", default=None, help="restrict to these dataset ids")
    ap.add_argument("--allow-eval-lines-for-code-testing", action="store_true")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.manifest), delimiter="\t") if r["coverage"] == args.coverage
            and (not args.datasets or r["dataset_id"] in args.datasets)]
    for r in rows:
        if lines_of(r["individual"]) & EVAL_LINES and not args.allow_eval_lines_for_code_testing:
            raise SystemExit(f"refusing evaluation-corpus line in {r['dataset_id']} {r['individual']}")

    P, ac, idx, fdel = panel_state()
    k = fdel.sum(1)
    res = {"panel": {"n_deleted_spectrum": (np.bincount(k, minlength=K + 1) / k.size).tolist(),
                     "mean_n_deleted": float(k.mean()),
                     "pair_both_deleted": float(np.mean((k / K) * (k - 1) / (K - 1))),
                     "pair_exactly_one": float(np.mean(2 * (k / K) * (K - k) / (K - 1)))},
           "lines": {}, "samples": {}}

    seen = set()
    for r in rows:
        for tp in {r["truth_h1"], r["truth_h2"]}:
            if "maize_v2" in tp and tp not in seen and "RIL" not in r["dataset_id"]:
                seen.add(tp)
                res["lines"][Path(tp).name.replace(".g.vcf.gz", "")] = gvcf_blocks(tp)
                print(f"blocks {Path(tp).name}", flush=True)

    for r in rows:
        ds, ind = r["dataset_id"], r["individual"]
        d = [Path(p) for p in glob.glob(f"{args.align_root}/*__{ds}__{ind}__{args.coverage}x")
             if (Path(p) / "raw.tsv").exists()]
        if not d:
            print(f"SKIP {ds} {ind}: not aligned", flush=True)
            continue
        d = d[0]
        t1, t2 = fss.load_truth(r["truth_h1"]), fss.load_truth(r["truth_h2"])
        d1, o1 = truth_del(t1, ac, idx)
        d2, o2 = truth_del(t2, ac, idx)
        dosage = np.where(o1 & o2, d1.astype(int) + d2, -1)
        mix, reads = read_mix(d / "raw.tsv", P, fdel, dosage)
        s = {"read_mix_by_dosage": mix}
        if (d / "windowed_k25native_wcount.npy").exists():
            s["features"] = feature_stats(d / "windowed_k25native_wcount.npy")
            bins = pd.read_csv(d / "raw.npy.bins.tsv", sep="\t")
            s["bp_per_512_rows"] = float(256 * bins.groupby("contig")["bin"].agg(lambda b: (b.max() - b.min()) / max(len(b), 1)).mean() * T)
        if ds.endswith("INBRED"):
            s["placement"] = placement_shift(reads, r["truth_h1"])
        res["samples"][f"{ds}__{ind}"] = s
        print(f"sample {ds} {ind}", flush=True)

    Path(args.out).write_text(json.dumps(res, indent=1))       # keep measurements even if the next step fails

    # ---------- suggested simulator parameters
    sites_per_window = args.sim_sites_per_window
    if sites_per_window is None:
        # load this branch's simulator by path: the `python` package may already
        # resolve to another checkout via fast_snprc_score's sys.path insert
        import importlib.util
        sim_path = Path(__file__).resolve().parents[3] / "src/python/crf/simulate_alleles.py"
        spec = importlib.util.spec_from_file_location("simulate_alleles_branch", sim_path)
        sim = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sim)
        simulate = sim.simulate
        out, *rest = simulate(np.random.default_rng(0), windows=4, sites=60000, founders=K, min_cross=2,
                              max_cross=10, inbreeding=1.0, allele_sharing=0.2, bad_frac=0.05,
                              sharing_model="coalescent", ancestors=6, sharing_theta=4.0,
                              simulate_indels=True, indel_model="overlay", indel_coverage=2.0,
                              emit_read_counts=True, collapse_rows=True, indel_region_mult=2)
        refpos = rest[5]
        spans = [np.ptp(refpos[i, j:j + T][refpos[i, j:j + T] >= 0]) for i in range(refpos.shape[0])
                 for j in range(0, refpos.shape[1] - T, T)]
        sites_per_window = float(np.median(spans))
    bp512 = [s["bp_per_512_rows"] for s in res["samples"].values() if "bp_per_512_rows" in s]
    bp_per_site = (np.median(bp512) / sites_per_window) if bp512 else None
    lines = [v for kname, v in res["lines"].items() if kname not in EVAL_LINES]
    sug = {"bp_per_site": bp_per_site, "sim_sites_per_512_rows": sites_per_window}
    if lines and bp_per_site:
        med_len = float(np.median([v["del_len_bp_weighted_median"] for v in lines if v["del_len_bp_weighted_median"]]))
        del_frac = float(np.mean([v["bp_frac"].get("D", 0) for v in lines]))
        max_share = int(max(i for i, v in enumerate(res["panel"]["n_deleted_spectrum"]) if v >= 0.01))
        mean_len_sites = med_len / bp_per_site
        e_share = (1 + max_share) / 2
        rate = del_frac / (mean_len_sites * np.exp(0.7 ** 2 / 2) * e_share / K)
        sug.update(repl_mean_len=mean_len_sites, repl_max_share=max_share, repl_rate=float(rate),
                   target_deleted_bp_frac=del_frac, del_len_bp_weighted_median_bp=med_len)
    shifts = [s["placement"].get("PLACED_replacement", {}).get("shift_median_bp") for s in res["samples"].values()
              if "placement" in s]
    shifts = [v for v in shifts if v]
    if shifts and bp_per_site:
        sug["repl_shift_sites"] = float(np.median(shifts) / np.log(2) / bp_per_site)
    cp = [s["read_mix_by_dosage"]["2"]["placed_named_present_per_present_founder"]
          for s in res["samples"].values() if "2" in s.get("read_mix_by_dosage", {})]
    if cp:
        sug["repl_cross_present"] = float(np.nanmedian(cp))
    res["suggested_sim_params"] = sug
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(sug, indent=1))


if __name__ == "__main__":
    main()
