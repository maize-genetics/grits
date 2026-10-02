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

SCRIPTS = Path(str(Path(__file__).resolve().parents[2] / "simval-corpus/scripts"))
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


# ---------------------------------------------------------------- refmap ternary vs truth
LEN_EDGES_BP = [0, 100, 500, 1000, 4000, 20000, 10 ** 12]
BIN_BP = 256


def _del_runlen(dele, pos):
    """Per panel record and founder: bp length of the contiguous deleted run covering it (0 if present)."""
    out = np.zeros(dele.shape, np.int64)
    for f in range(dele.shape[1]):
        dx = np.diff(np.r_[0, dele[:, f].astype(np.int8), 0])
        s, e = np.where(dx == 1)[0], np.where(dx == -1)[0]
        if len(s):
            L = pos[e - 1] - pos[s] + 1
            idx = np.concatenate([np.arange(a, b) for a, b in zip(s, e)])
            out[idx, f] = np.repeat(L, e - s)
    return out


def ternary_obs_stats(dirs, P):
    """Pooled over aligned rows (refmap raw.npy, flat [n_bins, 3K+2]): refmap's per-founder
    ternary state at each row's bin vs the panel truth (founder lacks the B73 sequence at
    >= 50% of the bin's panel records = deleted; 0% = present; otherwise skipped), split
    by deleted-run length, plus distance-code histograms per (truth, state). Truth here is
    the index panel's own assemblies, the same for every sample, so only the calibration
    rows' positions enter. Returns raw counts so rows pool by addition."""
    nb = len(LEN_EDGES_BP) - 1
    cnt_del = np.zeros((nb, 3), np.int64)            # [len bucket, state -1/0/1]
    cnt_pres = np.zeros(3, np.int64)
    hist = {f"{tr}|{st}": np.zeros(DIST_CODES, np.int64) for tr in ("del", "present") for st in (-1, 0, 1)}
    srcs = []
    for d in dirs:
        raw = np.load(d / "raw.npy", mmap_mode="r")
        bins = pd.read_csv(d / "raw.npy.bins.tsv", sep="\t")
        gam = pd.read_csv(d / "raw.npy.gametes.tsv", sep="\t").sort_values("gameteIndex").sampleName.tolist()
        srcs.append((raw, bins, [P.fidx[g] for g in gam]))
    for ci, c in enumerate(fss.AUTOSOMES):
        lo, hi = P.bounds[ci]
        pos = P.pos[lo:hi]
        fa = np.asarray(P.fa[lo:hi])
        ac = np.asarray(P.acls[lo:hi])
        valid = fa >= 0
        dele = valid & (np.take_along_axis(ac, np.clip(fa, 0, None), 1) == 3)
        runlen = _del_runlen(dele, pos)
        cv = np.vstack([np.zeros((1, K), np.int32), np.cumsum(valid, 0, dtype=np.int32)])
        cd = np.vstack([np.zeros((1, K), np.int32), np.cumsum(dele, 0, dtype=np.int32)])
        del valid, fa, ac
        for raw, bins, col in srcs:
            sel = np.where(bins.contig.values == c)[0]
            if not len(sel):
                continue
            st = bins.bin.values[sel] * BIN_BP
            a, b = np.searchsorted(pos, st, "left"), np.searchsorted(pos, st + BIN_BP, "left")
            mid = np.clip(np.searchsorted(pos, st + BIN_BP // 2), 0, len(pos) - 1)
            nv, nd = (cv[b] - cv[a])[:, col], (cd[b] - cd[a])[:, col]
            tern = np.asarray(raw[sel, K:2 * K])
            code = _dist_code(np.asarray(raw[sel, 2 * K:3 * K]))
            is_del = (nv > 0) & (nd >= 0.5 * nv)
            is_pres = (nv > 0) & (nd == 0)
            bucket = np.clip(np.searchsorted(LEN_EDGES_BP, runlen[mid][:, col], "right") - 1, 0, nb - 1)
            for si, s in enumerate((-1, 0, 1)):
                m = tern == s
                np.add.at(cnt_del[:, si], bucket[m & is_del], 1)
                cnt_pres[si] += int((m & is_pres).sum())
                hist[f"del|{s}"] += np.bincount(code[m & is_del] + 1, minlength=DIST_CODES)
                hist[f"present|{s}"] += np.bincount(code[m & is_pres] + 1, minlength=DIST_CODES)
        print(f"  ternary obs {c}", flush=True)
    return {"len_edges_bp": LEN_EDGES_BP, "states": [-1, 0, 1],
            "del_by_len_counts": cnt_del.tolist(), "present_counts": cnt_pres.tolist(),
            "dist_hist_offset": 1, "dist_hist": {k: v.tolist() for k, v in hist.items()}}


DIST_CODES = 129        # code -1 (no anchor) .. 127, stored at index code + 1


def _dist_code(dist_bp, scale=8.0):
    """refmap bp distance -> the same log code ropebwt_npy_to_matrix.py writes (-1 = no anchor)."""
    code = np.clip(np.rint(scale * np.log2(1.0 + np.maximum(dist_bp, 0))), 0, 127).astype(np.int64)
    return np.where(dist_bp < 0, -1, code)


def obs_table_from_stats(s):
    """Simulator observation table (simulate_alleles --obs-table): among NON-matching
    founders, P(-1) for deleted founders by run length and for present founders, and the
    distance-code distribution for each (truth, observed state)."""
    cd = np.asarray(s["del_by_len_counts"], float)
    cp = np.asarray(s["present_counts"], float)
    nm_del = cd[:, 0] + cd[:, 1]
    p_del = np.where(nm_del > 0, cd[:, 0] / np.maximum(nm_del, 1), np.nan)
    hist = {k: (np.asarray(v, float) / max(sum(v), 1)).tolist() for k, v in s["dist_hist"].items()
            if not k.endswith("|1")}
    return {"len_edges_bp": s["len_edges_bp"],
            "p_minus1_given_del_by_len": p_del.tolist(),
            "p_minus1_given_present": float(cp[0] / max(cp[0] + cp[1], 1)),
            "p_match_given_del": float(cd[:, 2].sum() / max(cd.sum(), 1)),
            "dist_code_offset": 1, "dist_code_pmf": hist}


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
    ap.add_argument("--skip-ternary-obs", action="store_true", help="skip the refmap-ternary-vs-truth tables")
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

    aligned = []
    for r in rows:
        ds, ind = r["dataset_id"], r["individual"]
        d = [Path(p) for p in glob.glob(f"{args.align_root}/*__{ds}__{ind}__{args.coverage}x")
             if (Path(p) / "raw.tsv").exists()]
        if not d:
            print(f"SKIP {ds} {ind}: not aligned", flush=True)
            continue
        d = d[0]
        if (d / "raw.npy").exists():
            aligned.append(d)
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

    if aligned and not args.skip_ternary_obs:
        res["ternary_obs"] = ternary_obs_stats(aligned, P)
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
    if "ternary_obs" in res and bp_per_site:
        sug["obs_table"] = dict(obs_table_from_stats(res["ternary_obs"]), bp_per_site=bp_per_site)
    res["suggested_sim_params"] = sug
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(sug, indent=1))


if __name__ == "__main__":
    main()
