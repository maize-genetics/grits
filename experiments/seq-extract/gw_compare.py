"""Gene-window eval: old (v1) vs new (v2) run comparison tables.

v1 = ropebwt3 (no snap) + external gw_filter_proj.py (excess > 50 kb); v2 = ropebwt3_v3 (snap on, built-in
span check). Writes into OUT_DIR: compare_projection.tsv (failure reasons, model-segment vs panel
queries), compare_uncalled.tsv (per class: % N bp, % all-N windows), compare_anchors.tsv (IDX-INBRED
exactness), compare_switch_artifact.tsv, compare_per_class.tsv (full + trimmed metrics side by side).

Usage: python gw_compare.py V1_DIR V2_DIR OUT_DIR
  V1_DIR: proj_filt.tsv, queries.tsv, switch_artifact/, results_v1trim/ ; V2_DIR: proj.tsv, queries.tsv,
  switch_artifact/, results/
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

CLASSES = ["IDX-INBRED", "IDX-HYB", "IDX-RIL2", "MIX-HYB", "MIX-RIL2", "OUT-INBRED", "OUT-HYB", "OUT-RIL2"]


def reason(st):
    if st in ("ok", "ok_clamped"):
        return st
    if st == "implausible_length":
        return "projected span implausibly long"
    st = st.replace("projection_failed: ", "")
    return re.sub(r"\s*\(.*\)$", "", st)


def proj_stats(proj, queries, panel_keys):
    model = set()
    with open(queries) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            k = (r["ref_contig"], r["ref_start"], r["ref_end"], r["assembly"])
            if k not in panel_keys:
                model.add(k)
    c = {"model_segments": Counter(), "full_window_panel": Counter()}
    with open(proj) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            k = (r["ref_contig"], r["ref_start"], r["ref_end"], r["assembly"])
            st = reason(r["status"])
            if k in model:
                c["model_segments"][st] += 1
            if k in panel_keys:
                c["full_window_panel"][st] += 1
    return c


def panel_keys(windows):
    founders = ["B97", "CML103", "CML228", "CML247", "CML277", "CML322", "CML333", "CML52", "CML69", "HP301",
                "Il14H", "Ki11", "Ki3", "Ky21", "M162W", "M37W", "Mo18W", "Ms71", "NC350", "NC358", "Oh43", "Oh7B",
                "P39", "Tzi8"]
    keys = set()
    for r in csv.DictReader(open(windows), delimiter="\t"):
        for f in founders:
            keys.add((f"B73_{r['chrom']}", r["start"], r["end"], f))
    return keys


def anchors(phw):
    d = pd.read_csv(phw, sep="\t", dtype={"depth": str})
    rows = []
    for (dp, cls), x in d[d["class"].str.startswith("IDX")].groupby(["depth", "class"]):
        right = x[(x.n_switches == 0) & (x.model_founders == x.truth)]
        sc = right[right.state == "scored"]
        rows.append(dict(depth=dp, cls=cls, correct_founder_noswitch=len(right), scored=len(sc),
                         exact=(sc.errors == 0).sum(), pct_exact=100 * (sc.errors == 0).mean(),
                         nonexact_errors=sc.errors.sum(), t50_exact=(sc.t50_errors == 0).sum(),
                         mislocated=(right.state == "mislocated").sum(), all_N=(right.state == "all_N").sum()))
    return pd.DataFrame(rows)


def uncalled(phw):
    d = pd.read_csv(phw, sep="\t", dtype={"depth": str})
    g = d.groupby(["depth", "class"])
    return pd.DataFrame({"pct_N_bp": 100 * g.query_N.sum() / g.qlen.sum(),
                         "pct_projfail_bp": 100 * g.failed_bases.sum() / g.qlen.sum(),
                         "pct_all_N_windows": 100 * g.state.apply(lambda s: (s == "all_N").mean())}).reset_index()


def switch_summary(path):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    full = [r for r in rows if r["full_errors"] != ""]
    sp = [r for r in rows if r["split_errors"] != ""]
    se = [int(r["split_errors"]) for r in sp if r["split_chrom_ok"] == "True"]
    off = Counter()
    for r in sp:
        o = r["junction_offset"]
        if o in ("NA", ""):
            off["NA"] += 1
            continue
        o = abs(int(o))
        off["0" if o == 0 else "<=10" if o <= 10 else "<=100" if o <= 100 else ">100"] += 1
    return dict(n=len(rows), full_ok=len(full),
                full_exact=sum(int(r["full_errors"]) == 0 and r["full_chrom_ok"] == "True" for r in full),
                split_ok=len(sp), split_exact=sum(e == 0 for e in se), split_err_total=sum(se),
                offset_0=off["0"], offset_le10=off["<=10"], offset_le100=off["<=100"], offset_gt100=off[">100"],
                offset_NA=off["NA"])


def main():
    v1, v2, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    pk = panel_keys(v2 / "windows.tsv")
    rows = []
    for tag, d, proj in (("v1", v1, "proj_filt.tsv"), ("v2", v2, "proj.tsv"), ("v2_nosnap", v2, "proj_nosnap.tsv")):
        if not (d / proj).exists():
            continue
        for qset, c in proj_stats(d / proj, d / "queries.tsv", pk).items():
            n = sum(c.values())
            for st, k in sorted(c.items()):
                rows.append(dict(run=tag, query_set=qset, status=st, n=k, pct=100 * k / n))
    pd.DataFrame(rows).to_csv(out / "compare_projection.tsv", sep="\t", index=False, float_format="%.3f")

    u = uncalled(v1 / "results_v1trim/per_hapwin.tsv").merge(uncalled(v2 / "results/per_hapwin.tsv"),
                                                             on=["depth", "class"], suffixes=("_v1", "_v2"))
    u.to_csv(out / "compare_uncalled.tsv", sep="\t", index=False, float_format="%.3f")
    a = anchors(v1 / "results_v1trim/per_hapwin.tsv").merge(anchors(v2 / "results/per_hapwin.tsv"),
                                                            on=["depth", "cls"], suffixes=("_v1", "_v2"))
    a.to_csv(out / "compare_anchors.tsv", sep="\t", index=False, float_format="%.3f")
    sw = pd.DataFrame([dict(run="v1", **switch_summary(v1 / "switch_artifact/switch_artifact.tsv")),
                       dict(run="v2", **switch_summary(v2 / "switch_artifact/switch_artifact.tsv"))])
    sw.to_csv(out / "compare_switch_artifact.tsv", sep="\t", index=False)
    cols = ["depth", "class", "err_per_kb", "t50_err_per_kb", "t20_err_per_kb", "mean_identity", "t50_identity",
            "pct_exact_called", "t50_pct_exact", "t50_mean_trimmed_bp", "t50_pct_unanchored", "t20_pct_unanchored",
            "ceil_err_per_kb", "t50_ceil_err_per_kb", "pct_mislocated", "pct_uncalled_bp", "pct_all_N",
            "err_per_kb_noswitch", "err_per_kb_switch", "t50_pct_diff_from_switch_win"]
    p1 = pd.read_csv(v1 / "results_v1trim/per_class.tsv", sep="\t", dtype={"depth": str})[cols]
    p2 = pd.read_csv(v2 / "results/per_class.tsv", sep="\t", dtype={"depth": str})[cols]
    m = p1.merge(p2, on=["depth", "class"], suffixes=("_v1", "_v2"))
    m["order"] = m["class"].map(CLASSES.index)
    m = m.sort_values(["depth", "order"]).drop(columns="order")
    m.to_csv(out / "compare_per_class.tsv", sep="\t", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    for f in ("compare_projection.tsv", "compare_uncalled.tsv", "compare_anchors.tsv", "compare_switch_artifact.tsv"):
        print(f"== {f}")
        print(pd.read_csv(out / f, sep="\t").to_string(index=False))


if __name__ == "__main__":
    main()
