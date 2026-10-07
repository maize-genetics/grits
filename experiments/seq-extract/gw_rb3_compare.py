"""Gene-window eval pilot: minimap2 vs ropebwt3 sw on the same queries, for a subset of truth lines.

Truth assignment (hap -> line, incl. the hybrid pairing) is taken from the minimap2 v2 run's per_hapwin.tsv
so both aligners are scored on identical (query, truth line) pairs. The ceiling is recomputed per aligner
with gw_score's rule (best founder by full errors among on-chromosome hits; trimmed: best anchored founder
by t50 errors).

Usage: python gw_rb3_compare.py --v2 V2_DIR --rb3 NAME=DIR [NAME=DIR ...] --lines Oh43 Tx303 --out DIR
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gw_score as S  # noqa: E402

CLASSES = S.CLASSES


def load_scores(d, lines):
    sc = {}
    for line in lines:
        for x in ("", ".model", ".panel"):
            p = Path(d) / f"scores_{line}{x}.tsv"
            if p.exists():
                for r in csv.DictReader(open(p), delimiter="\t"):
                    for k in S.INT:
                        r[k] = int(r[k])
                    sc[(r["seq_hash"], line)] = r
    return sc


def ceilings(panel, wins, sc, lines):
    out = {}
    for wid, w in wins.items():
        for line in lines:
            best = tb = None
            for f, pr in panel[wid].items():
                if pr["status"] != "ok":
                    continue
                s = sc.get((pr["seq_hash"], line))
                if S.state(s, w["chrom"]) != "scored":
                    continue
                k = (s["errors"], s["query_N"], -s["matches"])
                if best is None or k < best[0]:
                    best = (k, s)
                if s["t50_anchored"]:
                    k2 = (s["t50_errors"], s["t50_trimmed_bp"], -s["t50_matches"])
                    if tb is None or k2 < tb[0]:
                        tb = (k2, s)
            out[(wid, line)] = (best[1] if best else None, tb[1] if tb else None)
    return out


def rows_for(aligner, sc, base, hashes, wins, ceil):
    out = []
    for r in base:
        key = (r["depth"], r["class"], r["sample"], r["wid"], str(r["hap"]))
        h = hashes[key]
        s = sc.get((h, r["truth"]))
        st = S.state(s, r["chrom"])
        c, tc = ceil.get((r["wid"], r["truth"]), (None, None))
        row = dict(aligner=aligner, depth=r["depth"], cls=r["class"], sample=r["sample"], wid=r["wid"], hap=r["hap"],
                   truth=r["truth"], seq_hash=h, n_switches=r["n_switches"], state=st)
        for k in ("qlen", "query_N", "matches", "mismatches", "ins", "dels", "unaligned", "errors", "n_blocks",
                  "aligned_q", "t50_anchored", "t50_errors", "t50_called", "t50_matches", "t50_trimmed_bp"):
            row[k] = s[k] if s else 0
        row["ceil_errors"] = c["errors"] if c else np.nan
        row["ceil_called"] = (c["qlen"] - c["query_N"]) if c else np.nan
        row["ceil_t50_errors"] = tc["t50_errors"] if tc else np.nan
        row["ceil_t50_called"] = tc["t50_called"] if tc else np.nan
        out.append(row)
    return out


def summarize(d):
    sc = d[d.state == "scored"]
    called = (sc.qlen - sc.query_N).sum()
    an = sc[sc.t50_anchored == 1]
    ce = sc.dropna(subset=["ceil_errors"])
    ct = an.dropna(subset=["ceil_t50_errors"])
    has = d[d.state.isin(["scored", "mislocated"])]
    return pd.Series({
        "n_hapwin": len(d), "scored": len(sc),
        "pct_mislocated": 100 * (has.state == "mislocated").mean(),
        "err_per_kb": 1000 * sc.errors.sum() / called,
        "t50_err_per_kb": 1000 * an.t50_errors.sum() / an.t50_called.sum(),
        "identity": sc.matches.sum() / (sc.matches.sum() + sc.errors.sum()),
        "pct_exact_called": 100 * (sc.errors == 0).mean(),
        "t50_pct_exact": 100 * (an.t50_errors == 0).mean(),
        "unaln_per_kb": 1000 * sc.unaligned.sum() / called,
        "unaln_share_of_errors": 100 * sc.unaligned.sum() / max(1, sc.errors.sum()),
        "mean_blocks": sc.n_blocks.mean(),
        "ceil_err_per_kb": 1000 * ce.ceil_errors.sum() / ce.ceil_called.sum(),
        "ceil_t50_err_per_kb": 1000 * ct.ceil_t50_errors.sum() / ct.ceil_t50_called.sum(),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--rb3", nargs="+", required=True)
    ap.add_argument("--lines", nargs="+", default=["Oh43", "Tx303"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    v2, out = Path(a.v2), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wins = {r["wid"]: r for r in csv.DictReader(open(v2 / "windows.tsv"), delimiter="\t")}
    panel = defaultdict(dict)
    for r in csv.DictReader(open(v2 / "stitch/panel.tsv"), delimiter="\t"):
        panel[r["wid"]][r["founder"]] = r
    hashes = {(r["depth"], r["class"], r["sample"], r["wid"], r["hap"]): r["seq_hash"]
              for r in csv.DictReader(open(v2 / "stitch/model_haps.tsv"), delimiter="\t")}
    base = [r for r in csv.DictReader(open(v2 / "results/per_hapwin.tsv"), delimiter="\t") if r["truth"] in a.lines]
    for r in base:
        r["n_switches"] = int(r["n_switches"])
    aligners = {"minimap2": str(v2 / "align")}
    for x in a.rb3:
        k, d = x.split("=", 1)
        aligners[k] = d
    rows, scores = [], {}
    for name, d in aligners.items():
        sc = load_scores(d, a.lines)
        scores[name] = sc
        rows += rows_for(name, sc, base, hashes, wins, ceilings(panel, wins, sc, a.lines))
    df = pd.DataFrame(rows)
    df.to_csv(out / "pilot_per_hapwin.tsv", sep="\t", index=False)
    df["cls"] = pd.Categorical(df.cls, CLASSES, ordered=True)
    pc = df.groupby(["aligner", "depth", "cls"], observed=True).apply(summarize, include_groups=False).reset_index()
    tot = df.groupby(["aligner", "depth"]).apply(summarize, include_groups=False).reset_index()
    tot["cls"] = "ALL(Oh43,Tx303 truth)"
    pc = pd.concat([pc, tot])
    pc.to_csv(out / "pilot_per_class.tsv", sep="\t", index=False, float_format="%.4g")

    # per query agreement (unique (hash, line): model haps + panel), both scored on the right chromosome
    agree_rows, top = [], []
    mm = scores["minimap2"]
    for name in aligners:
        if name == "minimap2":
            continue
        rb = scores[name]
        for line in a.lines:
            pairs = []
            for (h, l), s in mm.items():
                if l != line or (h, l) not in rb:
                    continue
                t = rb[(h, l)]
                if s["status"] != "aligned" or t["status"] != "aligned" or s["contig"] != t["contig"]:
                    continue
                pairs.append((h, s, t))
            e1 = np.array([p[1]["errors"] for p in pairs])
            e2 = np.array([p[2]["errors"] for p in pairs])
            m = np.maximum(e1, e2)
            d = np.abs(e1 - e2)
            agree_rows.append(dict(rb3=name, line=line, n=len(pairs), pct_equal=100 * (d == 0).mean(),
                                   pct_within_1pct=100 * ((d <= 0.01 * m) | (m == 0)).mean(),
                                   pct_diff_gt_100bp=100 * (d > 100).mean(), pct_diff_gt_1kb=100 * (d > 1000).mean(),
                                   pct_rb3_higher=100 * (e2 > e1).mean(), pct_rb3_lower=100 * (e2 < e1).mean(),
                                   sum_err_mm2=int(e1.sum()), sum_err_rb3=int(e2.sum()),
                                   t50_pct_equal=100 * np.mean([p[1]["t50_errors"] == p[2]["t50_errors"]
                                                                for p in pairs])))
            for i in np.argsort(-d)[:8]:
                h, s, t = pairs[i]
                top.append(dict(rb3=name, line=line, seq_hash=h, qlen=s["qlen"], query_N=s["query_N"],
                                **{f"mm2_{k}": s[k] for k in ("errors", "mismatches", "ins", "dels", "unaligned",
                                                               "n_blocks", "aligned_q", "tstart", "tend")},
                                **{f"rb3_{k}": t[k] for k in ("errors", "mismatches", "ins", "dels", "unaligned",
                                                               "n_blocks", "aligned_q", "tstart", "tend")}))
            # status disagreements
            st = defaultdict(int)
            for (h, l), s in mm.items():
                if l == line and (h, l) in rb:
                    t = rb[(h, l)]
                    st[(s["status"], t["status"], s["contig"] == t["contig"])] += 1
            agree_rows[-1]["status_pairs"] = "; ".join(f"{k}:{v}" for k, v in sorted(st.items(), key=lambda x: -x[1]))
    pd.DataFrame(agree_rows).to_csv(out / "pilot_agreement.tsv", sep="\t", index=False, float_format="%.3f")
    pd.DataFrame(top).to_csv(out / "pilot_top_disagreements.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(pc.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(pd.DataFrame(agree_rows).drop(columns="status_pairs").to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
