"""Gene-window eval, steps 4-6: assign truth, pick the panel ceiling, aggregate.

Truth per (sample, window, haplotype):
  inbred X   both haplotypes -> X
  hybrid AxB the one-to-one pairing (h1->A, h2->B) or (h1->B, h2->A) with fewer total errors
  ril2 AxB   both haplotypes -> the mosaic parent of the window (gw_ril_truth.py); windows containing a
             mosaic breakpoint are excluded
Panel ceiling per (window, truth line): the founder (of 25) whose whole-window sequence has the fewest
errors against that line, among founders whose primary hit is on the window's chromosome.
Event ceiling (ceil_ev_*): the same choice by event errors (gw_align EVENT_COLS: an indel or unaligned
run counts once); ceil_event_errors / ceil_t<K>_event_errors are the bp-best founder's event errors.
Random founder (rand_*): the exact expectation over the same scorable founders (status ok, primary hit on the
window's chromosome) of min(errors, called), min(event_errors, called) and called; rand_n = their number.

In-panel extraction rule: when the truth line is itself a panel founder whose own whole-window panel sequence
failed extraction (panel status != ok, or no row) or whose alignment to itself is not scored on the window's
chromosome, the panel cannot represent the truth there, so the haplotype window is "truth_unextractable":
excluded from the model, the ceiling and the random baseline, and counted separately. Out-of-panel truth
lines are never affected.

A haplotype window is
  all_N       sequence entirely N (uncalled / projection failed): not scored, counted as uncalled
  mislocated  primary hit not on the window's chromosome: not scored, counted
  truth_unextractable  in-panel truth line whose own panel sequence is unusable (rule above): not scored,
              counted (overrides the other states)
  scored      otherwise (an unaligned sequence counts all its non-N bases as unaligned errors)
errors = mismatches + inserted + deleted + unaligned (bases); N never counts as an error.

Writes per_hapwin.tsv, per_sample.tsv, per_class.tsv, ceiling.tsv, ril_regions_check.tsv into --out.
"""
import os
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gw_align import ALL_LINES, EVENT_COLS, norm_chrom, split_pair  # noqa: E402

CLASSES = ["IDX-INBRED", "IDX-HYB", "IDX-RIL2", "MIX-HYB", "MIX-RIL2", "OUT-INBRED", "OUT-HYB", "OUT-RIL2"]
READS = os.path.join(os.environ.get("GRITS_WORKDIR", "/nonexistent-GRITS_WORKDIR-unset"), "scratch/read_datasets")
SNPRC = os.path.join(os.environ.get("GRITS_WORKDIR", "/nonexistent-GRITS_WORKDIR-unset"), "results/eval_suite/v6cfullfe_homo/per_sample.tsv")
TRIM_KS = (50, 20)
TCOLS = ("anchored", "matches", "errors", "trimmed_bp", "called")
INT = ["qlen", "query_N", "n_blocks", "aligned_q", "matches", "mismatches", "ins", "dels", "unaligned", "truth_N",
       "errors"] + [f"t{K}_{x}" for K in TRIM_KS for x in TCOLS] + EVENT_COLS + \
    [f"t{K}_{x}" for K in TRIM_KS for x in EVENT_COLS]


def load_scores(align_dir):
    sc = {}
    for line in ALL_LINES:
        ps = [p for p in (Path(align_dir) / f"scores_{line}{x}.tsv" for x in ("", ".model", ".panel")) if p.exists()]
        if not ps:
            print(f"missing scores for {line}", file=sys.stderr)
        for p in ps:
            with open(p) as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    for k in INT:
                        r[k] = int(r[k])
                    sc[(r["seq_hash"], line)] = r
    return sc


def state(s, chrom):
    if s is None:
        return "missing"
    if s["status"] == "all_N":
        return "all_N"
    if s["status"] == "aligned" and s["contig"] != chrom:
        return "mislocated"
    return "scored"


def truth_unextractable(panel_w, line, chrom, sc, panel_founders):
    """True when `line` is a panel founder and its own panel sequence for this window (panel_w: {founder: panel
    row}) failed extraction or does not score against `line` itself on `chrom`."""
    if line not in panel_founders:
        return False
    pr = panel_w.get(line)
    if pr is None or pr["status"] != "ok":
        return True
    return state(sc.get((pr["seq_hash"], line)), chrom) != "scored"


def random_founder(panel_w, line, chrom, sc):
    """Exact expectation of a uniformly random scorable panel founder for (window, truth line): returns
    (n_founders, mean capped bp errors, mean capped event errors, mean called bases); n = 0 -> zeros."""
    n = e = ev = c = 0
    for pr in panel_w.values():
        if pr["status"] != "ok":
            continue
        s = sc.get((pr["seq_hash"], line))
        if state(s, chrom) != "scored":
            continue
        called = s["qlen"] - s["query_N"]
        n += 1
        e += min(s["errors"], called)
        ev += min(s["event_errors"], called)
        c += called
    return (n, e / n, ev / n, c / n) if n else (0, 0.0, 0.0, 0.0)


def regions(path):
    by = defaultdict(list)
    for line in open(path):
        c, rng = line.strip().rsplit(":", 1)
        s, e = rng.split("-")
        by[norm_chrom(c)].append((int(s) - 1, int(e)))
    return by


def in_regions(reg, contig, ts, te):
    ov = sum(max(0, min(e, te) - max(s, ts)) for s, e in reg.get(contig, []))
    return ov >= 0.5 * (te - ts)


def score(windows, panel_tsv, model_haps_tsv, sc, ril_truth, out):
    """Write ceiling.tsv, per_hapwin.tsv and ril_regions_check.tsv into `out` from the window table, the
    panel stitch table, the model haplotype table and the score lookup sc {(seq_hash, line): row}."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    wins = {r["wid"]: r for r in csv.DictReader(open(windows), delimiter="\t")}
    ril = {(r["class"], r["sample"], r["wid"]): r["truth"] for r in csv.DictReader(open(ril_truth), delimiter="\t")}

    # ---- panel ceiling
    panel = defaultdict(dict)
    for r in csv.DictReader(open(panel_tsv), delimiter="\t"):
        panel[r["wid"]][r["founder"]] = r
    panel_founders = {f for pw in panel.values() for f in pw}
    ceil, tceil, ceil_ev, rand, unext = {}, {}, {}, {}, set()
    with open(out / "ceiling.tsv", "w") as fo:
        fo.write("wid\tline\tbest_founder\terrors\tcalled\tmatches\tn_ok_founders\tn_scored_founders\t"
                 "self_in_panel_errors\ttruth_unextractable\trand_n\trand_err_capped\trand_event_errors\t"
                 "rand_called\n")
        for wid, w in wins.items():
            for line in ALL_LINES:
                ux = truth_unextractable(panel[wid], line, w["chrom"], sc, panel_founders)
                if ux:
                    unext.add((wid, line))
                rnd = random_founder(panel[wid], line, w["chrom"], sc)
                rand[(wid, line)] = rnd
                rtxt = f"{int(ux)}\t{rnd[0]}\t{rnd[1]:.6g}\t{rnd[2]:.6g}\t{rnd[3]:.6g}"
                best = None
                nsc = 0
                evb = None
                for f, pr in panel[wid].items():
                    if pr["status"] != "ok":
                        continue
                    s = sc.get((pr["seq_hash"], line))
                    if state(s, w["chrom"]) != "scored":
                        continue
                    nsc += 1
                    ek = (s["event_errors"], s["query_N"], -s["matches"])
                    if evb is None or ek < evb[0]:
                        evb = (ek, f, s)
                    key = (s["errors"], s["query_N"], -s["matches"])
                    if best is None or key < best[0]:
                        best = (key, f, s)
                for K in TRIM_KS:
                    tb = None
                    for f, pr in panel[wid].items():
                        if pr["status"] != "ok":
                            continue
                        s = sc.get((pr["seq_hash"], line))
                        if state(s, w["chrom"]) != "scored" or not s[f"t{K}_anchored"]:
                            continue
                        key = (s[f"t{K}_errors"], s[f"t{K}_trimmed_bp"], -s[f"t{K}_matches"])
                        if tb is None or key < tb[0]:
                            tb = (key, f, s)
                    if tb:
                        tceil[(wid, line, K)] = (tb[1], tb[2])
                nok = sum(1 for pr in panel[wid].values() if pr["status"] == "ok")
                selfe = "."
                if line in panel[wid] and panel[wid][line]["status"] == "ok":
                    s = sc.get((panel[wid][line]["seq_hash"], line))
                    selfe = s["errors"] if s and state(s, w["chrom"]) == "scored" else state(s, w["chrom"])
                if best:
                    ceil[(wid, line)] = (best[1], best[2])
                    ceil_ev[(wid, line)] = (evb[1], evb[2])
                    s = best[2]
                    fo.write(f"{wid}\t{line}\t{best[1]}\t{s['errors']}\t{s['qlen'] - s['query_N']}\t{s['matches']}\t"
                             f"{nok}\t{nsc}\t{selfe}\t{rtxt}\n")
                else:
                    fo.write(f"{wid}\t{line}\t.\t.\t.\t.\t{nok}\t{nsc}\t{selfe}\t{rtxt}\n")

    # ---- model haplotypes
    haps = defaultdict(dict)
    for r in csv.DictReader(open(model_haps_tsv), delimiter="\t"):
        haps[(r["depth"], r["class"], r["sample"], r["kind"], r["wid"])][int(r["hap"])] = r

    cols = ["depth", "class", "sample", "wid", "chrom", "hap", "truth", "model_founders", "n_switches", "gap_bases",
            "failed_bases", "state", "qlen", "query_N", "asm_N", "n_blocks", "aligned_q", "matches", "mismatches",
            "ins", "dels", "unaligned", "errors", "contig", "mapq", "ceil_founder", "ceil_state", "ceil_errors",
            "ceil_called", "ceil_matches", "founder_eq_ceil"] + EVENT_COLS + \
        ["ceil_event_errors", "ceil_ev_founder", "ceil_ev_errors", "ceil_ev_called"] + \
        [f"t{K}_event_errors" for K in TRIM_KS] + [f"ceil_t{K}_event_errors" for K in TRIM_KS] + \
        [f"t{K}_{x}" for K in TRIM_KS for x in TCOLS] + \
        [f"ceil_t{K}_{x}" for K in TRIM_KS for x in ("founder", "matches", "errors", "trimmed_bp", "called")] + \
        ["rand_n", "rand_err_capped", "rand_event_errors", "rand_called"]
    rows = []
    rilchk = defaultdict(lambda: defaultdict(int))
    first_depth = sorted({k[0] for k in haps})[0]
    regcache = {}
    for (depth, cls, ind, kind, wid), hh in sorted(haps.items()):
        w = wins[wid]
        chrom = w["chrom"]
        if kind == "inbred":
            assign = {1: ind, 2: ind}
        elif kind == "hybrid":
            A, B = split_pair(ind)

            def tot(m):
                t = 0
                for h, line in m.items():
                    s = sc.get((hh[h]["seq_hash"], line))
                    st = state(s, chrom)
                    t += s["errors"] if st == "scored" else (s["qlen"] - s["query_N"] if st == "mislocated" else 0)
                return t
            m1, m2 = {1: A, 2: B}, {1: B, 2: A}
            assign = m1 if tot(m1) <= tot(m2) else m2
        else:
            t = ril[(cls, ind, wid)]
            if t == "BP":
                continue
            assign = {1: t, 2: t}
            A, B = split_pair(ind)
            if depth == first_depth:
                for lab, line in (("A", A), ("B", B)):
                    if (cls, ind, lab) not in regcache:
                        regcache[(cls, ind, lab)] = regions(f"{READS}/{cls}/{ind}/{lab}_{line}.segments.regions.txt")
                # regions rule (independent of the mosaic replay): the parent whose primary hit of the hap-1
                # sequence (>= 2 kb aligned, on the window's chromosome) lies inside that parent's listed segments
                hits = {}
                for lab, line in (("A", A), ("B", B)):
                    s = sc.get((hh[1]["seq_hash"], line))
                    if state(s, chrom) == "scored" and s["status"] == "aligned" and s["aligned_q"] >= 2000:
                        hits[line] = in_regions(regcache[(cls, ind, lab)], s["contig"], int(s["tstart"]),
                                                int(s["tend"]))
                inside = [l for l, v in hits.items() if v]
                rule = inside[0] if len(inside) == 1 else ("both" if len(inside) == 2 else
                                                           ("none" if hits else "no_hit"))
                rilchk[cls][("agree" if rule == t else rule if rule in ("both", "none", "no_hit") else "disagree")] += 1
        for h in (1, 2):
            r = hh[h]
            line = assign[h]
            s = sc.get((r["seq_hash"], line))
            st = "truth_unextractable" if (wid, line) in unext else state(s, chrom)
            c = ceil.get((wid, line))
            cs_ = c[1] if c else None
            founders = r["founders"]
            row = dict(depth=depth, **{"class": cls}, sample=ind, wid=wid, chrom=chrom, hap=h, truth=line,
                       model_founders=founders, n_switches=int(r["n_switches"]), gap_bases=int(r["gap_bases"]),
                       failed_bases=int(r["failed_bases"]), state=st)
            for k in ["qlen", "query_N", "n_blocks", "aligned_q", "matches", "mismatches", "ins", "dels", "unaligned",
                      "errors"]:
                row[k] = s[k] if s else 0
            row["asm_N"] = row["query_N"] - row["gap_bases"] - row["failed_bases"]
            row["contig"] = s["contig"] if s else "."
            row["mapq"] = s["mapq"] if s else "."
            row["ceil_founder"] = c[0] if c else "."
            row["ceil_state"] = "scored" if c else "none"
            row["ceil_errors"] = cs_["errors"] if cs_ else ""
            row["ceil_called"] = (cs_["qlen"] - cs_["query_N"]) if cs_ else ""
            row["ceil_matches"] = cs_["matches"] if cs_ else ""
            row["founder_eq_ceil"] = (founders == row["ceil_founder"]) if row["n_switches"] == 0 else ""
            for x in EVENT_COLS:
                row[x] = s[x] if s else 0
            row["ceil_event_errors"] = cs_["event_errors"] if cs_ else ""     # event errors of the bp-best founder
            ce = ceil_ev.get((wid, line))                                    # the event-best founder
            row["ceil_ev_founder"] = ce[0] if ce else "."
            row["ceil_ev_errors"] = ce[1]["event_errors"] if ce else ""
            row["ceil_ev_called"] = (ce[1]["qlen"] - ce[1]["query_N"]) if ce else ""
            for K in TRIM_KS:
                for x in TCOLS:
                    row[f"t{K}_{x}"] = s[f"t{K}_{x}"] if s else 0
                tc = tceil.get((wid, line, K))
                row[f"ceil_t{K}_founder"] = tc[0] if tc else "."
                for x in ("matches", "errors", "trimmed_bp", "called"):
                    row[f"ceil_t{K}_{x}"] = tc[1][f"t{K}_{x}"] if tc else ""
                row[f"t{K}_event_errors"] = s[f"t{K}_event_errors"] if s else 0
                row[f"ceil_t{K}_event_errors"] = tc[1][f"t{K}_event_errors"] if tc else ""
            rn = rand[(wid, line)]
            row["rand_n"] = rn[0]
            for k, v in zip(("rand_err_capped", "rand_event_errors", "rand_called"), rn[1:]):
                row[k] = f"{v:.6g}" if rn[0] else ""
            rows.append(row)
    with open(out / "per_hapwin.tsv", "w") as fo:
        fo.write("\t".join(cols) + "\n")
        for r in rows:
            fo.write("\t".join(str(r[c]) for c in cols) + "\n")
    with open(out / "ril_regions_check.tsv", "w") as fo:
        fo.write("class\toutcome\tn_windows\n")
        for cls, d in sorted(rilchk.items()):
            for k, v in sorted(d.items()):
                fo.write(f"{cls}\t{k}\t{v}\n")
    print("wrote", len(rows), "haplotype-window rows", file=sys.stderr)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--stitch", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--ril-truth", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    score(a.windows, Path(a.stitch) / "panel.tsv", Path(a.stitch) / "model_haps.tsv", load_scores(a.align),
          a.ril_truth, a.out)


if __name__ == "__main__":
    main()
