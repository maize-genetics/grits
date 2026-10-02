#!/usr/bin/env python
"""
Aggregate simval_batch.py's results TSV into the error-rate matrix (8
dataset types x 5 coverages) plus total processing time reported two ways
(summed per-row wall time, and the last batch run's observed makespan), and
the free diagnostic signals (het_scale by kind, dropped-founder indices
seen). See /home/zrm22/.claude/plans/swift-chasing-melody.md section 6.

Usage: simval_report.py [--json-out PATH]
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import simval_paths as P  # noqa: E402

DATASETS = ["IDX-INBRED", "IDX-HYB", "IDX-RIL", "OUT-INBRED", "OUT-HYB", "OUT-RIL",
            "MIX-HYB", "MIX-RIL"]
COVERAGES = ["0.01", "0.1", "0.5", "1.0", "2.0"]


def load_results():
    if not P.RESULTS_TSV.exists():
        return []
    with open(P.RESULTS_TSV) as f:
        return list(csv.DictReader(f, delimiter="\t"))


EVENTS_TSV = P.RESULTS_DIR / "simval_events_results.tsv"


def load_event_results():
    """Rows from simval_rescore_events.py (see
    /home/zrm22/.claude/plans/swift-chasing-melody.md) -- the per-EVENT
    companion to the per-panel-site metric above. Returns [] if that batch
    hasn't been run/finished yet, so this script keeps working before and
    after it lands."""
    if not EVENTS_TSV.exists():
        return []
    with open(EVENTS_TSV) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def dedupe_by_row_key(rows):
    """A row can appear more than once across repeated/resumed batch runs;
    last occurrence wins (most recent write)."""
    by_key = {}
    for r in rows:
        by_key[r.get("row_key", id(r))] = r
    return list(by_key.values())


def _float_or_none(v):
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def build_matrix(rows, field="error_rate"):
    """field='error_rate' is the headline strict-diploid-genotype metric (one
    misassigned haplotype scores the WHOLE site wrong); field='partial_error_rate'
    is, to within ~3% (measured: partial/strict ratio is 0.516 mean for
    hybrid, 0.557 for ril, exactly 1.0 for inbred -- see
    memory/simval_full_corpus_eval.md), the PER-HAPLOTYPE error rate -- the
    only version of this metric that's comparable across inbred (where one
    miss corrupts both alleles) and hybrid/RIL (where one miss corrupts
    exactly one)."""
    cells = {}
    for r in rows:
        er = _float_or_none(r.get(field))
        if er is None:
            continue
        cells.setdefault((r["dataset_id"], r["coverage"]), []).append(er)
    out = {}
    for key, vals in cells.items():
        mean = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        out[key] = {"mean": mean, "sd": sd, "n": len(vals)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-out", default=str(P.RESULTS_DIR / "simval_report.json"))
    args = ap.parse_args()

    rows = dedupe_by_row_key(load_results())
    if not rows:
        print(f"No results found at {P.RESULTS_TSV} -- run simval_batch.py first.")
        return

    mat = build_matrix(rows, "error_rate")
    mat_partial = build_matrix(rows, "partial_error_rate")

    event_rows = dedupe_by_row_key(load_event_results())
    mat_event_strict = build_matrix(event_rows, "event_error_rate_strict") if event_rows else {}
    mat_event_frac = build_matrix(event_rows, "event_error_rate_fractional") if event_rows else {}

    print("Error rate (%) by dataset type x coverage, mean +/- population sd over individuals")
    print("(STRICT diploid genotype match -- one misassigned haplotype scores the whole site wrong)\n")
    header = "dataset_id".ljust(12) + "".join(c.rjust(20) for c in COVERAGES)
    print(header)
    for d in DATASETS:
        cells = []
        for c in COVERAGES:
            v = mat.get((d, c))
            cells.append(f"{v['mean']*100:6.3f}+/-{v['sd']*100:5.3f}(n={v['n']})".rjust(20)
                         if v else "n/a".rjust(20))
        print(d.ljust(12) + "".join(cells))

    print("\nPer-haplotype error rate (%) -- same matrix, using partial_error_rate, the fair")
    print("comparison across inbred/hybrid/ril (inbred: one miss = both alleles; hybrid/ril:")
    print("one miss = one allele -- see memory/simval_full_corpus_eval.md)\n")
    print(header)
    for d in DATASETS:
        cells = []
        for c in COVERAGES:
            v = mat_partial.get((d, c))
            cells.append(f"{v['mean']*100:6.3f}+/-{v['sd']*100:5.3f}(n={v['n']})".rjust(20)
                         if v else "n/a".rjust(20))
        print(d.ljust(12) + "".join(cells))

    if event_rows:
        print("\n\nEVENT error rate (%) by dataset type x coverage, mean +/- population sd")
        print("(one unit per truth VARIANT EVENT, per haplotype -- NOT per panel site; STRICT: "
              "any compared site inside the event wrong -> whole event wrong)\n")
        print(header)
        for d in DATASETS:
            cells = []
            for c in COVERAGES:
                v = mat_event_strict.get((d, c))
                cells.append(f"{v['mean']*100:6.3f}+/-{v['sd']*100:5.3f}(n={v['n']})".rjust(20)
                             if v else "n/a".rjust(20))
            print(d.ljust(12) + "".join(cells))

        print("\nEVENT error rate (%), FRACTIONAL (mean of correct/compared sites per event)\n")
        print(header)
        for d in DATASETS:
            cells = []
            for c in COVERAGES:
                v = mat_event_frac.get((d, c))
                cells.append(f"{v['mean']*100:6.3f}+/-{v['sd']*100:5.3f}(n={v['n']})".rjust(20)
                             if v else "n/a".rjust(20))
            print(d.ljust(12) + "".join(cells))

        print("\nEvent-level per-class breakdown, aggregated across ALL scored rows:")
        for cls in ("SNP", "INS", "DEL"):
            tot = sum(int(r.get(f"events_total_{cls}", 0) or 0) for r in event_rows)
            obs = sum(int(r.get(f"events_observable_{cls}", 0) or 0) for r in event_rows)
            corr = sum(int(r.get(f"events_correct_strict_{cls}", 0) or 0) for r in event_rows)
            err = 100 * (1 - corr / obs) if obs else float("nan")
            print(f"  {cls:4s} total={tot:>12,d}  observable={obs:>12,d}  "
                  f"strict_error%={err:6.3f}")

        n_bad_inv = sum(1 for r in event_rows
                         if r.get("invariant_check_delta") not in (None, "")
                         and abs(float(r["invariant_check_delta"])) > 1e-3)
        n_bad_h1h2 = sum(1 for r in event_rows
                          if r.get("kind") == "inbred" and r.get("h1_eq_h2_events") == "False")
        print(f"\nEvent self-checks: invariant_delta>1e-3 in {n_bad_inv}/{len(event_rows)} rows; "
              f"inbred h1!=h2 events in {n_bad_h1h2} rows (both should be 0)")
    else:
        print(f"\n\n(no {EVENTS_TSV} found -- event-level metrics not yet computed; "
              f"run simval_rescore_events.py)")

    n_total = len(rows)
    n_scored = sum(1 for r in rows if _float_or_none(r.get("error_rate")) is not None)
    print(f"\nRows with a scored error_rate: {n_scored}/{n_total}")
    if n_scored < n_total:
        missing = [r["row_key"] for r in rows if _float_or_none(r.get("error_rate")) is None]
        print(f"  NOT scored (first 10 shown): {missing[:10]}")

    summed_wall = sum(_float_or_none(r.get("wall_seconds")) or 0.0 for r in rows)
    print(f"\nTotal processing time:")
    print(f"  summed per-row wall time : {summed_wall:.1f}s ({summed_wall/3600:.2f}h)")

    summary_path = P.RESULTS_DIR / "simval_batch_summary.json"
    makespan_s = None
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        makespan_s = s.get("makespan_s")
        print(f"  observed batch makespan (last run) : {makespan_s:.1f}s "
              f"({makespan_s/3600:.2f}h)")
    else:
        print(f"  (no {summary_path.name} found -- makespan not available)")

    # Free diagnostics.
    het_by_kind = {}
    for r in rows:
        hs = _float_or_none(r.get("align_het_scale"))
        if hs is None:
            continue
        het_by_kind.setdefault(r["kind"], []).append(hs)
    print("\nhet_scale by kind (expect ~0 for inbred, ~1 for hybrid/ril):")
    for k in sorted(het_by_kind):
        vals = het_by_kind[k]
        print(f"  {k:10s} mean={statistics.mean(vals):.3f}  n={len(vals)}")

    dropped_idx_seen = sorted({r.get("align_dropped_idx") for r in rows
                                if r.get("align_dropped_idx") not in (None, "")})
    print(f"\nfounder-drop index used (should be one fixed value across the whole batch): "
          f"{dropped_idx_seen}")

    event_class_totals = {}
    if event_rows:
        for cls in ("SNP", "INS", "DEL"):
            tot = sum(int(r.get(f"events_total_{cls}", 0) or 0) for r in event_rows)
            obs = sum(int(r.get(f"events_observable_{cls}", 0) or 0) for r in event_rows)
            corr = sum(int(r.get(f"events_correct_strict_{cls}", 0) or 0) for r in event_rows)
            event_class_totals[cls] = {
                "total": tot, "observable": obs, "correct_strict": corr,
                "strict_error_rate": (1 - corr / obs) if obs else None,
            }

    out_json = {
        "matrix": {f"{d}|{c}": v for (d, c), v in mat.items()},
        "matrix_partial_per_haplotype": {f"{d}|{c}": v for (d, c), v in mat_partial.items()},
        "matrix_event_strict": {f"{d}|{c}": v for (d, c), v in mat_event_strict.items()},
        "matrix_event_fractional": {f"{d}|{c}": v for (d, c), v in mat_event_frac.items()},
        "event_class_totals": event_class_totals,
        "n_rows_total": n_total,
        "n_rows_scored": n_scored,
        "n_event_rows_scored": len(event_rows),
        "summed_wall_seconds": summed_wall,
        "makespan_seconds": makespan_s,
        "het_scale_by_kind": {k: statistics.mean(v) for k, v in het_by_kind.items()},
        "dropped_idx_seen": dropped_idx_seen,
    }
    Path(args.json_out).write_text(json.dumps(out_json, indent=1))
    print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
