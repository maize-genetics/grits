"""Gene-window scorer: in-panel extraction rule and random-founder baseline (gw_score + eval_genewindow._rates)."""
import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT / "experiments" / "seq-extract"))
sys.path.insert(0, str(WT / "experiments" / "simval-corpus" / "scripts"))
import gw_score  # noqa: E402
from gw_align import SCORE_COLS  # noqa: E402

PANEL_FOUNDERS = {"B73", "B97", "CML228"}


def srow(h, line, errors, events, qlen=1000, query_N=0, contig="chr1", status="aligned"):
    r = {c: 0 for c in SCORE_COLS}
    r.update(seq_hash=h, line=line, qlen=qlen, query_N=query_N, status=status, contig=contig, tstart=0, tend=qlen,
             strand="+", mapq=60, matches=qlen - query_N - errors, mismatches=errors, errors=errors,
             event_errors=events)
    return r


def prow(wid, founder, h, status="ok", chrom="chr1"):
    return {"wid": wid, "chrom": chrom, "founder": founder, "status": status, "seq_hash": h}


def test_truth_unextractable_rule():
    sc = {("b73", "B73"): srow("b73", "B73", 0, 0), ("b97", "B97"): srow("b97", "B97", 0, 0, contig="chr5")}
    pw = {"B73": prow("w", "B73", "b73"), "B97": prow("w", "B97", "b97"),
          "CML228": prow("w", "CML228", ".", status="projection_failed")}
    ux = lambda line: gw_score.truth_unextractable(pw, line, "chr1", sc, PANEL_FOUNDERS)  # noqa: E731
    assert not ux("B73")             # own sequence ok and scored on the window's chromosome
    assert ux("B97")                 # own sequence aligns off-chromosome
    assert ux("CML228")              # own sequence failed extraction
    assert not ux("A188")            # out-of-panel truth is never affected
    assert gw_score.truth_unextractable({}, "B73", "chr1", sc, PANEL_FOUNDERS)    # no panel row


def test_random_founder_exact_expectation():
    sc = {("a", "A188"): srow("a", "A188", 10, 2),
          ("b", "A188"): srow("b", "A188", 2000, 30, qlen=1000, query_N=200),     # capped at called = 800
          ("c", "A188"): srow("c", "A188", 5, 1, contig="chr9"),                 # mislocated: not scorable
          }
    pw = {"B73": prow("w", "B73", "a"), "B97": prow("w", "B97", "b"), "CML228": prow("w", "CML228", "c"),
          "Ki3": prow("w", "Ki3", "d", status="projection_failed")}
    n, e, ev, called = gw_score.random_founder(pw, "A188", "chr1", sc)
    assert n == 2
    assert e == pytest.approx((10 + 800) / 2)
    assert ev == pytest.approx((2 + 30) / 2)
    assert called == pytest.approx((1000 + 800) / 2)
    assert gw_score.random_founder({}, "A188", "chr1", sc) == (0, 0.0, 0.0, 0.0)


def _tsv(path, rows, cols):
    with open(path, "w") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def test_score_end_to_end(tmp_path):
    """Two windows, panel B73/B97/CML228; inbred samples B97 (in panel) and A188 (out of panel).
    w2: B97's own panel sequence failed extraction -> B97 haplotype windows there are truth_unextractable."""
    import eval_genewindow
    _tsv(tmp_path / "windows.tsv", [{"wid": "w1", "chrom": "chr1"}, {"wid": "w2", "chrom": "chr1"}], ["wid", "chrom"])
    _tsv(tmp_path / "ril.tsv", [], ["class", "sample", "wid", "truth"])
    panel = [prow("w1", "B73", "p1a"), prow("w1", "B97", "p1b"), prow("w1", "CML228", "p1c"),
             prow("w2", "B73", "p2a"), prow("w2", "B97", ".", status="projection_failed"), prow("w2", "CML228", "p2c")]
    _tsv(tmp_path / "panel.tsv", panel, ["wid", "chrom", "founder", "status", "seq_hash"])
    errs = {  # (panel seq, truth line) -> (bp errors, event errors)
        ("p1a", "B97"): (40, 4), ("p1b", "B97"): (0, 0), ("p1c", "B97"): (20, 2),
        ("p2a", "B97"): (60, 6), ("p2c", "B97"): (30, 3),
        ("p1a", "A188"): (50, 5), ("p1b", "A188"): (10, 1), ("p1c", "A188"): (30, 3),
        ("p2a", "A188"): (8, 2), ("p2c", "A188"): (12, 4)}
    sc = {(h, ln): srow(h, ln, e, ev) for (h, ln), (e, ev) in errs.items()}
    haps = []
    for cls, ind, model in (("IDX-INBRED", "B97", "m_b97"), ("OUT-INBRED", "A188", "m_a188")):
        for wid in ("w1", "w2"):
            h = f"{model}_{wid}"
            sc[(h, ind)] = srow(h, ind, 5 if ind == "B97" else 10, 1 if ind == "B97" else 3)
            for hap in (1, 2):
                haps.append({"depth": "0.1", "class": cls, "sample": ind, "kind": "inbred", "wid": wid, "hap": hap,
                             "founders": "B73", "n_switches": 0, "gap_bases": 0, "failed_bases": 0, "seq_hash": h})
    _tsv(tmp_path / "model_haps.tsv", haps, list(haps[0]))
    out = tmp_path / "out"
    gw_score.score(tmp_path / "windows.tsv", tmp_path / "panel.tsv", tmp_path / "model_haps.tsv", sc,
                   tmp_path / "ril.tsv", out)
    d = pd.read_csv(out / "per_hapwin.tsv", sep="\t", dtype={"depth": str})
    st = d.set_index(["sample", "wid", "hap"]).sort_index().state
    assert (st.loc["B97", "w2"] == "truth_unextractable").all()
    assert (st.loc["B97", "w1"] == "scored").all()
    assert (st.loc["A188"] == "scored").all()          # out-of-panel truth unaffected by B97's failure
    r = d.set_index(["sample", "wid"]).groupby(level=[0, 1]).first()
    assert r.loc[("A188", "w1"), "rand_n"] == 3
    assert r.loc[("A188", "w1"), "rand_err_capped"] == pytest.approx(30)       # (50 + 10 + 30) / 3
    assert r.loc[("A188", "w2"), "rand_n"] == 2
    assert r.loc[("A188", "w2"), "rand_event_errors"] == pytest.approx(3)      # (2 + 4) / 2
    assert r.loc[("B97", "w1"), "rand_err_capped"] == pytest.approx(20)       # (40 + 0 + 20) / 3
    cl = pd.read_csv(out / "ceiling.tsv", sep="\t").set_index(["wid", "line"])
    assert cl.loc[("w2", "B97"), "truth_unextractable"] == 1
    assert cl.loc[("w2", "A188"), "truth_unextractable"] == 0

    for c in ("pick_rank", "n_ranked", "pick_panel_errors"):
        d[c] = float("nan")
    rates = d.groupby("class").apply(eval_genewindow._rates, include_groups=False)
    b = rates.loc["IDX-INBRED"]
    # only w1 is scored for B97: model 5/1000 bp, ceiling = B97 itself (0), random = 20/1000
    assert b.pct_truth_unextractable == pytest.approx(50)
    assert b.err_pct_capped == pytest.approx(0.5)
    assert b.ceil_err_pct_capped == pytest.approx(0)
    assert b.rand_err_pct_capped == pytest.approx(2.0)
    assert b.rand_event_err_pct == pytest.approx(0.2)                          # (4+0+2)/3 / 1000
    assert b.r2c_bp_pct == pytest.approx(100 * (2.0 - 0.5) / 2.0)
    assert b.r2c_event_pct == pytest.approx(100 * (0.2 - 0.1) / 0.2)
    a = rates.loc["OUT-INBRED"]
    # A188: both windows; random pooled = sum(mean errors) / sum(mean called) = (30 + 10) / 2000
    assert a.pct_truth_unextractable == 0
    assert a.rand_err_pct_capped == pytest.approx(100 * 40 / 2000)
    assert a.rand_event_err_pct == pytest.approx(100 * 6 / 2000)               # (3 + 3) / 2000
    assert a.err_pct_capped == pytest.approx(1.0)                              # 10 per 1000 per window
    assert a.ceil_err_pct_capped == pytest.approx(100 * (10 + 8) / 2000)
    assert a.r2c_bp_pct == pytest.approx(100 * (2.0 - 1.0) / (2.0 - 0.9))
