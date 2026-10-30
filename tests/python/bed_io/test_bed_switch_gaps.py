import os
import tempfile

import pandas as pd

from python.bed_io.bed import output_collapse_bed


def _collapse(rows, **kw):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".bed") as f:
        path = f.name
    try:
        output_collapse_bed(pd.DataFrame(rows), path, **kw)
        return pd.read_csv(path, sep="\t")
    finally:
        os.remove(path)


def _rows(spec, with_row=True):
    """spec: list of (chrom, pos, p1, p2[, row])"""
    out = []
    for i, t in enumerate(spec):
        r = {"chrom": t[0], "start": t[1], "end": t[1] + 1, "parent1": t[2], "parent2": t[3]}
        if with_row:
            r["row"] = t[4] if len(t) > 4 else i
        out.append(r)
    return out


SWITCH = [("chr1", 100, "A", "A"), ("chr1", 200, "A", "A"), ("chr1", 1000, "B", "B"), ("chr1", 1100, "B", "B")]


def test_default_leaves_switch_gap():
    df = _collapse(_rows(SWITCH))
    assert list(df.columns) == ["chrom", "start", "end", "parent1", "parent2"]
    assert df[["start", "end"]].values.tolist() == [[100, 201], [1000, 1101]]


def test_fill_splits_switch_gap_at_midpoint():
    df = _collapse(_rows(SWITCH), fill_switch_gaps=True)
    assert df[["start", "end"]].values.tolist() == [[100, 600], [600, 1101]]
    assert list(df.columns) == ["chrom", "start", "end", "parent1", "parent2"]


def test_fill_keeps_gap_left_by_skipped_rows():
    spec = [("chr1", 100, "A", "A", 0), ("chr1", 200, "A", "A", 1), ("chr1", 1000, "B", "B", 5)]
    df = _collapse(_rows(spec), fill_switch_gaps=True)
    assert df[["start", "end"]].values.tolist() == [[100, 201], [1000, 1001]]


def test_fill_never_crosses_chromosomes():
    spec = [("chr1", 100, "A", "A"), ("chr2", 50, "B", "B")]
    df = _collapse(_rows(spec), fill_switch_gaps=True)
    assert df[["chrom", "start", "end"]].values.tolist() == [["chr1", 100, 101], ["chr2", 50, 51]]


def test_fill_without_row_column_treats_runs_as_adjacent():
    df = _collapse(_rows(SWITCH, with_row=False), fill_switch_gaps=True)
    assert df[["start", "end"]].values.tolist() == [[100, 600], [600, 1101]]


def test_fill_leaves_touching_bins_alone():
    rows = [{"chrom": "chr1", "start": 0, "end": 256, "parent1": "A", "parent2": "A"},
            {"chrom": "chr1", "start": 256, "end": 512, "parent1": "B", "parent2": "B"}]
    df = _collapse(rows, fill_switch_gaps=True)
    assert df[["start", "end"]].values.tolist() == [[0, 256], [256, 512]]


def test_fill_total_span_is_tiled_without_overlap():
    spec = [("chr1", p, "AB"[i // 3 % 2], "A") for i, p in enumerate(range(0, 9000, 500))]
    df = _collapse(_rows(spec), fill_switch_gaps=True)
    assert (df["start"].to_numpy()[1:] == df["end"].to_numpy()[:-1]).all()
    assert df["start"].iloc[0] == 0 and df["end"].iloc[-1] == 8501
