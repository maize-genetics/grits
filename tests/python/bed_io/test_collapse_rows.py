import pandas as pd

from src.python.bed_io.bed import output_collapse_bed


def _collapse(tmp_path, rows, **kw):
    out = tmp_path / "x.bed"
    output_collapse_bed(pd.DataFrame(rows), str(out), **kw)
    return pd.read_csv(out, sep="\t").values.tolist()


def _row(r, pos, a, b):
    return {"chrom": "chr1", "start": pos, "end": pos + 1, "parent1": a, "parent2": b, "row": r}


def test_skipped_row_between_same_pair_stays_uncalled(tmp_path):
    # row 1 was a null-state call and is not in the input
    rows = [_row(0, 100, "A", "A"), _row(2, 300, "A", "A"), _row(3, 400, "A", "A")]
    got = _collapse(tmp_path, rows, fill_switch_gaps=True)
    assert got == [["chr1", 100, 101, "A", "A"], ["chr1", 300, 401, "A", "A"]]


def test_consecutive_same_pair_merges(tmp_path):
    rows = [_row(0, 100, "A", "A"), _row(1, 200, "A", "A"), _row(2, 300, "A", "A")]
    assert _collapse(tmp_path, rows, fill_switch_gaps=True) == [["chr1", 100, 301, "A", "A"]]


def test_switch_between_consecutive_rows_is_filled(tmp_path):
    rows = [_row(0, 100, "A", "A"), _row(1, 300, "B", "B")]
    assert _collapse(tmp_path, rows, fill_switch_gaps=True) == [
        ["chr1", 100, 200, "A", "A"], ["chr1", 200, 301, "B", "B"]]


def test_without_row_column_unchanged(tmp_path):
    rows = [{k: v for k, v in _row(r, p, "A", "A").items() if k != "row"}
            for r, p in [(0, 100), (2, 300)]]
    assert _collapse(tmp_path, rows) == [["chr1", 100, 301, "A", "A"]]
