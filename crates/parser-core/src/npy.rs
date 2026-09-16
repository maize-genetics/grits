use crate::types::NpyOverlayResult;
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use std::io::Read;

/// Try reading an .npy file as various numeric types from a reader, returning f64 values.
pub fn read_npy_as_f64_from_reader(mut reader: impl Read) -> Result<Array2<f64>, String> {
    let mut buf = Vec::new();
    reader
        .read_to_end(&mut buf)
        .map_err(|e| format!("Failed to read npy data: {}", e))?;

    if let Ok(arr) = try_read_typed::<f64>(&buf) {
        return Ok(arr);
    }
    if let Ok(arr) = try_read_typed::<f32>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(arr) = try_read_typed::<i64>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(arr) = try_read_typed::<i32>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(arr) = try_read_typed::<i16>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(arr) = try_read_typed::<u8>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(arr) = try_read_typed::<i8>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(arr) = try_read_typed::<u64>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(arr) = try_read_typed::<u32>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(arr) = try_read_typed::<u16>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }

    Err("Unsupported .npy dtype".to_string())
}

fn try_read_typed<T: ndarray_npy::ReadableElement>(buf: &[u8]) -> Result<Array2<T>, String> {
    let cursor = std::io::Cursor::new(buf);
    Array2::<T>::read_npy(cursor).map_err(|e| format!("{}", e))
}

/// How to line up an `.npy` file's rows with the heatmap's columns when its
/// row count doesn't equal the column count directly — e.g. a genome-wide
/// per-PS4G-row `.npy` being overlaid on a single chromosome. `source_rows`
/// names, per heatmap column, which `.npy` row to read (see
/// `ChromosomeMatrixResult::source_rows`); `total_rows` is the row count the
/// `.npy` must have for that mapping to apply.
pub struct NpyRowMapping<'a> {
    pub source_rows: &'a [u32],
    pub total_rows: usize,
}

/// Round a column to path-index values, preserving `-1` for values that
/// round to negative (a producer's "no label" sentinel) rather than
/// saturating them to `0`, which would silently draw a path point on
/// gamete row 0. `gather`, when present, selects and reorders rows first
/// (see `NpyRowMapping`).
fn extract_path_columns(
    arr: &Array2<f64>,
    start_col: usize,
    gather: Option<&[u32]>,
) -> Vec<Vec<i64>> {
    let ncols = arr.ncols();
    let mut paths = Vec::new();
    for col_idx in start_col..ncols {
        let column = arr.column(col_idx);
        let col: Vec<i64> = match gather {
            Some(source_rows) => source_rows
                .iter()
                .map(|&r| column[r as usize].round() as i64)
                .collect(),
            None => column.iter().map(|&v| v.round() as i64).collect(),
        };
        paths.push(col);
    }
    paths
}

/// Resolve how a single `.npy` file's row count should be interpreted:
/// gathered through `mapping` (checked first — see below), read directly, or
/// rejected as neither. Returns the number of heatmap columns the result
/// will have.
///
/// Gather is checked *before* direct equality. If direct were checked first,
/// a single-chromosome file where `expected_num_positions == mapping.total_rows`
/// would take the direct path even though `source_rows` is a permutation (not
/// necessarily the identity) — silently producing a scrambled overlay instead
/// of erroring or gathering correctly. Gathering with an identity mapping
/// produces the same result as direct indexing, so gather-first is never
/// worse.
fn resolve_alignment<'a>(
    nrows: usize,
    expected_num_positions: usize,
    mapping: Option<&NpyRowMapping<'a>>,
) -> Result<Option<&'a [u32]>, String> {
    if let Some(m) = mapping {
        if nrows == m.total_rows {
            return Ok(Some(m.source_rows));
        }
    }
    if nrows == expected_num_positions {
        return Ok(None);
    }
    match mapping {
        Some(m) => Err(format!(
            "file has {} rows but expected {} positions (direct) or {} rows (genome-wide, gathered)",
            nrows, expected_num_positions, m.total_rows
        )),
        None => Err(format!(
            "file has {} rows but expected {} positions (from PS4G chromosome data)",
            nrows, expected_num_positions
        )),
    }
}

/// Load NPY overlay data from readers. Pass `None` for either reader if that
/// file is not available. `row_mapping`, when present, lets a genome-wide
/// `.npy` (one row per PS4G data row, across every chromosome) align to the
/// current chromosome's columns — see [`NpyRowMapping`].
pub fn load_npy_overlay_from_readers(
    observed_reader: Option<impl Read>,
    predictions_reader: Option<impl Read>,
    expected_num_positions: usize,
    expected_num_gametes: usize,
    row_mapping: Option<NpyRowMapping<'_>>,
) -> Result<NpyOverlayResult, String> {
    let mut true_paths: Vec<Vec<i64>> = vec![];
    let mut predicted_paths: Vec<Vec<i64>> = vec![];
    let mut is_diploid_true = false;
    let mut is_diploid_predicted = false;
    let mut num_positions = expected_num_positions;

    if observed_reader.is_none() && predictions_reader.is_none() {
        return Ok(NpyOverlayResult::error(
            "At least one file (observed or predictions) must be provided".to_string(),
        ));
    }

    if let Some(reader) = observed_reader {
        let observed = read_npy_as_f64_from_reader(reader)?;
        let (nrows, ncols) = (observed.nrows(), observed.ncols());

        let gather = match resolve_alignment(nrows, expected_num_positions, row_mapping.as_ref()) {
            Ok(g) => g,
            Err(msg) => return Ok(NpyOverlayResult::error(format!("Observed {}", msg))),
        };
        let out_positions = gather.map(|g| g.len()).unwrap_or(nrows);

        let label_cols = ncols.saturating_sub(expected_num_gametes);
        if label_cols == 0 {
            return Ok(NpyOverlayResult::error(format!(
                "Observed file has {} columns but expected at least {} gamete columns + label columns",
                ncols, expected_num_gametes
            )));
        }

        if label_cols > 2 {
            return Ok(NpyOverlayResult::error(format!(
                "Observed file has {} label columns (expected 1 for haploid or 2 for diploid). \
                 Total cols: {}, gametes: {}",
                label_cols, ncols, expected_num_gametes
            )));
        }

        if let Some(source_rows) = gather {
            if let Some(&bad) = source_rows.iter().find(|&&r| r as usize >= nrows) {
                return Ok(NpyOverlayResult::error(format!(
                    "Observed file has {} rows but a mapped column references row {}",
                    nrows, bad
                )));
            }
        }

        true_paths = extract_path_columns(&observed, expected_num_gametes, gather);
        is_diploid_true = true_paths.len() == 2;
        num_positions = out_positions;
    }

    if let Some(reader) = predictions_reader {
        let preds = read_npy_as_f64_from_reader(reader)?;
        let (nrows, ncols) = (preds.nrows(), preds.ncols());

        let gather = match resolve_alignment(nrows, expected_num_positions, row_mapping.as_ref()) {
            Ok(g) => g,
            Err(msg) => return Ok(NpyOverlayResult::error(format!("Predictions {}", msg))),
        };
        let out_positions = gather.map(|g| g.len()).unwrap_or(nrows);

        if ncols == 0 || ncols > 2 {
            return Ok(NpyOverlayResult::error(format!(
                "Predictions file has {} columns (expected 1 for haploid or 2 for diploid)",
                ncols
            )));
        }

        if let Some(source_rows) = gather {
            if let Some(&bad) = source_rows.iter().find(|&&r| r as usize >= nrows) {
                return Ok(NpyOverlayResult::error(format!(
                    "Predictions file has {} rows but a mapped column references row {}",
                    nrows, bad
                )));
            }
        }

        predicted_paths = extract_path_columns(&preds, 0, gather);
        is_diploid_predicted = predicted_paths.len() == 2;
        num_positions = out_positions;
    }

    Ok(NpyOverlayResult {
        success: true,
        true_paths,
        predicted_paths,
        num_positions,
        is_diploid_true,
        is_diploid_predicted,
        error: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use ndarray_npy::WriteNpyExt;
    use std::io::Cursor;

    fn npy_bytes(arr: &Array2<f64>) -> Vec<u8> {
        let mut buf = Vec::new();
        arr.write_npy(&mut buf).unwrap();
        buf
    }

    #[test]
    fn direct_alignment_still_accepted_when_rows_match_columns() {
        // 3 positions, 1 gamete column, 1 label column.
        let arr = array![[0.0, 1.0], [0.0, 0.0], [0.0, 1.0]];
        let bytes = npy_bytes(&arr);
        let result = load_npy_overlay_from_readers(
            Some(Cursor::new(bytes)),
            None::<Cursor<Vec<u8>>>,
            3,
            1,
            None,
        )
        .unwrap();

        assert!(result.success);
        assert_eq!(result.true_paths, vec![vec![1, 0, 1]]);
        assert_eq!(result.num_positions, 3);
    }

    #[test]
    fn gather_alignment_selects_mapped_rows() {
        // Genome-wide file: 5 rows, 1 gamete column + 1 label column.
        // Chromosome has 3 columns, mapped to genome-wide rows [3, 1, 4].
        let arr = array![
            [0.0, 10.0],
            [0.0, 11.0],
            [0.0, 12.0],
            [0.0, 13.0],
            [0.0, 14.0],
        ];
        let bytes = npy_bytes(&arr);
        let source_rows = [3u32, 1, 4];
        let mapping = NpyRowMapping { source_rows: &source_rows, total_rows: 5 };
        let result = load_npy_overlay_from_readers(
            Some(Cursor::new(bytes)),
            None::<Cursor<Vec<u8>>>,
            3, // expected_num_positions -- deliberately != total_rows
            1,
            Some(mapping),
        )
        .unwrap();

        assert!(result.success);
        assert_eq!(result.true_paths, vec![vec![13, 11, 14]]);
        assert_eq!(result.num_positions, 3);
    }

    #[test]
    fn gather_wins_when_both_counts_coincide() {
        // total_rows == expected_num_positions (both 3), but source_rows is
        // NOT the identity permutation -- direct-first would silently
        // scramble this; gather-first must still reorder correctly.
        let arr = array![[0.0, 100.0], [0.0, 200.0], [0.0, 300.0]];
        let bytes = npy_bytes(&arr);
        let source_rows = [2u32, 0, 1];
        let mapping = NpyRowMapping { source_rows: &source_rows, total_rows: 3 };
        let result = load_npy_overlay_from_readers(
            Some(Cursor::new(bytes)),
            None::<Cursor<Vec<u8>>>,
            3,
            1,
            Some(mapping),
        )
        .unwrap();

        assert!(result.success);
        assert_eq!(result.true_paths, vec![vec![300, 100, 200]]);
    }

    #[test]
    fn out_of_bounds_source_row_errors_not_panics() {
        let arr = array![[0.0, 1.0], [0.0, 2.0]];
        let bytes = npy_bytes(&arr);
        let source_rows = [0u32, 5]; // 5 is out of bounds for a 2-row file
        let mapping = NpyRowMapping { source_rows: &source_rows, total_rows: 2 };
        let result = load_npy_overlay_from_readers(
            Some(Cursor::new(bytes)),
            None::<Cursor<Vec<u8>>>,
            2,
            1,
            Some(mapping),
        )
        .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("row 5"));
    }

    #[test]
    fn mismatched_rows_error_names_both_accepted_counts() {
        let arr = array![[0.0, 1.0], [0.0, 2.0]]; // 2 rows
        let bytes = npy_bytes(&arr);
        let source_rows = [0u32, 1, 2];
        let mapping = NpyRowMapping { source_rows: &source_rows, total_rows: 999 };
        let result = load_npy_overlay_from_readers(
            Some(Cursor::new(bytes)),
            None::<Cursor<Vec<u8>>>,
            42,
            1,
            Some(mapping),
        )
        .unwrap();

        assert!(!result.success);
        let err = result.error.unwrap();
        assert!(err.contains("42"));
        assert!(err.contains("999"));
    }

    #[test]
    fn negative_label_becomes_minus_one_not_row_zero() {
        let arr = array![[0.0, -1.0], [0.0, 0.0]];
        let bytes = npy_bytes(&arr);
        let result = load_npy_overlay_from_readers(
            Some(Cursor::new(bytes)),
            None::<Cursor<Vec<u8>>>,
            2,
            1,
            None,
        )
        .unwrap();

        assert_eq!(result.true_paths, vec![vec![-1, 0]]);
    }

    #[test]
    fn gather_sets_num_positions_to_column_count_not_npy_row_count() {
        let arr = array![[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]];
        let bytes = npy_bytes(&arr);
        let source_rows = [1u32, 3];
        let mapping = NpyRowMapping { source_rows: &source_rows, total_rows: 4 };
        let result = load_npy_overlay_from_readers(
            Some(Cursor::new(bytes)),
            None::<Cursor<Vec<u8>>>,
            2,
            1,
            Some(mapping),
        )
        .unwrap();

        assert_eq!(result.num_positions, 2); // column count, not the npy's 4 rows
    }
}
