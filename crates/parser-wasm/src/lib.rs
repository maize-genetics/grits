use js_sys::Function;
use parser_core::types::*;
use serde::Serialize;
use std::io::Cursor;
use wasm_bindgen::prelude::*;

fn to_js_value<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    value
        .serialize(&serializer)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ============================================================================
// PS4G
// ============================================================================

/// Opaque handle holding parsed PS4G data in WASM memory.
/// Allows chromosome matrix retrieval without re-parsing.
#[wasm_bindgen]
pub struct PS4GFileHandle {
    cached: CachedPS4GData,
}

#[wasm_bindgen]
impl PS4GFileHandle {
    /// Build a chromosome matrix from the cached data. `column_mode` is
    /// `"binned"` or `"row"`; anything else (including omission) defaults to
    /// `"binned"`.
    pub fn get_chromosome_matrix(
        &self,
        chromosome: &str,
        column_mode: Option<String>,
    ) -> Result<JsValue, JsValue> {
        let mode = ColumnMode::from_wire(column_mode.as_deref());
        let result = parser_core::build_chromosome_matrix(&self.cached, chromosome, mode)
            .map_err(|e| JsValue::from_str(&e))?;
        to_js_value(&result)
    }

    /// Encode a chromosome matrix as a compact base64 binary payload.
    pub fn get_chromosome_matrix_binary(
        &self,
        chromosome: &str,
        column_mode: Option<String>,
    ) -> Result<JsValue, JsValue> {
        let mode = ColumnMode::from_wire(column_mode.as_deref());
        let matrix = parser_core::build_chromosome_matrix(&self.cached, chromosome, mode)
            .map_err(|e| JsValue::from_str(&e))?;
        let binary = parser_core::encode_matrix_binary(&matrix);
        to_js_value(&binary)
    }
}

/// Parse a PS4G file from raw bytes.
/// Returns a `PS4GParseResult` as a JS object. The `PS4GFileHandle` for
/// subsequent chromosome matrix calls is returned via `get_ps4g_handle`
/// on the same data.
#[wasm_bindgen]
pub fn parse_ps4g_file(
    data: &[u8],
    file_size: u64,
    on_progress: Option<Function>,
) -> Result<JsValue, JsValue> {
    let cursor = Cursor::new(data);
    let (result, _cached) = parser_core::parse_ps4g(
        cursor,
        Some(file_size),
        |p: PS4GProgress| {
            if let Some(ref f) = on_progress {
                if let Ok(val) = to_js_value(&p) {
                    let _ = f.call1(&JsValue::NULL, &val);
                }
            }
        },
    )
    .map_err(|e| JsValue::from_str(&e))?;

    to_js_value(&result)
}

/// Parse a PS4G file and return an opaque handle for chromosome matrix queries.
#[wasm_bindgen]
pub fn parse_ps4g_to_handle(
    data: &[u8],
    file_size: u64,
    on_progress: Option<Function>,
) -> Result<PS4GFileHandle, JsValue> {
    let cursor = Cursor::new(data);
    let (_result, cached) = parser_core::parse_ps4g(
        cursor,
        Some(file_size),
        |p: PS4GProgress| {
            if let Some(ref f) = on_progress {
                if let Ok(val) = to_js_value(&p) {
                    let _ = f.call1(&JsValue::NULL, &val);
                }
            }
        },
    )
    .map_err(|e| JsValue::from_str(&e))?;

    Ok(PS4GFileHandle { cached })
}

// ============================================================================
// BED
// ============================================================================

/// Opaque handle holding raw BED file bytes in WASM memory.
/// BED parsing does not cache intermediate data the way PS4G does, so the
/// handle stores the raw bytes for re-reading during matrix construction.
#[wasm_bindgen]
pub struct BEDFileHandle {
    data: Vec<u8>,
    file_size: u64,
}

#[wasm_bindgen]
impl BEDFileHandle {
    /// Build a chromosome matrix for the given chromosome.
    pub fn get_chromosome_matrix(
        &self,
        chromosome: &str,
        on_progress: Option<Function>,
    ) -> Result<JsValue, JsValue> {
        let cursor = Cursor::new(&self.data);
        let result = parser_core::build_bed_chromosome_matrix(
            cursor,
            Some(self.file_size),
            chromosome,
            |p: BEDMatrixProgress| {
                if let Some(ref f) = on_progress {
                    if let Ok(val) = to_js_value(&p) {
                        let _ = f.call1(&JsValue::NULL, &val);
                    }
                }
            },
        )
        .map_err(|e| JsValue::from_str(&e))?;

        to_js_value(&result)
    }
}

/// Parse a BED file from raw bytes.
#[wasm_bindgen]
pub fn parse_bed_file(
    data: &[u8],
    file_size: u64,
    on_progress: Option<Function>,
) -> Result<JsValue, JsValue> {
    let cursor = Cursor::new(data);
    let result = parser_core::parse_bed(
        cursor,
        Some(file_size),
        |p: BEDProgress| {
            if let Some(ref f) = on_progress {
                if let Ok(val) = to_js_value(&p) {
                    let _ = f.call1(&JsValue::NULL, &val);
                }
            }
        },
    )
    .map_err(|e| JsValue::from_str(&e))?;

    to_js_value(&result)
}

/// Parse a BED file and return an opaque handle for chromosome matrix queries.
#[wasm_bindgen]
pub fn parse_bed_to_handle(
    data: &[u8],
    file_size: u64,
    on_progress: Option<Function>,
) -> Result<BEDFileHandle, JsValue> {
    let cursor = Cursor::new(data);
    // Run the initial parse to validate the data and report progress
    let _result = parser_core::parse_bed(
        cursor,
        Some(file_size),
        |p: BEDProgress| {
            if let Some(ref f) = on_progress {
                if let Ok(val) = to_js_value(&p) {
                    let _ = f.call1(&JsValue::NULL, &val);
                }
            }
        },
    )
    .map_err(|e| JsValue::from_str(&e))?;

    Ok(BEDFileHandle {
        data: data.to_vec(),
        file_size,
    })
}

// ============================================================================
// NPY Overlay
// ============================================================================

/// Load NPY overlay data from raw bytes.
/// Pass empty `Uint8Array` (length 0) for files that should be skipped.
/// `source_rows` + `total_rows` optionally let a genome-wide `.npy` (one row
/// per PS4G data row, across every chromosome) align to the current
/// chromosome's columns — see `NpyRowMapping`. Pass an empty `source_rows`
/// (length 0) to skip that and rely on direct row-count alignment only,
/// mirroring the empty-buffer-means-skip convention used for the file data
/// above.
#[wasm_bindgen]
pub fn load_npy_overlay(
    observed_data: &[u8],
    predictions_data: &[u8],
    expected_num_positions: usize,
    expected_num_gametes: usize,
    source_rows: Vec<u32>,
    total_rows: usize,
) -> Result<JsValue, JsValue> {
    let observed_reader: Option<Cursor<&[u8]>> = if observed_data.is_empty() {
        None
    } else {
        Some(Cursor::new(observed_data))
    };

    let predictions_reader: Option<Cursor<&[u8]>> = if predictions_data.is_empty() {
        None
    } else {
        Some(Cursor::new(predictions_data))
    };

    let row_mapping = if source_rows.is_empty() {
        None
    } else {
        Some(parser_core::NpyRowMapping {
            source_rows: &source_rows,
            total_rows,
        })
    };

    let result = parser_core::load_npy_overlay_from_readers(
        observed_reader,
        predictions_reader,
        expected_num_positions,
        expected_num_gametes,
        row_mapping,
    )
    .map_err(|e| JsValue::from_str(&e))?;

    to_js_value(&result)
}
