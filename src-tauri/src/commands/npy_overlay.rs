use parser_core::types::NpyOverlayResult;
use parser_core::NpyRowMapping;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[tauri::command]
pub async fn load_npy_overlay(
    observed_path: Option<String>,
    predictions_path: Option<String>,
    expected_num_positions: usize,
    expected_num_gametes: usize,
    source_rows: Option<Vec<u32>>,
    total_rows: usize,
) -> Result<NpyOverlayResult, String> {
    if observed_path.is_none() && predictions_path.is_none() {
        return Ok(NpyOverlayResult::error(
            "At least one file path (observed or predictions) must be provided".to_string(),
        ));
    }

    let observed_reader = match &observed_path {
        Some(opath) => {
            let path = Path::new(opath);
            if !path.exists() {
                return Ok(NpyOverlayResult::error(format!(
                    "Observed file not found: {}",
                    opath
                )));
            }
            let file = File::open(path)
                .map_err(|e| format!("Failed to open observed file: {}", e))?;
            Some(BufReader::new(file))
        }
        None => None,
    };

    let predictions_reader = match &predictions_path {
        Some(ppath) => {
            let path = Path::new(ppath);
            if !path.exists() {
                return Ok(NpyOverlayResult::error(format!(
                    "Predictions file not found: {}",
                    ppath
                )));
            }
            let file = File::open(path)
                .map_err(|e| format!("Failed to open predictions file: {}", e))?;
            Some(BufReader::new(file))
        }
        None => None,
    };

    let row_mapping = source_rows
        .as_deref()
        .map(|source_rows| NpyRowMapping { source_rows, total_rows });

    parser_core::load_npy_overlay_from_readers(
        observed_reader,
        predictions_reader,
        expected_num_positions,
        expected_num_gametes,
        row_mapping,
    )
}
