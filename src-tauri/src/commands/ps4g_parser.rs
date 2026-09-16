use parser_core::types::*;
use rustc_hash::FxHashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tauri::{Emitter, State};

/// Tauri-level cache that wraps `CachedPS4GData` with filesystem metadata
/// for cache invalidation based on file modification time.
#[derive(Debug, Clone)]
pub struct CachedPS4GFile {
    pub file_path: String,
    pub modified_time: std::time::SystemTime,
    pub data: CachedPS4GData,
}

pub struct PS4GCache {
    // `Arc` so a cache hit clones a reference rather than the whole parsed
    // file (tens of MB of row/gamete data on a real file) — this fires on
    // every chromosome load and every column-mode toggle.
    pub cache: Mutex<FxHashMap<String, Arc<CachedPS4GFile>>>,
}

impl PS4GCache {
    pub fn new() -> Self {
        PS4GCache {
            cache: Mutex::new(FxHashMap::default()),
        }
    }

    pub fn get_cached(&self, file_path: &str) -> Option<Arc<CachedPS4GFile>> {
        let cache = self.cache.lock().ok()?;
        let cached = cache.get(file_path)?;

        if let Ok(metadata) = std::fs::metadata(file_path) {
            if let Ok(modified) = metadata.modified() {
                if modified == cached.modified_time {
                    return Some(Arc::clone(cached));
                }
            }
        }
        None
    }

    pub fn store(&self, data: CachedPS4GFile) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(data.file_path.clone(), Arc::new(data));
        }
    }

    #[allow(dead_code)]
    pub fn invalidate(&self, file_path: &str) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.remove(file_path);
        }
    }
}

impl Default for PS4GCache {
    fn default() -> Self {
        Self::new()
    }
}

#[tauri::command]
pub async fn parse_ps4g_file(
    file_path: String,
    cache: State<'_, PS4GCache>,
    window: tauri::Window,
) -> Result<PS4GParseResult, String> {
    let path = Path::new(&file_path);

    if !path.exists() {
        return Err(format!("File not found: {}", file_path));
    }

    let file_metadata = std::fs::metadata(path)
        .map_err(|e| format!("Failed to get file metadata: {}", e))?;
    let file_size = file_metadata.len();
    let modified_time = file_metadata
        .modified()
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = BufReader::with_capacity(1024 * 1024, file);

    let window_clone = window.clone();
    let (result, cached_data) = parser_core::parse_ps4g(reader, Some(file_size), move |progress| {
        let _ = window_clone.emit("ps4g-progress", &progress);
    })?;

    cache.store(CachedPS4GFile {
        file_path,
        modified_time,
        data: cached_data,
    });

    Ok(result)
}

#[tauri::command]
pub async fn get_chromosome_matrix(
    file_path: String,
    chromosome: String,
    column_mode: Option<String>,
    cache: State<'_, PS4GCache>,
    window: tauri::Window,
) -> Result<ChromosomeMatrixResult, String> {
    let mode = ColumnMode::from_wire(column_mode.as_deref());
    let path = Path::new(&file_path);

    if !path.exists() {
        return Ok(ChromosomeMatrixResult {
            success: false,
            chromosome: chromosome.clone(),
            matrix: vec![],
            positions: vec![],
            gamete_names: vec![],
            num_gametes: 0,
            num_positions: 0,
            position_range: (0, 0),
            column_mode: mode,
            source_rows: vec![],
            error: Some(format!("File not found: {}", file_path)),
        });
    }

    if let Some(cached) = cache.get_cached(&file_path) {
        let _ = window.emit(
            "chromosome-matrix-progress",
            &ChromosomeMatrixProgress {
                rows_processed: cached.data.total_rows,
                chromosome: chromosome.clone(),
                percent: 100.0,
            },
        );
        return parser_core::build_chromosome_matrix(&cached.data, &chromosome, mode);
    }

    // No cache -- full parse then build matrix
    let file_metadata = std::fs::metadata(path)
        .map_err(|e| format!("Failed to get file metadata: {}", e))?;
    let file_size = file_metadata.len();
    let modified_time = file_metadata
        .modified()
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = BufReader::with_capacity(1024 * 1024, file);

    let window_clone = window.clone();
    let chr_for_progress = chromosome.clone();
    let (_, cached_data) = parser_core::parse_ps4g(reader, Some(file_size), move |progress| {
        let _ = window_clone.emit(
            "chromosome-matrix-progress",
            &ChromosomeMatrixProgress {
                rows_processed: progress.rows_processed,
                chromosome: chr_for_progress.clone(),
                percent: progress.percent,
            },
        );
    })?;

    let result = parser_core::build_chromosome_matrix(&cached_data, &chromosome, mode);

    cache.store(CachedPS4GFile {
        file_path,
        modified_time,
        data: cached_data,
    });

    result
}

#[tauri::command]
pub async fn get_chromosome_matrix_binary(
    file_path: String,
    chromosome: String,
    column_mode: Option<String>,
    cache: State<'_, PS4GCache>,
    window: tauri::Window,
) -> Result<ChromosomeMatrixBinaryResult, String> {
    let result = get_chromosome_matrix(file_path, chromosome, column_mode, cache, window).await?;
    Ok(parser_core::encode_matrix_binary(&result))
}
