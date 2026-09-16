use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// PS4G Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameteInfo {
    /// Display label for this gamete. Bare sample name by default (e.g.
    /// "B73"); see `build_chromosome_matrix` for when a ":idx" suffix is
    /// added to disambiguate a collision.
    pub gamete: String,
    /// Sample name only, with any ":idx" suffix stripped (e.g. "B73").
    pub sample_name: String,
    /// Haplotype/gamete index parsed from a ":idx" suffix, or 0 when the
    /// header field had no suffix (per the PS4G spec, both forms are valid).
    pub gamete_idx: u32,
    pub gamete_index: u32,
    pub read_count: u64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PS4GDataRow {
    pub gamete_set: Vec<u32>,
    pub ref_contig: String,
    pub ref_pos_binned: u64,
    pub count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PS4GMetadata {
    pub version: Option<String>,
    pub command: Option<String>,
    pub total_unique_counts: Option<u64>,
    pub gametes: Vec<GameteInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PS4GSummary {
    pub total_rows: usize,
    /// Sum of the data-section `count` column across the whole file — the
    /// true read total. Distinct from summing per-gamete `read_count`,
    /// which double-counts reads whose `gameteSet` names several gametes.
    pub total_read_count: u64,
    pub unique_positions: usize,
    pub chromosomes: Vec<String>,
    pub chromosome_counts: HashMap<String, usize>,
    pub gamete_count: usize,
    pub position_range: HashMap<String, (u64, u64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PS4GProgress {
    pub rows_processed: usize,
    pub bytes_processed: u64,
    pub total_bytes: u64,
    pub percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PS4GParseResult {
    pub success: bool,
    pub metadata: PS4GMetadata,
    pub summary: PS4GSummary,
    pub data_preview: Vec<PS4GDataRow>,
    pub error: Option<String>,
}

/// Which axis a chromosome matrix's columns run over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ColumnMode {
    /// One column per distinct `refPosBinned` value; rows sharing a bin are
    /// summed together. This is the historical/default behavior.
    #[default]
    Binned,
    /// One column per PS4G data row, matching the file's own layout.
    Row,
}

impl ColumnMode {
    /// Parse from the wire representation used at the JS boundary
    /// (`"binned"` / `"row"`), defaulting to `Binned` for anything else so a
    /// stale or missing frontend value degrades safely.
    pub fn from_wire(value: Option<&str>) -> Self {
        match value {
            Some("row") => ColumnMode::Row,
            _ => ColumnMode::Binned,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromosomeMatrixResult {
    pub success: bool,
    pub chromosome: String,
    pub matrix: Vec<Vec<u32>>,
    pub positions: Vec<u64>,
    pub gamete_names: Vec<String>,
    pub num_gametes: usize,
    pub num_positions: usize,
    pub position_range: (u64, u64),
    /// Column model used to build this matrix.
    pub column_mode: ColumnMode,
    /// The global PS4G data-row index (0-based, across the whole file) each
    /// column was built from. In `Row` mode this is exact, one entry per
    /// column. In `Binned` mode it names the lowest-indexed row that fell
    /// into that bin — a representative, not an aggregate — which is enough
    /// to align a genome-wide per-row `.npy` overlay in either mode.
    pub source_rows: Vec<u32>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromosomeMatrixProgress {
    pub rows_processed: usize,
    pub chromosome: String,
    pub percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromosomeMatrixBinaryResult {
    pub success: bool,
    pub chromosome: String,
    pub matrix_data: String,
    pub shape: [usize; 2],
    pub dtype: String,
    pub positions: Vec<u64>,
    pub gamete_names: Vec<String>,
    pub position_range: (u64, u64),
    pub error: Option<String>,
}

/// One PS4G data row's position/gamete-set/count, plus its index among all
/// data rows in the file (`global_row_index`) — needed to align a
/// genome-wide `.npy` overlay to a per-chromosome matrix. `gamete_start` /
/// `gamete_len` slice into the owning `ChromosomeRowData::gamete_flat` arena
/// rather than each row owning its own `Vec<u32>`, since a real file has on
/// the order of a million rows and per-row heap allocations dominate.
/// Exactly 24 bytes (align 8): `u64` + four `u32`s, no padding.
#[derive(Debug, Clone, Copy)]
pub struct PS4GRowEntry {
    pub ref_pos_binned: u64,
    pub global_row_index: u32,
    pub gamete_start: u32,
    pub gamete_len: u32,
    pub count: u32,
}

/// Per-chromosome PS4G data rows, preserved in file order. Same-bin
/// aggregation (the historical `Binned` behavior) and per-row layout
/// (`Row` mode) are both derived from this at matrix-build time — see
/// `build_chromosome_matrix` — rather than one being computed at parse time
/// and the other being unrecoverable.
#[derive(Debug, Clone, Default)]
pub struct ChromosomeRowData {
    pub rows: Vec<PS4GRowEntry>,
    pub gamete_flat: Vec<u32>,
}

/// Platform-agnostic cached PS4G data (no filesystem types)
#[derive(Debug, Clone)]
pub struct CachedPS4GData {
    pub metadata: PS4GMetadata,
    pub chromosome_data: FxHashMap<String, ChromosomeRowData>,
    pub chromosomes: Vec<String>,
    pub chromosome_counts: FxHashMap<String, usize>,
    pub position_ranges: FxHashMap<String, (u64, u64)>,
    pub total_rows: usize,
    pub unique_positions: usize,
}

// ============================================================================
// BED Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BEDDataRow {
    pub chrom: String,
    pub start: u64,
    pub end: u64,
    pub parent1: String,
    pub parent2: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParentStats {
    pub parent_id: String,
    pub regions_as_parent1: usize,
    pub regions_as_parent2: usize,
    pub total_regions: usize,
    pub coverage_bp_as_parent1: u64,
    pub coverage_bp_as_parent2: u64,
    pub total_coverage_bp: u64,
    pub chromosome_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BEDSummary {
    pub total_rows: usize,
    pub chromosomes: Vec<String>,
    pub chromosome_counts: HashMap<String, usize>,
    pub position_range: HashMap<String, (u64, u64)>,
    pub total_coverage_bp: u64,
    pub avg_region_size_bp: f64,
    pub unique_parents: Vec<String>,
    pub unique_parent_pairs: usize,
    pub parent_stats: Vec<ParentStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BEDProgress {
    pub rows_processed: usize,
    pub bytes_processed: u64,
    pub total_bytes: u64,
    pub percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BEDParseResult {
    pub success: bool,
    pub summary: BEDSummary,
    pub data_preview: Vec<BEDDataRow>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BEDRegion {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BEDMatrixProgress {
    pub rows_processed: usize,
    pub chromosome: String,
    pub percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BEDChromosomeMatrixResult {
    pub success: bool,
    pub chromosome: String,
    pub matrix: Vec<Vec<u8>>,
    pub parent_names: Vec<String>,
    pub regions: Vec<BEDRegion>,
    pub num_parents: usize,
    pub num_regions: usize,
    pub parent1_path: Vec<usize>,
    pub parent2_path: Vec<usize>,
    pub error: Option<String>,
}

// ============================================================================
// NPY Overlay Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpyOverlayResult {
    pub success: bool,
    /// Row index per column, or `-1` for an unlabeled/missing position
    /// (round-tripped from the source `.npy`, not saturated to `0`).
    pub true_paths: Vec<Vec<i64>>,
    pub predicted_paths: Vec<Vec<i64>>,
    pub num_positions: usize,
    pub is_diploid_true: bool,
    pub is_diploid_predicted: bool,
    pub error: Option<String>,
}

impl NpyOverlayResult {
    pub fn error(msg: String) -> Self {
        Self {
            success: false,
            true_paths: vec![],
            predicted_paths: vec![],
            num_positions: 0,
            is_diploid_true: false,
            is_diploid_predicted: false,
            error: Some(msg),
        }
    }
}
