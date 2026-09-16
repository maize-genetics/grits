// ============================================================================
// Shared type definitions mirroring the Rust parser-core types
// ============================================================================

export interface GameteInfo {
  /** Display label: bare sample name, unless it collides within the panel. */
  gamete: string;
  /** Sample name only, with any ":idx" suffix stripped (e.g. "B73"). */
  sample_name: string;
  /** Haplotype/gamete index parsed from a ":idx" suffix (0 if absent). */
  gamete_idx: number;
  gamete_index: number;
  read_count: number;
  /** read_count / summary.total_read_count (the true read total). */
  weight: number;
}

export interface PS4GDataRow {
  gamete_set: number[];
  ref_contig: string;
  ref_pos_binned: number;
  count: number;
}

export interface PS4GMetadata {
  version: string | null;
  command: string | null;
  total_unique_counts: number | null;
  gametes: GameteInfo[];
}

export interface PS4GSummary {
  total_rows: number;
  /** Sum of the data-section `count` column — the true read total. */
  total_read_count: number;
  unique_positions: number;
  chromosomes: string[];
  chromosome_counts: Record<string, number>;
  gamete_count: number;
  position_range: Record<string, [number, number]>;
}

export interface PS4GProgress {
  rows_processed: number;
  bytes_processed: number;
  total_bytes: number;
  percent: number;
}

export interface PS4GParseResult {
  success: boolean;
  metadata: PS4GMetadata;
  summary: PS4GSummary;
  data_preview: PS4GDataRow[];
  error: string | null;
}

/** Which axis a chromosome matrix's columns run over. */
export type ColumnMode = 'binned' | 'row';

export interface ChromosomeMatrixResult {
  success: boolean;
  chromosome: string;
  matrix: number[][];
  positions: number[];
  gamete_names: string[];
  num_gametes: number;
  num_positions: number;
  position_range: [number, number];
  /** Column model used to build this matrix. */
  column_mode: ColumnMode;
  /**
   * The global PS4G data-row index (0-based, across the whole file) each
   * column was built from. In 'row' mode this is exact, one entry per
   * column. In 'binned' mode it names the lowest-indexed row that fell into
   * that bin -- enough to align a genome-wide per-row .npy overlay in
   * either mode.
   */
  source_rows: number[];
  error: string | null;
}

export interface ChromosomeMatrixProgress {
  rows_processed: number;
  chromosome: string;
  percent: number;
}

export interface BEDDataRow {
  chrom: string;
  start: number;
  end: number;
  parent1: string;
  parent2: string;
}

export interface ParentStats {
  parent_id: string;
  regions_as_parent1: number;
  regions_as_parent2: number;
  total_regions: number;
  coverage_bp_as_parent1: number;
  coverage_bp_as_parent2: number;
  total_coverage_bp: number;
  chromosome_count: number;
}

export interface BEDSummary {
  total_rows: number;
  chromosomes: string[];
  chromosome_counts: Record<string, number>;
  position_range: Record<string, [number, number]>;
  total_coverage_bp: number;
  avg_region_size_bp: number;
  unique_parents: string[];
  unique_parent_pairs: number;
  parent_stats: ParentStats[];
}

export interface BEDProgress {
  rows_processed: number;
  bytes_processed: number;
  total_bytes: number;
  percent: number;
}

export interface BEDParseResult {
  success: boolean;
  summary: BEDSummary;
  data_preview: BEDDataRow[];
  error: string | null;
}

export interface BEDRegion {
  start: number;
  end: number;
}

export interface BEDMatrixProgress {
  rows_processed: number;
  chromosome: string;
  percent: number;
}

export interface BEDChromosomeMatrixResult {
  success: boolean;
  chromosome: string;
  matrix: number[][];
  parent_names: string[];
  regions: BEDRegion[];
  num_parents: number;
  num_regions: number;
  parent1_path: number[];
  parent2_path: number[];
  error: string | null;
}

export interface NpyOverlayResult {
  success: boolean;
  true_paths: number[][];
  predicted_paths: number[][];
  num_positions: number;
  is_diploid_true: boolean;
  is_diploid_predicted: boolean;
  error: string | null;
}

// ============================================================================
// Platform abstraction
// ============================================================================

export type ProgressCallback<T> = (progress: T) => void;

export type FileInput = string | File;

export interface OpenFileOptions {
  title?: string;
  filters?: Array<{ name: string; extensions: string[] }>;
  multiple?: boolean;
}

export interface SaveFileOptions {
  defaultName?: string;
  filters?: Array<{ name: string; extensions: string[] }>;
}

/**
 * Opaque handle representing a parsed file that can be queried for
 * chromosome-level data without re-parsing.
 */
export type FileHandle = string | object;

export interface PlatformBackend {
  parsePS4GFile(
    file: FileInput,
    onProgress?: ProgressCallback<PS4GProgress>,
  ): Promise<{ result: PS4GParseResult; handle: FileHandle }>;

  getChromosomeMatrix(
    handle: FileHandle,
    chromosome: string,
    columnMode: ColumnMode,
    onProgress?: ProgressCallback<ChromosomeMatrixProgress>,
  ): Promise<ChromosomeMatrixResult>;

  parseBEDFile(
    file: FileInput,
    onProgress?: ProgressCallback<BEDProgress>,
  ): Promise<{ result: BEDParseResult; handle: FileHandle }>;

  getBEDChromosomeMatrix(
    handle: FileHandle,
    chromosome: string,
    onProgress?: ProgressCallback<BEDMatrixProgress>,
  ): Promise<BEDChromosomeMatrixResult>;

  loadNpyOverlay(
    observed: FileInput | null,
    predictions: FileInput | null,
    expectedNumPositions: number,
    expectedNumGametes: number,
    /**
     * Optional genome-wide row mapping: `sourceRows[col]` names which row of
     * a genome-wide (all-chromosome) .npy corresponds to matrix column
     * `col` (see `ChromosomeMatrixResult.source_rows`), and `totalRows` is
     * the row count that mapping applies to (the PS4G file's total row
     * count). Pass `null`/`0` to accept only a chromosome-sized .npy.
     */
    sourceRows: number[] | null,
    totalRows: number,
  ): Promise<NpyOverlayResult>;

  openFile(options?: OpenFileOptions): Promise<FileInput | null>;

  saveFile(data: Uint8Array, options?: SaveFileOptions): Promise<void>;
}
