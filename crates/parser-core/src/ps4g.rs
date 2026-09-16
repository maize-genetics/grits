use crate::types::*;
use crate::util::natural_sort;
use base64::{engine::general_purpose, Engine as _};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;
use std::io::BufRead;

const PREVIEW_ROW_LIMIT: usize = 100;
const PROGRESS_UPDATE_INTERVAL: usize = 100_000;

/// Parse a PS4G file from any `BufRead` source and return structured data plus
/// cached chromosome data for fast matrix retrieval.
pub fn parse_ps4g(
    mut reader: impl BufRead,
    file_size: Option<u64>,
    mut on_progress: impl FnMut(PS4GProgress),
) -> Result<(PS4GParseResult, CachedPS4GData), String> {
    let total_bytes = file_size.unwrap_or(0);

    let mut metadata = PS4GMetadata {
        version: None,
        command: None,
        total_unique_counts: None,
        gametes: Vec::new(),
    };

    let mut data_preview: Vec<PS4GDataRow> = Vec::with_capacity(PREVIEW_ROW_LIMIT);
    let mut total_rows: usize = 0;
    let mut total_read_count: u64 = 0;

    let mut chromosome_to_index: FxHashMap<String, u16> = FxHashMap::default();
    let mut index_to_chromosome: Vec<String> = Vec::new();
    let mut unique_positions: FxHashSet<(u16, u64)> = FxHashSet::default();

    let mut chromosome_counts: Vec<usize> = Vec::new();
    let mut position_ranges: Vec<(u64, u64)> = Vec::new();
    let mut chromosome_row_data: Vec<ChromosomeRowData> = Vec::new();

    // `in_header` tracks the leading run of `#`-prefixed (and blank) lines,
    // which is the only place metadata and gamete records are recognized.
    // `in_gamete_section` additionally tracks whether we're between the
    // `#gamete` tag line and the end of that header block — gamete records
    // are only ever parsed there, never by line shape alone (see
    // `parse_gamete_record`).
    let mut in_header = true;
    let mut in_gamete_section = false;
    // If the header block ends with no gamete records recognized (no
    // `#gamete` tag was ever seen), fall back to synthesizing one gamete per
    // distinct index that actually appears in the data section's
    // `gameteSet` column, named by that index. `need_fallback_gametes` is
    // decided once, right when the header block ends.
    let mut need_fallback_gametes = false;
    let mut fallback_tally: FxHashMap<u32, u64> = FxHashMap::default();
    let mut bytes_processed: u64 = 0;
    let mut line_buf = String::with_capacity(256);

    loop {
        line_buf.clear();
        let bytes_read = reader
            .read_line(&mut line_buf)
            .map_err(|e| format!("Failed to read line: {}", e))?;

        if bytes_read == 0 {
            break;
        }

        bytes_processed += bytes_read as u64;
        let line = line_buf.trim();

        if line.is_empty() {
            continue;
        }

        if line.starts_with('#') {
            if in_header {
                if is_gamete_section_tag(line) {
                    in_gamete_section = true;
                } else if !parse_metadata_line(line, &mut metadata) && in_gamete_section {
                    // Inside the gamete section, every non-metadata `#` line
                    // is expected to be a record. A malformed one is skipped
                    // (not fatal, section stays open) rather than treated as
                    // "not a gamete record" the way the old shape sniff did.
                    if let Some(g) = parse_gamete_record(line) {
                        metadata.gametes.push(g);
                    }
                }
            }
            // `#` lines outside the header block are inert trailing
            // comments — never scanned for metadata or gamete records.
            continue;
        }

        if in_header {
            // First non-'#' line ends the header block (and with it, the
            // gamete section, if one was ever opened).
            in_header = false;
            in_gamete_section = false;
            need_fallback_gametes = metadata.gametes.is_empty();

            if line.starts_with("gameteSet") {
                continue;
            }
        }

        if let Some(row) = parse_data_row(line) {
            let chr_idx = get_or_create_chromosome_index(
                row.ref_contig,
                &mut chromosome_to_index,
                &mut index_to_chromosome,
                &mut chromosome_counts,
                &mut position_ranges,
                &mut chromosome_row_data,
            );

            // Captured before `total_rows` is incremented below, so this is
            // the 0-based index of this row among all data rows in the file
            // — the identity a genome-wide `.npy` overlay is keyed on.
            let global_row_index = total_rows as u32;

            total_read_count += row.count as u64;

            if need_fallback_gametes {
                for gamete_idx in &row.gamete_set {
                    *fallback_tally.entry(*gamete_idx).or_insert(0) += row.count as u64;
                }
            }

            unique_positions.insert((chr_idx, row.ref_pos_binned));
            chromosome_counts[chr_idx as usize] += 1;

            let range = &mut position_ranges[chr_idx as usize];
            if row.ref_pos_binned < range.0 {
                range.0 = row.ref_pos_binned;
            }
            if row.ref_pos_binned > range.1 {
                range.1 = row.ref_pos_binned;
            }

            let chr_data = &mut chromosome_row_data[chr_idx as usize];
            let gamete_start = chr_data.gamete_flat.len() as u32;
            chr_data.gamete_flat.extend_from_slice(&row.gamete_set);
            chr_data.rows.push(PS4GRowEntry {
                ref_pos_binned: row.ref_pos_binned,
                global_row_index,
                gamete_start,
                gamete_len: row.gamete_set.len() as u32,
                count: row.count,
            });

            if data_preview.len() < PREVIEW_ROW_LIMIT {
                data_preview.push(PS4GDataRow {
                    gamete_set: row.gamete_set,
                    ref_contig: row.ref_contig.to_string(),
                    ref_pos_binned: row.ref_pos_binned,
                    count: row.count,
                });
            }

            total_rows += 1;

            if total_rows % PROGRESS_UPDATE_INTERVAL == 0 {
                let percent = if total_bytes > 0 {
                    (bytes_processed as f64 / total_bytes as f64) * 100.0
                } else {
                    0.0
                };
                on_progress(PS4GProgress {
                    rows_processed: total_rows,
                    bytes_processed,
                    total_bytes,
                    percent,
                });
            }
        }
    }

    let mut chromosome_data_map: FxHashMap<String, ChromosomeRowData> = FxHashMap::default();
    let mut chromosome_counts_fx: FxHashMap<String, usize> = FxHashMap::default();
    let mut position_range_fx: FxHashMap<String, (u64, u64)> = FxHashMap::default();

    for (idx, chr_name) in index_to_chromosome.iter().enumerate() {
        let mut chr_data = std::mem::take(&mut chromosome_row_data[idx]);
        // `Vec` growth can leave up to 2x slack; this cache is held for the
        // life of the file, so reclaim it rather than carry the slack.
        chr_data.rows.shrink_to_fit();
        chr_data.gamete_flat.shrink_to_fit();
        chromosome_data_map.insert(chr_name.clone(), chr_data);
        chromosome_counts_fx.insert(chr_name.clone(), chromosome_counts[idx]);
        position_range_fx.insert(chr_name.clone(), position_ranges[idx]);
    }

    let mut chromosomes: Vec<String> = index_to_chromosome;
    chromosomes.sort_by(|a, b| natural_sort(a, b));

    let chromosome_counts_map: HashMap<String, usize> =
        chromosome_counts_fx.iter().map(|(k, v)| (k.clone(), *v)).collect();
    let position_range_map: HashMap<String, (u64, u64)> =
        position_range_fx.iter().map(|(k, v)| (k.clone(), *v)).collect();

    // No `#gamete` section was found in the header — synthesize one gamete
    // per distinct index seen in the data section's `gameteSet` column,
    // named by that index, rather than falling back to shape-sniffing `#`
    // lines (the thing this section-gated design replaces).
    if need_fallback_gametes && !fallback_tally.is_empty() {
        let mut indices: Vec<u32> = fallback_tally.keys().copied().collect();
        indices.sort_unstable();
        for idx in indices {
            let name = idx.to_string();
            metadata.gametes.push(GameteInfo {
                gamete: name.clone(),
                sample_name: name,
                gamete_idx: 0,
                gamete_index: idx,
                read_count: fallback_tally[&idx],
                weight: 0.0,
            });
        }
    }

    // Weights are normalized against the true read total, which is only
    // known after the whole data section has been read. Computing this
    // here (rather than while scanning the header) removes any dependency
    // on "#TotalUniqueCounts:" appearing before the gamete records, and
    // uses the recomputed total rather than a producer-declared header
    // value that may disagree with it.
    if total_read_count > 0 {
        for g in &mut metadata.gametes {
            g.weight = g.read_count as f64 / total_read_count as f64;
        }
    }

    let summary = PS4GSummary {
        total_rows,
        total_read_count,
        unique_positions: unique_positions.len(),
        chromosomes: chromosomes.clone(),
        chromosome_counts: chromosome_counts_map,
        gamete_count: metadata.gametes.len(),
        position_range: position_range_map,
    };

    on_progress(PS4GProgress {
        rows_processed: total_rows,
        bytes_processed: total_bytes,
        total_bytes,
        percent: 100.0,
    });

    let cached = CachedPS4GData {
        metadata: metadata.clone(),
        chromosome_data: chromosome_data_map,
        chromosomes,
        chromosome_counts: chromosome_counts_fx,
        position_ranges: position_range_fx,
        total_rows,
        unique_positions: unique_positions.len(),
    };

    let result = PS4GParseResult {
        success: true,
        metadata,
        summary,
        data_preview,
        error: None,
    };

    Ok((result, cached))
}

/// Build a chromosome matrix from previously cached PS4G data.
///
/// `mode` selects the column model: [`ColumnMode::Binned`] (the historical
/// behavior — one column per distinct `refPosBinned`, same-bin rows summed)
/// or [`ColumnMode::Row`] (one column per PS4G data row, matching the file's
/// own layout). Both are derived here from the same per-row cache rather
/// than one being computed at parse time and the other being unrecoverable.
pub fn build_chromosome_matrix(
    cached: &CachedPS4GData,
    chromosome: &str,
    mode: ColumnMode,
) -> Result<ChromosomeMatrixResult, String> {
    let chr_data = match cached.chromosome_data.get(chromosome) {
        Some(data) if !data.rows.is_empty() => data,
        _ => {
            return Ok(ChromosomeMatrixResult {
                success: false,
                chromosome: chromosome.to_string(),
                matrix: vec![],
                positions: vec![],
                gamete_names: vec![],
                num_gametes: 0,
                num_positions: 0,
                position_range: (0, 0),
                column_mode: mode,
                source_rows: vec![],
                error: Some(format!("No data found for chromosome: {}", chromosome)),
            });
        }
    };

    let mut gametes = cached.metadata.gametes.clone();
    gametes.sort_by_key(|g| g.gamete_index);

    let gamete_idx_to_row: FxHashMap<u32, usize> = gametes
        .iter()
        .enumerate()
        .map(|(row, g)| (g.gamete_index, row))
        .collect();

    let num_gametes = gametes.len();

    let (positions, source_rows, matrix) = match mode {
        ColumnMode::Binned => {
            // Same-bin rows are summed together. Rows arrive in file order,
            // so the first row seen for a given position is already its
            // lowest global index — no need to track a running minimum.
            let mut position_first_row: FxHashMap<u64, u32> = FxHashMap::default();
            let mut position_gamete_counts: FxHashMap<u64, FxHashMap<u32, u32>> =
                FxHashMap::default();

            for row in &chr_data.rows {
                position_first_row
                    .entry(row.ref_pos_binned)
                    .or_insert(row.global_row_index);
                let gamete_counts = position_gamete_counts.entry(row.ref_pos_binned).or_default();
                let gamete_slice = &chr_data.gamete_flat
                    [row.gamete_start as usize..(row.gamete_start + row.gamete_len) as usize];
                for &gamete_idx in gamete_slice {
                    *gamete_counts.entry(gamete_idx).or_insert(0) += row.count;
                }
            }

            let mut positions: Vec<u64> = position_gamete_counts.keys().copied().collect();
            positions.sort_unstable();

            let mut matrix: Vec<Vec<u32>> = vec![vec![0; positions.len()]; num_gametes];
            let mut source_rows: Vec<u32> = Vec::with_capacity(positions.len());
            for (col, &pos) in positions.iter().enumerate() {
                source_rows.push(position_first_row[&pos]);
                for (&gamete_idx, &count) in &position_gamete_counts[&pos] {
                    if let Some(&row) = gamete_idx_to_row.get(&gamete_idx) {
                        matrix[row][col] = count;
                    }
                }
            }

            (positions, source_rows, matrix)
        }
        ColumnMode::Row => {
            // Sort (position, original-index) pairs rather than the rows
            // themselves: the original index is unique, so this is a total
            // order that ties-break to file order without depending on a
            // stable-sort guarantee, and it's cache-friendlier than sorting
            // the 24-byte `PS4GRowEntry`s directly.
            let mut order: Vec<(u64, usize)> = chr_data
                .rows
                .iter()
                .enumerate()
                .map(|(i, r)| (r.ref_pos_binned, i))
                .collect();
            order.sort_unstable();

            let num_cols = order.len();
            let mut positions: Vec<u64> = Vec::with_capacity(num_cols);
            let mut source_rows: Vec<u32> = Vec::with_capacity(num_cols);
            let mut matrix: Vec<Vec<u32>> = vec![vec![0; num_cols]; num_gametes];

            for (col, &(pos, seq_idx)) in order.iter().enumerate() {
                let row = &chr_data.rows[seq_idx];
                positions.push(pos);
                source_rows.push(row.global_row_index);
                let gamete_slice = &chr_data.gamete_flat
                    [row.gamete_start as usize..(row.gamete_start + row.gamete_len) as usize];
                for &gamete_idx in gamete_slice {
                    if let Some(&r) = gamete_idx_to_row.get(&gamete_idx) {
                        matrix[r][col] += row.count;
                    }
                }
            }

            (positions, source_rows, matrix)
        }
    };

    let num_positions = positions.len();

    let position_range = if !positions.is_empty() {
        (*positions.first().unwrap(), *positions.last().unwrap())
    } else {
        (0, 0)
    };

    // Name gametes by bare sample name, except when two gametes in this
    // panel share a sample name (e.g. "B73:0" and "B73:1") — then fall back
    // to "sample:idx" for just those so rows stay distinguishable.
    let mut sample_name_counts: FxHashMap<&str, usize> = FxHashMap::default();
    for g in &gametes {
        *sample_name_counts.entry(g.sample_name.as_str()).or_insert(0) += 1;
    }
    let gamete_names: Vec<String> = gametes
        .iter()
        .map(|g| {
            if sample_name_counts.get(g.sample_name.as_str()).copied().unwrap_or(0) > 1 {
                format!("{}:{}", g.sample_name, g.gamete_idx)
            } else {
                g.sample_name.clone()
            }
        })
        .collect();

    Ok(ChromosomeMatrixResult {
        success: true,
        chromosome: chromosome.to_string(),
        matrix,
        positions,
        gamete_names,
        num_gametes,
        num_positions,
        position_range,
        column_mode: mode,
        source_rows,
        error: None,
    })
}

/// Encode a chromosome matrix result as a compact base64 binary payload.
pub fn encode_matrix_binary(
    result: &ChromosomeMatrixResult,
) -> ChromosomeMatrixBinaryResult {
    if !result.success {
        return ChromosomeMatrixBinaryResult {
            success: false,
            chromosome: result.chromosome.clone(),
            matrix_data: String::new(),
            shape: [0, 0],
            dtype: "uint32".to_string(),
            positions: vec![],
            gamete_names: vec![],
            position_range: (0, 0),
            error: result.error.clone(),
        };
    }

    let num_rows = result.matrix.len();
    let num_cols = if num_rows > 0 { result.matrix[0].len() } else { 0 };

    let flat_data: Vec<u8> = result
        .matrix
        .iter()
        .flat_map(|row| row.iter().flat_map(|&val| val.to_le_bytes()))
        .collect();

    let matrix_data = general_purpose::STANDARD.encode(&flat_data);

    ChromosomeMatrixBinaryResult {
        success: true,
        chromosome: result.chromosome.clone(),
        matrix_data,
        shape: [num_rows, num_cols],
        dtype: "uint32".to_string(),
        positions: result.positions.clone(),
        gamete_names: result.gamete_names.clone(),
        position_range: result.position_range,
        error: None,
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Parse a gamete header field into (sample_name, gamete_idx).
///
/// Per the PS4G spec, the field is either `<sampleName>` (index implicitly
/// 0) or `<sampleName>:<gameteIdx>`. Only a trailing `:<digits>` suffix is
/// treated as an index; a sample name that itself contains `:` but has no
/// pure-digit suffix is kept whole.
#[inline]
fn parse_sample_gamete(field: &str) -> (String, u32) {
    if let Some((name, suffix)) = field.rsplit_once(':') {
        if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) {
            if let Ok(idx) = suffix.parse::<u32>() {
                return (name.to_string(), idx);
            }
        }
    }
    (field.to_string(), 0)
}

/// True if `line` is the `#gamete\tgameteIndex\tcount` tag line that opens
/// the gamete section (see `parse_ps4g`). Only the first tab field is
/// checked — the column names after it are informational — so this matches
/// regardless of exact header spelling, and matching is case-insensitive
/// and tolerant of extra leading `#`s (mirroring the `#PS4G`/`##PS4G` magic
/// line).
#[inline]
pub(crate) fn is_gamete_section_tag(line: &str) -> bool {
    line.trim_start_matches('#')
        .split('\t')
        .next()
        .map(|field| field.trim().eq_ignore_ascii_case("gamete"))
        .unwrap_or(false)
}

/// Consume a keyed metadata line (`#PS4G`, `#version=`, `#Command:`,
/// `#TotalUniqueCounts:`). Returns `true` if `line` was one of these, so the
/// caller knows not to also try `parse_gamete_record` on it — these keys
/// take precedence even inside the gamete section (a producer-declared
/// `#TotalUniqueCounts:` line interleaved among gamete records shouldn't be
/// mistaken for a malformed record).
#[inline]
pub(crate) fn parse_metadata_line(line: &str, metadata: &mut PS4GMetadata) -> bool {
    if line.trim_start_matches('#') == "PS4G" {
        // Magic line, e.g. "#PS4G" (v2.0 form) or "##PS4G" (legacy form).
        true
    } else if let Some(version) = line.strip_prefix("#version=") {
        metadata.version = Some(version.to_string());
        true
    } else if let Some(command) = line.strip_prefix("#Command:") {
        metadata.command = Some(command.trim().to_string());
        true
    } else if let Some(count_str) = line.strip_prefix("#TotalUniqueCounts:") {
        if let Ok(count) = count_str.trim().parse::<u64>() {
            metadata.total_unique_counts = Some(count);
        }
        true
    } else {
        false
    }
}

/// Parse a line already known — by its position under the `#gamete` tag —
/// to be a gamete record, e.g. `"#B73\t0\t784970"` or `"#B73:0\t0\t784970"`.
/// Returns `None` if the line is malformed (not a shape sniff: the caller
/// has already established this line *should* be a record).
#[inline]
pub(crate) fn parse_gamete_record(line: &str) -> Option<GameteInfo> {
    let content = line.trim_start_matches('#');
    let parts: Vec<&str> = content.split('\t').collect();
    if parts.len() < 3 {
        return None;
    }
    let gamete_full = parts[0];
    let idx = parts[1].parse::<u32>().ok()?;
    let count = parts[2].parse::<u64>().ok()?;
    let (sample_name, gamete_idx) = parse_sample_gamete(gamete_full);

    Some(GameteInfo {
        gamete: sample_name.clone(),
        sample_name,
        gamete_idx,
        gamete_index: idx,
        read_count: count,
        // Filled in by parse_ps4g once the data section's true read total
        // is known; see the post-loop weight pass there.
        weight: 0.0,
    })
}

struct ParsedDataRow<'a> {
    gamete_set: Vec<u32>,
    ref_contig: &'a str,
    ref_pos_binned: u64,
    count: u32,
}

#[inline]
fn parse_data_row(line: &str) -> Option<ParsedDataRow<'_>> {
    let mut parts = line.split('\t');

    let gamete_set_str = parts.next()?;
    let ref_contig = parts.next()?;
    let ref_pos_str = parts.next()?;
    let count_str = parts.next()?;

    let gamete_set: Vec<u32> = gamete_set_str
        .split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect();

    let ref_pos_binned = ref_pos_str.parse::<u64>().ok()?;
    let count = count_str.parse::<u32>().ok()?;

    Some(ParsedDataRow {
        gamete_set,
        ref_contig,
        ref_pos_binned,
        count,
    })
}

#[inline]
fn get_or_create_chromosome_index(
    chr_name: &str,
    chromosome_to_index: &mut FxHashMap<String, u16>,
    index_to_chromosome: &mut Vec<String>,
    chromosome_counts: &mut Vec<usize>,
    position_ranges: &mut Vec<(u64, u64)>,
    chromosome_row_data: &mut Vec<ChromosomeRowData>,
) -> u16 {
    if let Some(&idx) = chromosome_to_index.get(chr_name) {
        idx
    } else {
        let idx = index_to_chromosome.len() as u16;
        chromosome_to_index.insert(chr_name.to_string(), idx);
        index_to_chromosome.push(chr_name.to_string());
        chromosome_counts.push(0);
        position_ranges.push((u64::MAX, 0));
        chromosome_row_data.push(ChromosomeRowData::default());
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fresh_metadata() -> PS4GMetadata {
        PS4GMetadata {
            version: None,
            command: None,
            total_unique_counts: None,
            gametes: Vec::new(),
        }
    }

    #[test]
    fn parses_bare_gamete_name() {
        let g = parse_gamete_record("#B73\t0\t60").unwrap();
        assert_eq!(g.gamete, "B73");
        assert_eq!(g.sample_name, "B73");
        assert_eq!(g.gamete_idx, 0);
        assert_eq!(g.gamete_index, 0);
        assert_eq!(g.read_count, 60);
    }

    #[test]
    fn parses_colon_suffixed_gamete_name() {
        let g = parse_gamete_record("#B73:1\t2\t40").unwrap();
        assert_eq!(g.gamete, "B73");
        assert_eq!(g.sample_name, "B73");
        assert_eq!(g.gamete_idx, 1);
        assert_eq!(g.gamete_index, 2);
    }

    #[test]
    fn parse_gamete_record_rejects_malformed_lines() {
        assert!(parse_gamete_record("#B73\t0").is_none()); // too few fields
        assert!(parse_gamete_record("#B73\tx\t4").is_none()); // non-integer index
        assert!(parse_gamete_record("#B73\t0\tx").is_none()); // non-integer count
    }

    #[test]
    fn gamete_tag_recognized_in_all_forms() {
        assert!(is_gamete_section_tag("#gamete\tgameteIndex\tcount"));
        assert!(is_gamete_section_tag("#gamete"));
        assert!(is_gamete_section_tag("##gamete\tgameteIndex\tcount"));
        assert!(is_gamete_section_tag("#GAMETE\tgameteIndex\tcount"));
        assert!(!is_gamete_section_tag("#B73\t0\t4"));
        assert!(!is_gamete_section_tag("#Command: ropebwt3 refmap"));
        assert!(!is_gamete_section_tag("#version=2.0"));
    }

    #[test]
    fn command_line_with_colon_is_recognized_as_metadata() {
        let mut metadata = fresh_metadata();
        assert!(parse_metadata_line(
            "#Command: ropebwt3 refmap --ref-prefix=B73 --max-occ=-1",
            &mut metadata,
        ));
        assert_eq!(
            metadata.command.as_deref(),
            Some("ropebwt3 refmap --ref-prefix=B73 --max-occ=-1")
        );
    }

    #[test]
    fn magic_and_version_lines_are_metadata_not_gamete_records() {
        let mut metadata = fresh_metadata();
        assert!(parse_metadata_line("#PS4G", &mut metadata));
        assert!(parse_metadata_line("##PS4G", &mut metadata));
        assert!(parse_metadata_line("#version=2.0", &mut metadata));
        assert!(metadata.gametes.is_empty());
        assert_eq!(metadata.version.as_deref(), Some("2.0"));
    }

    #[test]
    fn gamete_record_parsing_does_not_set_weight() {
        // Weight is computed post-parse in parse_ps4g, once the data
        // section's true read total is known. See the parse_ps4g tests
        // below for the real weight computation.
        let g = parse_gamete_record("#B73\t0\t250").unwrap();
        assert_eq!(g.weight, 0.0);
    }

    fn sample_ps4g_bytes() -> &'static [u8] {
        b"#TotalUniqueCounts: 4\n\
          #gamete\tgameteIndex\tcount\n\
          #B73:0\t0\t4\n\
          #CML247:0\t1\t2\n\
          #W22:0\t2\t1\n\
          gameteSet\trefContig\trefPosBinned\tcount\n\
          0\tchr1\t1000\t1\n\
          0,1\tchr1\t1000\t1\n\
          0\tchr1\t2000\t1\n\
          0,1,2\tchr1\t2000\t1\n"
    }

    #[test]
    fn total_read_count_sums_data_column_not_gamete_counts() {
        let (result, _cached) =
            parse_ps4g(Cursor::new(sample_ps4g_bytes()), None, |_| {}).unwrap();

        // 4 data rows, count 1 each -> the true read total.
        assert_eq!(result.summary.total_read_count, 4);
        assert_eq!(result.summary.total_rows, 4);

        // Per-gamete counts sum to 7 (B73 hits every row, CML247 two rows,
        // W22 one row) -- strictly more than the true read total, because
        // rows with a multi-gamete gameteSet credit every gamete in it.
        let summed_gamete_counts: u64 =
            result.metadata.gametes.iter().map(|g| g.read_count).sum();
        assert_eq!(summed_gamete_counts, 7);
    }

    #[test]
    fn weight_uses_computed_read_total() {
        let (result, _cached) =
            parse_ps4g(Cursor::new(sample_ps4g_bytes()), None, |_| {}).unwrap();

        let by_name: FxHashMap<&str, f64> = result
            .metadata
            .gametes
            .iter()
            .map(|g| (g.sample_name.as_str(), g.weight))
            .collect();
        assert!((by_name["B73"] - 1.0).abs() < 1e-9);
        assert!((by_name["CML247"] - 0.5).abs() < 1e-9);
        assert!((by_name["W22"] - 0.25).abs() < 1e-9);
    }

    #[test]
    fn weight_is_independent_of_header_order() {
        // Also covers: a keyed metadata line (#TotalUniqueCounts:)
        // interleaved *after* the gamete records, inside the section,
        // doesn't close the section or get mistaken for a malformed record.
        let reordered = b"#gamete\tgameteIndex\tcount\n\
                           #B73:0\t0\t4\n\
                           #CML247:0\t1\t2\n\
                           #W22:0\t2\t1\n\
                           #TotalUniqueCounts: 4\n\
                           gameteSet\trefContig\trefPosBinned\tcount\n\
                           0\tchr1\t1000\t1\n\
                           0,1\tchr1\t1000\t1\n\
                           0\tchr1\t2000\t1\n\
                           0,1,2\tchr1\t2000\t1\n";
        let (result, _cached) = parse_ps4g(Cursor::new(&reordered[..]), None, |_| {}).unwrap();

        let by_name: FxHashMap<&str, f64> = result
            .metadata
            .gametes
            .iter()
            .map(|g| (g.sample_name.as_str(), g.weight))
            .collect();
        assert!((by_name["B73"] - 1.0).abs() < 1e-9);
        assert!((by_name["CML247"] - 0.5).abs() < 1e-9);
        assert!((by_name["W22"] - 0.25).abs() < 1e-9);
    }

    #[test]
    fn computed_total_ignores_disagreeing_header() {
        let disagreeing = b"#TotalUniqueCounts: 999\n\
                             #gamete\tgameteIndex\tcount\n\
                             #B73:0\t0\t4\n\
                             #CML247:0\t1\t2\n\
                             #W22:0\t2\t1\n\
                             gameteSet\trefContig\trefPosBinned\tcount\n\
                             0\tchr1\t1000\t1\n\
                             0,1\tchr1\t1000\t1\n\
                             0\tchr1\t2000\t1\n\
                             0,1,2\tchr1\t2000\t1\n";
        let (result, _cached) = parse_ps4g(Cursor::new(&disagreeing[..]), None, |_| {}).unwrap();

        assert_eq!(result.summary.total_read_count, 4);
        assert_eq!(result.metadata.total_unique_counts, Some(999));
    }

    #[test]
    fn header_only_file_has_zero_total_and_zero_weights() {
        let header_only = b"#TotalUniqueCounts: 0\n\
                             #gamete\tgameteIndex\tcount\n\
                             #B73:0\t0\t0\n\
                             gameteSet\trefContig\trefPosBinned\tcount\n";
        let (result, _cached) = parse_ps4g(Cursor::new(&header_only[..]), None, |_| {}).unwrap();

        assert_eq!(result.summary.total_read_count, 0);
        assert_eq!(result.summary.total_rows, 0);
        for g in &result.metadata.gametes {
            assert_eq!(g.weight, 0.0);
            assert!(!g.weight.is_nan());
        }
    }

    #[test]
    fn multi_count_rows_sum_correctly() {
        let multi = b"#TotalUniqueCounts: 5\n\
                       #gamete\tgameteIndex\tcount\n\
                       #B73:0\t0\t5\n\
                       gameteSet\trefContig\trefPosBinned\tcount\n\
                       0\tchr1\t1000\t5\n";
        let (result, _cached) = parse_ps4g(Cursor::new(&multi[..]), None, |_| {}).unwrap();

        assert_eq!(result.summary.total_read_count, 5);
        assert_eq!(result.summary.total_rows, 1);
    }

    #[test]
    fn records_before_gamete_tag_are_ignored() {
        // A gamete-shaped line appearing before the #gamete tag opens the
        // section is not a record -- position, not shape, is what matters.
        let bytes = b"#B73:0\t0\t4\n\
                       #gamete\tgameteIndex\tcount\n\
                       #CML247:0\t1\t2\n\
                       #W22:0\t2\t1\n\
                       gameteSet\trefContig\trefPosBinned\tcount\n\
                       0\tchr1\t1000\t1\n\
                       0,1\tchr1\t1000\t1\n\
                       0\tchr1\t2000\t1\n\
                       0,1,2\tchr1\t2000\t1\n";
        let (result, _cached) = parse_ps4g(Cursor::new(&bytes[..]), None, |_| {}).unwrap();

        assert_eq!(result.summary.gamete_count, 2);
        assert!(result
            .metadata
            .gametes
            .iter()
            .all(|g| g.sample_name != "B73"));
    }

    #[test]
    fn three_column_comment_outside_section_is_not_a_gamete() {
        // The reviewer's regression case: a #-line with the old gamete
        // shape (3 tab fields, integer cols 2-3) that is NOT a gamete
        // record, placed both before the tag and after the data section.
        let bytes = b"#binSize\t256\t1\n\
                       #gamete\tgameteIndex\tcount\n\
                       #B73:0\t0\t4\n\
                       #CML247:0\t1\t2\n\
                       #W22:0\t2\t1\n\
                       gameteSet\trefContig\trefPosBinned\tcount\n\
                       0\tchr1\t1000\t1\n\
                       0,1\tchr1\t1000\t1\n\
                       0\tchr1\t2000\t1\n\
                       0,1,2\tchr1\t2000\t1\n\
                       #binSize\t256\t1\n";
        let (result, _cached) = parse_ps4g(Cursor::new(&bytes[..]), None, |_| {}).unwrap();

        assert_eq!(result.summary.gamete_count, 3);
    }

    #[test]
    fn comment_after_data_section_is_ignored() {
        // '#' lines are only recognized inside the leading header block --
        // once data rows start, trailing '#' lines (metadata or gamete
        // shaped) are inert comments.
        let bytes = b"#gamete\tgameteIndex\tcount\n\
                       #B73:0\t0\t4\n\
                       gameteSet\trefContig\trefPosBinned\tcount\n\
                       0\tchr1\t1000\t4\n\
                       #B99\t9\t9\n\
                       #version=9.9\n";
        let (result, _cached) = parse_ps4g(Cursor::new(&bytes[..]), None, |_| {}).unwrap();

        assert_eq!(result.summary.gamete_count, 1);
        assert_eq!(result.metadata.version, None);
    }

    #[test]
    fn file_without_gamete_tag_synthesizes_ids_from_data() {
        // No #gamete tag anywhere -- rather than falling back to shape
        // sniffing, gametes are synthesized from the indices actually used
        // in the data section's gameteSet column, named by that index.
        let bytes = b"#TotalUniqueCounts: 4\n\
                       gameteSet\trefContig\trefPosBinned\tcount\n\
                       0\tchr1\t1000\t1\n\
                       0,1\tchr1\t1000\t1\n\
                       0\tchr1\t2000\t1\n\
                       0,1,2\tchr1\t2000\t1\n";
        let (result, _cached) = parse_ps4g(Cursor::new(&bytes[..]), None, |_| {}).unwrap();

        assert_eq!(result.summary.gamete_count, 3);
        let mut names: Vec<&str> = result
            .metadata
            .gametes
            .iter()
            .map(|g| g.sample_name.as_str())
            .collect();
        names.sort();
        assert_eq!(names, vec!["0", "1", "2"]);

        let by_index: FxHashMap<u32, u64> = result
            .metadata
            .gametes
            .iter()
            .map(|g| (g.gamete_index, g.read_count))
            .collect();
        assert_eq!(by_index[&0], 4); // hits all 4 rows
        assert_eq!(by_index[&1], 2); // hits rows 2, 4
        assert_eq!(by_index[&2], 1); // hits row 4
    }

    #[test]
    fn malformed_record_inside_section_is_skipped() {
        let bytes = b"#gamete\tgameteIndex\tcount\n\
                       #B73:0\t0\t4\n\
                       #bad\tnot-a-number\t2\n\
                       #W22:0\t2\t1\n\
                       gameteSet\trefContig\trefPosBinned\tcount\n\
                       0\tchr1\t1000\t1\n";
        let (result, _cached) = parse_ps4g(Cursor::new(&bytes[..]), None, |_| {}).unwrap();

        assert_eq!(result.summary.gamete_count, 2);
    }

    // ========================================================================
    // build_chromosome_matrix / ColumnMode tests
    // ========================================================================

    #[test]
    fn binned_matrix_matches_pre_refactor_values() {
        // Golden values for sample_ps4g_bytes(), matching the collapsed
        // (position -> summed-gamete-counts) behavior the old
        // FxHashMap<u64, FxHashMap<u32,u32>> cache produced directly.
        let (_result, cached) = parse_ps4g(Cursor::new(sample_ps4g_bytes()), None, |_| {}).unwrap();
        let matrix = build_chromosome_matrix(&cached, "chr1", ColumnMode::Binned).unwrap();

        assert!(matrix.success);
        assert_eq!(matrix.column_mode, ColumnMode::Binned);
        assert_eq!(matrix.positions, vec![1000, 2000]);
        assert_eq!(matrix.gamete_names, vec!["B73", "CML247", "W22"]);
        assert_eq!(
            matrix.matrix,
            vec![
                vec![2, 2], // B73:  hit by every row
                vec![1, 1], // CML247: rows 2 and 4
                vec![0, 1], // W22: row 4 only
            ]
        );
        // Lowest global row index landing in each bin: row 0 for bin 1000,
        // row 2 for bin 2000 (0-based, file order).
        assert_eq!(matrix.source_rows, vec![0, 2]);
    }

    #[test]
    fn row_mode_matches_expected_per_row_columns() {
        let (_result, cached) = parse_ps4g(Cursor::new(sample_ps4g_bytes()), None, |_| {}).unwrap();
        let matrix = build_chromosome_matrix(&cached, "chr1", ColumnMode::Row).unwrap();

        assert!(matrix.success);
        assert_eq!(matrix.column_mode, ColumnMode::Row);
        assert_eq!(matrix.positions, vec![1000, 1000, 2000, 2000]);
        assert_eq!(matrix.source_rows, vec![0, 1, 2, 3]);
        assert_eq!(
            matrix.matrix,
            vec![
                vec![1, 1, 1, 1], // B73: every row
                vec![0, 1, 0, 1], // CML247: rows 1 and 3 (0-indexed)
                vec![0, 0, 0, 1], // W22: row 3 only
            ]
        );
    }

    #[test]
    fn row_mode_column_count_equals_chromosome_row_count() {
        let (_result, cached) = parse_ps4g(Cursor::new(sample_ps4g_bytes()), None, |_| {}).unwrap();
        let matrix = build_chromosome_matrix(&cached, "chr1", ColumnMode::Row).unwrap();
        assert_eq!(matrix.num_positions, cached.chromosome_counts["chr1"]);
        assert_eq!(matrix.num_positions, 4);
    }

    /// A fixture with two gametes, three chromosomes interleaved in file
    /// order, and out-of-order positions within chr1 (500 before 100) --
    /// deliberately hostile to any implementation that assumes sortedness
    /// or per-chromosome row contiguity.
    fn interleaved_ps4g_bytes() -> &'static [u8] {
        b"#gamete\tgameteIndex\tcount\n\
          #B73:0\t0\t3\n\
          #CML247:0\t1\t2\n\
          gameteSet\trefContig\trefPosBinned\tcount\n\
          0\tchr1\t500\t1\n\
          0\tchr2\t10\t1\n\
          0\tchr1\t100\t1\n\
          1\tchr2\t5\t1\n\
          0,1\tchr1\t100\t1\n\
          0\tchr3\t777\t1\n"
    }

    #[test]
    fn row_mode_source_rows_are_global_indices() {
        // chr1's data rows are global indices 0, 2, 4 -- not contiguous --
        // so an implementation keyed on a per-chromosome row offset (rather
        // than each row's own recorded global_row_index) would fail this.
        let (_result, cached) =
            parse_ps4g(Cursor::new(interleaved_ps4g_bytes()), None, |_| {}).unwrap();
        let matrix = build_chromosome_matrix(&cached, "chr1", ColumnMode::Row).unwrap();

        assert_eq!(matrix.positions, vec![100, 100, 500]);
        assert_eq!(matrix.source_rows, vec![2, 4, 0]);
    }

    #[test]
    fn row_mode_ties_preserve_file_order() {
        // Rows 2 (global) and 4 (global) both land on refPosBinned=100 for
        // chr1; row 2 appears earlier in the file and must sort first.
        let (_result, cached) =
            parse_ps4g(Cursor::new(interleaved_ps4g_bytes()), None, |_| {}).unwrap();
        let matrix = build_chromosome_matrix(&cached, "chr1", ColumnMode::Row).unwrap();

        let tied: Vec<u32> = matrix
            .positions
            .iter()
            .zip(&matrix.source_rows)
            .filter(|(&pos, _)| pos == 100)
            .map(|(_, &row)| row)
            .collect();
        assert_eq!(tied, vec![2, 4]);
    }

    #[test]
    fn row_mode_positions_are_non_decreasing_with_duplicates() {
        let (_result, cached) =
            parse_ps4g(Cursor::new(interleaved_ps4g_bytes()), None, |_| {}).unwrap();
        let matrix = build_chromosome_matrix(&cached, "chr1", ColumnMode::Row).unwrap();
        assert!(matrix.positions.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn unsorted_input_produces_sorted_columns() {
        // chr2's rows appear in the file as pos 10 then pos 5 (descending);
        // both column models must still emit ascending positions.
        let (_result, cached) =
            parse_ps4g(Cursor::new(interleaved_ps4g_bytes()), None, |_| {}).unwrap();

        let row_matrix = build_chromosome_matrix(&cached, "chr2", ColumnMode::Row).unwrap();
        assert_eq!(row_matrix.positions, vec![5, 10]);
        assert_eq!(row_matrix.source_rows, vec![3, 1]); // global indices

        let binned_matrix = build_chromosome_matrix(&cached, "chr2", ColumnMode::Binned).unwrap();
        assert_eq!(binned_matrix.positions, vec![5, 10]);
    }

    #[test]
    fn binned_source_rows_pick_lowest_row_per_bin() {
        let (_result, cached) =
            parse_ps4g(Cursor::new(interleaved_ps4g_bytes()), None, |_| {}).unwrap();
        let matrix = build_chromosome_matrix(&cached, "chr1", ColumnMode::Binned).unwrap();

        assert_eq!(matrix.positions, vec![100, 500]);
        // Bin 100 first appears at global row 2; bin 500 is global row 0.
        assert_eq!(matrix.source_rows, vec![2, 0]);
    }

    #[test]
    fn both_modes_agree_on_position_range_and_gamete_names() {
        let (_result, cached) =
            parse_ps4g(Cursor::new(interleaved_ps4g_bytes()), None, |_| {}).unwrap();
        let binned = build_chromosome_matrix(&cached, "chr1", ColumnMode::Binned).unwrap();
        let row = build_chromosome_matrix(&cached, "chr1", ColumnMode::Row).unwrap();

        assert_eq!(binned.position_range, row.position_range);
        assert_eq!(binned.gamete_names, row.gamete_names);
        assert_eq!(binned.num_gametes, row.num_gametes);
    }

    #[test]
    fn empty_chromosome_returns_error_result_in_both_modes() {
        let (_result, cached) =
            parse_ps4g(Cursor::new(interleaved_ps4g_bytes()), None, |_| {}).unwrap();

        for mode in [ColumnMode::Binned, ColumnMode::Row] {
            let matrix = build_chromosome_matrix(&cached, "chrNope", mode).unwrap();
            assert!(!matrix.success);
            assert!(matrix.error.is_some());
            assert_eq!(matrix.column_mode, mode);
            assert!(matrix.source_rows.is_empty());
        }
    }

    #[test]
    fn single_row_chromosome() {
        // chr3 has exactly one data row (the Pt/Mt case in real files).
        let (_result, cached) =
            parse_ps4g(Cursor::new(interleaved_ps4g_bytes()), None, |_| {}).unwrap();

        for mode in [ColumnMode::Binned, ColumnMode::Row] {
            let matrix = build_chromosome_matrix(&cached, "chr3", mode).unwrap();
            assert!(matrix.success);
            assert_eq!(matrix.num_positions, 1);
            assert_eq!(matrix.positions, vec![777]);
            assert_eq!(matrix.source_rows, vec![5]);
        }
    }
}
