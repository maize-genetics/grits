import numpy as np
import pandas as pd
from src.python.ps4g_io.ps4g import decode_position, build_index_lookup


def output_bed_file_deprecated(output_bed, chroms, final_predictions, index_array, positions, collapse_bed_regions=True):
    bed_df = pd.DataFrame({
        # TODO: convert chr_idx to chr
        "chrom_idx": chroms[:len(final_predictions)],
        "pos": positions[:len(final_predictions)],
        "parent1": np.array(index_array)[final_predictions[:, 0]],
        "parent2": np.array(index_array)[final_predictions[:, 1]],
    })
    # Save to BED file
    if not collapse_bed_regions:
        # If not collapsing, we can save directly
        bed_df.to_csv(output_bed, sep="\t", index=False)
    else:
        output_collapse_bed_deprecated(bed_df, output_bed)


def output_collapse_bed_deprecated(bed_df, output_bed):
    # Define group boundaries where parent1, parent2, or chrom changes
    group_change = (
            (bed_df["parent1"] != bed_df["parent1"].shift()) |
            (bed_df["parent2"] != bed_df["parent2"].shift()) |
            (bed_df["chrom_idx"] != bed_df["chrom_idx"].shift())
    )
    group_id = group_change.cumsum()
    # Collapse into ranges
    ranges_df = bed_df.groupby(group_id).agg({
        "chrom_idx": "first",
        "pos": ["min", "max"],
        "parent1": "first",
        "parent2": "first"
    }).reset_index(drop=True)
    # Clean up MultiIndex columns
    ranges_df.columns = ["chrom_idx", "start", "end", "parent1", "parent2"]
    # Save to BED file
    ranges_df.to_csv(output_bed, sep="\t", index=False)


def output_predictions_deprecated(ps4g_file, output_bed, final_predictions, collapse_bed_regions = True):
    spline_pos = pd.read_csv(ps4g_file, sep="\t", comment="#")['pos']
    decoded = np.vstack(np.vectorize(decode_position)(spline_pos)).T
    chroms, positions = zip(*decoded)
    index_array = build_index_lookup(ps4g_file)
    output_bed_file_deprecated(output_bed, chroms, final_predictions, index_array, positions, collapse_bed_regions)


def output_bed_file(output_bed, chroms, final_predictions, index_array, positions, collapse_bed_regions=True):
    '''
    output_bed: name of the desired output bed of imputed path
    chroms: chromosomes/contigs of predictions
    final_predictions: numpy array of predictions of shape [# predicted bins, 2]
    index_array: function to convert predicted gamete idx to gamete name
    positions: positions of predictions
    collapse_bed_regions: bool value of whether to consolidate consecutive predictions

    writes predictions to bed file
    '''
    bed_df = pd.DataFrame({
        "chrom": chroms[:len(final_predictions)],
        "start": positions[:len(final_predictions)],
        "end": positions[:len(final_predictions)] + 1,
        "parent1": np.array(index_array)[final_predictions[:, 0]],
        "parent2": np.array(index_array)[final_predictions[:, 1]],
    })
    # Save to BED file
    if not collapse_bed_regions:
        # If not collapsing, we can save directly
        bed_df.to_csv(output_bed, sep="\t", index=False)
    else:
        output_collapse_bed(bed_df, output_bed)


def output_collapse_bed(bed_df, output_bed):
    '''
    bed_df: pandas dataframe containing non-collapsed bed file data
    output_bed: name of the desired output bed of imputed path
    '''
    # Define group boundaries where parent1, parent2, or chrom changes
    group_change = (
            (bed_df["parent1"] != bed_df["parent1"].shift()) |
            (bed_df["parent2"] != bed_df["parent2"].shift()) |
            (bed_df["chrom"] != bed_df["chrom"].shift())
    )
    group_id = group_change.cumsum()
    # Collapse into ranges. NOTE: "end" must come from the "end" column's own
    # max, not "start"'s max -- aggregating "start" with both min/max (as a
    # prior version of this function did) gives [min(start), max(start)],
    # which for a group of exactly one row degenerates to start==end (an
    # invalid, zero-width interval that crashes downstream BED->VCF
    # conversion), and for multi-row groups silently under-reports the true
    # end by exactly one base (the last row's own 1bp span was never
    # included).
    ranges_df = bed_df.groupby(group_id).agg({
        "chrom": "first",
        "start": "min",
        "end": "max",
        "parent1": "first",
        "parent2": "first"
    }).reset_index(drop=True)
    # Clean up column order/names
    ranges_df = ranges_df[["chrom", "start", "end", "parent1", "parent2"]]
    # Save to BED file
    ranges_df.to_csv(output_bed, sep="\t", index=False)


def output_predictions(ps4g_file, output_bed, final_predictions, collapse_bed_regions = True, bin_size=256):
    '''
    ps4g_file: name of the ps4g file for the imputed sample
    output_bed: name of the desired output bed of imputed path
    final_predictions: numpy array of predictions of shape [# predicted bins, 2]
    collapse_bed_regions: bool value of whether to consolidate consecutive predictions

    writes predictions to bed file
    '''
    chroms = pd.read_csv(ps4g_file, sep="\t", comment="#")['refContig']
    positions = pd.read_csv(ps4g_file, sep="\t", comment="#")['refPosBinned'] * bin_size
    index_array = build_index_lookup(ps4g_file)
    index_array.append(None) # add extra index to represent "unlabelled" prediction
    output_bed_file(output_bed, chroms, final_predictions, index_array, positions, collapse_bed_regions)


def load_saved_predictions(sample_name, contigs, file_dir):
    '''
    sample_name: name of the imputed sample
    contigs: list of chromosome/contig names
    file_dir: directory containing saved numpy prediction files

    returns: a numpy array containing all chromosome/contig predictions,
            reshaped to [# predicted bins, 2]
    '''
    predictions = []
    for c in contigs:
        file_name = f"{file_dir}/{sample_name}_{c}.npy"
        preds = np.load(file_name, allow_pickle=True)
        predictions.append(preds)
    all_predictions = np.concatenate(predictions, axis=0)
    return all_predictions.reshape(-1, all_predictions.shape[-1])