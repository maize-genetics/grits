#!/usr/bin/env python
"""
Ground-truth source labels for the maize simulated-validation corpus, added to
answer "how much PS4G noise is real vs. an artifact" by making per-read and
per-bin truth attribution possible, which the corpus does not otherwise carry
(hybrid/RIL reads from both parents share the same chr1..chr10 naming and are
binary-concatenated, so wgsim read names alone don't say which parent a read
came from -- see /home/zrm22/.claude/plans/deep-jumping-salamander.md).

Two independent label layers:

  L1 (per-read):  rename_and_concat() inserts "<label>|" after the leading
                   '@' of every FASTQ read name during simulation-time concat,
                   leaving wgsim's own error-count suffix byte-identical (so
                   existing error-QC parsing keeps working). Delimiter '|'
                   never appears in wgsim names (which use ':' and '_').

  L2 (per-bin):    bin_truth_labels() returns the true founder index for h1/h2
                   at every row of a completed alignment's raw.npy, aligned
                   1:1 by construction (both are built by iterating raw.npy.bins.tsv
                   in file order). For RIL kind this delegates the B73-coordinate
                   mosaic reconstruction to simval_oracle_bed.build_ril_mosaics/
                   _founder_at rather than re-deriving it -- same validated
                   crc32-seeded RNG replay, already checked against the real
                   corpus truth gVCFs (oracle max identity error 0.0235%).

Deliberately kept free of any one run's specifics (row keys, coverage, error
rate) so the corpus generator can adopt these directly if the full datasets
are ever remade with labels built in from the start.
"""
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import simval_oracle_bed as oracle  # noqa: E402  (build_ril_mosaics, _founder_at, load_contig_layout_positions)

CORPUS_SCRIPTS = Path(
    "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/scripts"
)
if str(CORPUS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CORPUS_SCRIPTS))
import mosaic  # noqa: E402

LABEL_SEP = "|"


# ---------------------------------------------------------------------------
# L1 -- per-read source label
# ---------------------------------------------------------------------------

def _open_fastq(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def rename_and_concat(parts, out_gz_path):
    """parts: ordered list of (fastq_path, label) pairs -- e.g. a hybrid's two
    parents' R1 files, each with its own founder label. Streams every part,
    prefixes each read's header with "<label>|" right after the leading '@',
    and gzip-writes the concatenation to out_gz_path. Plain or .gz inputs both
    accepted. Not resumable by design (unlike the corpus's own concat_files/
    simulate_reads_pe) -- callers should route through a fresh --outdir per
    arm instead, per the existing eval-pipeline caching convention.

    Returns (out_gz_path, n_reads_by_label: dict).
    """
    out_gz_path = Path(out_gz_path)
    out_gz_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {label: 0 for _, label in parts}
    with gzip.open(out_gz_path, "wt") as out_f:
        for path, label in parts:
            prefix = f"@{label}{LABEL_SEP}"
            with _open_fastq(path) as in_f:
                for i, line in enumerate(in_f):
                    if i % 4 == 0:
                        if not line.startswith("@"):
                            raise ValueError(
                                f"{path}: expected FASTQ header at record start, got {line[:60]!r}"
                            )
                        out_f.write(prefix + line[1:])
                        counts[label] += 1
                    else:
                        out_f.write(line)
    return out_gz_path, counts


def parse_read_label(read_name):
    """Inverse of rename_and_concat's header prefix. read_name may or may not
    include the leading '@' and may or may not include the /1 or /2 mate
    suffix or trailing whitespace -- all tolerated. Returns (label, rest) or
    (None, read_name) if unlabeled."""
    name = read_name.lstrip("@").rstrip("\n")
    if LABEL_SEP not in name:
        return None, name
    label, rest = name.split(LABEL_SEP, 1)
    return label, rest


# ---------------------------------------------------------------------------
# L2 -- per-bin truth labels, aligned to raw.npy row order
# ---------------------------------------------------------------------------

def _vectorized_founder_at(sorted_segs, positions):
    """sorted_segs: [(start, end, label), ...] sorted by start, contiguous
    coverage of the chromosome. positions: 1-D int array. Returns array of
    labels (object dtype), vectorized via searchsorted instead of per-position
    bisect (matters at ~1M rows)."""
    starts = np.array([s[0] for s in sorted_segs], dtype=np.int64)
    labels = np.array([s[2] for s in sorted_segs], dtype=object)
    idx = np.searchsorted(starts, positions, side="right") - 1
    idx = np.clip(idx, 0, len(sorted_segs) - 1)
    return labels[idx]


def bin_truth_labels(bins_tsv_path, kind, gamete_names, dataset_id=None,
                      individual=None, parent_a=None, parent_b=None,
                      replicate=0):
    """Returns (labels: np.int16[n_rows, 2], founder_names: list[str]).

    labels[:, 0] / labels[:, 1] are the true founder's index into
    `gamete_names` for h1/h2 respectively, -1 if the founder is not in
    gamete_names (e.g. dropped by the fixed founder-drop index) or position
    is unresolvable. Row order matches raw.npy exactly -- both are built by
    iterating raw.npy.bins.tsv in file order (verified: bins.tsv row count -
    1 header == raw.npy row count on every existing row).

    kind == "inbred": parent_a required (== the sample), parent_b ignored;
        constant label every row.
    kind == "hybrid": parent_a, parent_b required; constant labels every row
        (0 breakpoints by this corpus's own hybrid-truth construction --
        truth.py's hybrid_truth is literally the two parent gVCFs, no mosaic).
    kind == "ril"/"ril2": dataset_id, individual, parent_a, parent_b required;
        delegates the B73-coordinate mosaic to
        simval_oracle_bed.build_ril_mosaics(dataset_id, parent_a, parent_b,
        replicate) -- which itself dispatches on dataset_id's kind, so "ril2"
        naturally comes back with mosaic_h1 is mosaic_h2 (homozygous) -- reads
        THIS row's own bins_tsv (not a reference layout) so labels line up
        with whichever alignment produced it.
    """
    bins_df = pd.read_csv(bins_tsv_path, sep="\t")
    n_rows = len(bins_df)
    name_to_idx = {n: i for i, n in enumerate(gamete_names)}
    labels = np.full((n_rows, 2), -1, dtype=np.int16)

    if kind == "inbred":
        if parent_a is None:
            raise ValueError("kind='inbred' requires parent_a")
        idx = name_to_idx.get(parent_a, -1)
        labels[:, 0] = idx
        labels[:, 1] = idx
        return labels, gamete_names

    if kind == "hybrid":
        if parent_a is None or parent_b is None:
            raise ValueError("kind='hybrid' requires parent_a and parent_b")
        labels[:, 0] = name_to_idx.get(parent_a, -1)
        labels[:, 1] = name_to_idx.get(parent_b, -1)
        return labels, gamete_names

    if kind in ("ril", "ril2"):
        if not all([dataset_id, individual, parent_a, parent_b]):
            raise ValueError(f"kind={kind!r} requires dataset_id, individual, parent_a, parent_b")
        mosaic_h1, mosaic_h2, label_to_name = oracle.build_ril_mosaics(
            dataset_id, parent_a, parent_b, replicate=replicate)
        for contig, sub in bins_df.groupby("contig", sort=False):
            if contig not in mosaic_h1:
                continue  # organelle/unplaced contigs carry no mosaic -> stay -1
            positions = (sub["bin"].to_numpy() * 256).astype(np.int64)
            segs1 = mosaic_h1[contig]
            segs2 = mosaic_h2[contig]
            lab1 = _vectorized_founder_at(segs1, positions)
            lab2 = _vectorized_founder_at(segs2, positions)
            name1 = np.array([name_to_idx.get(label_to_name[l], -1) for l in lab1], dtype=np.int16)
            name2 = np.array([name_to_idx.get(label_to_name[l], -1) for l in lab2], dtype=np.int16)
            labels[sub.index.to_numpy(), 0] = name1
            labels[sub.index.to_numpy(), 1] = name2
        return labels, gamete_names

    raise ValueError(f"unknown kind {kind!r}, expected inbred/hybrid/ril/ril2")


def write_truth_labels(outdir, bins_tsv_path, kind, gamete_names, **kwargs):
    """Computes bin_truth_labels(...) and writes truth_labels.npy +
    truth_labels.tsv next to raw.npy in `outdir`. Returns the labels array."""
    outdir = Path(outdir)
    labels, names = bin_truth_labels(bins_tsv_path, kind, gamete_names, **kwargs)
    np.save(outdir / "truth_labels.npy", labels)

    bins_df = pd.read_csv(bins_tsv_path, sep="\t")
    name_arr = np.array(names + ["NA"])
    h1_names = name_arr[np.where(labels[:, 0] >= 0, labels[:, 0], len(names))]
    h2_names = name_arr[np.where(labels[:, 1] >= 0, labels[:, 1], len(names))]
    out_df = pd.DataFrame({
        "row": bins_df["row"] if "row" in bins_df.columns else np.arange(len(labels)),
        "contig": bins_df["contig"],
        "bin": bins_df["bin"],
        "truth_h1_idx": labels[:, 0],
        "truth_h1_name": h1_names,
        "truth_h2_idx": labels[:, 1],
        "truth_h2_name": h2_names,
    })
    out_df.to_csv(outdir / "truth_labels.tsv", sep="\t", index=False)
    return labels
