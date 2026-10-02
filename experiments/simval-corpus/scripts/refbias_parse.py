#!/usr/bin/env python
"""
PS4G parsing + per-index-sample attribution metrics for the reference-bias
eval (see /home/zrm22/.claude/plans/dreamy-booping-sutton.md).

Why PS4G and not raw.tsv: `refmap`'s plain TSV blanks the assembly-hit column
for EXACT reads (col 5 = "."), so it under-reports sample attribution for the
very reads a bias study cares most about. The `.ps4g` file's `#gamete` header
block is computed by the tool itself, with the SAME accumulator semantics for
refmap and chain (ps4g.c:345-351: "gamete_total[g] += row.count for every row
whose gameteSet contains g"), and never blanks anything. `raw.tsv` is
correspondingly not touched by this module at all -- placement rate is
derived from `#TotalUniqueCounts` (== PLACED+EXACT for refmap; every emitted
chain row, for chain) versus the manifest's exact read count, which is
cheaper and avoids streaming a 50-200MB file per sample.

Two things this module deliberately keeps separate rather than conflating:
- gamete_totals (from the header): reads-crediting-sample-S, DOUBLE-COUNTED
  across a read's whole gameteSet. Sum over founders >> TotalUniqueCounts.
  This is "marginal hit ratio" -- exactly what the task asked for ("ratio of
  how many reads hit each index sample").
- singleton_totals (from the body): reads whose gameteSet has cardinality 1,
  i.e. UNIQUELY attributed to that one sample. Not double-counted. Sharper
  signal, smaller denominator.
"""
from collections import defaultdict
from pathlib import Path

# Fixed 25-founder alphabetical panel -- matches nam_baseline.PANEL_ORDER /
# every raw.npy.gametes.tsv on disk. Used only as a cross-check; the real
# index->name mapping is always read from the run's own header.
PANEL_ORDER = [
    "B73", "B97", "CML103", "CML228", "CML247", "CML277", "CML322", "CML333",
    "CML52", "CML69", "HP301", "Il14H", "Ki11", "Ki3", "Ky21", "M162W", "M37W",
    "Mo18W", "Ms71", "NC350", "NC358", "Oh43", "Oh7B", "P39", "Tzi8",
]

# All 30 possible parent names in this corpus (25 index founders + 5 held-out
# assemblies) -- used to split hybrid/RIL individual names like "B73xTx303"
# or "Tx303xA188" correctly. A naive split on the literal "x" breaks on any
# name containing a lowercase x (Tx303) -- confirmed present in 8 of the 30
# names' cross products. No name here is a prefix of another, so first-match
# is unambiguous.
HELDOUT_NAMES = ["Tx303", "A188", "EP1", "CML459", "Ia453"]
ALL_PARENT_NAMES = PANEL_ORDER + HELDOUT_NAMES


def split_individual_name(individual, kind):
    """('B73',) for inbred; (parentA, parentB) for hybrid/ril, order-preserving."""
    if kind == "inbred":
        return (individual,)
    for name in ALL_PARENT_NAMES:
        if individual.startswith(name):
            rest = individual[len(name):]
            if rest.startswith("x") and rest[1:] in ALL_PARENT_NAMES:
                return (name, rest[1:])
    raise ValueError(f"could not split parent names out of individual={individual!r} kind={kind!r}")


def parse_ps4g_header(ps4g_path):
    """Returns dict: total_unique_counts (int), gamete_totals (dict name->int,
    insertion order == gameteIndex order), command (str)."""
    total_unique_counts = None
    command = None
    gamete_totals = {}
    with open(ps4g_path) as f:
        for line in f:
            if not line.startswith("#"):
                break  # header block always precedes the data rows
            if line.startswith("#Command:"):
                command = line[len("#Command:"):].strip()
            elif line.startswith("#TotalUniqueCounts:"):
                total_unique_counts = int(line.split(":", 1)[1].strip())
            elif line.startswith("#gamete\t"):
                continue  # column-name line
            elif line.startswith("#version") or line == "#PS4G\n":
                continue
            else:
                # "#<name>\t<gameteIndex>\t<count>"
                parts = line[1:].rstrip("\n").split("\t")
                if len(parts) == 3 and parts[1].isdigit():
                    gamete_totals[parts[0]] = int(parts[2])
    if total_unique_counts is None or not gamete_totals:
        raise ValueError(f"malformed or empty ps4g header: {ps4g_path}")
    return {
        "total_unique_counts": total_unique_counts,
        "gamete_totals": gamete_totals,
        "command": command,
    }


def parse_ps4g_body(ps4g_path, gamete_names):
    """Single streaming pass over the PS4G data rows. Returns dict:
    - gamete_totals_recomputed: name -> int (cross-check against header)
    - singleton_totals: name -> int (reads uniquely attributed to that sample)
    - cardinality_hist: {set_size: read_count} (mean tells you how "shared"
      hits typically are -- refmap's union-style sets vs chain's
      intersection-style sets should differ sharply here)
    - n_rows, n_reads (== sum of count column, should equal header's
      total_unique_counts -- second cross-check)
    """
    n_idx = len(gamete_names)
    gamete_totals = [0] * n_idx
    singleton_totals = [0] * n_idx
    cardinality_hist = defaultdict(int)
    n_rows = 0
    n_reads = 0
    with open(ps4g_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            if line.startswith("gameteSet\t"):
                continue  # column header
            gset_str, _contig, _bin, count_str = line.rstrip("\n").split("\t")
            count = int(count_str)
            idxs = [int(x) for x in gset_str.split(",")]
            n_rows += 1
            n_reads += count
            cardinality_hist[len(idxs)] += count
            for i in idxs:
                gamete_totals[i] += count
            if len(idxs) == 1:
                singleton_totals[idxs[0]] += count
    return {
        "gamete_totals_recomputed": dict(zip(gamete_names, gamete_totals)),
        "singleton_totals": dict(zip(gamete_names, singleton_totals)),
        "cardinality_hist": dict(cardinality_hist),
        "n_rows": n_rows,
        "n_reads": n_reads,
    }


def compute_row_metrics(ps4g_path, total_input_reads):
    """Full per-(sample,arm) metric bundle for one ps4g file.

    total_input_reads: exact read count fed to this run (from the manifest,
    NOT re-derived from raw.tsv), used as the denominator for placement rate.
    """
    header = parse_ps4g_header(ps4g_path)
    gamete_names = list(header["gamete_totals"].keys())
    body = parse_ps4g_body(ps4g_path, gamete_names)

    # Cross-checks -- fail loudly rather than silently trusting a truncated file.
    mismatches = {
        n: (header["gamete_totals"][n], body["gamete_totals_recomputed"][n])
        for n in gamete_names
        if header["gamete_totals"][n] != body["gamete_totals_recomputed"][n]
    }
    if mismatches:
        raise ValueError(f"{ps4g_path}: header/body gamete-total mismatch: {mismatches}")
    if body["n_reads"] != header["total_unique_counts"]:
        raise ValueError(
            f"{ps4g_path}: body row-count sum {body['n_reads']} != "
            f"header TotalUniqueCounts {header['total_unique_counts']}"
        )

    total_unique = header["total_unique_counts"]
    placement_rate = total_unique / total_input_reads if total_input_reads else None

    # Two denominators, kept distinct: hit_ratio (of PLACED reads only) is the
    # natural per-arm number; hit_ratio_of_input (of ALL input reads) is what
    # makes refmap and chain comparable despite their different placement
    # rates -- a founder's hit_ratio can rise simply because an arm placed
    # fewer, more-confident reads, while hit_ratio_of_input cannot.
    hit_ratio = {
        n: (header["gamete_totals"][n] / total_unique if total_unique else 0.0)
        for n in gamete_names
    }
    hit_ratio_of_input = {
        n: (header["gamete_totals"][n] / total_input_reads if total_input_reads else 0.0)
        for n in gamete_names
    }
    singleton_ratio = {
        n: (body["singleton_totals"][n] / total_unique if total_unique else 0.0)
        for n in gamete_names
    }
    mean_cardinality = (
        sum(k * v for k, v in body["cardinality_hist"].items()) / body["n_rows"]
        if body["n_rows"] else 0.0
    )

    return {
        "gamete_names": gamete_names,
        "total_input_reads": total_input_reads,
        "total_unique_counts": total_unique,
        "placement_rate": placement_rate,
        "gamete_totals": header["gamete_totals"],
        "hit_ratio": hit_ratio,
        "hit_ratio_of_input": hit_ratio_of_input,
        "singleton_totals": body["singleton_totals"],
        "singleton_ratio": singleton_ratio,
        "cardinality_hist": body["cardinality_hist"],
        "mean_cardinality": mean_cardinality,
        "n_rows": body["n_rows"],
    }
