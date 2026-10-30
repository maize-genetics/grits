"""Promoted GRITS decode for one aligned sample: windowed model input -> founder-pair BED files.

    decode_sample.py --dir OUTDIR --sample NAME [--device cuda|cpu] [--batch-size 256]

OUTDIR is an align_sample.sh output directory. Writes OUTDIR/bed/NAME_<chrom>_imputed.bed (one file per
chromosome, columns chrom start end parent1 parent2, founder names) and OUTDIR/decode_summary.txt.

This is the same call the promoted evaluation makes (infer_real_founder_pairs with preset
placedfix_hetdel_hs_peel, route=True, knockout="homoroute"); only the corpus bookkeeping is left out.
"""
import argparse
import contextlib
import subprocess
import io
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))          # bed.py imports src.python.*
sys.path.insert(0, str(REPO / "src"))  # model code imports python.*

BIN_SIZE = 256   # refmap's default; align_sample.sh never changes it
WINDOW = 512


def contig_layout(bins_tsv, window=WINDOW):
    """(contig, bp positions, n_windows) per contig, in the order the windowing script concatenated them:
    whole 512-row windows only, so a contig's trailing partial window is never decoded."""
    bins = pd.read_csv(bins_tsv, sep="\t")
    out = []
    for contig, idx in bins.groupby("contig", sort=False).indices.items():
        idx = np.sort(idx)
        n = len(idx) // window
        if n:
            out.append((contig, bins.loc[idx[: n * window], "bin"].to_numpy() * BIN_SIZE, n))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="align_sample.sh output directory")
    ap.add_argument("--sample", required=True, help="sample name used in the BED file names")
    ap.add_argument("--ckpt", default=os.environ.get("GRITS_CKPT"), help="checkpoint [$GRITS_CKPT]")
    ap.add_argument("--device", default=None, help="cuda or cpu [cuda if available]")
    ap.add_argument("--batch-size", type=int, default=256, help="windows per GPU batch; lower it if out of memory")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for torch/numpy [4]")
    a = ap.parse_args()
    if re.search(r"_chr", a.sample):
        sys.exit("sample name must not contain '_chr': the BED->VCF tool splits file names on it")
    if not a.ckpt:
        sys.exit("no checkpoint: pass --ckpt or source grits.env")

    import torch
    torch.set_num_threads(a.threads)
    from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, INFERENCE_PRESETS, infer_real_founder_pairs
    if "placedfix_hetdel_hs_peel" not in INFERENCE_PRESETS:
        sys.exit("unknown preset placedfix_hetdel_hs_peel: this GRITS checkout is older than the promoted model")
    from src.python.bed_io.bed import output_collapse_bed

    d = Path(a.dir)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    data = np.load(d / "windowed_k25native_wcount.npy")
    names = pd.read_csv(d / "raw.npy.gametes.tsv", sep="\t").sort_values("gameteIndex")["sampleName"].tolist()
    if len(names) != 25 or data.shape[2] != 3 * 25 + 2:
        sys.exit(f"expected the 25-founder panel (input width 77); got {len(names)} founders, width {data.shape[2]}")
    layout = contig_layout(d / "raw.npy.bins.tsv")
    if sum(n for _, _, n in layout) != data.shape[0]:
        sys.exit("window count does not match raw.npy.bins.tsv: windowing and bins are out of sync")
    row_bin = np.concatenate([np.asarray(p, np.int64) + (ci << 40) for ci, (_, p, _) in enumerate(layout)])
    row_bin = row_bin.reshape(data.shape[:2])

    model = GRITSCRFDiploidIndel.load_from_checkpoint(a.ckpt, map_location=dev, strict=False).eval().to(dev)
    print(f"decoding {data.shape[0]:,} windows on {dev}", flush=True)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        lo, hi = infer_real_founder_pairs(model, data, 25, device=dev, batch_size=a.batch_size,
                                          preset="placedfix_hetdel_hs_peel", route=True, row_bin=row_bin,
                                          knockout="homoroute")
    print(log.getvalue(), end="")
    (d / "decode.log").write_text(log.getvalue())
    route = [l.strip() for l in log.getvalue().splitlines() if l.strip().startswith("route:")]

    bed_dir = d / "bed"
    bed_dir.mkdir(exist_ok=True)
    lo, hi = lo.reshape(-1), hi.reshape(-1)
    cur, lines = 0, route[:1] or ["route: (not reported)"]
    for contig, pos, n in layout:
        m = n * WINDOW
        p1, p2 = lo[cur:cur + m].astype(int), hi[cur:cur + m].astype(int)
        cur += m
        keep = (p1 < len(names)) & (p2 < len(names))       # null-state rows (no founder fits) stay uncalled
        if not keep.any():
            lines.append(f"{contig}: no founder calls")
            continue
        rows = pd.DataFrame({"chrom": contig, "start": pos[keep], "end": pos[keep] + 1,
                             "parent1": [names[i] for i in p1[keep]], "parent2": [names[i] for i in p2[keep]],
                             "row": np.flatnonzero(keep)})
        out = bed_dir / f"{a.sample}_{contig}_imputed.bed"
        output_collapse_bed(rows, str(out), fill_switch_gaps=True)
        het = float((rows.parent1 != rows.parent2).mean())
        lines.append(f"{contig}: {keep.sum():,} sites called, {(~keep).sum():,} uncalled, het fraction {het:.3f}")
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        commit = "unknown"
    lines.append(f"decode wall time {time.time() - t0:.0f} s")
    lines.append(f"provenance: checkpoint {a.ckpt}; preset placedfix_hetdel_hs_peel, route, knockout homoroute; "
                 f"GRITS commit {commit}")
    (d / "decode_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
