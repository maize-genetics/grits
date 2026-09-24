#!/usr/bin/env python
"""
Minutes-scale OUT/IDX 0.1x genotype eval of a checkpoint: reuses each row's
existing model input (windowed_k25native_wcount.npy + raw.npy.bins.tsv +
raw.npy.gametes.tsv -- produced by read alignment, independent of the
checkpoint), runs the canonical infer_real_founder_pairs once per row with
the model loaded once, writes the BED with the pipeline's own
hae.write_imputed_bed, and scores it with fast_snprc_score (validated
identical to the full bed-to-vcf + compare path).

Results: results/<tag>_<scope>_fast/<tag>__<DS>__<IND>__0.1x.json and a
per-class summary printed at the end.
"""
import argparse
import contextlib
import csv
import glob
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS))
import simval_eval_indel as sei  # noqa: E402
# The model code must be imported BEFORE fast_snprc_score: the comparator it
# wraps puts another checkout's src/ first on sys.path, and whichever
# `python` package is imported first wins for the whole process.
sys.path.insert(0, sei.CRF_SRC)
from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, infer_real_founder_pairs  # noqa: E402
import fast_snprc_score as fss  # noqa: E402
import heldout_assembly_eval as hae  # noqa: E402

WORK = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
MANIFEST = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv"
SCOPES = {"out": ["OUT-INBRED", "OUT-HYB", "OUT-RIL2"],
          "idx": ["IDX-INBRED", "IDX-HYB", "IDX-RIL2"],
          "mix": ["MIX-HYB", "MIX-RIL", "MIX-RIL2"]}


def source_dir(ds, ind, DEPTH):
    """Any existing row dir for this sample with a complete model input."""
    for d in sorted(glob.glob(str(WORK / f"scratch/*_snprc/*__{ds}__{ind}__{DEPTH}x"))):
        d = Path(d)
        if all((d / f).exists() for f in ("windowed_k25native_wcount.npy",
                                           "raw.npy.bins.tsv", "raw.npy.gametes.tsv")):
            return d
    raise FileNotFoundError(f"no existing model input for {ds} {ind} {DEPTH}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scope", nargs="+", choices=sorted(SCOPES), default=["out", "idx"])
    ap.add_argument("--oracle-homo", action="store_true",
                    help="force the true inbred/hybrid homo_scale (diagnostic)")
    ap.add_argument("--switch-scale", type=float, default=None,
                    help="scale the decoder's transition score (diagnostic; unset = 1.0, "
                         "or the router's choice with --route)")
    ap.add_argument("--decode", choices=["viterbi", "marginal"], default="viterbi")
    ap.add_argument("--depth", default="0.1")
    ap.add_argument("--route", action="store_true",
                    help="pick homo_scale and switch_scale per sample with route_real_sample")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GRITSCRFDiploidIndel.load_from_checkpoint(
        args.ckpt, map_location=device, strict=False).eval().to(device)
    panel = fss.Panel()
    DEPTH = args.depth
    rows = [r for r in csv.DictReader(open(MANIFEST), delimiter="\t") if r["coverage"] == DEPTH]

    summary = {}
    for scope in args.scope:
        out_root = WORK / "scratch" / f"{args.tag}_{scope}_fast"
        res_dir = WORK / "results" / f"{args.tag}_{scope}_fast"
        res_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            ds, ind = r["dataset_id"], r["individual"]
            if ds not in SCOPES[scope]:
                continue
            t0 = time.time()
            try:
                src = source_dir(ds, ind, DEPTH)
            except FileNotFoundError as e:
                print(f"SKIP {e}", flush=True)
                continue
            key = f"{args.tag}__{ds}__{ind}__{DEPTH}x"
            data = np.load(src / "windowed_k25native_wcount.npy")
            # oracle kind: MIX-RIL keeps heterozygous stretches, so it has no single true call
            hs = ((1.0 if ds.endswith("-HYB") else None if ds.endswith("-RIL") else 0.0)
                  if args.oracle_homo else None)
            row_bin = None
            if args.route:
                layout = hae.load_contig_layout(src / "raw.npy.bins.tsv", data.shape[1], bin_size=256)
                row_bin = np.concatenate([np.asarray(pos, np.int64) + (ci << 40)
                                          for ci, (_c, pos, _n) in enumerate(layout)]
                                         ).reshape(data.shape[0], data.shape[1])
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                pred_lo, pred_hi = infer_real_founder_pairs(model, data, sei.K, device=device,
                                                            homo_scale=hs,
                                                            switch_scale=args.switch_scale,
                                                            route=args.route, row_bin=row_bin,
                                                            decode=args.decode)
                bed_dir = out_root / key / "bed"
                hae.write_imputed_bed(ind, pred_lo, pred_hi, None,
                                      hae.load_gamete_names(src / "raw.npy.gametes.tsv"),
                                      src / "raw.npy.bins.tsv", bed_dir, bin_size=256)
            chosen = [l for l in buf.getvalue().splitlines() if "homo_scale=" in l or "route:" in l]
            f1, f2 = fss.pair_from_bed(panel, bed_dir)
            res = fss.score_arrays(panel, fss.load_truth(r["truth_h1"]),
                                   fss.load_truth(r["truth_h2"]), f1, f2)
            res.update(row_key=key, status="ok", source_input=str(src),
                       homo_scale_log=" | ".join(c.strip() for c in chosen) if chosen else None,
                       wall_seconds=round(time.time() - t0, 1))
            (res_dir / f"{key}.json").write_text(json.dumps(res, indent=1))
            summary.setdefault(ds, []).append(res["error_rate"])
            print(f"{key:<55} err={100 * res['error_rate']:6.2f}%  snprc={100 * res['snprc_error_rate']:6.2f}%  "
                  f"{res['homo_scale_log'] or ''}  ({res['wall_seconds']}s)", flush=True)
    print()
    for ds, errs in summary.items():
        print(f"{ds:<12} n={len(errs)}  mean_error_rate={100 * np.mean(errs):.2f}%")


if __name__ == "__main__":
    main()
