#!/usr/bin/env python
"""
Fast SNP+RefCall eval of the OLD July "diploid-affinity" model (E12,
checkpoints/diploid-affinity-sim512-h3, class GRITSCRFDiploid in
train_diploid.py -- NOT GRITSCRFDiploidIndel) on the same aligned inputs and
with the same scorer as fast_eval_ckpt.py, so its numbers are comparable with
the current models.

Model input (weights unchanged). GRITSCRFDiploid takes K=24 per-founder match
features [N,T,24] (P39, gamete index 23, dropped -- the July K25->K24 fixed
drop), a genome-wide founder-affinity ext_emb (_founder_affinity of those
features) and a per-sample homo_scale. It predates the ternary/distance/count
layout. From each row's windowed_k25native_wcount.npy ([N,T,3K+2], K=25):
  --features bin   (default): X = (ternary == MATCH)[..., keep24]; 0/1 as in
                   training. MATCH == (read count > 0) exactly on this data.
  --features count : X = read count per founder, decoded from the count block
                   (code = rint(8*log2(1+c)) -> c = rint(2**(code/8)-1)); this is
                   what the July eval actually fed (its windowed_k24_fixdrop23.npy
                   predates binarize-by-default), kept to separate that effect.
  --old-input      : read the July eval's own windowed_k24_fixdrop23.npy (raw
                   counts) + raw.npy.bins.tsv from scratch/simval_eval/<DS>__<IND>__<cov>x
                   (sanity check that this script reproduces the July rescore).
Rows whose reads matched only P39 become all-zero rows, identical to the July
window_fixed_drop. BEDs are written with dropped_idx=23 so founder names map
back through hae.k_target_to_name exactly as in July.

Kind / homo_scale (the model's only per-sample knob; binary 0 = inbred/RIL,
1 = hybrid):
  --kind-source route  (default): route_real_sample (train_diploid_indel.py)
                   on the K25 MATCH view -> homo_scale = 1 if hybrid (same-bin
                   disjoint-pair fraction > ROUTE_DISJ_HYBRID) else 0. Only the
                   homo_scale is taken from the router; the decoder's switch
                   cost is left as trained (switch_scale 1.0).
  --kind-source p90    : homo_scale_from_affinity (train_diploid.py, p90 of the
                   per-window inbreeding estimate, threshold 0.5) on the K25
                   MATCH view -- the default infer_real_founder_pairs call.
  --kind-source oracle : the TRUE kind, simval_eval_one.py's KIND_HOMO_SCALE
                   (inbred 0, hybrid 1, ril2 0; ril 0.5).
Both classifier calls are always computed and stored in the JSON.

Results: results/<tag>_<scope>_fast/<tag>__<DS>__<IND>__<cov>x.json with the
same keys as fast_eval_ckpt.py.
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
# Model code BEFORE fast_snprc_score / heldout_assembly_eval (they put another
# checkout's `python` package on sys.path; the first import wins).
sys.path.insert(0, sei.CRF_SRC)
from python.crf.train_diploid import (GRITSCRFDiploid, _founder_affinity,  # noqa: E402
                                      _dcrf_viterbi, homo_scale_from_affinity)
from python.crf.train_diploid_indel import route_real_sample, TERN_MATCH  # noqa: E402
import fast_snprc_score as fss  # noqa: E402
import heldout_assembly_eval as hae  # noqa: E402

WORK = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
CKPT = WORK / "checkpoints/diploid-affinity-sim512-h3/d-epoch=04-val_pair_acc=0.6179.ckpt"
MANIFEST = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv"
SCOPES = {"out": ["OUT-INBRED", "OUT-HYB", "OUT-RIL2"],
          "idx": ["IDX-INBRED", "IDX-HYB", "IDX-RIL2"],
          "mix": ["MIX-HYB", "MIX-RIL", "MIX-RIL2"]}
K25, DROP = 25, 23
KEEP = np.array([i for i in range(K25) if i != DROP])
KIND_HOMO_SCALE = {"inbred": 0.0, "hybrid": 1.0, "ril": 0.5, "ril2": 0.0}  # simval_eval_one.py


def s200_features(data, mode):
    tern = data[:, :, :K25]
    if mode == "bin":
        X = (tern == TERN_MATCH).astype(np.float32)
    else:
        code = data[:, :, 2 * K25 + 2:3 * K25 + 2].astype(np.float64)
        X = np.rint(2.0 ** (code / 8.0) - 1.0).astype(np.float32)
    return X[:, :, KEEP]


def classify(data, bins_path):
    """Model-independent kind calls on the K25 MATCH view (as the current models)."""
    M = data[:, :, :K25] == TERN_MATCH
    p90_hs = float(homo_scale_from_affinity(M.astype(np.float32)))
    layout = hae.load_contig_layout(bins_path, data.shape[1], bin_size=256)
    row_bin = np.concatenate([np.asarray(pos, np.int64) + (ci << 40)
                              for ci, (_c, pos, _n) in enumerate(layout)]
                             ).reshape(data.shape[0], data.shape[1])
    r = route_real_sample(M, row_bin)
    return p90_hs, r


def decode(model, X, homo_scale, device, batch_size=128):
    import torch
    ext = torch.tensor(_founder_affinity(X), dtype=torch.float32, device=device)
    lo, hi = [], []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.tensor(X[s:s + batch_size], dtype=torch.float32, device=device)
            hs = torch.full((xb.shape[0],), float(homo_scale), device=device)
            emis_p, _, c = model(xb, hs, ext.unsqueeze(0).expand(xb.shape[0], -1, -1))
            pred = _dcrf_viterbi(emis_p, c, model.nsw_pair, model.stay_bonus)
            lo.append(model.pi[pred].cpu().numpy())
            hi.append(model.pj[pred].cpu().numpy())
    return np.concatenate(lo), np.concatenate(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--scope", nargs="+", choices=sorted(SCOPES), default=["idx", "out", "mix"])
    ap.add_argument("--depth", default="0.1")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--kind-source", choices=["route", "p90", "oracle"], default="route")
    ap.add_argument("--features", choices=["bin", "count"], default="bin")
    ap.add_argument("--input-glob", default="scratch/lift_s200_local20k")
    ap.add_argument("--old-input", action="store_true",
                    help="use the July scratch/simval_eval windowed_k24_fixdrop23.npy instead")
    ap.add_argument("--rows", nargs="*", default=None, help="optional DS__IND filter")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GRITSCRFDiploid.load_from_checkpoint(args.ckpt, map_location=device).eval().to(device)
    assert model.num_parents == 24 and getattr(model, "founder_affinity", False)
    panel = fss.Panel()
    D = args.depth
    rows = [r for r in csv.DictReader(open(args.manifest), delimiter="\t") if r["coverage"] == D]
    summary = {}
    for scope in args.scope:
        res_dir = WORK / "results" / f"{args.tag}_{scope}_fast"
        res_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            ds, ind, kind = r["dataset_id"], r["individual"], r["kind"]
            if ds not in SCOPES[scope] or (args.rows and f"{ds}__{ind}" not in args.rows):
                continue
            t0 = time.time()
            src = WORK / args.input_glob / f"eval__{ds}__{ind}__{D}x"
            wc = src / "windowed_k25native_wcount.npy"
            if not wc.exists():
                print(f"SKIP no s200 input {src}", flush=True)
                continue
            data = np.load(wc)
            p90_hs, rt = classify(data, src / "raw.npy.bins.tsv")
            if args.old_input:
                old = WORK / "scratch/simval_eval" / f"{ds}__{ind}__{D}x"
                X = np.load(old / "windowed_k24_fixdrop23.npy")[:, :, :24].astype(np.float32)
                bins, gam = old / "raw.npy.bins.tsv", old / "raw.npy.gametes.tsv"
            else:
                X = s200_features(data, args.features)
                bins, gam = src / "raw.npy.bins.tsv", src / "raw.npy.gametes.tsv"
            hs = {"route": rt["homo_scale"], "p90": p90_hs,
                  "oracle": KIND_HOMO_SCALE[kind]}[args.kind_source]
            true_hyb = kind == "hybrid"
            key = f"{args.tag}__{ds}__{ind}__{D}x"
            lo, hi = decode(model, X, hs, device)
            bed_dir = WORK / "scratch" / f"{args.tag}_{scope}_fast" / key / "bed"
            with contextlib.redirect_stdout(io.StringIO()):
                hae.write_imputed_bed(ind, lo, hi, DROP, hae.load_gamete_names(gam),
                                      bins, bed_dir, bin_size=256)
            f1, f2 = fss.pair_from_bed(panel, bed_dir)
            res = fss.score_arrays(panel, fss.load_truth(r["truth_h1"]),
                                   fss.load_truth(r["truth_h2"]), f1, f2)
            log = (f"kind_source={args.kind_source} homo_scale={hs} | route: disj={rt['disj']:.4f} "
                   f"p90={rt['p90']:.3f} hybrid={rt['hybrid']} | p90 classifier chose {p90_hs}")
            res.update(row_key=key, status="ok", source_input=str(old if args.old_input else src),
                       features=("old_k24_counts" if args.old_input else args.features),
                       kind=kind, kind_source=args.kind_source, homo_scale=hs,
                       route_disj=rt["disj"], route_p90=rt["p90"], route_hybrid=rt["hybrid"],
                       route_correct=(rt["hybrid"] == true_hyb),
                       p90_homo_scale=p90_hs, p90_correct=((p90_hs == 1.0) == true_hyb),
                       homo_scale_log=log, wall_seconds=round(time.time() - t0, 1))
            (res_dir / f"{key}.json").write_text(json.dumps(res, indent=1))
            summary.setdefault(ds, []).append((res["snprc_error_rate"], res["error_rate"]))
            print(f"{key:<60} snprc={100 * res['snprc_error_rate']:6.2f}%  "
                  f"err={100 * res['error_rate']:6.2f}%  {log}  ({res['wall_seconds']}s)", flush=True)
    print()
    for ds, v in summary.items():
        v = np.array(v)
        print(f"{ds:<12} n={len(v)}  snprc={100 * v[:, 0].mean():.2f}%  all={100 * v[:, 1].mean():.2f}%")


if __name__ == "__main__":
    main()
