"""
Whole-chromosome CRF decode for REAL sparse corpus data (grits_workdir's
simval_eval rows: raw.npy + raw.npy.bins.tsv, ragged per-contig bin counts,
most 256bp bins empty at low coverage) -- reuses infer_wholegenome.py's
sliding-window encode + center-crop stitch + single-chromosome CRF decode
(decode_chrom/_starts/_ownership/affinity_keep, imported UNMODIFIED) instead
of duplicating it. That module was built only for the dense, rectangular
[NC,L,K+2] arrays simulate_wholegenome.py produces; two real gaps here:

  1. Sparse per-contig loading: ragged L per contig from raw.npy/bins.tsv,
     not a dense array -- see load_real_contigs().
  2. homo_scale plumbed through to the model -- infer_wholegenome._encode()
     hardcodes homo_scale=None (applies the checkpoint's FULL fixed
     homo_penalty regardless of the individual's true zygosity). Confirmed
     elsewhere (chromosome-mosaic RIL sanity check) that this mismatch alone
     causes ~39% per-base founder error vs ~0.04% with the correct prior.
     Fixed here via _encode_real() + a module-global monkey-patch of
     infer_wholegenome._encode (no edit to that shared file -- same pattern
     this codebase already uses for the v1->v2 index cutover, hae.nb.FMD=...).

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/grits-wholechrom-worktree/src \
      /home/zrm22/mambaforge/envs/phg-ml/bin/python \
      src/python/crf/infer_wholegenome_real.py \
        --row-dir /local/workdir/.../scratch/simval_eval/ROW_ID --sample NAME \
        --ckpt /local/workdir/.../checkpoints/diploid-affinity-sim512-h3/d-epoch=04-val_pair_acc=0.6179.ckpt \
        --homo-scale 0.5 --drop-idx 23 --win 512 --stride 256 \
        --out-bed-dir /local/workdir/.../scratch/simval_eval/ROW_ID/bed_wholechrom
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# This codebase has two inconsistent import-root conventions across its own
# files: crf/*.py imports as `python.crf...` (needs .../src on sys.path,
# already satisfied by this script's own PYTHONPATH), while bed_io/*.py
# imports as `src.python.bed_io...` (needs .../src's PARENT, the repo root,
# on sys.path too) -- same fix heldout_assembly_eval.py already applies.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # .../src/python/crf/<file> -> repo root

import python.crf.infer_wholegenome as iwg  # noqa: E402
from python.crf.train_diploid import (  # noqa: E402
    GRITSCRFDiploid, _founder_affinity, estimate_inbreeding_coef, homo_scale_from_affinity)
from python.bed_io.bed import output_collapse_bed  # noqa: E402

SOURCE_K = 25
TARGET_K = 24


@torch.no_grad()
def _encode_real(model, feats, starts, win, ext_emb, device, bs, homo_scale):
    """Same body as infer_wholegenome._encode, except homo_scale is passed
    through to the model instead of hardcoded None."""
    Xs = torch.stack([feats[s:s + win] for s in starts])          # [nwin,win,K]
    em, cc = [], []
    for i in range(0, len(starts), bs):
        Xb = Xs[i:i + bs].to(device)
        eb = ext_emb.expand(Xb.shape[0], -1, -1) if ext_emb is not None else None
        hs = torch.full((Xb.shape[0],), homo_scale, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emis_p, _g, c = model(Xb, hs, eb)
        em.append(emis_p.float().cpu())
        cc.append(c.float().cpu())
    return torch.cat(em), torch.cat(cc)


def make_encode_real(homo_scale):
    """Factory closing over homo_scale, matching infer_wholegenome._encode's
    exact call signature so decode_chrom's internal call site (which knows
    nothing about homo_scale) works unmodified after the monkey-patch."""
    def _encode(model, feats, starts, win, ext_emb, device, bs):
        return _encode_real(model, feats, starts, win, ext_emb, device, bs, homo_scale)
    return _encode


def load_real_contigs(row_dir, drop_idx):
    """raw.npy [n_rows, SOURCE_K+2] + raw.npy.bins.tsv -> per-contig ragged
    {contig: (feats [n_rows_c, TARGET_K] float32 tensor, bp [n_rows_c] int64)},
    row order preserved (on-disk/file order, matching every other script's
    convention this session), K25->K24 drop applied with the SAME remap
    math ropebwt_npy_to_matrix.py / simval_eval_one.window_fixed_drop use.
    No windowing/truncation here -- whole-chromosome decode wants every
    real bin, not just window-aligned ones."""
    row_dir = Path(row_dir)
    raw = np.load(row_dir / "raw.npy", mmap_mode="r")
    bins_df = pd.read_csv(row_dir / "raw.npy.bins.tsv", sep="\t")
    if len(bins_df) != raw.shape[0]:
        raise ValueError(f"raw.npy rows ({raw.shape[0]}) != bins.tsv rows ({len(bins_df)})")

    keep_idx = np.array([i for i in range(SOURCE_K) if i != drop_idx])
    contigs = {}
    for contig, rows in bins_df.groupby("contig", sort=False):
        idx = np.sort(rows.index.to_numpy())
        feats = np.asarray(raw[idx][:, keep_idx]).astype(np.float32)
        bp = (bins_df.loc[idx, "bin"].to_numpy() * 256).astype(np.int64)
        contigs[contig] = (torch.tensor(feats), bp)
    return contigs


def k_target_to_name(idx_target, dropped_idx, gamete_names):
    """Map a TARGET_K-space founder index back to its SOURCE_K-space name --
    identical logic to heldout_assembly_eval.k_target_to_name, reproduced
    locally so this module has no dependency on grits_workdir's (non-git)
    scripts/ path."""
    idx_source = idx_target
    if dropped_idx is not None and idx_target >= dropped_idx:
        idx_source += 1
    return gamete_names[idx_source]


def write_wholechrom_bed(sample, contig_preds, model, gamete_names, dropped_idx, out_dir):
    """contig_preds: {contig: (pred [L] pair-idx tensor, bp [L] int64 array)}.
    Same chrom/start/end/parent1/parent2 contract as heldout_assembly_eval.
    write_imputed_bed (same output_collapse_bed helper), but from a flat
    per-contig decoded path instead of a [n_windows,T] reshape."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pi_c, pj_c = model.pi.cpu().numpy(), model.pj.cpu().numpy()
    written = []
    for contig, (pred, bp) in contig_preds.items():
        pred_np = pred.numpy()
        p1 = [k_target_to_name(int(pi_c[p]), dropped_idx, gamete_names) for p in pred_np]
        p2 = [k_target_to_name(int(pj_c[p]), dropped_idx, gamete_names) for p in pred_np]
        df = pd.DataFrame({"chrom": contig, "start": bp, "end": bp + 256,
                            "parent1": p1, "parent2": p2})
        out_path = out_dir / f"{sample}_{contig}_imputed.bed"
        output_collapse_bed(df, str(out_path))
        written.append(out_path)
        print(f"  {contig}: {len(pred_np):,} real sites -> {out_path}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-dir", required=True, help="scratch/simval_eval/<ROW_ID> directory")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--homo-scale", type=float, default=None,
                     help="0.0 inbred / 0.5 ril / 1.0 hybrid -- passed straight to the model, "
                          "unlike infer_wholegenome.py's hardcoded None (=full fixed penalty). "
                          "Omit to auto-derive from this individual's genome-wide founder "
                          "affinity via homo_scale_from_affinity (train_diploid.py) -- a smooth "
                          "0/1 threshold on the background-corrected top1/top2 affinity gap.")
    ap.add_argument("--drop-idx", type=int, default=23, help="SOURCE_K index dropped (P39=23)")
    ap.add_argument("--win", type=int, default=512,
                     help="encoder window length -- match the checkpoint's training T "
                          "(512 for diploid-affinity-sim512-h3), NOT infer_wholegenome.py's "
                          "1024 default -- the encoder is O(T^2) with no length extrapolation")
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--decode", choices=["viterbi", "marginal", "viterbi-factored"],
                     default="viterbi")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-bed-dir", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRITSCRFDiploid.load_from_checkpoint(args.ckpt, map_location=device).eval().to(device)

    row_dir = Path(args.row_dir)
    gametes = pd.read_csv(row_dir / "raw.npy.gametes.tsv", sep="\t").sort_values("gameteIndex")
    gamete_names = gametes["sampleName"].tolist()
    if len(gamete_names) != SOURCE_K:
        raise ValueError(f"expected {SOURCE_K} gametes, got {len(gamete_names)}")

    contigs = load_real_contigs(row_dir, args.drop_idx)

    # genome-wide founder affinity, over ALL real contigs concatenated (matches
    # simval_eval_one.run_inference_diploid's own genome-wide-pooled convention)
    all_feats = torch.cat([f for f, _ in contigs.values()], dim=0).numpy()
    aff = _founder_affinity(all_feats)                              # [TARGET_K,2]
    ext = torch.tensor(aff, dtype=torch.float32, device=device).unsqueeze(0)

    if args.homo_scale is None:
        homo_scale = homo_scale_from_affinity(aff[:, 0])
        print(f"[{args.sample}] homo_scale not given -- auto-derived {homo_scale} "
              f"(estimated inbreeding coef={estimate_inbreeding_coef(aff[:, 0]):.3f})")
    else:
        homo_scale = args.homo_scale

    print(f"[{args.sample}] loaded {len(contigs)} contigs, homo_scale={homo_scale}, "
          f"win={args.win} stride={args.stride} decode={args.decode}")
    for c, (feats, bp) in contigs.items():
        print(f"  {c}: {feats.shape[0]:,} real bins  bp=[{bp.min():,}-{bp.max():,}]")

    iwg._encode = make_encode_real(homo_scale)                      # module-global monkey-patch

    contig_preds = {}
    for contig, (feats, bp) in contigs.items():
        L = feats.shape[0]
        if L < args.win:
            print(f"  SKIP {contig}: {L} real bins < win={args.win}")
            continue
        out = iwg.decode_chrom(model, feats, ext, mode="whole-chrom", decode=args.decode,
                                win=args.win, stride=args.stride, device=device,
                                bs=args.batch_size, variants=None)
        pred, dt = out["full"]
        contig_preds[contig] = (pred, bp)
        print(f"  {contig}: decoded {L:,} sites in {dt*1e3:.1f}ms")

    write_wholechrom_bed(args.sample, contig_preds, model, gamete_names, args.drop_idx,
                          args.out_bed_dir)


if __name__ == "__main__":
    main()
