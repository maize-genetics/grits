"""
Unified held-out test-set evaluator for GRITS-CRF checkpoints — haploid
(GRITSCRFHaploid, train_haploid.py) and diploid (GRITSCRFDiploid,
train_diploid.py) in one entry point. Supersedes the separate eval_test.py /
eval_test_diploid.py (kept for reference; not modified).

Auto-detects the checkpoint type from its state_dict (diploid registers
`pair_table`; haploid does not) unless --mode overrides it. Rebuilds the
*identical* deterministic split used in training (make_splits / make_
diploid_splits are head-slices, no shuffle) and reports founder-path accuracy:
  haploid  — Viterbi vs emission-only argmax accuracy + breakpoint P/R/F1
             (±0, ±2 sites), via metrics.breakpoint_counts/prf.
  diploid  — pair_acc, sorted hap_acc, predicted-homozygous fraction.

Usage:
    python src/python/crf/eval.py --ckpt <path.ckpt> --data <path.npy> \
        --num-parents 24 --split test
    # --mode {auto,haploid,diploid} to force detection (default auto)

Haploid checkpoints trained with --relatedness (ext_dim > 0, train_haploid.py) are
auto-detected from the checkpoint's own hyperparameters and fed the affinity signal
`forward()` needs (silently zeroed otherwise, since forward(X, ext_emb=None) skips the
affinity term) — two modes, matching the two situations that come up in practice:
  - `--windows-per-individual G` given: the --data file holds MULTIPLE simulated
    individuals (G contiguous windows each, matching how it was built and trained);
    reuses train_haploid.py's own make_individual_splits/IndividualRelatednessDataset
    so each window gets its own individual's affinity, exactly as in training.
  - omitted: --data is treated as ONE real individual's genome (e.g. real reads with
    no per-individual grouping) — one genome-wide affinity vector is computed from the
    *entire* file (train_haploid.py's _founder_affinity is label-free, so this isn't
    leakage) and broadcast to every scored window.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from python.crf.train_haploid import (
    GRITSCRFHaploid, make_splits, make_individual_splits, _founder_affinity)
from python.crf.train_diploid import GRITSCRFDiploid, make_diploid_splits, _dcrf_viterbi
from python.crf.metrics import breakpoint_counts, prf

TOLS = (0, 2)


def parse_args():
    p = argparse.ArgumentParser(description="Unified GRITS-CRF held-out evaluator "
                                             "(haploid + diploid)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--mode", choices=["auto", "haploid", "diploid"], default="auto",
                   help="Model type. auto (default) sniffs the checkpoint's state_dict.")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--limit-n", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--split", choices=["test", "val"], default="test",
                   help="Which held-out split to score (val for the sanity check).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--workdir", default="/workdir/esb33",
                   help="Results summary is appended under <workdir>/results/.")
    p.add_argument("--tag", default="",
                   help="Optional label for the results line (e.g. arm name).")
    p.add_argument("--windows-per-individual", type=int, default=None,
                   help="Haploid --relatedness checkpoints only: G windows/individual "
                        "in --data. Given -> multi-individual sim split (matches "
                        "training). Omitted -> --data is treated as one real "
                        "individual's genome-wide data. Ignored for ext_dim=0 checkpoints.")
    return p.parse_args()


def detect_mode(ckpt_path):
    """Sniff haploid vs diploid from the checkpoint's own state_dict, no model
    construction needed. Diploid registers pi/pj/pair_table/nsw_pair
    (train_diploid.py); haploid registers only nsw (train_haploid.py)."""
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    return "diploid" if any(k.endswith("pair_table") for k in sd) else "haploid"


@torch.no_grad()
def evaluate_haploid(model, ds, device, batch_size, num_workers, fixed_ext_emb=None):
    """fixed_ext_emb: [K,2] single-individual affinity, broadcast to every batch
    (single real-individual mode). None + no "ext_emb" in the batch => no affinity
    (ext_dim=0 checkpoints, unchanged behavior). Batch already carries "ext_emb" =>
    multi-individual sim mode (IndividualRelatednessDataset), used as-is."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    n = vit_correct = emis_correct = 0
    bp = {t: {"tp_prec": 0, "n_pred": 0, "tp_rec": 0, "n_true": 0} for t in TOLS}
    for batch in loader:
        X = batch["input_embeds"].to(device)
        tags = batch["labels"].to(device)
        if fixed_ext_emb is not None:
            ext_emb = fixed_ext_emb.unsqueeze(0).expand(X.shape[0], -1, -1).to(device)
        else:
            ext_emb = batch["ext_emb"].to(device) if "ext_emb" in batch else None
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emis_f, g, c = model(X, ext_emb=ext_emb)
        else:
            emis_f, g, c = model(X, ext_emb=ext_emb)
        pred_vit = model.decode(emis_f, c)              # respects no_transition
        pred_emis = emis_f.argmax(dim=-1)                # emission-only
        vit_correct += (pred_vit == tags).sum().item()
        emis_correct += (pred_emis == tags).sum().item()
        n += tags.numel()
        for t in TOLS:
            for k, v in breakpoint_counts(pred_vit, tags, tol=t).items():
                bp[t][k] += v
    return {"vit_acc": vit_correct / n, "emis_acc": emis_correct / n, "n": n, "bp": bp}


@torch.no_grad()
def evaluate_diploid(model, ds, device, batch_size, num_workers):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    pair_correct = hap_correct = total = 0
    homo_pred = 0
    for batch in loader:
        X = batch["input_embeds"].to(device)
        h1 = batch["h1"].to(device)
        h2 = batch["h2"].to(device)
        emis_p, _, c = model(X)
        pred = _dcrf_viterbi(emis_p, c, model.nsw_pair, model.stay_bonus)
        pair_true = model.pair_table[h1, h2]
        pair_correct += (pred == pair_true).sum().item()
        pred_lo, pred_hi = model.pi[pred], model.pj[pred]
        t_lo = torch.minimum(h1, h2)
        t_hi = torch.maximum(h1, h2)
        hap_correct += ((pred_lo == t_lo).sum() + (pred_hi == t_hi).sum()).item()
        homo_pred += (pred_lo == pred_hi).sum().item()
        total += pair_true.numel()
    return {"pair_acc": pair_correct / total, "hap_acc": hap_correct / (2 * total),
            "homo_pred": homo_pred / total, "n": total}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = args.mode if args.mode != "auto" else detect_mode(args.ckpt)
    tag = args.tag or Path(args.ckpt).parent.name

    if mode == "haploid":
        model = GRITSCRFHaploid.load_from_checkpoint(args.ckpt, map_location=device)
        model.eval().to(device)
        needs_affinity = getattr(model, "ext_dim", 0) > 0

        fixed_ext_emb = None
        if needs_affinity and args.windows_per_individual:
            _, val_ds, test_ds = make_individual_splits(
                args.data, args.num_parents, args.val_frac, args.test_frac,
                args.windows_per_individual, limit_n=args.limit_n)
            ds = test_ds if args.split == "test" else val_ds
        else:
            _, val_ds, test_ds = make_splits(
                args.data, args.num_parents, args.val_frac, args.test_frac,
                limit_n=args.limit_n)
            ds = test_ds if args.split == "test" else val_ds
            if needs_affinity:
                data_full = np.load(args.data, allow_pickle=True, mmap_mode="r")
                feats_full = np.asarray(data_full[:, :, :args.num_parents])
                fixed_ext_emb = torch.tensor(
                    _founder_affinity(feats_full), dtype=torch.float32)
                print(f"  ext_dim={model.ext_dim}: computed one genome-wide affinity "
                      f"vector from the full file ({data_full.shape[0]:,} windows) — "
                      f"single real-individual mode (--windows-per-individual not given)")

        r = evaluate_haploid(model, ds, device, args.batch_size, args.num_workers,
                             fixed_ext_emb=fixed_ext_emb)
        print(f"\n[{tag}] mode=haploid split={args.split}  N_sites={r['n']:,}  "
              f"affinity={'on' if needs_affinity else 'off'}")
        print(f"  Viterbi (full decode) acc : {r['vit_acc']:.4f}")
        print(f"  emission-only argmax  acc : {r['emis_acc']:.4f}")
        print(f"  Viterbi - emission        : {r['vit_acc'] - r['emis_acc']:+.4f}")
        print(f"  no_transition arm         : {model.no_transition}")
        bp_str = {}
        for t in TOLS:
            p, rec, f1 = prf(r["bp"][t])
            bp_str[t] = (p, rec, f1)
            print(f"  breakpoint P/R/F1 (±{t})  : {p:.3f} / {rec:.3f} / {f1:.3f}  "
                  f"(pred={r['bp'][t]['n_pred']:,} true={r['bp'][t]['n_true']:,})")

        res_dir = Path(args.workdir) / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        bp_fields = "\t".join(
            f"bpP{t}={bp_str[t][0]:.3f}\tbpR{t}={bp_str[t][1]:.3f}\t"
            f"bpF{t}={bp_str[t][2]:.3f}" for t in TOLS)
        line = (f"{tag}\tmode=haploid\tsplit={args.split}\taffinity={needs_affinity}\t"
                f"N={r['n']}\t"
                f"viterbi={r['vit_acc']:.4f}\temis_only={r['emis_acc']:.4f}\t"
                f"{bp_fields}\tckpt={args.ckpt}\n")

    elif mode == "diploid":
        model = GRITSCRFDiploid.load_from_checkpoint(args.ckpt, map_location=device)
        model.eval().to(device)
        _, val_ds, test_ds = make_diploid_splits(
            args.data, args.num_parents, args.val_frac, args.test_frac,
            limit_n=args.limit_n)
        ds = test_ds if args.split == "test" else val_ds

        r = evaluate_diploid(model, ds, device, args.batch_size, args.num_workers)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[{tag}] mode=diploid split={args.split}  N_sites={r['n']:,}  "
              f"params={n_params:,}  homo_penalty={model.homo_penalty}")
        print(f"  pair_acc = {r['pair_acc']:.4f}")
        print(f"  hap_acc  = {r['hap_acc']:.4f}")
        print(f"  predicted-homozygous fraction = {r['homo_pred']:.4f}")

        res_dir = Path(args.workdir) / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        line = (f"{tag}\tmode=diploid\tsplit={args.split}\tN={r['n']}\t"
                f"params={n_params}\tpair_acc={r['pair_acc']:.4f}\t"
                f"hap_acc={r['hap_acc']:.4f}\thomo_pred={r['homo_pred']:.4f}\t"
                f"ckpt={args.ckpt}\n")

    else:
        raise ValueError(f"Unknown mode: {mode}")

    res_path = Path(args.workdir) / "results" / "eval.tsv"
    with open(res_path, "a") as f:
        f.write(line)
    print(f"  appended → {res_path}")


if __name__ == "__main__":
    main()
