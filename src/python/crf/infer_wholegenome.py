"""
Whole-chromosome diploid inference — edge-free decoding at genome scale.

The encoder is O(T^2) and capped at ~1024-site context, but the pair-state CRF
decode is LINEAR in length, so it can span a whole chromosome. Tiling a 100k-site
chromosome into independent 1024 windows leaves a decode discontinuity at every
boundary (the state at a tile edge has no neighbouring context). This module fixes
that by running the encoder on OVERLAPPED windows, stitching the per-site emissions
into a full-chromosome track, and running ONE CRF decode over the entire chromosome.

Three modes (set `--mode`):
  tiled         independent non-overlap 1024 windows, decoded separately (baseline;
                edge errors at each boundary).
  overlap-stitch overlapped encoder windows, decoded separately, paths center-cropped
                and stitched (encoder overlap, but still per-window decode).
  whole-chrom   overlapped encoder emissions stitched into one [L,P] track, decoded
                ONCE per chromosome (the edge-free target).

Reuses GRITSCRFDiploid.forward, _dcrf_viterbi/_dcrf_marginal, build_pair_tables,
_founder_affinity (affinity model). Per-chromosome decode (no inter-chromosome
transition). Reports per-class pair/hap and boundary-site accuracy (the edge metric).

Usage:
    LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
      .pixi/envs/gpu/bin/python src/python/crf/infer_wholegenome.py \
        --ckpt <affinity.ckpt> --glob '/workdir/esb33/data/held-out/wg_*_sparse.npy' \
        --mode whole-chrom --decode viterbi --stride 512 --founder-affinity
"""

import argparse
import glob as globmod
from pathlib import Path

import numpy as np
import torch

from python.crf.train_diploid import (GRITSCRFDiploid, _dcrf_viterbi,
                                       _dcrf_marginal, _dcrf_viterbi_factored,
                                       _founder_affinity)

DECODERS = {"viterbi": _dcrf_viterbi, "marginal": _dcrf_marginal}


def _t(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    import time
    return time.perf_counter()


def _starts(L, win, stride):
    s = list(range(0, max(1, L - win + 1), stride))
    if s[-1] != L - win:
        s.append(L - win)
    return s


@torch.no_grad()
def _encode(model, feats, starts, win, ext_emb, device, bs):
    """feats [L,K] -> per-window emis_p [nwin,win,P], c [nwin,win]."""
    Xs = torch.stack([feats[s:s + win] for s in starts])          # [nwin,win,K]
    em, cc = [], []
    for i in range(0, len(starts), bs):
        Xb = Xs[i:i + bs].to(device)
        eb = ext_emb.expand(Xb.shape[0], -1, -1) if ext_emb is not None else None
        dev_type = torch.device(device).type
        with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=dev_type == "cuda"):
            emis_p, _g, c = model(Xb, None, eb)
        em.append(emis_p.float().cpu())
        cc.append(c.float().cpu())
    return torch.cat(em), torch.cat(cc)


def _ownership(starts, L, win, stride):
    """Center-crop boundaries: window i owns output [b_prev, b_i)."""
    half = (stride + win) // 2
    bounds, b_prev = [], 0
    for i, s in enumerate(starts):
        b = L if i == len(starts) - 1 else s + half
        bounds.append((b_prev, b, s))                              # out lo, out hi, win start
        b_prev = b
    return bounds


def affinity_keep(rate, min_gap=0.03, margin=0.02):
    """Sorted-rate largest-gap split: founders above the carried/background step.
    Returns kept founder indices (np). No clear step (gap<min_gap) -> keep all."""
    order = np.argsort(rate)[::-1]
    sr = rate[order]
    gaps = sr[:-1] - sr[1:]
    cut = int(np.argmax(gaps))
    if gaps[cut] < min_gap:                       # flat -> all founders carried
        return np.arange(len(rate))
    return np.where(rate >= sr[cut + 1] + margin)[0]


def decode_chrom(model, feats, ext_emb, mode, decode, win, stride, device, bs,
                 variants=None):
    """feats [L,K] -> {name: (pred [L] orig-pair idx, decode_seconds)}. The encoder
    runs ONCE; each variant (name -> sel_pairs LongTensor or None=full) is decoded
    from the same stitched emissions, so comparing full vs pruned costs one encode."""
    if variants is None:
        variants = {"full": None}
    L = feats.shape[0]
    stay = model.stay_bonus
    eff_stride = win if mode == "tiled" else stride
    starts = _starts(L, win, eff_stride)
    emis_w, c_w = _encode(model, feats, starts, win, ext_emb, device, bs)   # [nwin,win,*]
    bounds = _ownership(starts, L, win, eff_stride)

    def _decoder_for(sel_pairs):
        dec = decode
        if sel_pairs is not None and dec == "viterbi-factored":
            dec = "viterbi"                          # factored needs full pi/pj; tiny P here
        if dec == "viterbi-factored":
            return lambda e, cc, n, s: _dcrf_viterbi_factored(e, cc, n, s, model.pi, model.pj)
        return DECODERS[dec]

    def _decode(emis_b, c_b, sel_pairs, decoder, nsw):   # [1,len,P] -> [len], seconds
        e = emis_b[..., sel_pairs] if sel_pairs is not None else emis_b
        t0 = _t(device)
        p = decoder(e.to(device), c_b.to(device), nsw, stay)[0].cpu()
        dt = _t(device) - t0
        if sel_pairs is not None:
            p = sel_pairs.cpu()[p]                   # remap reduced -> original pair idx
        return p, dt

    if mode == "whole-chrom":                        # stitch once, decode each variant
        P = emis_w.shape[-1]
        emis_full = torch.empty(L, P); c_full = torch.empty(L)
        for (lo, hi, s), ew, cw in zip(bounds, emis_w, c_w):
            emis_full[lo:hi] = ew[lo - s:hi - s]
            c_full[lo:hi] = cw[lo - s:hi - s]
        eb, cb = emis_full.unsqueeze(0), c_full.unsqueeze(0)
        out = {}
        for name, sp in variants.items():
            nsw = model.nsw_pair[sp][:, sp] if sp is not None else model.nsw_pair
            out[name] = _decode(eb, cb, sp, _decoder_for(sp), nsw)
        return out

    # tiled / overlap-stitch: decode each window, center-crop stitch the PATHS
    out = {name: (torch.empty(L, dtype=torch.long), 0.0) for name in variants}
    for (lo, hi, s), ew, cw in zip(bounds, emis_w, c_w):
        for name, sp in variants.items():
            nsw = model.nsw_pair[sp][:, sp] if sp is not None else model.nsw_pair
            p, d = _decode(ew.unsqueeze(0), cw.unsqueeze(0), sp, _decoder_for(sp), nsw)
            pred, tot = out[name]
            pred[lo:hi] = p[lo - s:hi - s]; out[name] = (pred, tot + d)
    return out


def acc(model, pred, h1, h2):
    """pred [L] pair idx; h1,h2 [L]. Returns (pair_correct[L] bool, hap_half[L])."""
    pair_true = model.pair_table.cpu()[h1, h2]
    pc = (pred == pair_true)
    lo, hi = model.pi.cpu()[pred], model.pj.cpu()[pred]
    tlo, thi = torch.minimum(h1, h2), torch.maximum(h1, h2)
    hh = ((lo == tlo).float() + (hi == thi).float()) / 2
    return pc, hh


def masked_snp(model, pred, feats_held, mask_idx, K):
    """Label-free held-out SNP accuracy. At each masked site the read was hidden from
    the encoder; the imputation is correct if EITHER decoded founder reproduces the
    held-out read (feats_held[t,a] OR feats_held[t,b]) — single read, unknown gamete,
    so either-match. Ceiling = sites where ANY founder matches the read (matchable at
    all). Needs no true-founder labels, so it transfers to real data. Returns
    (n_correct, n_matchable, n_masked)."""
    if mask_idx.numel() == 0:
        return 0.0, 0.0, 0
    a = model.pi.cpu()[pred][mask_idx]
    b = model.pj.cpu()[pred][mask_idx]
    fh = feats_held[mask_idx]                          # [m,K] held-out match rows
    m = torch.arange(fh.shape[0])

    def col(fidx):                                     # gather, null founder (>=K) -> 0
        return fh[m, fidx.clamp(max=K - 1)] * (fidx < K).float()

    either = ((col(a) + col(b)) > 0).float()
    matchable = (fh > 0).any(-1).float()
    return either.sum().item(), matchable.sum().item(), int(fh.shape[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--glob", required=True, help="WG individual .npy files")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--mode", choices=["tiled", "overlap-stitch", "whole-chrom"],
                   default="whole-chrom")
    p.add_argument("--decode", choices=["viterbi", "marginal", "viterbi-factored"],
                   default="viterbi")
    p.add_argument("--win", type=int, default=1024)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--founder-affinity", action="store_true")
    p.add_argument("--prune-affinity", action="store_true",
                   help="affinity largest-gap select carried founders -> decode the "
                        "reduced pair-state subset (the smaller-CRF speed lever)")
    p.add_argument("--prune-min-gap", type=float, default=0.03)
    p.add_argument("--prune-margin", type=float, default=0.02)
    p.add_argument("--compare", action="store_true",
                   help="decode BOTH full (P=325) and affinity-pruned from the same "
                        "encode and report them side by side with the delta")
    p.add_argument("--mask-frac", type=float, default=0.0,
                   help="hold out this fraction of WHOLE sites (zero the K-row before "
                        "encode), then score label-free either-match SNP accuracy at the "
                        "masked sites — the imputation metric that transfers to real data")
    p.add_argument("--mask-seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--boundary-d", type=int, default=16,
                   help="report accuracy within +/-d of tile (win) boundaries")
    p.add_argument("--workdir", default="/workdir/esb33")
    args = p.parse_args()
    device = torch.device("cuda")
    model = GRITSCRFDiploid.load_from_checkpoint(args.ckpt, map_location=device).eval().to(device)
    K = args.num_parents

    files = sorted(globmod.glob(args.glob))
    if not files:
        raise SystemExit(f"no files match {args.glob}")
    # which decode variants to run: compare = both; else single (pruned or full)
    if args.compare:
        vnames = ["full", "pruned"]
    elif args.prune_affinity:
        vnames = ["pruned"]
    else:
        vnames = ["full"]
    use_aff = args.founder_affinity or args.prune_affinity or args.compare
    print(f"\n[whole-genome infer] mode={args.mode} decode={args.decode} "
          f"win={args.win} stride={args.stride} affinity={use_aff} variants={vnames}")
    if args.compare:
        hdr = (f"{'individual':30s} {'full_pair':>9} {'prn_pair':>9} {'Δpair':>7} "
               f"{'full_hap':>9} {'prn_hap':>9} {'P_red':>6} {'full_ms':>8} {'prn_ms':>8}")
        if args.mask_frac > 0:
            hdr += f" {'f_msnp':>7} {'p_msnp':>7}"
    else:
        hdr = (f"{'individual':32s} {'pair':>7} {'hap':>7} {'bnd_pair':>9} {'pos':>10} "
               f"{'P_red':>6} {'dec_ms':>7}")
        if args.mask_frac > 0:
            hdr += f" {'msnp':>7} {'msnp/ceil':>9}"
    print(hdr); print("-" * len(hdr))

    pi_c, pj_c = model.pi.cpu().numpy(), model.pj.cpu().numpy()
    agg = {v: dict(pc=0.0, hh=0.0, n=0.0, bpc=0.0, bn=0.0, dt=0.0, sc=0.0, scn=0.0, sm=0.0)
           for v in vnames}
    rng_mask = np.random.default_rng(args.mask_seed)
    nchrom = 0
    rows = []
    for f in files:
        arr = np.load(f)                                          # [NC, L, K+2]
        feats_all = torch.tensor(arr[:, :, :K], dtype=torch.float32)
        h1_all = torch.tensor(np.clip(arr[:, :, K], 0, K).astype(np.int64))
        h2_all = torch.tensor(np.clip(arr[:, :, K + 1], 0, K).astype(np.int64))
        ext = None
        sel_pairs = None
        if use_aff:
            aff = _founder_affinity(arr[:, :, :K].astype(np.float32))   # [K,2] over whole indiv
            ext = torch.tensor(aff, dtype=torch.float32, device=device).unsqueeze(0)
        if "pruned" in vnames:
            sel = affinity_keep(aff[:, 0], args.prune_min_gap, args.prune_margin)
            keep = set(int(i) for i in sel) | {K}    # always retain the null founder
            sp = [pp for pp in range(model.P) if pi_c[pp] in keep and pj_c[pp] in keep]
            sel_pairs = torch.tensor(sp, dtype=torch.long)
        P_red = len(sel_pairs) if sel_pairs is not None else model.P
        variants = {v: (sel_pairs if v == "pruned" else None) for v in vnames}

        iv = {v: dict(pc=0.0, hh=0.0, n=0.0, bpc=0.0, bn=0.0, dt=0.0, sc=0.0, scn=0.0, sm=0.0)
              for v in vnames}
        for ch in range(feats_all.shape[0]):
            feats = feats_all[ch]; L = feats.shape[0]
            mask_idx = torch.empty(0, dtype=torch.long)
            feats_in = feats
            if args.mask_frac > 0:                       # hold out whole sites
                nmask = max(1, int(round(args.mask_frac * L)))
                mi = rng_mask.choice(L, size=nmask, replace=False)
                mask_idx = torch.tensor(np.sort(mi), dtype=torch.long)
                feats_in = feats.clone()
                feats_in[mask_idx] = 0.0                 # encoder no longer sees the read
            preds = decode_chrom(model, feats_in, ext, args.mode, args.decode,
                                 args.win, args.stride, device, args.batch_size,
                                 variants=variants)
            t = torch.arange(L)
            bnd = ((t % args.win) <= args.boundary_d) | ((t % args.win) >= args.win - args.boundary_d)
            for v, (pred, dt) in preds.items():
                pc, hh = acc(model, pred, h1_all[ch], h2_all[ch])
                sc, scn, sm = masked_snp(model, pred, feats, mask_idx, K)
                d = iv[v]
                d["pc"] += pc.sum().item(); d["hh"] += hh.sum().item(); d["n"] += L
                d["dt"] += dt; d["bpc"] += pc[bnd].sum().item(); d["bn"] += int(bnd.sum())
                d["sc"] += sc; d["scn"] += scn; d["sm"] += sm
        name = Path(f).stem
        nc = feats_all.shape[0]
        for v in vnames:                              # roll into global aggregate
            for k in agg[v]:
                agg[v][k] += iv[v][k]
        if args.compare:
            a, b = iv["full"], iv["pruned"]
            line = (f"{name:30s} {a['pc']/a['n']:>9.4f} {b['pc']/b['n']:>9.4f} "
                    f"{(b['pc']-a['pc'])/a['n']:>+7.4f} {a['hh']/a['n']:>9.4f} "
                    f"{b['hh']/b['n']:>9.4f} {P_red:>6d} {a['dt']/nc*1e3:>8.1f} {b['dt']/nc*1e3:>8.1f}")
            if args.mask_frac > 0:
                line += f" {a['sc']/a['sm']:>7.4f} {b['sc']/b['sm']:>7.4f}"
            print(line)
            rows.append(f"{name}=full {a['pc']/a['n']:.4f}/prn {b['pc']/b['n']:.4f}")
        else:
            d = iv[vnames[0]]
            line = (f"{name:32s} {d['pc']/d['n']:>7.4f} {d['hh']/d['n']:>7.4f} "
                    f"{d['bpc']/d['bn']:>9.4f} {int(d['n']):>10,} {P_red:>6d} {d['dt']/nc*1e3:>7.1f}")
            if args.mask_frac > 0:
                line += f" {d['sc']/d['sm']:>7.4f} {d['sc']/d['scn']:>9.4f}"
            print(line)
            rows.append(f"{name}={d['pc']/d['n']:.4f}/{d['hh']/d['n']:.4f}")
        nchrom += nc

    print("-" * len(hdr))
    if args.compare:
        a, b = agg["full"], agg["pruned"]
        line = (f"{'ALL':30s} {a['pc']/a['n']:>9.4f} {b['pc']/b['n']:>9.4f} "
                f"{(b['pc']-a['pc'])/a['n']:>+7.4f} {a['hh']/a['n']:>9.4f} {b['hh']/b['n']:>9.4f} "
                f"{'':>6} {a['dt']/nchrom*1e3:>8.1f} {b['dt']/nchrom*1e3:>8.1f}")
        if args.mask_frac > 0:
            line += f" {a['sc']/a['sm']:>7.4f} {b['sc']/b['sm']:>7.4f}"
        print(line)
    else:
        d = agg[vnames[0]]
        line = (f"{'ALL':32s} {d['pc']/d['n']:>7.4f} {d['hh']/d['n']:>7.4f} {d['bpc']/d['bn']:>9.4f} "
                f"{int(d['n']):>10,} {'':>6} {d['dt']/nchrom*1e3:>7.1f}")
        if args.mask_frac > 0:
            line += f" {d['sc']/d['sm']:>7.4f} {d['sc']/d['scn']:>9.4f}"
        print(line)
    out = Path(args.workdir) / "results" / "wg_infer.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    tag = "compare" if args.compare else ("prune" if args.prune_affinity else "full")
    with open(out, "a") as fh:
        fh.write(f"\n[{Path(args.ckpt).parent.parent.name} mode={args.mode} "
                 f"decode={args.decode} {tag} stride={args.stride}] {Path(args.glob).name}: "
                 + " | ".join(rows))
        for v in vnames:
            d = agg[v]
            fh.write(f" | ALL[{v}]={d['pc']/d['n']:.4f}/{d['hh']/d['n']:.4f} "
                     f"dec_ms/chrom={d['dt']/nchrom*1e3:.1f}")
            if args.mask_frac > 0:
                fh.write(f" msnp[{v}]={d['sc']/d['sm']:.4f} "
                         f"(/ceil {d['sc']/d['scn']:.4f}, mask={args.mask_frac})")
        fh.write("\n")
    print(f"\nappended → {out}")


if __name__ == "__main__":
    main()
