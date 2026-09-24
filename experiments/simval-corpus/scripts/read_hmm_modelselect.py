#!/usr/bin/env python
"""
Cutoff-free choice between a homozygous (haploid-path) and a diploid
(two-path) read HMM per sample, by comparing their forward log-likelihoods on
the sample's own reads -- Bayesian model comparison, no tuned threshold.

Both models are properly normalized (unlike read_hmm_ceiling.py's Viterbi
penalties): per row a haplotype keeps its founder with prob 1-r or moves to
one of the other K-1 founders uniformly. Homozygous model: one path, emission
P(row|f). Diploid model: two independent ordered paths (K*K states), emission
0.5 P(row|i) + 0.5 P(row|j). The diploid model contains homozygous stretches
but pays an Occam cost for them (625 states, and staying homozygous across a
switch needs both paths to switch together), so an inbred's reads favour the
simpler model while a hybrid's reads -- two distinct haplotypes -- can only be
explained well by the diploid one.

After selection the sample is decoded with read_hmm_ceiling.decode in the
chosen mode (haploid or 325-pair) at --lam, and scored exactly.
"""
import argparse
import contextlib
import csv
import glob
import io
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).parent))
import read_hmm_ceiling as rh  # noqa: E402

fss, hae = rh.fss, rh.hae
K, T = rh.K, rh.T


@njit(cache=True)
def _loglik_haploid(em, r):
    n, Kf = em.shape
    a = np.full(Kf, 1.0 / Kf)
    ll = 0.0
    q = r / (Kf - 1)
    for t in range(n):
        s = a.sum()
        for k in range(Kf):
            a[k] = ((1 - r) * a[k] + q * (s - a[k])) * em[t, k]
        z = a.sum()
        ll += np.log(z)
        a /= z
    return ll


@njit(cache=True)
def _loglik_diploid(em, r):
    n, Kf = em.shape
    a = np.full((Kf, Kf), 1.0 / (Kf * Kf))
    ll = 0.0
    q = r / (Kf - 1)
    for t in range(n):
        for j in range(Kf):                      # transition on haplotype 1 (axis 0)
            s = 0.0
            for i in range(Kf):
                s += a[i, j]
            for i in range(Kf):
                a[i, j] = (1 - r) * a[i, j] + q * (s - a[i, j])
        for i in range(Kf):                      # transition on haplotype 2 (axis 1)
            s = 0.0
            for j in range(Kf):
                s += a[i, j]
            for j in range(Kf):
                a[i, j] = (1 - r) * a[i, j] + q * (s - a[i, j])
        z = 0.0
        for i in range(Kf):
            for j in range(Kf):
                a[i, j] *= 0.5 * em[t, i] + 0.5 * em[t, j]
                z += a[i, j]
        ll += np.log(z)
        a /= z
    return ll


def loglik_pair(src, r, e):
    x = np.load(src / "windowed_k25native_wcount.npy")
    tern = x[..., :K].reshape(-1, K)
    em = np.where(tern == 1, 1 - e, e).astype(np.float64)
    layout = hae.load_contig_layout(src / "raw.npy.bins.tsv", T, bin_size=256)
    lh = ld = 0.0
    s = 0
    for _c, _p, nw in layout:
        rows = slice(s, s + nw * T)
        lh += _loglik_haploid(em[rows], r)
        ld += _loglik_diploid(em[rows], r)
        s += nw * T
    return lh, ld


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--depth", default="0.1")
    ap.add_argument("--rs", nargs="+", type=float, default=[1e-3, 1e-4, 1e-5])
    ap.add_argument("--e", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=20)
    ap.add_argument("--decode-r", type=float, default=1e-4,
                    help="which r's decision drives the decode+score")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    panel = fss.Panel()
    out = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    for r in csv.DictReader(open(rh.MANIFEST), delimiter="\t"):
        ds, ind = r["dataset_id"], r["individual"]
        if r["coverage"] != args.depth or ds not in args.classes:
            continue
        key = f"{ds}__{ind}__{args.depth}x"
        if key in out:
            continue
        cands = [Path(d) for d in sorted(glob.glob(str(rh.WORK / f"scratch/*_snprc/*__{ds}__{ind}__{args.depth}x")))
                 if (Path(d) / "windowed_k25native_wcount.npy").exists()]
        if not cands:
            print(f"SKIP {key}", flush=True)
            continue
        src = cands[0]
        rec = {}
        for rr in args.rs:
            lh, ld = loglik_pair(src, rr, args.e)
            rec[f"r{rr:g}"] = {"ll_haploid": lh, "ll_diploid": ld, "delta": ld - lh,
                               "choice": "diploid" if ld > lh else "haploid"}
        diploid = rec[f"r{args.decode_r:g}"]["choice"] == "diploid"
        lo, hi, _ = rh.decode(src, diploid, args.lam, args.e)
        bed = rh.WORK / "scratch/read_hmm_modelselect" / key / "bed"
        with contextlib.redirect_stdout(io.StringIO()):
            hae.write_imputed_bed(ind, lo, hi, None, hae.load_gamete_names(src / "raw.npy.gametes.tsv"),
                                  src / "raw.npy.bins.tsv", bed, bin_size=256)
        f1, f2 = fss.pair_from_bed(panel, bed)
        res = fss.score_arrays(panel, fss.load_truth(r["truth_h1"]), fss.load_truth(r["truth_h2"]), f1, f2)
        rec.update(error_rate=res["error_rate"], snprc_error_rate=res["snprc_error_rate"],
                   decoded_as="diploid" if diploid else "haploid")
        out[key] = rec
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"{ds:<11} {ind:<14} " + "  ".join(f"r={k[1:]}: d={v['delta']:+10.1f} {v['choice'][:3]}"
                                               for k, v in rec.items() if k.startswith("r"))
              + f"  -> {rec['decoded_as']:<8} err={100 * res['error_rate']:6.2f}%", flush=True)


if __name__ == "__main__":
    main()
