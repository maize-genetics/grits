#!/usr/bin/env python
"""
Read-limited reference decoder: an UNTRAINED HMM over the same observed read
rows the model sees (windowed_k25native_wcount.npy ternary block), decoded
along whole chromosomes, scored with fast_snprc_score. Answers "how low can
genotype error go from these reads alone with a trivially simple model?" --
a read-limited counterpart to panel_ceiling.py's truth-limited ceiling.

Emission per row, founder f: log(1-e) if the row MATCHes f, log(e) if f is
diverged or deleted there. Diploid (pair) mode: a row comes from either
haplotype, p = 0.5 P(row|i) + 0.5 P(row|j) over all 325 unordered founder
pairs; switching one founder costs lam, both 2*lam. Haploid mode (inbred /
RIL2): one founder path, predicted pair (f, f). --betas adds a per-row
log-odds bonus to homozygous pairs in pair mode (a soft homozygous prior:
a heterozygous stretch must be supported by enough reads to beat it), so one
pair decoder can serve every sample kind without a classifier.

lam / e are swept and the best is reported per class -- tuned on the test
samples, so this is a diagnostic ceiling-style number, not a model.
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS))
import simval_eval_indel as sei  # noqa: E402
sys.path.insert(0, sei.CRF_SRC)
import fast_snprc_score as fss  # noqa: E402
import heldout_assembly_eval as hae  # noqa: E402

K = 25
T = 512
WORK = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
MANIFEST = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv"
PI, PJ = np.triu_indices(K)                      # 325 unordered pairs incl. homozygous


@njit(cache=True)
def _haploid(ll, lam):
    n, Kf = ll.shape
    cost = np.zeros(Kf)
    stay = np.empty((n, Kf), np.bool_)
    bmin = np.empty(n, np.int16)
    for t in range(n):
        b = 0
        for k in range(1, Kf):
            if cost[k] > cost[b]:
                b = k
        bmin[t] = b
        sw = cost[b] - lam
        for k in range(Kf):
            if cost[k] >= sw:
                stay[t, k] = True
                c = cost[k]
            else:
                stay[t, k] = False
                c = sw
            cost[k] = c + ll[t, k]
    k = 0
    for j in range(1, Kf):
        if cost[j] > cost[k]:
            k = j
    path = np.empty(n, np.int16)
    for t in range(n - 1, -1, -1):
        path[t] = k
        if not stay[t, k]:
            k = bmin[t]
    return path


@njit(cache=True)
def _pair(ll, lam, pi, pj, beta):
    n, Kf = ll.shape
    P = pi.shape[0]
    cost = np.zeros(P)
    back = np.empty((n, P), np.int16)            # previous pair index
    m1 = np.empty(Kf)
    a1 = np.empty(Kf, np.int64)
    em = np.empty(P)
    for t in range(n):
        for f in range(Kf):
            m1[f] = -1e300
        g = 0
        for p in range(P):
            if cost[p] > cost[g]:
                g = p
            for f in (pi[p], pj[p]):
                if cost[p] > m1[f]:
                    m1[f] = cost[p]
                    a1[f] = p
        for p in range(P):
            i = pi[p]; j = pj[p]
            best = cost[p]; bp = p
            if m1[i] - lam > best:
                best = m1[i] - lam; bp = a1[i]
            if m1[j] - lam > best:
                best = m1[j] - lam; bp = a1[j]
            if cost[g] - 2 * lam > best:
                best = cost[g] - 2 * lam; bp = g
            a = ll[t, i]; b = ll[t, j]
            mx = a if a > b else b
            em[p] = mx + np.log(0.5 * np.exp(a - mx) + 0.5 * np.exp(b - mx))
            if i == j:
                em[p] += beta
            back[t, p] = bp
            cost[p] = best
        for p in range(P):
            cost[p] += em[p]
    p = 0
    for q in range(1, P):
        if cost[q] > cost[p]:
            p = q
    path = np.empty(n, np.int16)
    for t in range(n - 1, -1, -1):
        path[t] = p
        p = back[t, p]
    return path


def decode(src, diploid, lam, e, beta=0.0):
    x = np.load(src / "windowed_k25native_wcount.npy")
    N = x.shape[0]
    tern = x[..., :K].reshape(-1, K)
    ll = np.where(tern == 1, np.log(1 - e), np.log(e)).astype(np.float64)
    layout = hae.load_contig_layout(src / "raw.npy.bins.tsv", T, bin_size=256)
    lo = np.empty(N * T, np.int64)
    hi = np.empty(N * T, np.int64)
    s = 0
    for _c, _pos, nw in layout:
        rows = slice(s, s + nw * T)
        if diploid:
            p = _pair(ll[rows], lam, PI, PJ, beta)
            lo[rows], hi[rows] = PI[p], PJ[p]
        else:
            p = _haploid(ll[rows], lam)
            lo[rows] = hi[rows] = p
        s += nw * T
    return lo.reshape(N, T), hi.reshape(N, T), layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["OUT-INBRED", "OUT-HYB", "OUT-RIL2"])
    ap.add_argument("--depth", default="0.1")
    ap.add_argument("--lams", nargs="+", type=float, default=[2, 5, 10, 20, 40])
    ap.add_argument("--errs", nargs="+", type=float, default=[0.05])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.0],
                    help="per-row log-odds bonus for homozygous pairs (i,i) in pair mode")
    ap.add_argument("--pair-all", action="store_true",
                    help="325-pair decoding for every sample (no knowledge of the true kind)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    panel = fss.Panel()
    out = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    for r in csv.DictReader(open(MANIFEST), delimiter="\t"):
        ds, ind = r["dataset_id"], r["individual"]
        if r["coverage"] != args.depth or ds not in args.classes:
            continue
        cands = [Path(d) for d in sorted(glob.glob(str(WORK / f"scratch/*_snprc/*__{ds}__{ind}__{args.depth}x")))
                 if (Path(d) / "windowed_k25native_wcount.npy").exists()]
        if not cands:
            print(f"SKIP {ds} {ind}: no model input", flush=True)
            continue
        src = cands[0]
        diploid = args.pair_all or ds.endswith("-HYB") or ds.endswith("-RIL")
        t1, t2 = fss.load_truth(r["truth_h1"]), fss.load_truth(r["truth_h2"])
        gam = hae.load_gamete_names(src / "raw.npy.gametes.tsv")
        for e, lam, beta in [(e, l, b) for e in args.errs for l in args.lams for b in args.betas]:
            if True:
                key = (f"{ds}__{ind}__{args.depth}x__e{e:g}__lam{lam:g}" + ("__pairall" if args.pair_all else "")
                       + (f"__beta{beta:g}" if beta else ""))
                if key in out:
                    continue
                lo, hi, _ = decode(src, diploid, lam, e, beta)
                bed = WORK / "scratch/read_hmm_ceiling" / key / "bed"
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()):
                    hae.write_imputed_bed(ind, lo, hi, None, gam, src / "raw.npy.bins.tsv", bed, bin_size=256)
                f1, f2 = fss.pair_from_bed(panel, bed)
                res = fss.score_arrays(panel, t1, t2, f1, f2)
                sw = int(((np.diff(lo.reshape(-1)) != 0) | (np.diff(hi.reshape(-1)) != 0)).sum())
                res.update(pair_changes=sw, diploid=diploid, beta=beta)
                out[key] = res
                Path(args.out).write_text(json.dumps(out, indent=1))
                print(f"{ds:<11} {ind:<14} e={e:g} lam={lam:<4g} beta={beta:<5g} err={100 * res['error_rate']:6.2f}%  "
                      f"snprc={100 * res['snprc_error_rate']:6.2f}%  pair_changes={sw:,}", flush=True)


if __name__ == "__main__":
    main()
