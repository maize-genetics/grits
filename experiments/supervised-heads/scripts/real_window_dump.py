"""Real-read windows for real_window_plot.py: ternary input, every supervised head, the CRF decode
under each --het-prior mode, and truth from the sample's gVCFs (no labels.bed: its founder
labels are placeholders).

Truth per row (sites within +-FLANK bp of the row's bin, panel records with a usable truth call):
  compat_h1/h2 [T,K]  fraction of sites where founder f carries truth haplotype 1/2's allele
  err_<mode>   [T]    SNP+RefCall genotype mismatch rate of the decoded pair (as fast_snprc_score:
                      unordered allele pair, sites where truth and imputed classes are REF/SNP)
Windows: per sample, the window with the most SNP+RC errors under --select-mode (default row),
unless --windows given.
Evaluation data is only decoded and scored here, nothing is fitted."""
import argparse, csv, sys
from pathlib import Path
import numpy as np, torch
# model code first: heldout_assembly_eval puts another checkout's `python` package on sys.path
from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, _founder_affinity, pooled_affinity

SC = Path(__file__).resolve().parents[2] / "simval-corpus/scripts"
sys.path.insert(0, str(SC))
import heldout_assembly_eval as hae  # noqa: E402
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/grits_workdir/regionfix-out-snprc-wt/experiments/simval-corpus/scripts")
import fast_snprc_score as fss  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--samples", nargs="+", required=True, help="DATASET__INDIVIDUAL, e.g. IDX-INBRED__B73")
ap.add_argument("--windows", nargs="*", type=int, default=None, help="one window index per sample")
ap.add_argument("--depth", default="0.1")
ap.add_argument("--flank", type=int, default=2500)
ap.add_argument("--select-mode", default="row", choices=["row", "segment", "off"],
                help="pick the window with the most SNP+RC errors under this decode")
ap.add_argument("--out", required=True)
a = ap.parse_args()
K = 25
ROOT = "/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/lift_s200_local20k"
MAN = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv"
man = {(r["dataset_id"], r["individual"]): r for r in csv.DictReader(open(MAN), delimiter="\t")
       if r["coverage"] == a.depth}
MODES = ("row", "segment", "off")
panel = fss.Panel()
fa_all = panel.fa
acls = panel.acls
m = GRITSCRFDiploidIndel.load_from_checkpoint(a.ckpt, map_location="cpu", strict=False).cuda().eval()


def run(feats, count, ext, ap_, sel):
    """heads + decode per mode for the windows in sel"""
    B = len(sel)
    out = {}
    for mode in MODES:
        m.het_prior = mode
        with torch.no_grad():
            emis, g, c = m(feats[sel].cuda(), ext_emb=ext[sel].cuda(), count=count[sel].cuda(),
                           aff_pred=ap_.unsqueeze(0).expand(B, -1))
            pred = m.crf_decode(emis, c)
        out[mode] = (m.pi[pred].cpu().numpy(), m.pj[pred].cpu().numpy())
        if mode == "row":
            H = {k: v.float() for k, v in m._heads.items()}
            out["gate"] = g.float().cpu().numpy()
            out["het"] = torch.sigmoid(H["het_logit"]).cpu().numpy()
            out["sw"] = torch.sigmoid(-H["c"]).cpu().numpy()
            out["lam"] = (m.xo_scale * H["xo_lam"]).detach().cpu().numpy()
    m.het_prior = "row"
    return out


def row_truth(chrom_i, pos, t1, t2, preds):
    lo, hi = panel.bounds[chrom_i]
    P = panel.pos[lo:hi]
    T = len(pos)
    comp = np.full((2, T, K), np.nan)
    err = {md: np.full(T, np.nan) for md in preds}
    for r in range(T):
        a_, b_ = lo + np.searchsorted(P, pos[r] - a.flank), lo + np.searchsorted(P, pos[r] + a.flank + 256)
        if b_ <= a_:
            continue
        s = np.arange(a_, b_)
        fa = np.asarray(fa_all[s])                                   # [S,K]
        c1, c2 = t1["code"][s], t2["code"][s]
        ok = (c1 >= 0) & (c2 >= 0)
        if not ok.any():
            continue
        comp[0, r] = (fa[ok] == c1[ok, None]).mean(0)
        comp[1, r] = (fa[ok] == c2[ok, None]).mean(0)
        ac = np.asarray(acls[s])
        tc = fss.classify_pair(ac[np.arange(len(s)), np.clip(c1, 0, None)], ac[np.arange(len(s)), np.clip(c2, 0, None)])
        for md, (p1, p2) in preds.items():
            f1, f2 = p1[r], p2[r]
            if f1 >= K or f2 >= K:
                continue
            i1, i2 = fa[:, f1], fa[:, f2]
            okp = ok & (i1 >= 0) & (i2 >= 0)
            ic = fss.classify_pair(ac[np.arange(len(s)), np.clip(i1, 0, None)], ac[np.arange(len(s)), np.clip(i2, 0, None)])
            sn = okp & (tc <= 1) & (ic <= 1)
            if sn.any():
                mt = (np.minimum(i1, i2) == np.minimum(c1, c2)) & (np.maximum(i1, i2) == np.maximum(c1, c2))
                err[md][r] = 1.0 - mt[sn].mean()
    return comp, err


res = {"founders": np.array(panel.founders)}
for si, key in enumerate(a.samples):
    ds, ind = key.split("__")
    d = Path(f"{ROOT}/eval__{ds}__{ind}__{a.depth}x")
    data = np.load(d / "windowed_k25native_wcount.npy")
    assert list(hae.load_gamete_names(d / "raw.npy.gametes.tsv")) == list(panel.founders), \
        "model founder columns must be in panel order (truth is indexed by panel founder)"
    N, T = data.shape[:2]
    lay = hae.load_contig_layout(d / "raw.npy.bins.tsv", T, bin_size=256)
    wchrom = np.concatenate([np.full(n, fss.AUTOSOMES.index(c) if c in fss.AUTOSOMES else -1) for c, _p, n in lay])
    wpos = np.concatenate([p for _c, p, _n in lay]).reshape(N, T)
    tern = data[:, :, :K].astype(np.float32); dist = data[:, :, K + 2:2 * K + 2].astype(np.float32)
    feats = torch.tensor(np.stack([tern, dist], -1)); count = torch.tensor(data[:, :, 2*K+2:3*K+2].astype(np.float32))
    ext = torch.tensor(_founder_affinity((tern == 1).astype(np.float32).reshape(-1, K))).unsqueeze(0).expand(N, -1, -1)
    ap_ = pooled_affinity(m, feats, count, ext, "cuda", 128)
    r = man[(ds, ind)]
    t1, t2 = fss.load_truth(r["truth_h1"]), fss.load_truth(r["truth_h2"])
    if a.windows:
        w = a.windows[si]
    else:                           # most SNP+RC errors among a spread of candidate windows
        cand = np.linspace(0, N - 1, 60).astype(int)
        cand = cand[wchrom[cand] >= 0]
        o = run(feats, count, ext, ap_, cand)
        best = (-1, None)
        for j, wi in enumerate(cand):
            sm = a.select_mode
            _c, e = row_truth(wchrom[wi], wpos[wi], t1, t2, {sm: (o[sm][0][j], o[sm][1][j])})
            v = np.nansum(e[sm])
            if v > best[0]:
                best = (v, wi)
        w = int(best[1])
    o = run(feats, count, ext, ap_, np.array([w]))
    comp, err = row_truth(wchrom[w], wpos[w], t1, t2, {md: (o[md][0][0], o[md][1][0]) for md in MODES})
    p = f"{si}/"
    res.update({p + "label": np.array(f"{ds} {ind}  window {w}  {fss.AUTOSOMES[wchrom[w]]}:"
                                      f"{wpos[w][0] / 1e6:.2f}-{wpos[w][-1] / 1e6:.2f} Mb"),
                p + "tern": data[w, :, :K].astype(np.int8), p + "pos": wpos[w],
                p + "gate": o["gate"][0], p + "het": o["het"][0], p + "sw": o["sw"][0], p + "lam": o["lam"][0],
                p + "aff": ap_.cpu().numpy(), p + "comp": comp})
    for md in MODES:
        res[p + f"p1_{md}"], res[p + f"p2_{md}"] = o[md][0][0], o[md][1][0]
        res[p + f"err_{md}"] = err[md]
    print(key, "window", w, {md: float(np.nanmean(err[md])) for md in MODES}, flush=True)
res["n"] = len(a.samples)
np.savez(a.out, **res)
