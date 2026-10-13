"""Hybrid-depth investigation helpers (branch hybrid-depth).

decode_custom() reproduces infer_real_founder_pairs for the supervised-heads v6 model
(aff_source=reads, het_prior=off, extra_heads, no_null) with --route --knockout homoroute,
and adds inference-only knobs:

  T            re-chunk each contig's rows into windows of T rows (encoder has sinusoidal
               positions + full attention, so any T runs); trailing rows get a last window
               that overlaps the previous one.
  keep         'all' | ('thin', k) keep every k-th row | 'dedup' drop a row whose ternary
               vector equals the previous kept row in the same bin | ('bin', n) keep at most
               n rows per refmap bin. Dropped rows take the call of the nearest kept row of
               the same contig (calls projected back, every row still gets a call).
  lam_bp       None, or a reference window span in bp: lambda_window *= span_bp / lam_bp
               (switches per bp instead of per window).
  lam_mult     constant multiplier on lambda (diagnostic).
  het          None | ('seg', h) | ('row', h): het prior with a fixed het rate h (pooled
               rate), paid once per segment or per row, weight het_w.
  ew           None | 'count': weight each row's emission by its read count (rows are
               collapsed (bin, match set) groups; count code -> reads).

Every option off reproduces infer_real_founder_pairs exactly (checked by verify_repro.py).
"""
import contextlib
import csv
import io
import math
import sys
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[3]
SCRIPTS = WT / "experiments/simval-corpus/scripts"
sys.path.insert(0, str(SCRIPTS))
import simval_eval_indel as sei  # noqa: E402
sys.path.insert(0, sei.CRF_SRC)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from python.crf.train_diploid_indel import (GRITSCRFDiploidIndel, infer_real_founder_pairs,  # noqa: E402
                                            route_real_sample, crf_viterbi_prior,
                                            _founder_affinity, TERN_MATCH)
import fast_snprc_score as fss  # noqa: E402
import heldout_assembly_eval as hae  # noqa: E402

WORK = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
CKPT = "/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/v6cfullfe-stage2B/last.ckpt"
MANIFESTS = {"eval": "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv",
             "calib": "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_calibration/manifest.tsv"}
ROOTS = {"eval": WORK / "scratch/lift_s200_local20k", "calib": WORK / "scratch/calib_s200_local20k"}
K = 25
T0 = 512
_PANEL = None


def panel():
    global _PANEL
    if _PANEL is None:
        _PANEL = fss.Panel()
    return _PANEL


def load_model(device="cuda"):
    with contextlib.redirect_stdout(io.StringIO()):
        m = GRITSCRFDiploidIndel.load_from_checkpoint(CKPT, map_location=device, strict=False)
    return m.eval().to(device)


def manifest_row(corpus, ds, ind, depth):
    for r in csv.DictReader(open(MANIFESTS[corpus]), delimiter="\t"):
        if r["dataset_id"] == ds and r["individual"] == ind and float(r["coverage"]) == float(depth):
            return r
    raise KeyError((corpus, ds, ind, depth))


def rows_of(corpus, ds_list, depth):
    out = []
    for r in csv.DictReader(open(MANIFESTS[corpus]), delimiter="\t"):
        if r["dataset_id"] in ds_list and float(r["coverage"]) == float(depth):
            out.append(r)
    return out


def src_dir(corpus, ds, ind, depth):
    dep = r_depth(depth)
    return ROOTS[corpus] / f"{corpus}__{ds}__{ind}__{dep}x"


def r_depth(depth):
    return {0.01: "0.01", 0.1: "0.1", 0.5: "0.5", 1.0: "1.0", 2.0: "2.0"}[float(depth)]


def load_input(corpus, ds, ind, depth):
    src = src_dir(corpus, ds, ind, depth)
    data = np.load(src / "windowed_k25native_wcount.npy")
    layout = hae.load_contig_layout(src / "raw.npy.bins.tsv", data.shape[1], bin_size=256)
    pos = np.concatenate([np.asarray(p, np.int64) for _c, p, _n in layout])
    contig = np.concatenate([np.full(len(p), ci, np.int64) for ci, (_c, p, _n) in enumerate(layout)])
    return dict(data=data, src=src, pos=pos, contig=contig, layout=layout,
                gametes=hae.load_gamete_names(src / "raw.npy.gametes.tsv"))


def _rechunk(n_rows_by_contig, T):
    """Window index lists [n_win, T] over flat rows (contig-contiguous); last chunk per contig
    overlaps. Returns (idx [W,T], owner mask [W,T] = this window is the one that reports the row)."""
    idx, own = [], []
    start = 0
    for n in n_rows_by_contig:
        if n < T:      # too short for one window: pad by repeating the last row; only first n own
            ii = np.concatenate([np.arange(start, start + n), np.full(T - n, start + n - 1)])
            o = np.zeros(T, bool); o[:n] = True
            idx.append(ii); own.append(o)
            start += n
            continue
        s = 0
        while s + T <= n:
            idx.append(np.arange(start + s, start + s + T)); own.append(np.ones(T, bool))
            s += T
        if s < n:
            o = np.zeros(T, bool); o[T - (n - s):] = True
            idx.append(np.arange(start + n - T, start + n)); own.append(o)
        start += n
    return np.stack(idx), np.stack(own)


def select_rows(flat, contig, pos, keep):
    """Boolean keep mask over flat rows."""
    n = flat.shape[0]
    if keep == "all":
        return np.ones(n, bool)
    if isinstance(keep, tuple) and keep[0] == "thin":
        k = keep[1]
        m = np.zeros(n, bool)
        for c in np.unique(contig):
            ii = np.nonzero(contig == c)[0]
            m[ii[::k]] = True
        return m
    if isinstance(keep, tuple) and keep[0] == "bin":
        cap = keep[1]
        key = contig * (1 << 40) + pos
        first = np.r_[True, key[1:] != key[:-1]]
        grp = np.cumsum(first) - 1
        rank = np.arange(n) - np.nonzero(first)[0][grp]
        return rank < cap
    if keep == "dedup":
        tern = flat[:, :K]
        key = contig * (1 << 40) + pos
        same_bin = np.r_[False, key[1:] == key[:-1]]
        same_tern = np.r_[False, (tern[1:] == tern[:-1]).all(1)]
        return ~(same_bin & same_tern)
    raise ValueError(keep)


def _project(kept_idx, contig, n):
    """For each flat row, the index into kept_idx of the nearest kept row on the same contig."""
    out = np.empty(n, np.int64)
    for c in np.unique(contig):
        ii = np.nonzero(contig == c)[0]
        kk = np.nonzero(contig[kept_idx] == c)[0]          # positions in kept_idx
        kr = kept_idx[kk]
        j = np.searchsorted(kr, ii)
        j0 = np.clip(j - 1, 0, len(kr) - 1); j1 = np.clip(j, 0, len(kr) - 1)
        pick = np.where(np.abs(kr[j1] - ii) < np.abs(kr[j0] - ii), j1, j0)
        out[ii] = kk[pick]
    return out


COUNT_READS = None


def count_code_to_reads(code):
    """count code round(8*log2(1+n)) -> n."""
    c = np.asarray(code, np.float32)
    return np.where(c < 0, 0, np.round(2 ** (c / 8.0) - 1))


@torch.no_grad()
def decode_custom(model, inp, T=T0, keep="all", lam_bp=None, lam_mult=1.0, het=None, ew=None,
                  batch_size=128, device="cuda", capture=False, force_hybrid=None,
                  aff_mult=1.0, emis=None):
    data, contig, pos = inp["data"], inp["contig"], inp["pos"]
    N0, Tn, W = data.shape
    flat = data.reshape(-1, W)
    # route on the ORIGINAL windows, exactly as infer_real_founder_pairs
    tern0 = data[:, :, :K].astype(np.float32)
    M0 = (tern0 == TERN_MATCH)
    row_bin = (pos + (contig << 40)).reshape(N0, Tn)
    r = route_real_sample(M0, row_bin)
    hybrid = r["hybrid"] if force_hybrid is None else force_hybrid
    affinity = _founder_affinity(M0.reshape(-1, K).astype(np.float32))

    km = select_rows(flat, contig, pos, keep)
    kept = np.nonzero(km)[0]
    fk, ck, pk = flat[kept], contig[kept], pos[kept]
    nby = [int((ck == c).sum()) for c in np.unique(ck)]       # contigs are contiguous & ordered
    widx, own = _rechunk(nby, T)
    tern = fk[:, :K].astype(np.float32)
    dist = fk[:, K + 2:2 * K + 2].astype(np.float32)
    cnt = fk[:, 2 * K + 2:3 * K + 2].astype(np.float32)
    feats = torch.tensor(np.stack([tern, dist], -1))
    count = torch.tensor(cnt)
    ext = torch.tensor(affinity, dtype=torch.float32).unsqueeze(0)
    aff_pred = torch.tensor(affinity[:, 0])
    if ew == "count":
        nreads = count_code_to_reads(np.where(fk[:, :K] == 1, cnt, -1).max(1))
        nreads = torch.tensor(np.maximum(nreads, 1.0), dtype=torch.float32)
    span = (pk[widx[:, -1]] - pk[widx[:, 0]] + 256).astype(np.float64)

    model._knock = {"homo": not hybrid}
    model.het_prior = "off"
    pred = np.empty(len(kept), np.int64)
    caps = []
    for s in range(0, len(widx), batch_size):
        wi = torch.tensor(widx[s:s + batch_size])
        xb = feats[wi].to(device)
        B = xb.shape[0]
        eb = ext.expand(B, -1, -1).to(device)
        cb = count[wi].to(device)
        ab = aff_pred.unsqueeze(0).expand(B, -1).to(device)
        emis_p, g, c = model(xb, ext_emb=eb, count=cb, aff_pred=ab)
        h = model._heads
        ood = torch.sigmoid(h["ood_logit"].float()).mean(1)
        lam = model.xo_scale * h["xo_lam"].float() * torch.exp(model.ood_w * ood)
        if aff_mult != 1.0:
            model._trans_prior = model._trans_prior * aff_mult
            model._init_prior = model._init_prior * aff_mult
        if emis == "max":
            emis_p = _emission_weighted(model, xb, g, torch.ones(xb.shape[:2], device=device), hybrid, op="max")
        if lam_bp is not None or lam_mult != 1.0 or het is not None or ew is not None:
            fac = torch.full_like(lam, float(lam_mult))
            if lam_bp is not None:
                fac = fac * torch.tensor(span[s:s + B] / lam_bp, dtype=torch.float32, device=device)
            lam2 = lam * fac
            p = (lam2.unsqueeze(1) / (T - 1)).expand(B, T - 1).clamp(1e-12, 1 - 1e-4)
            p = torch.cat([p[:, :1], p], 1)
            c = -torch.log(p) + torch.log1p(-p) + model.c_offset
            if ew == "count":
                # rebuild the pair emission with the row weight on the read term
                emis_p = _emission_weighted(model, xb, g, nreads[wi].to(device), hybrid)
            if het is not None:
                hm = model.homo_mask
                hr = torch.tensor(het[1], device=device).clamp(1e-4, 1 - 1e-4)
                lq = model.het_w * (hm * torch.log1p(-hr) + (1 - hm) * torch.log(hr))
                lq = lq.view(1, 1, -1).expand(B, T, -1)
                if het[0] == "row":
                    emis_p = emis_p + lq
                    model._seg_prior = None
                else:
                    model._seg_prior = lq.contiguous()
        else:
            model._seg_prior = None
        pb = crf_viterbi_prior(emis_p, c, model.nsw_pair, model.stay_bonus, model._trans_prior,
                               model._init_prior, model._seg_prior)
        pb = pb.cpu().numpy()
        o = own[s:s + B]
        pred[widx[s:s + B][o]] = pb[o]
        if capture:
            caps.append(dict(g=g.float().mean(1).cpu().numpy(), lam=lam.cpu().numpy(),
                             xo_lam=h["xo_lam"].float().cpu().numpy(), ood=ood.cpu().numpy(),
                             het=torch.sigmoid(h["het_logit"].float()).mean(1).cpu().numpy(),
                             gate_row=g.float().cpu().numpy()))
    model._seg_prior = None
    full = pred[_project(kept, contig, len(flat))] if len(kept) < len(flat) else pred
    lo = model.pi[torch.tensor(full)].cpu().numpy().reshape(N0, Tn)
    hi = model.pj[torch.tensor(full)].cpu().numpy().reshape(N0, Tn)
    info = dict(route=r, hybrid=hybrid, n_kept=len(kept), n_rows=len(flat), n_win=len(widx),
                span_median=float(np.median(span)))
    if capture:
        info["caps"] = {k: np.concatenate([c_[k] for c_ in caps]) for k in caps[0]}
        info["widx"] = widx; info["own"] = own; info["kept"] = kept; info["span"] = span
    return lo, hi, info


def _emission_weighted(model, xb, g, wrow, hybrid, op="mix"):
    """Pair emission with each row's read term counted wrow times (rows = collapsed reads)."""
    Kf = model.num_parents
    tern_idx = (xb[..., 0].round().clamp(-1, 1) + 1).long()
    ll = model.tern_loglik[tern_idx].float()
    ll = torch.cat([ll, torch.zeros_like(ll[..., :1])], -1)
    if op == "max":
        mix = torch.maximum(ll[..., model.pi], ll[..., model.pj])
    else:
        mix = torch.logaddexp(ll[..., model.pi], ll[..., model.pj]) - math.log(2.0)
    gf = g.float().clamp(1e-6, 1 - 1e-6).unsqueeze(-1)
    emis = torch.logaddexp(torch.log(gf) + mix, torch.log1p(-gf)) * wrow.unsqueeze(-1)
    deleted = (torch.cat([xb[..., 0], torch.zeros_like(xb[..., :1, 0])], -1).round() == -1).long()
    dos_obs = deleted[..., model.pi] + deleted[..., model.pj]
    logp = F.log_softmax(model._heads["dos_logit"].float(), -1)
    emis = emis + model.dos_w * g.float().unsqueeze(-1) * logp.gather(2, dos_obs)
    emis = emis.masked_fill((model.pi == Kf) | (model.pj == Kf), -1e4)
    if not hybrid:
        emis = emis.masked_fill(model.homo_mask == 0, -1e4)
    return emis


def reference_decode(model, inp, device="cuda"):
    """The production path (fast_eval_ckpt --route --knockout homoroute)."""
    data = inp["data"]
    row_bin = (inp["pos"] + (inp["contig"] << 40)).reshape(data.shape[0], data.shape[1])
    with contextlib.redirect_stdout(io.StringIO()):
        return infer_real_founder_pairs(model, data, K, device=device, route=True, row_bin=row_bin,
                                        knockout="homoroute")


def score(inp, lo, hi, row, bed_dir, per_site=False):
    hae_out = io.StringIO()
    with contextlib.redirect_stdout(hae_out):
        hae.write_imputed_bed(row["individual"], lo, hi, None, inp["gametes"],
                              inp["src"] / "raw.npy.bins.tsv", Path(bed_dir), bin_size=256)
    P = panel()
    f1, f2 = fss.pair_from_bed(P, bed_dir)
    t1, t2 = fss.load_truth(row["truth_h1"]), fss.load_truth(row["truth_h2"])
    res = fss.score_arrays(P, t1, t2, f1, f2)
    truth_ok = ((t1["code"] != fss.T_NOINFO) & (t2["code"] != fss.T_NOINFO) &
                (t1["code"] != fss.T_MISSING) & (t2["code"] != fss.T_MISSING))
    res["coverage"] = res["compared_sites"] / max(int(truth_ok.sum()), 1)
    if per_site:
        res["_f1"], res["_f2"], res["_t1"], res["_t2"] = f1, f2, t1, t2
    return res
