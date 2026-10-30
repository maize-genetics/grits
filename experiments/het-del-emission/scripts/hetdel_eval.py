#!/usr/bin/env python
"""Inference-time knob sweep for a supervised-heads checkpoint (no retraining).

fast_eval_ckpt.py's per-row recipe (canonical infer_real_founder_pairs + write_imputed_bed with
this worktree's switch-gap-filling bed.py + fast_snprc_score), plus decode-time overrides of the
fitted CRF scalars, applied after loading:

  ll_del / ll_div / ll_match   the frozen read log-LR table (tern_loglik: DEL, DIV, MATCH)
  aff_w, ood_w, aff_ood_s, dos_w, het_w   CRF scalars as named in the checkpoint
  xo_add                       added to log_xo_scale for every sample (switch rate x e^xo_add)
  xo_add_homo                  same, only for samples the router calls homozygous
  oracle_kind=1                homoroute from the TRUE kind instead of the router (diagnostic)
  selfcal=N                    self-calibrated read table, N rounds: decode, re-estimate
                               log P(state | decoded pair's better founder) - log P(state | founders
                               outside the pair) from the sample's own reads, set ll_*, decode again.
                               selfcal_mix=w blends: ll = w*estimate + (1-w)*trained (default 1)
  xo_dens=REF[,xo_dens_a=A]    depth-normalised switch rate: log_xo_scale += A*log(REF / rows-per-Mb of
                               the sample) (A default 1), i.e. per-bp switch rate held at the REF density
  wholechrom=1                 decode each contig's windows as ONE chain (window boundaries pay the
                               normal switch cost instead of restarting free); per-window emissions,
                               switch costs and transition priors are the model's own
  del_rho=R[,del_win=W]        presence-aware pair emission (GRITSCRFDiploidIndel._presence_mix): a
                               founder deleted at the locus explains a row with weight R (not 1/2);
                               W = +-rows used to infer deletion status on MATCH rows (default 16)
  dk=N                         canonical depth knobs: infer_real_founder_pairs(depth_knobs="vN")
  out_only=1                   apply the overrides only when the router calls the sample out of panel
                               (route_real_sample in_panel=False); in-panel samples decode as trained

Several configs can be run in one process (--config NAME:k=v,k=v ...); each writes
results/<NAME>_<scope>_fast/<NAME>__<DS>__<IND>__<depth>x.json (+ genome_covered) and BEDs under
scratch/<NAME>_<scope>_fast/, the same layout as fast_eval_ckpt.py, so eval_suite-style tools work.
"""
import os
import argparse
import contextlib
import csv
import glob
import io
import json
import sys
import time
from pathlib import Path

WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
import src.python.bed_io.bed as _bed  # noqa: E402  pin this worktree's bed.py (fill_switch_gaps)
assert _bed.__file__.startswith(str(WT)), _bed.__file__
sys.path.remove(str(WT))
SIMVAL = WT / "experiments" / "simval-corpus" / "scripts"
sys.path.insert(0, str(SIMVAL))
import numpy as np  # noqa: E402
import simval_eval_indel as sei  # noqa: E402
sys.path.insert(0, sei.CRF_SRC)
from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, infer_real_founder_pairs  # noqa: E402
import python.crf.train_diploid_indel as tdi  # noqa: E402
import fast_snprc_score as fss  # noqa: E402
import heldout_assembly_eval as hae  # noqa: E402

WORK = Path(os.environ.get("GRITS_WORKDIR", "/nonexistent-GRITS_WORKDIR-unset"))
EVAL_MANIFEST = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv"
CALIB_MANIFEST = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_calibration/manifest.tsv"
CK = os.path.join(os.environ.get("GRITS_WORKDIR", "/nonexistent-GRITS_WORKDIR-unset"), "indel_baseline/checkpoints/v6cfullfe-stage2B/last.ckpt")
SCOPES = {"out": ["OUT-INBRED", "OUT-HYB", "OUT-RIL2"], "idx": ["IDX-INBRED", "IDX-HYB", "IDX-RIL2"],
          "mix": ["MIX-HYB", "MIX-RIL2"]}
GENOME_BP = 2.130e9
SCALARS = ("aff_w", "ood_w", "aff_ood_s", "dos_w", "het_w")


def bed_span(bed_dir):
    t = 0
    for f in glob.glob(str(Path(bed_dir) / "*.bed")):
        for l in open(f):
            if not l.startswith("chrom"):
                p = l.split("\t")
                t += int(p[2]) - int(p[1])
    return t / GENOME_BP


def parse_cfg(s):
    name, _, kv = s.partition(":")
    d = {}
    for item in filter(None, kv.split(",")):
        k, v = item.split("=")
        d[k] = float(v)
    return name, d


def apply_cfg(model, base, cfg, homo):
    import torch
    with torch.no_grad():
        for k, v in base.items():
            getattr(model, k).copy_(v)
        tl = model.tern_loglik
        for i, k in enumerate(("ll_del", "ll_div", "ll_match")):
            if k in cfg:
                tl[i] = cfg[k]
        for k in SCALARS:
            if k in cfg:
                getattr(model, k).fill_(cfg[k])
        add = cfg.get("xo_add", 0.0) + cfg.get("_xo_depth", 0.0) + (cfg.get("xo_add_homo", 0.0) if homo else 0.0)
        model.log_xo_scale.add_(add)
    model.del_rho = cfg.get("del_rho")
    model.del_win = int(cfg.get("del_win", 16))


def dosage_confusion(panel, t1, t2, f1, f2):
    """3x3 counts [truth #DEL haplotypes][predicted #DEL haplotypes] over compared sites
    (same site mask as fss.score_arrays; per-haplotype allele class == C_DEL)."""
    n = panel.pos.size
    idx = np.arange(n)
    hp = (f1 >= 0) & (f2 >= 0)
    i1 = np.full(n, -1, np.int16); i2 = np.full(n, -1, np.int16)
    i1[hp] = panel.fa[idx[hp], f1[hp]]; i2[hp] = panel.fa[idx[hp], f2[hp]]
    ok = (t1["code"] != fss.T_NOINFO) & (t2["code"] != fss.T_NOINFO) & \
         (t1["code"] != fss.T_MISSING) & (t2["code"] != fss.T_MISSING)
    m = ok & hp & (i1 >= 0) & (i2 >= 0)
    rows = idx[m]
    def tcls(t):
        c = t["code"][rows]
        return np.where(c == fss.T_NOVEL, t["ncls"][rows], panel.acls[rows, np.clip(c, 0, None)])
    td = (tcls(t1) == fss.C_DEL).astype(int) + (tcls(t2) == fss.C_DEL)
    pd_ = (panel.acls[rows, i1[m]] == fss.C_DEL).astype(int) + (panel.acls[rows, i2[m]] == fss.C_DEL)
    return np.bincount(td * 3 + pd_, minlength=9).reshape(3, 3).tolist()


def selfcal_table(data, lo, hi, K):
    t = np.rint(data[:, :, :K]).clip(-1, 1).astype(np.int8).reshape(-1, K)
    lo, hi = lo.reshape(-1), hi.reshape(-1)
    ok = (lo < K) & (hi < K)
    t, lo, hi = t[ok], lo[ok], hi[ok]
    r = np.arange(len(t))
    best = np.maximum(t[r, lo], t[r, hi])
    cnt_c = np.bincount(best + 1, minlength=3).astype(float)
    tot = np.array([(t == v).sum() for v in (-1, 0, 1)], float)
    inpair = np.zeros(3)
    for f in (lo, hi):
        np.add.at(inpair, t[r, f] + 1, 1)
    same = lo == hi
    np.subtract.at(inpair, t[r[same], lo[same]] + 1, 1)
    cnt_o = tot - inpair
    pc, po = (cnt_c + 1) / (cnt_c.sum() + 3), (cnt_o + 1) / (cnt_o.sum() + 3)
    return np.log(pc / po)


class _Collect:
    """Stands in for model.crf_decode: keeps each batch's CRF inputs, returns a dummy path."""
    def __init__(self, model):
        self.m, self.parts = model, []

    def __call__(self, emis_p, c, stay_bonus=None, marginal=False):
        m = self.m
        self.parts.append((emis_p.float().cpu(), c.float().cpu(), m._trans_prior.float().cpu(),
                           m._init_prior.float().cpu()))
        import torch
        return torch.zeros(emis_p.shape[:2], dtype=torch.long, device=emis_p.device)


def wholechrom_decode(model, col, layout, device):
    import torch
    E = torch.cat([p[0] for p in col.parts]); C = torch.cat([p[1] for p in col.parts])
    TP = torch.cat([p[2] for p in col.parts]); IP = torch.cat([p[3] for p in col.parts])
    N, T, P = E.shape
    nsw = model.nsw_pair.float().to(device)
    stay = (nsw == 0).float()
    out = torch.zeros(N, T, dtype=torch.long)
    w0 = 0
    for _c, _pos, nw in layout:
        ws = range(w0, w0 + nw)
        delta = None
        bps = []
        for w in ws:
            e = E[w].to(device); cc = C[w].to(device); tp = TP[w].to(device)
            for t in range(T):
                if delta is None:
                    delta = e[0] + IP[w].to(device)
                    continue
                ct = cc[t]
                tr = -ct * nsw + tp                       # stay_bonus is 0 in heads models
                best, idx = (delta.unsqueeze(1) + tr).max(dim=0)
                delta = e[t] + best
                bps.append(idx.to(torch.int16))
        bp = torch.stack(bps).cpu().numpy()
        path = np.empty(nw * T, np.int64)
        path[-1] = int(delta.argmax())
        for t in range(nw * T - 2, -1, -1):
            path[t] = bp[t, path[t + 1]]
        out[w0:w0 + nw] = torch.from_numpy(path).view(nw, T)
        w0 += nw
    return model.pi.cpu()[out].numpy(), model.pj.cpu()[out].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True, help="NAME:k=v,k=v (NAME is the result tag)")
    ap.add_argument("--scope", nargs="+", default=["out"])
    ap.add_argument("--depth", default="0.1")
    ap.add_argument("--corpus", choices=["calib", "eval"], required=True)
    ap.add_argument("--ckpt", default=CK)
    ap.add_argument("--input-glob", default=None, help="override the corpus's aligned-input dir glob")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these individuals")
    ap.add_argument("--datasets", nargs="*", default=None, help="restrict to these dataset ids (e.g. OUT-HYB)")
    a = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GRITSCRFDiploidIndel.load_from_checkpoint(a.ckpt, map_location=device, strict=False).eval().to(device)
    base = {k: getattr(model, k).detach().clone() for k in ("tern_loglik", "log_xo_scale") + SCALARS}
    panel = fss.Panel()
    manifest, glob_ = ((CALIB_MANIFEST, "scratch/calib_s200_local20k") if a.corpus == "calib"
                       else (EVAL_MANIFEST, "scratch/lift_s200_local20k"))
    glob_ = a.input_glob or glob_
    rows = [r for r in csv.DictReader(open(manifest), delimiter="\t") if r["coverage"] == a.depth
            and any(r["dataset_id"] in SCOPES[s] for s in a.scope) and (a.only is None or r["individual"] in a.only)
            and (a.datasets is None or r["dataset_id"] in a.datasets)]
    cfgs = [parse_cfg(s) for s in a.config]
    for r in rows:
        ds, ind = r["dataset_id"], r["individual"]
        scope = next(s for s in a.scope if ds in SCOPES[s])
        src = None
        for d in sorted(glob.glob(str(WORK / f"{glob_}/*__{ds}__{ind}__{a.depth}x"))):
            if (Path(d) / "windowed_k25native_wcount.npy").exists():
                src = Path(d)
        if src is None:
            print(f"SKIP {ds} {ind}", flush=True)
            continue
        data = np.load(src / "windowed_k25native_wcount.npy")
        layout = hae.load_contig_layout(src / "raw.npy.bins.tsv", data.shape[1], bin_size=256)
        row_bin = np.concatenate([np.asarray(pos, np.int64) + (ci << 40) for ci, (_c, pos, _n) in enumerate(layout)]
                                 ).reshape(data.shape[0], data.shape[1])
        t1, t2 = fss.load_truth(r["truth_h1"]), fss.load_truth(r["truth_h2"])
        # router call once per sample (independent of the knobs)
        rt = tdi.route_real_sample(data[:, :, :sei.K] == 1, row_bin)
        dens = data.shape[0] * data.shape[1] / (GENOME_BP / 1e6)
        for name, cfg in cfgs:
            t0 = time.time()
            if "xo_dens" in cfg:
                cfg = {**cfg, "_xo_depth": cfg.get("xo_dens_a", 1.0) * float(np.log(cfg["xo_dens"] / dens))}
            homo = not rt["hybrid"]
            if cfg.get("oracle_kind"):
                homo = not ds.endswith("-HYB")
            apply_cfg(model, base, {} if (cfg.get("out_only") and rt["in_panel"]) else cfg, homo)
            orig = tdi.route_real_sample
            if cfg.get("oracle_kind"):
                tdi.route_real_sample = lambda *x, **y: {**rt, "hybrid": not homo}
            key = f"{name}__{ds}__{ind}__{a.depth}x"
            if (WORK / "results" / f"{name}_{scope}_fast" / f"{key}.json").exists():
                continue                                   # resumable: row already scored
            bed_dir = WORK / "scratch" / f"{name}_{scope}_fast" / key / "bed"
            est = []
            dk = f"v{int(cfg['dk'])}" if cfg.get("dk") else None
            try:
                with contextlib.redirect_stdout(io.StringIO()) as buf:
                    if cfg.get("wholechrom"):
                        col = _Collect(model)
                        model.crf_decode = col
                        try:
                            infer_real_founder_pairs(model, data, sei.K, device=device, route=True,
                                                     row_bin=row_bin, knockout="homoroute", depth_knobs=dk)
                        finally:
                            del model.crf_decode
                        lo, hi = wholechrom_decode(model, col, layout, device)
                    else:
                        lo, hi = infer_real_founder_pairs(model, data, sei.K, device=device, route=True,
                                                          row_bin=row_bin, knockout="homoroute", depth_knobs=dk)
                    for _ in range(int(cfg.get("selfcal", 0))):
                        llr = selfcal_table(data, lo, hi, sei.K)
                        w = cfg.get("selfcal_mix", 1.0)
                        tr = base["tern_loglik"].cpu().numpy()
                        new = w * llr + (1 - w) * tr
                        est.append([round(float(x), 3) for x in new])
                        c2 = {k: v for k, v in cfg.items() if not k.startswith("ll_")}
                        c2.update(ll_del=float(new[0]), ll_div=float(new[1]), ll_match=float(new[2]))
                        apply_cfg(model, base, c2, homo)
                        lo, hi = infer_real_founder_pairs(model, data, sei.K, device=device, route=True,
                                                          row_bin=row_bin, knockout="homoroute", depth_knobs=dk)
                    hae.write_imputed_bed(ind, lo, hi, None, hae.load_gamete_names(src / "raw.npy.gametes.tsv"),
                                          src / "raw.npy.bins.tsv", bed_dir, bin_size=256)
            finally:
                tdi.route_real_sample = orig
            f1, f2 = fss.pair_from_bed(panel, bed_dir)
            res = fss.score_arrays(panel, t1, t2, f1, f2)
            res["dosage_confusion"] = dosage_confusion(panel, t1, t2, f1, f2)
            log = [l.strip() for l in buf.getvalue().splitlines() if "route:" in l or "homoroute" in l]
            res.update(router_in_panel=bool(rt["in_panel"]), router_hybrid=bool(rt["hybrid"]), row_key=key, status="ok", source_input=str(src), config=cfg, selfcal_tables=est, homo_scale_log=" | ".join(log),
                       genome_covered=bed_span(bed_dir), rows_per_mb=dens, wall_seconds=round(time.time() - t0, 1))
            rd = WORK / "results" / f"{name}_{scope}_fast"
            rd.mkdir(parents=True, exist_ok=True)
            (rd / f"{key}.json").write_text(json.dumps(res, indent=1))
            print(f"{key:<60} snprc={100 * res['snprc_error_rate']:6.3f}%  err={100 * res['error_rate']:6.2f}%  "
                  f"hetDEL_ok={100 * res['dosage_confusion'][1][1] / max(1, sum(res['dosage_confusion'][1])):.1f}%  "
                  f"cov={100 * res['genome_covered']:.1f}%  ({res['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
