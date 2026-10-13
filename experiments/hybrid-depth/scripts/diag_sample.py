"""Localise hybrid errors for one in-panel hybrid (IDX-HYB) across depths.

Per depth: production decode (decode_custom, all knobs off) with head capture, then
  row level  : read class of each row w.r.t. the true parents A,B (A-only / B-only / both /
               neither) and predicted pair class (AB / AA|BB / A+X|B+X / XY)
  window     : lambda, gate, ood, het head, span, #A-only/#B-only rows, wrong-row fraction
  site level : genotype errors (all-sites + SNP+RC) split by predicted pair class at the
               site, truth allele class, distance to the nearest model switch, and whether
               A and B carry the same allele there (IBD-like)
Usage: diag_sample.py DS IND [depths...]  -> results/hybrid_depth/diag_<DS>_<IND>.json
"""
import json
import sys
import time

import numpy as np

import hd_lib as L

ds, ind = sys.argv[1], sys.argv[2]
depths = [float(x) for x in sys.argv[3:]] or [0.1, 0.5, 1.0, 2.0]
A_name, B_name = ind.split("x")
out = L.WORK / "results/hybrid_depth"
out.mkdir(parents=True, exist_ok=True)
m = L.load_model()
P = L.panel()
summary = {}
for dep in depths:
    t0 = time.time()
    inp = L.load_input("eval", ds, ind, dep)
    lo, hi, info = L.decode_custom(m, inp, capture=True)
    gam = inp["gametes"]
    a, b = gam.index(A_name), gam.index(B_name)
    data = inp["data"]
    N, T, W = data.shape
    tern = data[:, :, :L.K]
    mA, mB = tern[:, :, a] == 1, tern[:, :, b] == 1
    nmatch = (tern == 1).sum(-1)
    rc = np.select([mA & ~mB, mB & ~mA, mA & mB], [0, 1, 2], 3)            # A-only,B-only,both,neither
    homo = lo == hi
    isA = (lo == a) | (hi == a)
    isB = (lo == b) | (hi == b)
    pc = np.select([isA & isB, homo & (isA | isB), isA | isB], [0, 1, 2], 3)  # AB, AA/BB, A+X/B+X, XY
    caps = info["caps"]
    # switches inside windows + at window boundaries (calls change between consecutive rows)
    fl_lo, fl_hi = lo.reshape(-1), hi.reshape(-1)
    contig = inp["contig"]; pos = inp["pos"]
    chg = np.r_[False, ((fl_lo[1:] != fl_lo[:-1]) | (fl_hi[1:] != fl_hi[:-1])) & (contig[1:] == contig[:-1])]
    in_win = chg.reshape(N, T)[:, 1:].sum()
    genome_bp = sum(int(p[-1] - p[0]) for _c, p, _n in inp["layout"])
    span = info["span"]
    wrong_w = (pc != 0).mean(1)
    nAo = (rc == 0).sum(1); nBo = (rc == 1).sum(1)
    minor = np.minimum(nAo, nBo)
    # head outputs vs wrong windows
    win = dict(
        span_median_kb=float(np.median(span) / 1e3),
        lam_mean=float(caps["lam"].mean()), lam_median=float(np.median(caps["lam"])),
        xo_lam_mean=float(caps["xo_lam"].mean()),
        lam_per_mb=float((caps["lam"] / (span / 1e6)).mean()),
        gate_mean=float(caps["g"].mean()), ood_mean=float(caps["ood"].mean()),
        het_head_mean=float(caps["het"].mean()),
        gate_rows_by_class={n: float(caps["gate_row"][rc == i].mean()) for i, n in
                            enumerate(["Aonly", "Bonly", "both", "neither"])},
        rows_per_window_Aonly=float(nAo.mean()), rows_per_window_Bonly=float(nBo.mean()),
        rows_per_window_minor_parent_median=float(np.median(minor)),
        frac_windows_minor_le2=float((minor <= 2).mean()),
        wrong_window_frac=float((wrong_w > 0).mean()),
        wrong_window_frac_gt50=float((wrong_w > 0.5).mean()),
        wrong_rows_in_windows_gt50=float(((pc != 0).sum(1) * (wrong_w > 0.5)).sum() / max((pc != 0).sum(), 1)),
        wrong_frac_by_minor={f"{lo_}-{hi_}": float(wrong_w[(minor >= lo_) & (minor <= hi_)].mean())
                             if ((minor >= lo_) & (minor <= hi_)).any() else None
                             for lo_, hi_ in [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 999)]},
        frac_windows_by_minor={f"{lo_}-{hi_}": float(((minor >= lo_) & (minor <= hi_)).mean())
                               for lo_, hi_ in [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 999)]},
    )
    row = dict(
        n_rows=int(N * T), n_windows=int(N),
        read_class={n: float((rc == i).mean()) for i, n in enumerate(["Aonly", "Bonly", "both", "neither"])},
        mean_matching_founders=float(nmatch.mean()),
        pred_class={n: float((pc == i).mean()) for i, n in enumerate(["AB", "AA_or_BB", "parent_plus_X", "XY"])},
        pred_homo_frac=float(homo.mean()),
        switches_in_window_per_mb=float(in_win / (genome_bp / 1e6)),
        switches_all_per_mb=float(chg.sum() / (genome_bp / 1e6)),
    )
    # site level
    row_meta = L.manifest_row("eval", ds, ind, dep)
    res = L.score(inp, lo, hi, row_meta, f"/tmp/hd_diag_bed_{ind}_{dep}", per_site=True)
    f1, f2, t1, t2 = res.pop("_f1"), res.pop("_f2"), res.pop("_t1"), res.pop("_t2")
    pa, pb_ = P.fidx[A_name], P.fidx[B_name]
    n = P.pos.size
    has = (f1 >= 0) & (f2 >= 0)
    idx = np.arange(n)
    i1 = np.full(n, -1, np.int16); i2 = np.full(n, -1, np.int16)
    i1[has] = P.fa[idx[has], f1[has]]; i2[has] = P.fa[idx[has], f2[has]]
    truth_ok = ((t1["code"] != L.fss.T_NOINFO) & (t2["code"] != L.fss.T_NOINFO) &
                (t1["code"] != L.fss.T_MISSING) & (t2["code"] != L.fss.T_MISSING))
    msk = truth_ok & has & (i1 >= 0) & (i2 >= 0)
    sel = np.nonzero(msk)[0]

    def ident(t):
        c = t["code"][sel].astype(np.int64)
        return np.where(c == L.fss.T_NOVEL, -(np.abs(t["nhash"][sel]) % (1 << 60)) - 10, c)
    ta, tb = ident(t1), ident(t2)
    x, y = i1[sel].astype(np.int64), i2[sel].astype(np.int64)
    match = (np.minimum(ta, tb) == np.minimum(x, y)) & (np.maximum(ta, tb) == np.maximum(x, y))
    acls = P.acls

    def tcls(t, rows):
        c = t["code"][rows]
        return np.where(c == L.fss.T_NOVEL, t["ncls"][rows], acls[rows, np.clip(c, 0, None)])
    t_cls = L.fss.classify_pair(tcls(t1, sel), tcls(t2, sel))
    i_cls = L.fss.classify_pair(acls[sel, i1[sel]], acls[sel, i2[sel]])
    snprc = (t_cls <= 1) & (i_cls <= 1)
    s1, s2 = f1[sel], f2[sel]
    sA = (s1 == pa) | (s2 == pa); sB = (s1 == pb_) | (s2 == pb_)
    shomo = s1 == s2
    spc = np.select([sA & sB, shomo & (sA | sB), sA | sB], [0, 1, 2], 3)
    alA = P.fa[sel, pa]; alB = P.fa[sel, pb_]
    ibd = alA == alB
    # distance to nearest model switch (row positions where the call changes), same chromosome
    chrom_rows = np.array([L.fss.AUTOSOMES.index(c) if c in L.fss.AUTOSOMES else -1
                           for c, _p, _n in inp["layout"]])[contig]
    sw_pos = pos[chg]; sw_ch = chrom_rows[chg]
    sch = P.chrom[sel]; spos = P.pos[sel].astype(np.int64)
    dsw = np.full(len(sel), 1 << 40, np.int64)
    for cc in np.unique(sch):
        ii = np.nonzero(sch == cc)[0]
        sp = np.sort(sw_pos[sw_ch == cc])
        if len(sp) == 0:
            continue
        j = np.searchsorted(sp, spos[ii])
        d0 = np.abs(spos[ii] - sp[np.clip(j - 1, 0, len(sp) - 1)])
        d1 = np.abs(spos[ii] - sp[np.clip(j, 0, len(sp) - 1)])
        dsw[ii] = np.minimum(d0, d1)
    err = ~match

    def brk(mask_scope):
        e = err & mask_scope
        tot_e = max(int(e.sum()), 1)
        return dict(
            n_err=int(e.sum()), err_rate=float(e.sum() / max(mask_scope.sum(), 1)),
            by_pred_class={nm: float((e & (spc == i)).sum() / tot_e) for i, nm in
                           enumerate(["AB", "AA_or_BB", "parent_plus_X", "XY"])},
            sites_by_pred_class={nm: float(((spc == i) & mask_scope).sum() / max(mask_scope.sum(), 1)) for i, nm in
                                 enumerate(["AB", "AA_or_BB", "parent_plus_X", "XY"])},
            by_truth_class={nm: float((e & (t_cls == i)).sum() / tot_e) for i, nm in
                            enumerate(["HOMREF", "SNP", "INS", "DEL", "HETMIX"])},
            by_switch_dist={nm: float((e & (dsw >= lo_) & (dsw < hi_)).sum() / tot_e) for nm, lo_, hi_ in
                            [("<10kb", 0, 10_000), ("10-100kb", 10_000, 100_000), ("100kb-1Mb", 100_000, 1_000_000),
                             (">1Mb", 1_000_000, 1 << 41)]},
            sites_by_switch_dist={nm: float(((dsw >= lo_) & (dsw < hi_) & mask_scope).sum() / max(mask_scope.sum(), 1))
                                  for nm, lo_, hi_ in
                                  [("<10kb", 0, 10_000), ("10-100kb", 10_000, 100_000),
                                   ("100kb-1Mb", 100_000, 1_000_000), (">1Mb", 1_000_000, 1 << 41)]},
            err_at_hetdel=float((e & (t_cls == 3) & (ta != tb)).sum() / tot_e),
            err_where_AB_same_allele=float((e & ibd).sum() / tot_e),
            err_rate_in_wrong_pair_sites=float(e[(spc != 0) & mask_scope].mean()) if ((spc != 0) & mask_scope).any() else None,
        )
    site = dict(all=brk(np.ones(len(sel), bool)), snprc=brk(snprc))
    summary[str(dep)] = dict(error_rate=res["error_rate"], snprc_error_rate=res["snprc_error_rate"],
                             coverage=res["coverage"], rows=row, windows=win, sites=site,
                             route=info["route"], secs=round(time.time() - t0))
    print(dep, json.dumps({k: summary[str(dep)][k] for k in ("error_rate", "snprc_error_rate", "coverage")}),
          json.dumps(row), flush=True)
    (out / f"diag_{ds}_{ind}.json").write_text(json.dumps(summary, indent=1))
