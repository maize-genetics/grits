#!/usr/bin/env python
"""Why does the supervised-heads CRF not take the panel-ceiling path's founder switches?

For one sample: decode with the canonical infer_real_founder_pairs (capture=... keeps the exact CRF
inputs: pair emissions, per-row switch cost c, founder transition prior), project the panel-ceiling
path (eval_suite reference, exact haploid Viterbi on truth, lam 50) onto the model's rows, and
score both paths under the model's own CRF:

  score(path) = init_prior[s0] + sum_t emis[t, s_t] + sum_{t>=1} (-c_t * nsw[s_{t-1}, s_t] + prior[s_{t-1}, s_t])

Per window: model vs ceiling switches (row level, and record level inside the window's bp span),
expected switches the model's own c implies (sum_t p_t), and the emission gain / transition cost
of the ceiling path relative to the Viterbi path. Per ceiling segment (row level): emission gain
of the ceiling founder over the model's call across the segment vs the cost of entering + leaving.

  switch_mechanism_diag.py --corpus validation --ds OUT-INBRED --ind Tx303 [--knockout homoroute]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import simval_eval_indel as sei  # noqa: E402
sys.path.insert(0, sei.CRF_SRC)
import torch  # noqa: E402
from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, infer_real_founder_pairs  # noqa: E402
import eval_suite as es  # noqa: E402
import fast_snprc_score as fss  # noqa: E402
import heldout_assembly_eval as hae  # noqa: E402
from switch_sweep_score import MANIFESTS  # noqa: E402

WORK = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
CKPT = "/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/checkpoints/v6cfullfe-stage2B/last.ckpt"
GLOBS = {"validation": "scratch/lift_s200_local20k/eval__{ds}__{ind}__{dep}x",
         "calibration": "scratch/calib_s200_local20k/calib__{ds}__{ind}__{dep}x"}


def q(a, ps=(10, 25, 50, 75, 90)):
    a = np.asarray(a, float)
    return {f"p{p}": round(float(np.percentile(a, p)), 3) for p in ps} if a.size else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="validation", choices=sorted(GLOBS))
    ap.add_argument("--ds", required=True)
    ap.add_argument("--ind", required=True)
    ap.add_argument("--depth", default="0.1")
    ap.add_argument("--knockout", default="homoroute")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    r = next(r for r in csv.DictReader(open(MANIFESTS[a.corpus]), delimiter="\t")
             if r["dataset_id"] == a.ds and r["individual"] == a.ind and r["coverage"] == a.depth)
    src = WORK / GLOBS[a.corpus].format(ds=a.ds, ind=a.ind, dep=a.depth)
    data = np.load(src / "windowed_k25native_wcount.npy")
    N, T = data.shape[:2]
    layout = hae.load_contig_layout(src / "raw.npy.bins.tsv", T, bin_size=256)
    row_bin = np.concatenate([np.asarray(pos, np.int64) + (ci << 40)
                              for ci, (_c, pos, _n) in enumerate(layout)]).reshape(N, T)
    contig_names = [c for c, _p, _n in layout]
    win_contig = (row_bin[:, 0] >> 40)
    bp = row_bin & ((1 << 40) - 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GRITSCRFDiploidIndel.load_from_checkpoint(a.ckpt, map_location=device, strict=False).eval().to(device)
    cap = {}
    lo_, hi_ = infer_real_founder_pairs(model, data, sei.K, device=device, route=True, row_bin=row_bin,
                                        knockout=a.knockout, capture=cap)
    emis = torch.cat(cap["emis"]).float().numpy()          # [N,T,P]
    c = torch.cat(cap["c"]).numpy()                          # [N,T]
    tp = torch.cat(cap["trans_prior"]).numpy()               # [N,P,P]
    ip = torch.cat(cap["init_prior"]).numpy()                # [N,P]
    pred = torch.cat(cap["pred"]).numpy()                    # [N,T]
    nsw = model.nsw_pair.cpu().numpy()
    pair_table = model.pair_table.cpu().numpy()
    pi, pj = model.pi.cpu().numpy(), model.pj.cpu().numpy()
    homo_only = bool(model._knock.get("homo"))

    # ceiling path on the model's rows
    P = es.panel()
    (g1, s1), (g2, s2) = es.ref_path(P, r["truth_h1"], 50.0), es.ref_path(P, r["truth_h2"], 50.0)
    rec = np.empty((N, T), np.int64)
    for w in range(N):
        cn = contig_names[win_contig[w]]
        ci = fss.AUTOSOMES.index(cn)
        lo, hi = P.bounds[ci]
        j = lo + np.searchsorted(P.pos[lo:hi], bp[w], "right") - 1
        rec[w] = np.clip(j, lo, hi - 1)
    k1, k2 = g1[rec].astype(np.int64), g2[rec].astype(np.int64)
    if homo_only:
        k2 = k1
    ceil_state = pair_table[k1, k2]

    def path_score(s):
        e = np.take_along_axis(emis, s[..., None], 2)[..., 0]           # [N,T]
        prev, nxt = s[:, :-1], s[:, 1:]
        ns = nsw[prev, nxt]
        pr = tp[np.arange(N)[:, None], prev, nxt]
        tr = -c[:, 1:] * ns + pr
        init = ip[np.arange(N), s[:, 0]]
        return e.sum(1), tr.sum(1) + init, e, ns

    em_m, tr_m, e_m, ns_m = path_score(pred)
    em_c, tr_c, e_c, ns_c = path_score(ceil_state)
    p_row = 1.0 / (1.0 + np.exp(c[:, 1:] - model.c_offset.item()))     # c = -logit p + log 2
    exp_sw = p_row.sum(1)

    # record-level ceiling switches inside each window's bp span (haplotype 1; and pair changes)
    ch1 = np.zeros(P.pos.size + 1, np.int64)
    same = np.r_[False, P.chrom[1:] == P.chrom[:-1]]
    d1 = same & np.r_[False, g1[1:] != g1[:-1]]
    ch1[1:] = np.cumsum(d1)
    rec_sw = ch1[rec[:, -1] + 1] - ch1[rec[:, 0] + 1]

    model_sw = (pred[:, 1:] != pred[:, :-1]).sum(1)
    model_hsw = ns_m.sum(1)
    ceil_sw = (ceil_state[:, 1:] != ceil_state[:, :-1]).sum(1)
    ceil_hsw = ns_c.sum(1)
    span_mb = (bp[:, -1] - bp[:, 0]) / 1e6

    # per ceiling segment (row level), homozygous case and hybrid alike: gain of the ceiling state
    # over the Viterbi state across the segment, vs the transition cost of the ceiling path there
    seg_gain, seg_len, seg_cost, seg_inf = [], [], [], []
    tern = data[:, :, :sei.K]
    for w in range(N):
        s = ceil_state[w]
        b = np.flatnonzero(s[1:] != s[:-1]) + 1
        starts = np.r_[0, b]
        ends = np.r_[b, T]
        for a0, a1 in zip(starts, ends):
            if np.all(pred[w, a0:a1] == s[a0]):
                continue                                     # model already on this state
            g = float((e_c[w, a0:a1] - e_m[w, a0:a1]).sum())
            # the ceiling's own transition cost at entry (a0) and exit (a1), if inside the window
            cost = 0.0
            for t in (a0, a1):
                if 0 < t < T:
                    cost += c[w, t] * nsw[s[t - 1] if t == a1 else ceil_state[w, t - 1], ceil_state[w, t]] \
                        - tp[w, ceil_state[w, t - 1], ceil_state[w, t]]
            # discriminating rows: ceiling founder and the model's founder differ in MATCH status
            fk = pi[s[a0]]
            fm = pi[pred[w, a0:a1]]
            mk = tern[w, a0:a1, fk] == 1 if fk < sei.K else np.zeros(a1 - a0, bool)
            mm = tern[w, np.arange(a0, a1), np.minimum(fm, sei.K - 1)] == 1
            seg_gain.append(g); seg_len.append(a1 - a0); seg_cost.append(cost)
            seg_inf.append(int((mk != mm).sum()))
    seg_gain, seg_len, seg_cost, seg_inf = map(np.asarray, (seg_gain, seg_len, seg_cost, seg_inf))

    # per-row emission magnitude on informative rows (homozygous pairs, founder MATCH vs not)
    hom_idx = pair_table[np.arange(sei.K), np.arange(sei.K)]
    e_hom = emis[:, :, hom_idx]                                         # [N,T,K]
    M = tern == 1
    gap = (np.where(M, e_hom, np.nan)).astype(np.float32)
    per_row_match = np.nanmean(gap)
    per_row_nomatch = float(np.nanmean(np.where(~M, e_hom, np.nan)))

    one_sw = 2 * c[:, 1:] if homo_only else c[:, 1:]
    out = dict(
        sample=f"{a.corpus}:{a.ds}:{a.ind}:{a.depth}x", knockout=a.knockout, homo_only=homo_only,
        windows=int(N), window_span_mb=q(span_mb),
        row_gap_kb=q(np.diff(bp, axis=1).ravel() / 1e3, (10, 50, 90, 99)),
        model_switch_rows_per_window=float(model_sw.mean()),
        model_hap_switches_per_window=float(model_hsw.mean()),
        expected_switches_per_window_from_c=float(exp_sw.mean()),
        expected_switches_per_window_q=q(exp_sw),
        ceiling_switch_rows_per_window=float(ceil_sw.mean()),
        ceiling_hap_switches_per_window=float(ceil_hsw.mean()),
        ceiling_record_switches_h1_per_window=float(rec_sw.mean()),
        ceiling_switches_h1_genome=int(s1), ceiling_switches_h2_genome=int(s2),
        switch_ratio_rows=float(model_sw.sum() / max(ceil_sw.sum(), 1)),
        c_per_row=q(c[:, 1:].ravel()),
        cost_one_founder_change_nats=q(one_sw.ravel()),
        trans_prior_on_ceiling_changes=q(np.concatenate([
            tp[w, ceil_state[w, :-1], ceil_state[w, 1:]][ceil_state[w, 1:] != ceil_state[w, :-1]]
            for w in range(N)])),
        emis_match_mean=float(per_row_match), emis_nomatch_mean=per_row_nomatch,
        window_delta_emission_ceiling_minus_model=q(em_c - em_m),
        window_delta_transition_ceiling_minus_model=q(tr_c - tr_m),
        window_delta_total=q((em_c + tr_c) - (em_m + tr_m)),
        total_delta_emission=float((em_c - em_m).sum()), total_delta_transition=float((tr_c - tr_m).sum()),
        frac_windows_ceiling_emission_better=float((em_c > em_m).mean()),
        segments_off_model=int(seg_gain.size),
        seg_len_rows=q(seg_len), seg_informative_rows=q(seg_inf),
        seg_gain_nats=q(seg_gain), seg_cost_nats=q(seg_cost),
        frac_seg_gain_pos=float((seg_gain > 0).mean()),
        frac_seg_gain_exceeds_cost=float((seg_gain > seg_cost).mean()),
        frac_seg_zero_informative=float((seg_inf == 0).mean()),
        seg_rows_in_gain_gt_cost=float(seg_len[seg_gain > seg_cost].sum() / max(seg_len.sum(), 1)),
        seg_rows_with_gain_pos=float(seg_len[seg_gain > 0].sum() / max(seg_len.sum(), 1)),
    )
    s = json.dumps(out, indent=1)
    print(s)
    if a.out:
        Path(a.out).write_text(s)
        np.savez_compressed(Path(a.out).with_suffix(".npz"), model_sw=model_sw, ceil_sw=ceil_sw,
                            rec_sw=rec_sw, exp_sw=exp_sw, em_m=em_m, em_c=em_c, tr_m=tr_m, tr_c=tr_c,
                            span_mb=span_mb, seg_gain=seg_gain, seg_len=seg_len, seg_cost=seg_cost,
                            seg_inf=seg_inf)


if __name__ == "__main__":
    main()
