"""End snapping: refine projected interval ends by aligning the reference flank to the founder.

A line-for-line port of ropebwt3 `extract --lift` snapping, so that results are base-identical.
snap_version="v4" (default) follows extract.c at 509b2bf (snap_boundary / snap_side / snap_walk): each
end is a reference boundary x, projected through base x-1, and placed from the flanks on both sides of x,
so pieces [a,x) and [x,b) always meet. snap_version="v3" keeps the a4df980 behaviour (each end snapped
with its own inner flank, projected through its own base). Common to both:

  * affine-gap DP (match +1, mismatch -4, gap open 6 + 1 per base), global on the flank (query),
    free at both ends of the founder window (target); N (or any non-ACGT) never matches;
  * the same cell priorities (diagonal > horizontal > vertical on ties; gap open preferred over
    extension on ties), leftmost best end, second best >= excl bp away, the same traceback;
  * acceptance: >= max(16, F/2) aligned flank bases, gap-compressed identity >= min_id, and no other
    placement within `margin` of the best -> otherwise proj_ambiguous / proj_lowid / proj_short;
  * fallbacks: radius2 window (snap_wide), then walking the flank inward (snap_walk);
  * the 10-mer prefilter for windows wider than 8x the flank (it can change results in corner cases,
    so it is reproduced exactly rather than skipped).

Founder windows are taken in reference orientation ('-' loci reverse-complemented), in "oriented"
coordinates u = L-1-x on '-' loci, exactly as in C.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .projection import OK, OK_CLAMPED, Projection, bare_chrom, is_usable

ST_OK, ST_AMBIG, ST_LOWID, ST_SHORT = 0, 1, 2, 3
INT32_MIN = -(1 << 31)
NEG_INF = -(1 << 29)            # INT32_MIN / 4
SEED_K = 10
SEED_RATIO = 8

LABEL_PROJ, LABEL_SNAP, LABEL_WIDE, LABEL_AMBIG = "proj", "snap", "snap_wide", "proj_ambiguous"
LABEL_LOWID, LABEL_SHORT, LABEL_IDENTITY, LABEL_WALK = "proj_lowid", "proj_short", "identity", "snap_walk"
LABEL_X_SNAP, LABEL_X_WIDE, LABEL_X_WALK = "snap_x", "snap_x_wide", "snap_x_walk"
SNAPPED_LABELS = frozenset((LABEL_SNAP, LABEL_WIDE, LABEL_WALK, LABEL_X_SNAP, LABEL_X_WIDE, LABEL_X_WALK))
_CROSS = {LABEL_SNAP: LABEL_X_SNAP, LABEL_WIDE: LABEL_X_WIDE}

_NT6 = np.full(256, 5, np.uint8)
for _c, _v in zip(b"ACGTacgt", (1, 2, 3, 4, 1, 2, 3, 4)):
    _NT6[_c] = _v
_COMP6 = np.array([0, 4, 3, 2, 1, 5], np.uint8)


def nt6(seq) -> np.ndarray:
    """ACGT (any case) -> 1..4, anything else -> 5 (as ropebwt3 stores non-ACGT as N)."""
    if isinstance(seq, str):
        seq = seq.encode("ascii")
    return _NT6[np.frombuffer(seq, np.uint8)]


def revcomp6(a: np.ndarray) -> np.ndarray:
    return _COMP6[a[::-1]]


@dataclass(frozen=True)
class SnapOpt:
    match: int = 1
    mismatch: int = 4
    gap_open: int = 6
    gap_ext: int = 1
    min_aligned: int = 16
    min_frac: float = 0.5
    min_id: float = 0.85
    margin: int = 5
    excl: int = 8
    clip5: int = -1
    no_seed: bool = False


@dataclass
class SnapRes:
    score: int = INT32_MIN
    second: int = INT32_MIN
    t_end: int = -1
    q_first: int = -1
    t_first: int = -1
    q_last: int = -1
    t_last: int = -1
    n_aligned: int = 0
    n_match: int = 0
    n_core: int = 0
    identity: float = 0.0
    status: int = ST_SHORT


def _dp(q: np.ndarray, t: np.ndarray, o: SnapOpt) -> SnapRes:
    n, m = len(q), len(t)
    r = SnapRes()
    need = int(o.min_frac * n)
    if need < o.min_aligned:
        need = o.min_aligned
    if n <= 0 or m <= 0 or n < o.min_aligned or m < need:
        r.status = ST_SHORT
        return r
    oe, ge = o.gap_open + o.gap_ext, o.gap_ext
    t64 = t.astype(np.int64)
    tvalid = (t64 >= 1) & (t64 <= 4)
    H = np.zeros(m + 1, np.int64)
    Fv = np.full(m + 1, NEG_INF, np.int64)
    dirm = np.zeros((n + 1, m + 1), np.uint8)
    ramp = np.arange(m, dtype=np.int64) * ge          # ge * k for k = 0..m-1
    neg_head = np.array([NEG_INF], np.int64)
    for i in range(1, n + 1):
        qc = int(q[i - 1])
        if 1 <= qc <= 4:
            sc = np.where(tvalid & (t64 == qc), o.match, -o.mismatch)
        else:
            sc = np.full(m, -o.mismatch, np.int64)
        f0 = -oe if i == 1 else int(Fv[0]) - ge
        fo = H[1:] - oe
        fe = Fv[1:] - ge
        b8 = fe > fo
        Fn = np.where(b8, fe, fo)
        h0 = H[:-1] + sc
        b16 = None
        if o.clip5 >= 0 and i > 1:
            c = sc - o.clip5
            b16 = c > h0
            h0 = np.where(b16, c, h0)
        G = np.maximum(h0, Fn)
        A = np.empty(m, np.int64)
        A[0] = f0
        A[1:] = G[:-1] + ramp[1:]
        E = np.maximum.accumulate(A) - oe - ramp      # E[j], j = 1..m (index j-1)
        selF = Fn > np.maximum(h0, E)
        selE = (~selF) & (E > h0)
        Hn = np.where(selF, Fn, np.where(selE, E, h0))
        Hprev = np.empty(m, np.int64)
        Hprev[0] = f0
        Hprev[1:] = Hn[:-1]
        Eprev = np.concatenate((neg_head, E[:-1]))
        b4 = (Eprev - ge) > (Hprev - oe)
        b = b4.astype(np.uint8) * 4 | b8.astype(np.uint8) * 8
        b |= selE.astype(np.uint8) | (selF.astype(np.uint8) * 2)
        if b16 is not None:
            b |= ((b16 & ~selE & ~selF).astype(np.uint8) * 16)
        dirm[i, 1:] = b
        dirm[i, 0] = 2 | (8 if i > 1 else 0)
        H = np.concatenate((np.array([f0], np.int64), Hn))
        Fv = np.concatenate((np.array([f0], np.int64), Fn))
    last = H[1:]
    best_j = int(np.argmax(last)) + 1
    r.score, r.t_end = int(H[best_j]), best_j
    js = np.arange(1, m + 1)
    far = np.abs(js - best_j) >= o.excl
    if far.any():
        r.second = int(last[far].max())

    col = []
    state, i, j, last_set = 0, n, best_j, False
    while i > 0:
        b = int(dirm[i, j])
        if state == 0:
            src = b & 3
            if src == 0:
                col.append("=" if (q[i - 1] == t[j - 1] and 1 <= q[i - 1] <= 4) else "X")
                if not last_set:
                    r.q_last, r.t_last, last_set = i - 1, j - 1, True
                r.q_first, r.t_first = i - 1, j - 1
                if b & 16:
                    break
                i -= 1
                j -= 1
            else:
                state = src
        elif state == 1:
            col.append("I")
            state = 1 if b & 4 else 0
            j -= 1
        else:
            col.append("D")
            state = 2 if b & 8 else 0
            i -= 1
    mpos = [k for k, c in enumerate(col) if c in "=X"]
    if mpos:
        first, lastk = mpos[0], mpos[-1]
        n_open = 0
        for k in range(first, lastk + 1):
            c = col[k]
            if c in "=X":
                r.n_aligned += 1
                r.n_match += c == "="
            elif k == first or col[k - 1] != c:
                n_open += 1
            r.n_core += 1
        r.identity = r.n_match / (r.n_aligned + n_open)
    if r.n_aligned < need or r.identity < o.min_id:
        r.status = ST_LOWID
    elif r.second != INT32_MIN and r.second >= r.score - o.margin:
        r.status = ST_AMBIG
    else:
        r.status = ST_OK
    return r


def _kmer_hashes(a: np.ndarray, k: int):
    """Hash of the k-mer ending at each position j >= k-1 and whether all its bases are ACGT."""
    if len(a) < k:
        return np.zeros(0, np.int64), np.zeros(0, bool)
    w = np.lib.stride_tricks.sliding_window_view(a.astype(np.int64), k)
    valid = ((w >= 1) & (w <= 4)).all(axis=1)
    h = ((w - 1) << (2 * np.arange(k - 1, -1, -1, dtype=np.int64))).sum(axis=1)
    return h, valid


def snap_align(q: np.ndarray, t: np.ndarray, o: SnapOpt = SnapOpt()) -> SnapRes:
    """rb3_snap_align: the DP, behind the same 10-mer prefilter for wide windows."""
    n, m, k = len(q), len(t), SEED_K
    if o.no_seed or m <= SEED_RATIO * n or n < k or n < o.min_aligned:
        return _dp(q, t, o)
    qh, qv = _kmer_hashes(q, k)
    qpos = np.flatnonzero(qv)
    order = np.lexsort((qpos, qh[qpos]))
    qkeys, qps = qh[qpos][order], qpos[order]
    th, tv = _kmer_hashes(t, k)
    hit = tv & np.isin(th, qkeys)
    lo_hi: list[list[int]] = []
    for jj in np.flatnonzero(hit):
        h = th[jj]
        a = int(np.searchsorted(qkeys, h, "left"))
        b = int(np.searchsorted(qkeys, h, "right"))
        for qp in qps[a:b]:
            diag = int(jj) - int(qp)                       # jj = j - k + 1
            s0, e0 = max(0, diag - n), min(m, diag + 2 * n)
            if lo_hi and s0 <= lo_hi[-1][1]:
                if e0 > lo_hi[-1][1]:
                    lo_hi[-1][1] = e0
            else:
                lo_hi.append([s0, e0])
    if not lo_hi:
        return SnapRes(status=ST_LOWID)
    lo_hi.sort(key=lambda x: x[0])                          # stable, as the C insertion sort
    merged = [lo_hi[0]]
    for s0, e0 in lo_hi[1:]:
        if s0 <= merged[-1][1]:
            if e0 > merged[-1][1]:
                merged[-1][1] = e0
        else:
            merged.append([s0, e0])
    sep = 2 * n + o.gap_open + 8
    cl = sum(e - s for s, e in merged) + sep * (len(merged) - 1)
    if cl >= m:
        return _dp(q, t, o)
    parts, segs, c = [], [], 0
    for idx, (s0, e0) in enumerate(merged):
        if idx:
            parts.append(np.full(sep, 5, np.uint8))
            c += sep
        segs.append((c, s0))
        parts.append(t[s0:e0])
        c += e0 - s0
    r = _dp(q, np.concatenate(parts), o)
    for name, off in (("t_first", 0), ("t_last", 0), ("t_end", 1)):
        x = getattr(r, name)
        if x < 0:
            continue
        s2 = len(segs) - 1
        while s2 > 0 and segs[s2][0] > x - off:
            s2 -= 1
        setattr(r, name, x - segs[s2][0] + segs[s2][1])
    return r


# --------------------------------------------------------------------------- end snapping

class SnapProvider:
    """Wrap a projection provider and snap both ends of every usable projection, as ropebwt3 does.

    ref: object with fetch(contig, start, end) for the reference assembly (e.g. an AssemblySet holding
    B73); assemblies: AssemblySet of the founders. The returned Projection carries
    snap = (start_label, end_label) and snap_shift = (start_shift, end_shift), in reference-end order,
    shifts in forward founder coordinates, as in the C `--project-only` columns.
    """

    def __init__(self, base, ref, assemblies, ref_assembly: str = "B73", flank: int = 64, radius: int = 200,
                 radius2: int = 6000, min_id: float = 0.85, margin: int = 5, opt: SnapOpt | None = None,
                 snap_version: str = "v4"):
        if snap_version not in ("v3", "v4"):
            raise ValueError("snap_version must be 'v3' or 'v4'")
        self.snap_version = snap_version
        o = opt or SnapOpt()
        self.opt = replace(o, min_id=min_id, margin=margin)
        if flank < self.opt.min_aligned or radius < 0 or radius2 < 0:
            raise ValueError(f"flank must be >= {self.opt.min_aligned} and the radii >= 0")
        self.base, self.ref, self.assemblies = base, ref, assemblies
        self.ref_assembly = ref_assembly
        self.F, self.r1, self.r2 = flank, radius, radius2

    # sequence access ---------------------------------------------------------------------------
    def _ref(self, contig, a, b):
        return nt6(self.ref.fetch(contig, a, b))

    def _window(self, contig, L, sign, wlo, whi, qs, o, reversed_=False):
        wlo, whi = max(wlo, 0), min(whi, L)
        if whi <= wlo:
            return ST_SHORT, SnapRes(t_first=0, t_last=0), wlo
        if sign > 0:
            win = nt6(self.assemblies.fetch(contig, wlo, whi))
        else:
            win = revcomp6(nt6(self.assemblies.fetch(contig, L - whi, L - wlo)))
        if reversed_:
            win = win[::-1]
        res = snap_align(qs, np.ascontiguousarray(win), o)
        if reversed_ and res.t_first >= 0:
            m, nq = whi - wlo, len(qs)
            tf, qf = res.t_first, res.q_first
            res.t_first, res.t_last = m - 1 - res.t_last, m - 1 - tf
            res.q_first, res.q_last = nq - 1 - res.q_last, nq - 1 - qf
        return res.status, res, wlo

    def _walk(self, i, sign, contig, L, ref_contig, rst, ren, p):
        F, r1, r2, o = self.F, self.r1, self.r2, self.opt
        ilen = ren - rst
        tmax, t = 0, F
        while t + F <= ilen and t <= r2:
            tmax, t = t, t << 1
        if tmax == 0:
            return ST_LOWID, -1
        ra, rb = (rst, rst + tmax + F) if i == 0 else (ren - tmax - F, ren)
        rs = self._ref(ref_contig, ra, rb)
        status, y, t = ST_LOWID, -1, F
        while t <= tmax:
            a = rst + t if i == 0 else ren - t - F
            qf = rs[a - ra:a - ra + F]
            if i == 0:
                status, res, wlo = self._window(contig, L, sign, p - r2, p + t + F + r2, qf, o)
            else:
                status, res, wlo = self._window(contig, L, sign, p - t - F - r2, p + r2 + 1, qf, o)
            if status == ST_OK:
                y = wlo + (res.t_first if i == 0 else res.t_last)
                break
            if status == ST_AMBIG:
                break
            t <<= 1
        if y < 0:
            return status, -1
        so = replace(o, min_frac=0.0, min_aligned=F // 2, clip5=o.gap_open + o.gap_ext)
        a = rst if i == 0 else ren - t - F
        n = t + F
        qs = rs[a - ra:a - ra + n]
        if i == 1:
            qs = qs[::-1]
        if i == 0:
            wlo2, whi2 = y - t - r1, y + F + r1
        else:
            wlo2, whi2 = y - F - r1, y + t + r1 + 1
        status, res, wlo = self._window(contig, L, sign, wlo2, whi2, np.ascontiguousarray(qs), so, i == 1)
        if status == ST_OK:
            return status, wlo + (res.t_first if i == 0 else res.t_last)
        return status, -1

    def snap_end(self, i: int, sign: int, contig: str, L: int, ref_contig: str, rst: int, ren: int, cp: int):
        """One end (i=0 start, 1 end). cp = forward founder coordinate of the first/last reference base.
        Returns (label, shift, new_cp)."""
        F, o = self.F, self.opt
        a = rst if i == 0 else max(ren - F, rst)
        b = min(rst + F, ren) if i == 0 else ren
        label = LABEL_SHORT
        if b - a < o.min_aligned:
            return label, 0, cp
        fl = self._ref(ref_contig, a, b)
        p = cp if sign > 0 else L - 1 - cp
        p = min(max(p, 0), L - 1)
        u = -1
        for k, rad in enumerate((self.r1, self.r2)):
            if k == 1 and self.r2 <= self.r1:
                break
            st, res, wlo = self._window(contig, L, sign, p - rad - F, p + rad + F + 1, fl, o)
            if st == ST_OK:
                u = wlo + (res.t_first if i == 0 else res.t_last)
                label = LABEL_SNAP if k == 0 else LABEL_WIDE
                break
            label = {ST_AMBIG: LABEL_AMBIG, ST_LOWID: LABEL_LOWID}.get(st, LABEL_SHORT)
            if st != ST_LOWID:
                break
        if u < 0 and label == LABEL_LOWID and self.r2 > 0:
            st, u = self._walk(i, sign, contig, L, ref_contig, rst, ren, p)
            if st == ST_OK:
                label = LABEL_WALK
            elif st == ST_AMBIG:
                label = LABEL_AMBIG
        if u >= 0:
            x = u if sign > 0 else L - 1 - u
            return label, x - cp, x
        return label, 0, cp

    def snap(self, pr: Projection) -> Projection:
        return self._snap_v4(pr) if self.snap_version == "v4" else self._snap_v3(pr)

    def _snap_v3(self, pr: Projection) -> Projection:
        chrom = bare_chrom(pr.ref_chrom, self.ref_assembly)
        if pr.assembly == self.ref_assembly:
            return replace(pr, snap=(LABEL_IDENTITY, LABEL_IDENTITY), snap_shift=(0, 0))
        if not is_usable(pr.status):
            return pr
        contig, L = pr.contig, self.assemblies.length(pr.contig)
        sign = 1 if pr.strand == "+" else -1
        cp = [pr.start, pr.end - 1] if sign > 0 else [pr.end - 1, pr.start]
        ref_contig = f"{self.ref_assembly}_{chrom}"
        labels, shifts = [None, None], [0, 0]
        for i in (0, 1):
            labels[i], shifts[i], cp[i] = self.snap_end(i, sign, contig, L, ref_contig, pr.ref_start,
                                                        pr.ref_end, cp[i])
        st, en = (cp[0], cp[1] + 1) if sign > 0 else (cp[1], cp[0] + 1)
        extra = dict(snap=tuple(labels), snap_shift=tuple(shifts))
        if en <= st:
            return replace(pr, contig=None, start=None, end=None,
                           status="projection_failed: projected end precedes projected start", **extra)
        if st >= L:
            return replace(pr, contig=None, start=None, end=None,
                           status="projection_failed: projected start is past the contig end", **extra)
        status = OK
        clamped_end = 1 if sign > 0 else 0
        if en > L:
            en, status = L, OK_CLAMPED
        elif pr.status == OK_CLAMPED and labels[clamped_end] not in (LABEL_SNAP, LABEL_WIDE, LABEL_WALK):
            status = OK_CLAMPED
        return replace(pr, start=st, end=en, status=status, **extra)


    # v4: one shared computation per reference boundary ------------------------------------------
    def _walk4(self, x, left, sign, contig, L, ref_contig, Lr, p):
        F, r1, r2, o = self.F, self.r1, self.r2, self.opt
        tmax, t = 0, F
        while t <= r2 and (x - t - F >= 0 if left else x + t + F <= Lr):
            tmax, t = t, t << 1
        if tmax == 0:
            return ST_LOWID, -1
        ra, rb = (x - tmax - F, x) if left else (x, x + tmax + F)
        rs = self._ref(ref_contig, ra, rb)
        status, y, t = ST_LOWID, -1, F
        while t <= tmax:
            a = x - t - F if left else x + t
            qf = rs[a - ra:a - ra + F]
            if not left:
                status, res, wlo = self._window(contig, L, sign, p - r2, p + t + F + r2, qf, o)
            else:
                status, res, wlo = self._window(contig, L, sign, p - t - F - r2, p + r2 + 1, qf, o)
            if status == ST_OK:
                y = wlo + (res.t_last if left else res.t_first)
                break
            if status == ST_AMBIG:
                break
            t <<= 1
        if y < 0:
            return status, -1
        so = replace(o, min_frac=0.0, min_aligned=F // 2, clip5=o.gap_open + o.gap_ext)
        a = x - t - F if left else x
        n = t + F
        qs = rs[a - ra:a - ra + n]
        if left:
            qs = qs[::-1]
        if not left:
            wlo2, whi2 = y - t - r1, y + F + r1
        else:
            wlo2, whi2 = y - F - r1, y + t + r1 + 1
        status, res, wlo = self._window(contig, L, sign, wlo2, whi2, np.ascontiguousarray(qs), so, left)
        if status == ST_OK:
            return status, (wlo + res.t_first if not left else wlo + res.t_last)
        return status, -1

    def _side(self, x, left, sign, contig, L, ref_contig, Lr, p, walk):
        """snap_side: returns (label, u, edge)."""
        F, o = self.F, self.opt
        a = max(x - F, 0) if left else x
        b = x if left else min(x + F, Lr)
        nq = b - a
        if nq < o.min_aligned:
            return LABEL_SHORT, -1, False
        p = min(max(p, 0), L - 1)
        if walk:
            lab, u = LABEL_LOWID, -1
            if self.r2 > 0:
                st, u = self._walk4(x, left, sign, contig, L, ref_contig, Lr, p)
                if st == ST_OK:
                    lab = LABEL_WALK
                elif st == ST_AMBIG:
                    lab = LABEL_AMBIG
            return lab, u, False
        fl = self._ref(ref_contig, a, b)
        lab = LABEL_SHORT
        for k, rad in enumerate((self.r1, self.r2)):
            if k == 1 and self.r2 <= self.r1:
                break
            st, res, wlo = self._window(contig, L, sign, p - rad - F, p + rad + F + 1, fl, o)
            if st == ST_OK:
                u = wlo + (res.t_last if left else res.t_first)
                edge = res.q_last == nq - 1 if left else res.q_first == 0
                return (LABEL_SNAP if k == 0 else LABEL_WIDE), u, edge
            lab = {ST_AMBIG: LABEL_AMBIG, ST_LOWID: LABEL_LOWID}.get(st, LABEL_SHORT)
            if st != ST_LOWID:
                break
        return lab, -1, False

    def snap_boundary(self, x: int, is_start: bool, sign: int, contig: str, L: int, ref_contig: str, Lr: int,
                      u_raw: int):
        """Oriented founder fencepost of reference boundary x. Returns (label, U)."""
        def cross(lab):
            return _CROSS.get(lab, LABEL_X_WALK)
        U, uL, uR = u_raw, -1, -1
        labL = labR = LABEL_SHORT
        eL = eR = False
        if x > 0:
            labL, uL, eL = self._side(x, True, sign, contig, L, ref_contig, Lr, u_raw - 1, False)
        if is_start or uL < 0:
            labR, uR, eR = self._side(x, False, sign, contig, L, ref_contig, Lr, u_raw, False)
        if uL >= 0:
            if is_start and uR >= 0 and eL and eR and uR >= uL + 1:
                U, lab = uR, labR
            else:
                U, lab = uL + 1, (cross(labL) if is_start else labL)
        elif uR >= 0:
            U, lab = uR, (labR if is_start else cross(labR))
        else:
            lab = labR if labL == LABEL_SHORT else labL
            if labL == LABEL_LOWID:
                labL, uL, eL = self._side(x, True, sign, contig, L, ref_contig, Lr, u_raw - 1, True)
                if uL >= 0:
                    U, lab = uL + 1, (cross(labL) if is_start else labL)
            if uL < 0 and labR == LABEL_LOWID:
                labR, uR, eR = self._side(x, False, sign, contig, L, ref_contig, Lr, u_raw, True)
                if uR >= 0:
                    U, lab = uR, (labR if is_start else cross(labR))
            if uL < 0 and uR < 0:
                lab = labR if labL == LABEL_SHORT else labL
        return lab, U

    def _snap_v4(self, pr: Projection) -> Projection:
        chrom = bare_chrom(pr.ref_chrom, self.ref_assembly)
        if pr.assembly == self.ref_assembly:
            return replace(pr, snap=(LABEL_IDENTITY, LABEL_IDENTITY), snap_shift=(0, 0))
        if not is_usable(pr.status):
            return pr
        contig, L = pr.contig, self.assemblies.length(pr.contig)
        ref_contig = f"{self.ref_assembly}_{chrom}"
        Lr = self.ref.length(ref_contig)
        sign = 1 if pr.strand == "+" else -1
        # unsnapped projection = the raw fenceposts (C --no-snap: st = U0, en = U1 on '+'; L-U1, L-U0 on '-')
        uraw = [pr.start, pr.end] if sign > 0 else [L - pr.end, L - pr.start]
        maybe_clamped = [False, False]
        if pr.status == OK_CLAMPED:      # a clamped raw fencepost sits on 0 or L; its true value is lost
            maybe_clamped = [uraw[0] in (0, L), uraw[1] in (0, L)]
        x = (pr.ref_start, pr.ref_end)
        labels, U, shifts = [None, None], [0, 0], [0, 0]
        for i in (0, 1):
            labels[i], U[i] = self.snap_boundary(x[i], i == 0, sign, contig, L, ref_contig, Lr, uraw[i])
            shifts[i] = U[i] - uraw[i] if sign > 0 else uraw[i] - U[i]
        st, en = (U[0], U[1]) if sign > 0 else (L - U[1], L - U[0])
        extra = dict(snap=tuple(labels), snap_shift=tuple(shifts))

        def fail(msg):
            return replace(pr, contig=None, start=None, end=None, status=f"projection_failed: {msg}", **extra)
        if en == st:
            return fail("interval deleted in the founder (empty projection)")
        if en < st:
            return fail("projected end precedes projected start")
        clamped = any(maybe_clamped[i] and labels[i] not in SNAPPED_LABELS for i in (0, 1))
        if st < 0:
            st, clamped = 0, True
        if st >= L:
            return fail("projected start is past the contig end")
        if en > L:
            en, clamped = L, True
        return replace(pr, start=st, end=en, status=OK_CLAMPED if clamped else OK, **extra)

    def project(self, ref_chrom, ref_start, ref_end, assembly) -> Projection:
        return self.snap(self.base.project(ref_chrom, ref_start, ref_end, assembly))
