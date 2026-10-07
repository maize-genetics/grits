"""SnapProvider / snap_align tests.

The aligner is checked against a literal scalar port of snap.c's snap_align_dp (below), and every
snapping case against the truth convention of simulate_pangenome.py on hand-built founders.
"""
import random

import numpy as np
import pytest

from python.seq_extract.assembly import AssemblySet
from python.seq_extract.fai import revcomp
from python.seq_extract.haplotype import BedInterval, HaplotypeBuilder, Window, group_bed
from python.seq_extract.projection import OK, OK_CLAMPED, Projection, TableProvider
from python.seq_extract.snap import (INT32_MIN, ST_AMBIG, ST_LOWID, ST_OK, ST_SHORT, SnapOpt, SnapProvider, _dp,
                                     nt6, revcomp6, snap_align)


# --------------------------------------------------------------------------- scalar reference port of snap.c

def ref_dp(q, t, o):
    NEG = -(1 << 29)
    n, m = len(q), len(t)
    res = dict(score=INT32_MIN, second=INT32_MIN, t_end=-1, q_first=-1, t_first=-1, q_last=-1, t_last=-1,
               n_aligned=0, n_match=0, identity=0.0)
    need = max(int(o.min_frac * n), o.min_aligned)
    if n <= 0 or m <= 0 or n < o.min_aligned or m < need:
        return res, ST_SHORT
    oe, ge = o.gap_open + o.gap_ext, o.gap_ext
    H0, F = [0] * (m + 1), [NEG] * (m + 1)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        H1 = [0] * (m + 1)
        E, qc = NEG, q[i - 1]
        F[0] = -oe if i == 1 else F[0] - ge
        H1[0] = F[0]
        d[i][0] = 2 | (8 if i > 1 else 0)
        for j in range(1, m + 1):
            eo, ee = H1[j - 1] - oe, E - ge
            fo, fe = H0[j] - oe, F[j] - ge
            sc = o.match if (qc == t[j - 1] and 1 <= qc <= 4) else -o.mismatch
            b = 0
            if ee > eo:
                E, b = ee, b | 4
            else:
                E = eo
            if fe > fo:
                F[j], b = fe, b | 8
            else:
                F[j] = fo
            h = H0[j - 1] + sc
            if o.clip5 >= 0 and i > 1 and -o.clip5 + sc > h:
                h, b = -o.clip5 + sc, b | 16
            if E > h:
                h, b = E, (b & ~19) | 1
            if F[j] > h:
                h, b = F[j], (b & ~19) | 2
            H1[j], d[i][j] = h, b
        H0 = H1
    best = -1
    for j in range(1, m + 1):
        if best < 0 or H0[j] > H0[best]:
            best = j
    res["score"], res["t_end"] = H0[best], best
    for j in range(1, m + 1):
        if abs(j - best) >= o.excl and H0[j] > res["second"]:
            res["second"] = H0[j]
    col, state, i, j, last = [], 0, n, best, False
    while i > 0:
        b = d[i][j]
        if state == 0:
            src = b & 3
            if src == 0:
                col.append("=" if (q[i - 1] == t[j - 1] and 1 <= q[i - 1] <= 4) else "X")
                if not last:
                    res["q_last"], res["t_last"], last = i - 1, j - 1, True
                res["q_first"], res["t_first"] = i - 1, j - 1
                if b & 16:
                    break
                i, j = i - 1, j - 1
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
    ms = [k for k, c in enumerate(col) if c in "=X"]
    if ms:
        n_open = 0
        for k in range(ms[0], ms[-1] + 1):
            if col[k] in "=X":
                res["n_aligned"] += 1
                res["n_match"] += col[k] == "="
            elif k == ms[0] or col[k - 1] != col[k]:
                n_open += 1
        res["identity"] = res["n_match"] / (res["n_aligned"] + n_open)
    if res["n_aligned"] < need or res["identity"] < o.min_id:
        st = ST_LOWID
    elif res["second"] != INT32_MIN and res["second"] >= res["score"] - o.margin:
        st = ST_AMBIG
    else:
        st = ST_OK
    return res, st


FIELDS = ["score", "second", "t_end", "q_first", "t_first", "q_last", "t_last", "n_aligned", "n_match", "identity"]


def rand6(rng, n, alphabet=(1, 2, 3, 4)):
    return np.array([rng.choice(alphabet) for _ in range(n)], np.uint8)


@pytest.mark.parametrize("clip5", [-1, 7])
@pytest.mark.parametrize("seed", range(6))
def test_vectorised_dp_equals_scalar_port(seed, clip5):
    rng = random.Random(seed)
    o = SnapOpt(clip5=clip5, min_aligned=4)
    for _ in range(25):
        alpha = rng.choice([(1, 2), (1, 2, 3, 4), (1, 2, 3, 4, 5)])
        n, m = rng.randint(4, 14), rng.randint(4, 30)
        q = rand6(rng, n, alpha)
        if rng.random() < 0.5 and m > n:
            t = rand6(rng, m, alpha)
            p = rng.randint(0, m - n)
            t[p:p + n] = q
            for _ in range(rng.randint(0, 3)):
                t[rng.randrange(m)] = rng.choice(alpha)
        else:
            t = rand6(rng, m, alpha)
        want, st = ref_dp(list(q), list(t), o)
        got = _dp(q, t, o)
        assert got.status == st, (q, t)
        if st != ST_SHORT:
            assert {f: getattr(got, f) for f in FIELDS} == want, (q, t)


def test_dp_basic_cases():
    o = SnapOpt()
    rng = random.Random(1)
    q = rand6(rng, 64)
    t = np.concatenate([rand6(rng, 100), q, rand6(rng, 100)])
    r = snap_align(q, t, o)
    assert (r.status, r.score, r.t_first, r.t_last, r.t_end, r.identity) == (ST_OK, 64, 100, 163, 164, 1.0)
    t2 = t.copy()
    t2[130] = 5                                                      # N never matches
    r = snap_align(q, t2, o)
    assert r.score == 64 - 1 - 4 and r.n_match == 63 and r.status == ST_OK
    assert snap_align(q[:10], t, o).status == ST_SHORT
    assert snap_align(q, t[:20], o).status == ST_SHORT
    rr = snap_align(q, rand6(rng, 300), o)
    assert rr.status == ST_LOWID


def test_dp_ambiguous_repeat_and_leftmost():
    o = SnapOpt()
    rng = random.Random(2)
    q = rand6(rng, 64)
    t = np.concatenate([rand6(rng, 50), q, rand6(rng, 30), q, rand6(rng, 50)])
    r = snap_align(q, t, o)
    assert r.status == ST_AMBIG and r.t_end == 50 + 64 and r.second == r.score


def test_dp_gap_and_identity():
    o = SnapOpt()
    rng = random.Random(3)
    q = rand6(rng, 64)
    t = np.concatenate([rand6(rng, 40), q[:30], rand6(rng, 3), q[30:], rand6(rng, 40)])
    r = snap_align(q, t, o)
    assert r.status == ST_OK and r.score == 64 - 6 - 3 and r.t_first == 40 and r.t_last == 40 + 66
    assert r.identity == pytest.approx(64 / 65)


def test_seed_path_matches_full_dp_on_planted_homologs():
    rng = random.Random(4)
    for k in range(30):
        q = rand6(rng, 64)
        m = rng.choice([600, 3000, 12000])
        t = rand6(rng, m)
        p = rng.randrange(0, m - 80)
        hom = list(q)
        for _ in range(rng.randint(0, 4)):
            hom[rng.randrange(64)] = rng.randint(1, 4)
        if rng.random() < 0.3:
            del hom[rng.randrange(64)]
        t[p:p + len(hom)] = hom
        a = snap_align(q, t, SnapOpt())
        b = snap_align(q, t, SnapOpt(no_seed=True))
        assert (a.status, a.t_first, a.t_last, a.t_end, a.score) == (b.status, b.t_first, b.t_last, b.t_end, b.score)


def test_seed_no_shared_kmer_is_lowid():
    q = np.array([1, 2] * 32, np.uint8)
    t = np.array([3, 4] * 400, np.uint8)
    assert snap_align(q, t).status == ST_LOWID


def test_nt6_and_revcomp6():
    assert list(nt6("ACGTacgtNRx")) == [1, 2, 3, 4, 1, 2, 3, 4, 5, 5, 5]
    assert list(revcomp6(nt6("AACGN"))) == list(nt6("NCGTT"))


# --------------------------------------------------------------------------- snapping cases

class Seqs:
    def __init__(self, d):
        self.d = d

    def fetch(self, contig, s, e, strand="+", upper=False):
        if s < 0 or e > len(self.d[contig]) or e <= s:
            raise ValueError((contig, s, e))
        x = self.d[contig][s:e]
        return revcomp(x) if strand == "-" else x

    def length(self, contig):
        return len(self.d[contig])


class Fixed:
    def __init__(self, pr):
        self.pr = pr

    def project(self, *a):
        return self.pr


def rseq(rng, n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def derive(R, edits):
    """edits: (pos, 'snp', base) | (pos, 'ins', seq) inserted before ref base pos | (pos, 'del', length)."""
    by = {}
    for e in edits:
        by.setdefault(e[0], []).append(e)
    out, mp, fpos, i = [], np.full(len(R) + 1, -1, np.int64), 0, 0
    while i < len(R):
        b = R[i]
        skip = 0
        for _, kind, x in by.get(i, []):
            if kind == "ins":
                out.append(x)
                fpos += len(x)
            elif kind == "del":
                skip = x
            elif kind == "snp":
                b = x
        if skip:
            i += skip
            continue
        mp[i] = fpos
        out.append(b)
        fpos += 1
        i += 1
    mp[len(R)] = fpos
    return "".join(out), mp


def truth(mp, s, e):
    pres = np.flatnonzero(mp[s:e] >= 0)
    return int(mp[s + pres[0]]), int(mp[s + pres[-1]]) + 1


RNG = random.Random(99)
R = rseq(RNG, 6000)


def run(edits, rst=1000, ren=3000, d0=0, d1=0, strand="+", **kw):
    fs, mp = derive(R, edits)
    ts, te = truth(mp, rst, ren)
    if strand == "+":
        seqs = Seqs({"B73_chr1": R, "A_chr1": fs})
        raw = Projection("chr1", rst, ren, "A", "A_chr1", ts + d0, te + d1, "+", OK)
        want = (ts, te)
    else:
        L = len(fs)
        seqs = Seqs({"B73_chr1": R, "A_chr1": revcomp(fs)})
        # '-': forward start = homolog of the reference end, forward end - 1 = homolog of the start
        raw = Projection("chr1", rst, ren, "A", "A_chr1", L - te - d1, L - ts - d0, "-", OK)
        want = (L - te, L - ts)
    sp = SnapProvider(Fixed(raw), seqs, seqs, **kw)
    return sp.project("chr1", rst, ren, "A"), want, raw


def test_exact_projection_unchanged():
    p, want, _ = run([])
    assert (p.start, p.end) == want and p.snap == ("snap", "snap") and p.snap_shift == (0, 0)


def test_snps_in_flank_and_offsets():
    p, want, _ = run([(1005, "snp", "A" if R[1005] != "A" else "C"), (1030, "snp", "G" if R[1030] != "G" else "T"),
                      (2970, "snp", "A" if R[2970] != "A" else "C")], d0=7, d1=-12)
    assert (p.start, p.end) == want and p.snap == ("snap", "snap") and p.snap_shift == (-7, 12)


@pytest.mark.parametrize("edit", [(2980, "ins", "ACGTTGCAAGTC"), (2975, "del", 8), (1020, "ins", "GGATCC"),
                                  (1010, "del", 5)])
def test_small_indel_near_end(edit):
    p, want, raw = run([edit], d0=3, d1=-4)
    assert (p.start, p.end) == want and p.snap == ("snap", "snap")


def test_deleted_end_base_walks_over():
    p, want, _ = run([(2995, "del", 10)], d1=3)
    assert (p.start, p.end) == want and p.snap[1] == "snap"
    p, want, _ = run([(997, "del", 10)], d0=-2)
    assert (p.start, p.end) == want and p.snap[0] == "snap"


def test_deletion_boundary_residual_v3():
    # v3 limitation (README: "deletion-boundary placements"): with ref [995,1005) deleted, flank bases
    # 1000-1004 align by chance to founder bases left of the true start, scoring better than the 5 bp
    # terminal deletion, so the start lands 3 bp early. v4 places it from the left flank instead.
    p, want, _ = run([(995, "del", 10)], d0=-2, snap_version="v3")
    assert want[0] == 995 and p.start == 992 and p.snap[0] == "snap"
    p, want, _ = run([(995, "del", 10)], d0=-2)
    assert p.start == want[0] == 995 and p.snap[0] == "snap_x"


def test_founder_only_bases_next_to_end_excluded():
    p, want, _ = run([(3000, "ins", "TTGACCA"), (1000, "ins", "CCAGT")])
    assert (p.start, p.end) == want


def test_equal_base_insertion_maps_to_own_copy():
    e_base, s_base = R[2999], R[1000]
    p, want, _ = run([(3000, "ins", e_base + "ACGTA"), (1000, "ins", "TGCA" + s_base)], d0=1, d1=1)
    fs, mp = derive(R, [(3000, "ins", e_base + "ACGTA"), (1000, "ins", "TGCA" + s_base)])
    assert want == (int(mp[1000]), int(mp[2999]) + 1)
    assert (p.start, p.end) == want


def test_repeat_gives_ambiguity_and_keeps_projection_v3():
    p, want, raw = run([(900, "ins", R[1000:1064])], d0=2, snap_version="v3")
    assert p.snap[0] == "proj_ambiguous" and p.start == raw.start and p.snap_shift[0] == 0
    assert p.snap[1] == "snap" and p.end == want[1]


def test_minus_strand_locus():
    edits = [(1003, "snp", "A" if R[1003] != "A" else "C"), (2980, "ins", "ACGTAC"), (1020, "del", 3)]
    p, want, raw = run(edits, d0=5, d1=-9, strand="-")
    assert p.strand == "-" and (p.start, p.end) == want and p.snap == ("snap", "snap")
    assert p.snap_shift == (-5, 9) or p.snap_shift == (5, -9)


def test_flank_shorter_than_F_and_too_short_v3():
    p, want, _ = run([], rst=1000, ren=1040, d0=3, d1=-3, snap_version="v3")
    assert (p.start, p.end) == want and p.snap == ("snap", "snap")
    p, want, raw = run([], rst=1000, ren=1010, d0=3, d1=-3, snap_version="v3")
    assert p.snap == ("proj_short", "proj_short") and (p.start, p.end) == (raw.start, raw.end)


def test_window_clamped_at_founder_contig_start():
    p, want, _ = run([], rst=5, ren=2000, d0=-5)
    assert (p.start, p.end) == want and p.snap[0] == "snap"


def test_wide_fallback():
    p, want, _ = run([(1500, "del", 50)], d0=1500, d1=-2500)
    assert (p.start, p.end) == want and p.snap == ("snap_wide", "snap_wide")


def test_low_identity_keeps_projection():
    p, want, raw = run([], d0=1500, radius2=0)
    assert p.snap[0] == "proj_lowid" and p.start == raw.start
    rng = random.Random(5)
    seqs = Seqs({"B73_chr1": R, "X_chr1": rseq(rng, 5000)})
    raw = Projection("chr1", 1000, 3000, "X", "X_chr1", 1000, 3000, "+", OK)
    p = SnapProvider(Fixed(raw), seqs, seqs, radius2=500).project("chr1", 1000, 3000, "X")
    assert p.snap == ("proj_lowid", "proj_lowid") and (p.start, p.end) == (1000, 3000)


@pytest.mark.parametrize("strand", ["+", "-"])
def test_walk_end_and_start_v3(strand):
    p, want, _ = run([(2880, "del", 150)], d1=5, strand=strand, snap_version="v3")
    assert (p.start, p.end) == want and p.snap == ("snap", "snap_walk")
    p, want, _ = run([(930, "del", 200)], d0=-3, strand=strand, snap_version="v3")
    assert (p.start, p.end) == want and p.snap == ("snap_walk", "snap")


def test_walk_partial_flank_falls_back_like_c():
    # same in v3 and v4
    # ref [2900,3100) deleted: the first inward flank [2872,2936) keeps 28 present bases plus 9 chance-aligned
    # ones (37 >= 32, accepted), so the stretch [2872,3000) has < F/2 = 32 alignable bases -> proj_lowid.
    p, want, raw = run([(2900, "del", 200)], d1=5)
    assert p.snap[1] == "proj_lowid" and p.end == raw.end and p.snap_shift[1] == 0


def test_identity_and_failed_passthrough():
    seqs = Seqs({"B73_chr1": R})
    pr = Projection("chr1", 10, 100, "B73", "B73_chr1", 10, 100, "+", OK)
    p = SnapProvider(Fixed(pr), seqs, seqs).project("chr1", 10, 100, "B73")
    assert p.snap == ("identity", "identity") and (p.start, p.end) == (10, 100)
    bad = Projection("chr1", 10, 100, "A", status="deleted")
    assert SnapProvider(Fixed(bad), seqs, seqs).project("chr1", 10, 100, "A") == bad


def test_clamped_end_resolution():
    fs, mp = derive(R, [])
    short = fs[:2950]
    seqs = Seqs({"B73_chr1": R, "A_chr1": short})
    raw = Projection("chr1", 1000, 3000, "A", "A_chr1", 1000, 2950, "+", OK_CLAMPED)
    p = SnapProvider(Fixed(raw), seqs, seqs).project("chr1", 1000, 3000, "A")
    assert p.snap[1] == "snap_walk" and p.end == 2950 and p.status == OK
    p = SnapProvider(Fixed(raw), seqs, seqs, radius2=0).project("chr1", 1000, 3000, "A")
    assert p.snap[1] == "proj_lowid" and p.status == OK_CLAMPED and p.clamped


def test_invalid_params():
    with pytest.raises(ValueError):
        SnapProvider(None, None, None, flank=8)
    with pytest.raises(ValueError):
        SnapProvider(None, None, None, radius=-1)


def test_snap_through_builder_with_table(tmp_path):
    fs, mp = derive(R, [(2000, "ins", "ACGTACGTAC"), (2500, "del", 7)])
    d = tmp_path / "fa"
    d.mkdir()
    for name, s in (("B73", R), ("A", fs)):
        (d / f"{name}.fa").write_text(f">{name}_chr1\n{s}\n")
    asm = AssemblySet.from_dir(d, "{asm}.fa", fai_dir=tmp_path / "fai", build_missing=True)
    ts, te = truth(mp, 1000, 3000)
    tab = TableProvider([Projection("chr1", 1000, 3000, "A", "A_chr1", ts - 4, te + 6, "+", OK)])
    b = HaplotypeBuilder(group_bed([BedInterval("chr1", 0, 6000, "A", "A")]), asm, SnapProvider(tab, asm, asm))
    r = b.build(Window("w", "chr1", 1000, 3000), 1)
    assert r.seq == fs[ts:te] and r.segments[0].snap == ("snap", "snap")
    assert "[snap,snap]" in r.segments[0].describe()


# --------------------------------------------------------------------------- v4: shared boundary computation

def pieces(edits, m, a=1000, b=3000, strand="+", d=(3, -4, 5, -6), **kw):
    """Project [a,m) and [m,b) separately (raw ends offset by d) and return both plus the truth fenceposts."""
    left, wl, _ = run(edits, a, m, d[0], d[1], strand, **kw)
    right, wr, _ = run(edits, m, b, d[2], d[3], strand, **kw)
    return left, right, wl, wr


def junction(left, right, strand):
    return (left.end, right.start) if strand == "+" else (left.start, right.end)


@pytest.mark.parametrize("strand", ["+", "-"])
def test_v4_deletion_spanning_boundary_shared_junction(strand):
    # exact case: both pieces meet at the deletion junction
    left, right, wl, wr = pieces([(1994, "del", 9)], 2000, strand=strand)
    assert (left.start, left.end) == wl and (right.start, right.end) == wr
    j = junction(left, right, strand)
    assert j[0] == j[1]
    # typical case (as in C): the deleted tail of the placed flank is aligned as mismatches/chance matches
    # (a terminal 1 bp gap costs 7, a mismatch 4), so the shared junction is a few bp off the truth --
    # but the two pieces still meet exactly
    left, right, wl, wr = pieces([(1985, "del", 25)], 2000, strand=strand)
    j = junction(left, right, strand)
    assert j[0] == j[1] and right.snap[0] == "snap"


@pytest.mark.parametrize("strand", ["+", "-"])
def test_v4_insertion_at_boundary_in_neither_piece(strand):
    ins = "ACGTTTGCA"
    left, right, wl, wr = pieces([(2000, "ins", ins)], 2000, strand=strand)
    assert (left.start, left.end) == wl and (right.start, right.end) == wr
    j = junction(left, right, strand)
    assert abs(j[1] - j[0]) == len(ins)
    assert left.snap[1] == "snap" and right.snap[0] == "snap"


class StubSides(SnapProvider):
    """snap_boundary with scripted flank placements: sides[(left, walk)] = (label, u, edge)."""

    def __init__(self, sides):
        super().__init__(None, None, None)
        self.sides, self.calls = sides, []

    def _side(self, x, left, sign, contig, L, ref_contig, Lr, p, walk):
        self.calls.append((left, walk, p))
        return self.sides.get((left, walk), ("proj_short", -1, False))


def boundary(sides, is_start, x=500, u_raw=1000):
    sp = StubSides(sides)
    lab, U = sp.snap_boundary(x, is_start, 1, "A_chr1", 5000, "B73_chr1", 5000, u_raw)
    return lab, U, sp.calls


def test_v4_rule1_both_flanks_insertion_gap_in_neither_piece():
    sides = {(True, False): ("snap", 900, True), (False, False): ("snap_wide", 910, True)}
    assert boundary(sides, True)[:2] == ("snap_wide", 910)          # start = right flank
    lab, U, calls = boundary(sides, False)
    assert (lab, U) == ("snap", 901) and [c[0] for c in calls] == [True]   # end: left only
    assert calls[0][2] == 999                                       # left side probes u_raw - 1


@pytest.mark.parametrize("eL,eR,uR", [(False, True, 910), (True, False, 910), (True, True, 900)])
def test_v4_rule2_left_defines_both_on_deletion_replacement_or_overlap(eL, eR, uR):
    # deletion at x (an edge base absent), replacement, or the right flank overlapping the left:
    # the left placement gives the single junction, so replacement bases go to the right piece
    sides = {(True, False): ("snap", 900, eL), (False, False): ("snap", uR, eR)}
    assert boundary(sides, True)[:2] == ("snap_x", 901)
    assert boundary(sides, False)[:2] == ("snap", 901)


def test_v4_rule2_right_only():
    sides = {(True, False): ("proj_ambiguous", -1, False), (False, False): ("snap_wide", 950, True)}
    assert boundary(sides, True)[:2] == ("snap_wide", 950)
    assert boundary(sides, False)[:2] == ("snap_x_wide", 950)


def test_v4_rule3_walks_left_then_right():
    low = {(True, False): ("proj_lowid", -1, False), (False, False): ("proj_lowid", -1, False)}
    lab, U, calls = boundary({**low, (True, True): ("snap_walk", 870, False)}, False)
    assert (lab, U) == ("snap_walk", 871) and (True, True, 999) in calls and (False, True, 1000) not in calls
    lab, U, _ = boundary({**low, (True, True): ("snap_walk", 870, False)}, True)
    assert (lab, U) == ("snap_x_walk", 871)
    lab, U, calls = boundary({**low, (True, True): ("proj_lowid", -1, False),
                              (False, True): ("snap_walk", 1030, False)}, False)
    assert (lab, U) == ("snap_x_walk", 1030) and (False, True, 1000) in calls
    lab, U, _ = boundary({**low, (True, True): ("proj_ambiguous", -1, False),
                          (False, True): ("proj_lowid", -1, False)}, True)
    assert (lab, U) == ("proj_ambiguous", 1000)


def test_v4_rule4_keeps_projection_and_x0():
    lab, U, calls = boundary({(True, False): ("proj_ambiguous", -1, False),
                              (False, False): ("proj_lowid", -1, False)}, True)
    # an ambiguous left side is not walked; the low-identity right side still is
    assert (lab, U) == ("proj_ambiguous", 1000) and [(c[0], c[1]) for c in calls] == [(True, False), (False, False), (False, True)]
    lab, U, calls = boundary({(False, False): ("snap", 1003, True)}, True, x=0, u_raw=1000)
    assert (lab, U) == ("snap", 1003) and [c[0] for c in calls] == [False]   # no left flank at x = 0
    lab, U, _ = boundary({}, False, x=0)
    assert (lab, U) == ("proj_short", 1000)


def test_v4_replacement_sequence_level():
    rng = random.Random(3)
    ins = "".join(rng.choice("ACGT") for _ in range(30))
    edits = [(1990, "del", 20), (2010, "ins", ins)]
    left, right, wl, wr = pieces(edits, 2000)
    assert left.start == wl[0] and right.end == wr[1]
    assert left.end <= right.start


def test_v4_low_identity_flank_on_one_side_uses_the_other():
    rng = random.Random(11)
    noisy = [(k, "snp", rng.choice([c for c in "ACGT" if c != R[k]])) for k in range(2000, 2064, 3)]
    left, right, wl, wr = pieces(noisy, 2000, d=(0, -9, 9, 0))
    assert right.start == wr[0] and left.end == wl[1]
    assert right.snap[0] == "snap_x" and left.snap[1] == "snap"
    noisy = [(k, "snp", rng.choice([c for c in "ACGT" if c != R[k]])) for k in range(1936, 2000, 3)]
    left, right, wl, wr = pieces(noisy, 2000, d=(0, -9, 9, 0))
    assert left.end == wl[1] and right.start == wr[0]
    assert left.snap[1] == "snap_x" and right.snap[0] == "snap"


def test_v4_start_projects_with_both_flanks_and_end_fallback():
    p, want, _ = run([(2970, "ins", "GGGG")], d0=4, d1=-3)
    assert (p.start, p.end) == want and p.snap == ("snap", "snap")


@pytest.mark.parametrize("strand", ["+", "-"])
def test_v4_interval_inside_deletion_fails(strand):
    fs, mp = derive(R, [(1000, "del", 300)])
    seqs = Seqs({"B73_chr1": R, "A_chr1": fs if strand == "+" else revcomp(fs)})
    L, j = len(fs), int(mp[1300])
    if strand == "+":
        raw = Projection("chr1", 1100, 1200, "A", "A_chr1", j - 30, j + 40, "+", OK)
    else:
        raw = Projection("chr1", 1100, 1200, "A", "A_chr1", L - j - 40, L - j + 30, "-", OK)
    p = SnapProvider(Fixed(raw), seqs, seqs).project("chr1", 1100, 1200, "A")
    assert p.status == "projection_failed: interval deleted in the founder (empty projection)"
    assert p.start is None and p.snap is not None


@pytest.mark.parametrize("strand", ["+", "-"])
def test_v4_walk_both_sides(strand):
    p, want, _ = run([(2800, "del", 300)], d1=7, strand=strand)
    assert (p.start, p.end) == want and p.snap[1] == "snap_walk"
    p, want, _ = run([(2900, "del", 300)], d1=7, strand=strand)
    assert (p.start, p.end) == want and p.snap[1] == "snap_x_walk"
    p, want, _ = run([(800, "del", 300)], d0=-4, strand=strand)
    assert (p.start, p.end) == want and p.snap[0] == "snap_x_walk"


def test_v4_walk_not_bounded_by_interval():
    # a 100 bp interval whose end sits in a 400 bp deletion: v3 cannot walk (t + F > interval), v4 can
    p3, want, _ = run([(1050, "del", 400)], rst=1000, ren=1100, d1=6, snap_version="v3")
    p4, want, _ = run([(1050, "del", 400)], rst=1000, ren=1100, d1=6)
    assert p3.snap[1] == "proj_lowid" and p3.end != want[1]
    assert p4.end == want[1] and p4.snap[1] in ("snap_walk", "snap_x_walk", "snap_x")


def test_v4_ambiguous_repeat_covering_both_flanks():
    p, want, raw = run([(900, "ins", R[930:1070])], d0=2)
    assert p.snap[0] == "proj_ambiguous" and p.start == raw.start and p.snap_shift[0] == 0


def test_v4_short_flanks_near_reference_contig_start():
    p, want, _ = run([], rst=5, ren=2000, d0=2)
    assert p.start == want[0] and p.snap[0] == "snap"          # left flank [0,5) too short, right placed
    p, want, _ = run([], rst=0, ren=2000, d0=0)
    assert p.start == 0 and p.snap[0] == "snap"
    seqs = Seqs({"B73_chr1": R[:20], "A_chr1": R[:20]})
    raw = Projection("chr1", 10, 15, "A", "A_chr1", 10, 15, "+", OK)
    p = SnapProvider(Fixed(raw), seqs, seqs).project("chr1", 10, 15, "A")
    assert p.snap == ("proj_short", "proj_short") and (p.start, p.end) == (10, 15)


def test_snap_version_validation():
    with pytest.raises(ValueError):
        SnapProvider(None, None, None, snap_version="v2")
