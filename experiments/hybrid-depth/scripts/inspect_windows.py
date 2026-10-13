"""Inspect wrong windows of an in-panel hybrid: per predicted segment, read support of the
predicted pair vs the true pair (rows matching each founder), and the emission score gap."""
import sys, numpy as np, torch
import hd_lib as L
ds, ind, dep = sys.argv[1], sys.argv[2], float(sys.argv[3])
m = L.load_model(); inp = L.load_input("eval", ds, ind, dep)
lo, hi, info = L.decode_custom(m, inp, capture=True)
gam = inp["gametes"]; A, B = [gam.index(x) for x in ind.split("x")]
tern = inp["data"][:, :, :L.K]
N, T = lo.shape
ok = ((lo == min(A, B)) & (hi == max(A, B)))
wrongw = np.nonzero((~ok).mean(1) > 0.5)[0]
rng = np.random.default_rng(0)
ll = m.tern_loglik.detach().cpu().numpy()   # del, div, match
def pair_score(t, i, j):
    a, b = ll[t[:, i] + 1], ll[t[:, j] + 1]
    return float((np.logaddexp(a, b) - np.log(2)).sum())
tot = dict(n=0)
agg = []
for w in rng.choice(wrongw, min(400, len(wrongw)), replace=False):
    t = tern[w].astype(int)
    pl, ph = lo[w], hi[w]
    # segments
    cut = np.r_[0, np.nonzero((pl[1:] != pl[:-1]) | (ph[1:] != ph[:-1]))[0] + 1, T]
    for s, e in zip(cut[:-1], cut[1:]):
        i, j = pl[s], ph[s]
        if (i, j) == (min(A, B), max(A, B)):
            continue
        seg = t[s:e]
        sp = pair_score(seg, i, j); st = pair_score(seg, A, B)
        agg.append((e - s, sp - st, (seg[:, A] == 1).sum(), (seg[:, B] == 1).sum(), (seg[:, i] == 1).sum(), (seg[:, j] == 1).sum(),
                    ((seg[:, A] == 1) & (seg[:, B] != 1)).sum(), ((seg[:, B] == 1) & (seg[:, A] != 1)).sum(), int(i == j)))
a = np.array(agg, float)
print(f"{ind} {dep}x: {len(wrongw)}/{N} windows >50% wrong; sampled {len(a)} wrong segments")
print("seg len rows: median %.0f mean %.0f | emission gap pred-true (nats): median %.2f mean %.2f, frac pred>=true %.3f" % (
    np.median(a[:, 0]), a[:, 0].mean(), np.median(a[:, 1]), a[:, 1].mean(), (a[:, 1] >= 0).mean()))
print("per segment rows: A-only %.1f  B-only %.1f (medians %.0f/%.0f) | homozygous segments %.2f" % (
    a[:, 6].mean(), a[:, 7].mean(), np.median(a[:, 6]), np.median(a[:, 7]), a[:, 8].mean()))
big = a[:, 0] >= 100
print("segments >=100 rows: n=%d, gap median %.2f, A-only %.1f B-only %.1f" % (big.sum(), np.median(a[big, 1]), a[big, 6].mean(), a[big, 7].mean()))
