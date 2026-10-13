#!/usr/bin/env python
"""Verify gap-filled BEDs (fill_switch_gaps=True) against the original BEDs of the same decode.

For every sample of OLD_TAG / NEW_TAG (per scope dir): per chromosome file, the interval list must have
the same length and the same founder pair in the same order (= same decoded path, since the BED is a
deterministic function of the preds); each new interval must contain the old one, with the first start
and last end of each chromosome unchanged, and every changed boundary must be a closed switch gap
(new end[i] == new start[i+1], split inside the old gap). Also checks that the panel-site founder calls
are identical on every site the old BED called. Prints per-sample gaps filled, gaps left open and bp
gained, and writes a TSV.
  verify_gapfill_beds.py OLD_TAG NEW_TAG OUT.tsv
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fast_snprc_score as fss  # noqa: E402

WORK = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")


def main(old_tag, new_tag, out_tsv):
    P = fss.Panel()
    recs = []
    for sc in ("idx", "mix", "out"):
        for od in sorted(glob.glob(str(WORK / "scratch" / f"{old_tag}_{sc}_fast" / "*" / "bed"))):
            key = Path(od).parent.name
            nd = WORK / "scratch" / f"{new_tag}_{sc}_fast" / key.replace(old_tag + "__", new_tag + "__", 1) / "bed"
            _, ds, ind, dep = key.split("__")
            r = dict(dataset=ds, individual=ind, depth=dep, ok=True, problems="")
            if not nd.is_dir():
                r.update(ok=False, problems="missing new bed dir"); recs.append(r); continue
            ofiles = sorted(Path(od).glob("*.bed")); nfiles = sorted(nd.glob("*.bed"))
            if [f.name for f in ofiles] != [f.name for f in nfiles]:
                r.update(ok=False, problems="file set differs"); recs.append(r); continue
            n_int = filled = open_gaps = 0
            old_bp = new_bp = 0
            probs = []
            for fo, fn in zip(ofiles, nfiles):
                o = pd.read_csv(fo, sep="\t"); n = pd.read_csv(fn, sep="\t")
                if len(o) != len(n) or not (o[["chrom", "parent1", "parent2"]].values == n[["chrom", "parent1", "parent2"]].values).all():
                    probs.append(f"{fo.name}: path differs"); continue
                os_, oe, ns, ne = (x.to_numpy(np.int64) for x in (o.start, o.end, n.start, n.end))
                n_int += len(o)
                old_bp += int((oe - os_).sum()); new_bp += int((ne - ns).sum())
                if (ns > os_).any() or (ne < oe).any():
                    probs.append(f"{fo.name}: new interval shrinks")
                if ns[0] != os_[0] or ne[-1] != oe[-1]:
                    probs.append(f"{fo.name}: chromosome ends moved")
                gap = os_[1:] > oe[:-1]
                closed = ne[:-1] == ns[1:]
                changed = (ne[:-1] != oe[:-1]) | (ns[1:] != os_[1:])
                if (changed & ~(gap & closed)).any():
                    probs.append(f"{fo.name}: boundary changed outside a closed gap")
                if (closed & gap & ((ne[:-1] < oe[:-1]) | (ne[:-1] > os_[1:]))).any():
                    probs.append(f"{fo.name}: split point outside old gap")
                filled += int((gap & closed).sum()); open_gaps += int((gap & ~closed).sum())
            # same founder calls on every site the old BED called
            a1, a2 = fss.pair_from_bed(P, Path(od)); b1, b2 = fss.pair_from_bed(P, nd)
            oc = (a1 >= 0) & (a2 >= 0); nc = (b1 >= 0) & (b2 >= 0)
            if not ((a1[oc] == b1[oc]).all() and (a2[oc] == b2[oc]).all()):
                probs.append("site calls differ on old-called sites")
            if (oc & ~nc).any():
                probs.append("sites lost")
            r.update(ok=not probs, problems="; ".join(probs), intervals=n_int, switch_gaps=filled + open_gaps,
                     gaps_filled=filled, gaps_open=open_gaps, old_bp=old_bp, new_bp=new_bp, bp_gained=new_bp - old_bp,
                     old_cov=old_bp / 2.130e9, new_cov=new_bp / 2.130e9,
                     sites_old=int(oc.sum()), sites_new=int(nc.sum()), sites_gained=int((nc & ~oc).sum()))
            recs.append(r)
            print(f"{ds:<11} {ind:<14} {dep:>5} ok={r['ok']} filled={filled:>5} open={open_gaps} "
                  f"+{(new_bp - old_bp) / 1e6:7.2f} Mb  cov {100 * old_bp / 2.13e9:5.2f}% -> {100 * new_bp / 2.13e9:5.2f}%  "
                  f"+{r['sites_gained']:,} sites {r['problems']}", flush=True)
    pd.DataFrame(recs).to_csv(out_tsv, sep="\t", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
