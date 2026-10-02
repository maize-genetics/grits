"""Share of panel sites whose decoded founder pair changes when one output is knocked out, vs the
baseline decode (from the BEDs fast_eval_ckpt.py wrote). Needs no truth. Per class mean over samples."""
import glob, sys, collections
import numpy as np
sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/grits_workdir/regionfix-out-snprc-wt/experiments/simval-corpus/scripts")
import fast_snprc_score as fss  # noqa: E402
S = "/workdir/zrm22/HackathonJun2026/grits_workdir/scratch"
base, kos = sys.argv[1], sys.argv[2:]
panel = fss.Panel()
cls_order = ["IDX-INBRED", "IDX-HYB", "IDX-RIL2", "MIX-HYB", "MIX-RIL2", "OUT-INBRED", "OUT-HYB", "OUT-RIL2"]
cache = {}


def pair(tag, key):
    d = glob.glob(f"{S}/{tag}_*_fast/{tag}__{key}__0.1x/bed")
    if not d:
        return None
    if (tag, key) not in cache:
        f1, f2 = (np.asarray(v) for v in fss.pair_from_bed(panel, d[0]))
        cache[(tag, key)] = (np.minimum(f1, f2), np.maximum(f1, f2))
    return cache[(tag, key)]


keys = sorted({p.split(f"{base}__")[1].rsplit("__", 1)[0] for p in glob.glob(f"{S}/{base}_*_fast/{base}__*__0.1x")})
print(f"{'knockout':<10}" + "".join(f"{c:>11}" for c in cls_order))
for ko in kos:
    out = collections.defaultdict(list)
    for k in keys:
        a, b = pair(base, k), pair(ko, k)
        if a is None or b is None:
            continue
        ok = (a[0] >= 0) & (b[0] >= 0)
        out[k.split("__")[0]].append(float(((a[0] != b[0]) | (a[1] != b[1]))[ok].mean()) * 100)
    print(f"{ko.replace('v6_B_ko_', ''):<10}" + "".join(f"{np.mean(out[c]):>10.2f}%" if out[c] else f"{'–':>11}" for c in cls_order), flush=True)
