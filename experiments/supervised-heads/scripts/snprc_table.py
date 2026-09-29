"""SNP+RefCall % per class for the supervised-heads runs vs baselines, 0.1x and 1x, from exact
results/<tag>_{out,idx,mix}_fast dirs (never a <tag>_* glob)."""
import glob, json, sys
from pathlib import Path
import numpy as np
W = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results")
RUNS = [("ternary", "s200_ternary_routed", "tern_1x"), ("v3-tie", "hetrepl_v3_tie", "v3tie_1x"),
        ("lik2", "hetrepl_v3_lik2", "hetrepl_v3_lik2_1x"), ("multi-likd", "hetrepl_multi_likd", "hetrepl_multi_likd_1x")]
RUNS += [(n, t, t + "_1x") for n, t in (a.split("=") for a in sys.argv[1:])]
CLS = ["IDX-INBRED", "IDX-HYB", "IDX-RIL2", "OUT-INBRED", "OUT-HYB", "OUT-RIL2", "MIX-HYB", "MIX-RIL2"]


def load(tag):
    r = {}
    for s in ("out", "idx", "mix"):
        for f in glob.glob(str(W / f"{tag}_{s}_fast" / "*.json")):
            j = json.load(open(f)); _, ds, ind, _ = j["row_key"].split("__"); r[(ds, ind)] = j
    return r


for di, depth in enumerate(("0.1x", "1x")):
    sets = [(n, load(t[di] if False else (t if di == 0 else t1))) for n, t, t1 in RUNS]
    print(f"\nSNP+RefCall error %, {depth}, router on (n rows)")
    print(f"{'class':<11}" + "".join(f"{n:>12}" for n, _ in sets))
    for c in CLS:
        common = None
        for _, r in sets:
            k = {i for (d, i) in r if d == c}
            common = k if common is None else common & k
        if not common:
            continue
        vals = [np.mean([r[(c, i)]["snprc_error_rate"] for i in common]) * 100 for _, r in sets]
        print(f"{c:<11}" + "".join(f"{v:12.2f}" for v in vals) + f"   ({len(common)})")
