#!/usr/bin/env python
"""
Per-variant-class breakdown of genotype error (classes by the TRUTH alleles,
exactly compare_diploid's class_breakdown) for several decoders on the OUT
0.1x samples: panel ceiling (lam=50 path per truth haplotype), the untrained
read HMM, the routed ternary model and diploid-affinity. Reports, per class,
share of compared sites, error rate within the class, and its contribution
(percentage points) to the headline all-sites error_rate.
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fast_snprc_score as fss  # noqa: E402
import panel_ceiling as pc  # noqa: E402

S = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch"
MANIFEST = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv"
CLASSES = ("HOMREF", "SNP", "INS", "DEL", "HET_MIXED")
BEST_LAM = {"OUT-INBRED": 20, "OUT-HYB": 10, "OUT-RIL2": 20}
DECODERS = {
    "read_hmm": S + "/read_hmm_ceiling/{ds}__{ind}__0.1x__e0.05__lam{lam}/bed",
    "routed_ternary": S + "/ternary_baseline_routed_out_fast/*__{ds}__{ind}__0.1x/bed",
    "diploid_affinity": S + "/simval_eval/{ds}__{ind}__0.1x/bed",
}


def main():
    out_path = Path(sys.argv[1])
    panel = fss.Panel()
    panel.fa = np.asarray(panel.fa)
    out = {}
    for r in csv.DictReader(open(MANIFEST), delimiter="\t"):
        ds, ind = r["dataset_id"], r["individual"]
        if r["coverage"] != "0.1" or ds not in BEST_LAM:
            continue
        t1, t2 = fss.load_truth(r["truth_h1"]), fss.load_truth(r["truth_h2"])
        p1, _ = pc.best_path(panel, t1["code"], 50)
        p2 = p1 if r["truth_h1"] == r["truth_h2"] else pc.best_path(panel, t2["code"], 50)[0]
        out[f"ceiling__{ds}__{ind}"] = fss.score_arrays(panel, t1, t2, p1, p2)
        for name, pat in DECODERS.items():
            beds = glob.glob(pat.format(ds=ds, ind=ind, lam=BEST_LAM[ds]))
            if beds:
                f1, f2 = fss.pair_from_bed(panel, beds[0])
                out[f"{name}__{ds}__{ind}"] = fss.score_arrays(panel, t1, t2, f1, f2)
        out_path.write_text(json.dumps(out, indent=1))
        print(f"done {ds} {ind}", flush=True)

    for ds in BEST_LAM:
        print(f"\n=== {ds} (means over samples; contribution = pp of all-sites error) ===")
        print(f"{'decoder':<17} {'all':>6} | " + " | ".join(f"{c:>21}" for c in CLASSES))
        print(f"{'':<17} {'':>6} | " + " | ".join(f"{'share  err  contrib':>21}" for _ in CLASSES))
        for name in ["ceiling"] + list(DECODERS):
            rows = [v for k, v in out.items() if k.startswith(f"{name}__{ds}__")]
            if not rows:
                continue
            cells = []
            for c in CLASSES:
                share = np.mean([v[f"class_total_{c}"] / v["compared_sites"] for v in rows])
                err = np.mean([v[f"class_mismatch_{c}"] / max(v[f"class_total_{c}"], 1) for v in rows])
                contrib = np.mean([v[f"class_mismatch_{c}"] / v["compared_sites"] for v in rows])
                cells.append(f"{100 * share:5.1f} {100 * err:5.1f} {100 * contrib:6.2f}")
            allerr = np.mean([v["error_rate"] for v in rows])
            print(f"{name:<17} {100 * allerr:6.2f} | " + " | ".join(f"{c:>21}" for c in cells))
        print(f"{'decoder':<17} {'all':>6} {'SNP+RC':>7} {'outsideDEL':>10} {'DELevents':>10} {'DEL strict':>10} {'DEL frac':>9} {'falseDEL':>9}")
        for name in ["ceiling"] + list(DECODERS):
            rows = [v for k, v in out.items() if k.startswith(f"{name}__{ds}__")]
            if not rows:
                continue
            g = lambda k: np.mean([v[k] for v in rows if v.get(k) is not None])
            print(f"{name:<17} {100 * g('error_rate'):6.2f} {100 * g('snprc_error_rate'):7.2f} "
                  f"{100 * g('error_rate_outside_truth_del'):10.2f} {g('del_events'):10.0f} "
                  f"{100 * g('del_event_strict_acc'):10.1f} {100 * g('del_event_mean_frac_correct'):9.1f} "
                  f"{g('false_del_events'):9.0f}")


if __name__ == "__main__":
    main()
