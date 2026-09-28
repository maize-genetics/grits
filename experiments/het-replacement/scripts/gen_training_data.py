#!/usr/bin/env python
"""
Stage 2.3/2.6 of experiments/het-replacement/PLAN.md: generate a training set with
`--indel-model replacement`, calibrated from measure_real_stats.py output.

Mirrors the 77-column recipe every model since ternary-baseline was trained and
evaluated with (the lowrate / region-fix family: read counts, collapsed rows,
indel_region_mult=2, separate inbred / outbred halves concatenated, sliced into
512-row windows) and changes ONLY the indel model plus its calibrated
repl_* parameters, so any difference in a model trained on it is attributable to
the replacement modelling. Simulated data only -- no real reads or features enter
training (standing rule).

Writes <out-dir>/<prefix>_inb{1.0,0.0}.npy (+ .ibd/.refpos sidecars), and
<prefix>_fullscale_sliced.npy for train_diploid_indel.py --windows-per-individual G.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from python.crf.simulate_alleles import simulate  # noqa: E402

T = 512
RECIPE = dict(sites=60000, founders=25, min_cross=2, max_cross=10, allele_sharing=0.2, bad_frac=0.05,
              sharing_model="coalescent", ancestors=6, sharing_theta=4.0, simulate_indels=True,
              indel_coverage=2.0, emit_read_counts=True, collapse_rows=True, indel_region_mult=2)
REPL_KEYS = ("repl_rate", "repl_mean_len", "repl_max_share", "repl_groups_mean", "repl_shift_sites",
             "repl_cross_del", "repl_cross_present")
REPL_STR_KEYS = ("repl_groups",)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True,
                    help="measure_real_stats.py JSON (uses its suggested_sim_params) or a flat JSON of repl_*")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--windows", type=int, default=500, help="individuals per half")
    ap.add_argument("--seeds", type=int, nargs=2, default=[601, 602], help="inbred, outbred")
    ap.add_argument("--override", nargs="*", default=[], help="repl_x=value overrides")
    ap.add_argument("--indel-model", default="replacement",
                    help="control runs: e.g. overlay (repl_* params are then unused)")
    ap.add_argument("--no-obs-table", action="store_true",
                    help="ignore the params' obs_table (exact -1/distance observation, as before)")
    args = ap.parse_args()

    p = json.loads(Path(args.params).read_text())
    p = p.get("suggested_sim_params", p)
    repl = {k: p[k] for k in REPL_KEYS + REPL_STR_KEYS if k in p and p[k] is not None}
    for kv in args.override:
        k, v = kv.split("=")
        repl[k] = v if k in REPL_STR_KEYS else float(v) if k != "repl_max_share" else int(v)
    if "repl_max_share" in repl:
        repl["repl_max_share"] = int(repl["repl_max_share"])
    obs = None if args.no_obs_table else p.get("obs_table")
    print("repl params:", repl, "| obs_table:", "yes" if obs else "no (exact)", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.prefix}.recipe.json").write_text(json.dumps({"recipe": RECIPE, "indel_model": args.indel_model, "repl": repl, "obs_table": obs,
                                                                "windows": args.windows, "seeds": args.seeds}, indent=1))
    parts = []
    for inb, seed in zip(("1.0", "0.0"), args.seeds):
        o, ibd, _i, _p, _h, _c, refpos, short, _tc = simulate(
            np.random.default_rng(seed), windows=args.windows, inbreeding=float(inb),
            indel_model=args.indel_model, obs_table=obs, **RECIPE, **repl)
        print(f"inb{inb}: {o.shape}  short windows {100 * short.mean():.2f}%", flush=True)
        np.save(out / f"{args.prefix}_inb{inb}.npy", o)
        np.save(out / f"{args.prefix}_inb{inb}.ibd.npy", ibd)
        np.save(out / f"{args.prefix}_inb{inb}.refpos.npy", refpos)
        parts.append(o)
        del ibd
    comb = np.concatenate(parts, axis=0)
    G = comb.shape[1] // T
    np.save(out / f"{args.prefix}_fullscale_sliced.npy", comb[:, :G * T].reshape(-1, T, comb.shape[2]))
    print(f"wrote {args.prefix}_fullscale_sliced.npy  (--windows-per-individual {G})", flush=True)
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).with_name("lineage_rows.py")), "--dir", str(out),
                    "--prefix", args.prefix], check=True)      # per-row lineage labels for --tie-aware-loss


if __name__ == "__main__":
    main()
