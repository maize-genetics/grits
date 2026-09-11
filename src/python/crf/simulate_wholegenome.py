"""
Whole-genome eval/benchmark set: the 6 sim_breedpop cells as whole genomes.

sim_breedpop trains on 1024-site windows; this emits, for each of the 6 breeding
cells — (k=2 F2, k=8 S1, outbred[12,24]) x (inbred, het) — ONE individual as a
whole genome: 10 chromosomes x 100 000 sites = ~1M read positions, written as one
`.npy` per individual `[10, 100000, K+2]` (col K=H1, K+1=H2; same layout as the
training sim). Two recombination densities are produced so we can measure the
window-density train/test gap:

  * sparse  — ~1-3 crossovers per 100k chromosome (realistic plant genome; most
              1024-windows are constant -> a real gap vs the dense training data).
  * dense   — training-matched crossover DENSITY (2-10 per 1024 -> ~195-977 per
              100k chromosome), so each 1024 sub-window matches what the model saw.

Reuse: this is a thin driver over `simulate()` (simulate_alleles.py) called once
per cell with a SINGLE-class `breeding_classes` list and `class_inbred_frac` forced
to 1.0 (inbred) or 0.0 (het) -> exactly one individual per cell, no duplication of
the breeding generative logic. Defaults match sim_breedpop (K=24, coalescent θ=6,
bad_frac 0.05).

Usage:
    LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src \
      .pixi/envs/gpu/bin/python src/python/crf/simulate_wholegenome.py \
        --workdir /workdir/esb33 [--chrom-len 100000 --n-chrom 10 --seed 0]
"""

import argparse
from pathlib import Path

import numpy as np

from python.crf.simulate_alleles import simulate

# (name, kmin, kmax, target_het) — the three sim_breedpop classes (RESULTS E11).
CLASSES = [("k2F2", 2, 2, 0.5), ("k8S1", 8, 8, 0.25), ("outbred", 12, 24, 0.9)]
HETS = [("inbred", 1.0), ("het", 0.0)]          # label -> class_inbred_frac


def cross_range(density, chrom_len):
    if density == "sparse":
        return 1, 3                              # ~1-3 crossovers per chromosome
    scale = chrom_len / 1024.0                   # training-matched density (2-10/1024)
    return max(1, int(round(2 * scale))), max(2, int(round(10 * scale)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default="/workdir/esb33")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--chrom-len", type=int, default=100_000)
    p.add_argument("--n-chrom", type=int, default=10)
    p.add_argument("--n-per-cell", type=int, default=1,
                   help="individuals per (class x het) cell, for stable estimates")
    p.add_argument("--bad-frac", type=float, default=0.05)
    p.add_argument("--sharing-theta", type=float, default=6.0)
    p.add_argument("--read-snps", type=int, default=8)
    p.add_argument("--anc-base", type=float, default=8.0,
                   help="founder lineage-mosaic crossovers per 1024 sites (training "
                        "rate). MUST scale with T or founder IBD tracts get ~T/1024x "
                        "too long -> out-of-distribution (verified: anc_cx=8 over 100k "
                        "gives k8S1-het 0.48 vs 0.75 when density-matched).")
    p.add_argument("--densities", default="sparse,dense")
    p.add_argument("--chunk", type=int, default=2,
                   help="chromosomes generated at once (memory: large T)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.workdir) / "data" / "held-out"
    out_dir.mkdir(parents=True, exist_ok=True)
    K, T, NC = args.num_parents, args.chrom_len, args.n_chrom
    rng = np.random.default_rng(args.seed)
    anc_cx = max(1, round(args.anc_base * T / 1024.0))   # scale lineage tracts with T

    print(f"whole-genome eval set -> {out_dir}")
    print(f"  {NC} chrom x {T:,} sites = {NC*T:,} positions/individual, K={K}, "
          f"anc_cx={anc_cx} (founder IBD tracts ~{T//anc_cx} sites)")
    manifest = []
    for density in args.densities.split(","):
        mn, mx = cross_range(density, T)
        print(f"\n[{density}] crossovers/chrom in [{mn},{mx}]")
        for cname, kmin, kmax, het in CLASSES:
            for hlabel, inbred_frac in HETS:
                het_t = 0.0 if hlabel == "inbred" else het
                for ind_i in range(args.n_per_cell):       # N individuals/cell -> stable
                    out, _ibd, _ind, _panel, het_tw, cls_w, _refpos, _short = simulate(
                        rng, windows=NC, sites=T, founders=K,
                        min_cross=mn, max_cross=mx, inbreeding=1.0,
                        allele_sharing=0.2, bad_frac=args.bad_frac,
                        sharing_model="coalescent", sharing_theta=args.sharing_theta,
                        read_snps=args.read_snps, gamete_balance=0.5,
                        ancestor_crossovers=anc_cx,
                        windows_per_individual=NC,
                        breeding_classes=[(1.0, kmin, kmax, het)],
                        class_inbred_frac=inbred_frac, chunk=args.chunk)
                    # one individual = NC chromosomes, shape [NC, T, K+2]
                    fname = f"wg_{cname}_{hlabel}_{density}_i{ind_i}.npy"
                    np.save(out_dir / fname, out)
                    founders = np.unique(out[:, :, K:K + 2]).tolist()
                    xrate = float((out[:, 1:, K] != out[:, :-1, K]).mean())   # H1 switch
                    realized_het = float((out[:, :, K] != out[:, :, K + 1]).mean())
                    np.savez(out_dir / f"wg_{cname}_{hlabel}_{density}_i{ind_i}.meta.npz",
                             cls=cls_w[0], het_target=het_t, founders=np.array(founders),
                             density=density, n_chrom=NC, chrom_len=T)
                    manifest.append((fname, cname, hlabel, density, len(founders),
                                     realized_het, xrate))
                    print(f"  wrote {fname:40s} k={len(founders):2d} "
                          f"het={realized_het:.3f} H1xrate={xrate:.5f}")

    print(f"\n{'file':38s} {'class':8s} {'het':6s} {'dens':6s} {'k':>3} "
          f"{'het':>6} {'xrate':>8}")
    for f, c, h, d, k, hr, xr in manifest:
        print(f"{f:38s} {c:8s} {h:6s} {d:6s} {k:>3} {hr:>6.3f} {xr:>8.5f}")
    print(f"\n{len(manifest)} individuals -> {out_dir}")


if __name__ == "__main__":
    main()
