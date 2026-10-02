#!/usr/bin/env python
"""
Synthetic-diploid baseline: combine two founders' real WGS read files into one
"individual" and run the resulting reads through the same ropebwt3 refmap ->
windowed-matrix pipeline as nam_baseline.py, then score with the diploid
GRITS-CRF (train_diploid.py's GRITSCRFDiploid) via crf/eval.py's
evaluate_diploid.

Motivation: the only real-data diploid eval run so far
(results/eval.tsv row `diploid-sim-on-ropebwt-oh43-k24`) fed single-founder
Oh43 reads, i.e. a HOMOZYGOUS individual (H1==H2 everywhere) -- the model's
worst case (its het prior actively resists predicting a homozygous pair;
pair_acc=0.04, homo_pred=0.04 there). Concatenating two DIFFERENT founders'
read files gives a genuinely heterozygous individual with known-by-construction
truth (H1=founderA, H2=founderB at every site, since there is no recombination
across combined whole read sets) -- the regime the diploid pair-CRF was
actually built for.

Reused, unmodified: nam_baseline.py's BIN/LIFT/FMD/WINDOW_SCRIPT constants,
discover_founders(), make_labels_bed(), NAME_TO_IDX_K25/PANEL_ORDER,
founder_density_stats(); crf/ropebwt_npy_to_matrix.py; crf/train_diploid.py's
GRITSCRFDiploid/make_diploid_splits; crf/eval.py's evaluate_diploid. New here:
per-depth read subsampling (nam_baseline.make_subsample hardcodes 1M and a
fixed filename, so isn't depth-parameterized), combining two founders' reads,
a generic outdir-parameterized window() wrapper, and overwriting the two
label columns with the known constant (idxA, idxB) truth after windowing.

Affinity checkpoint support (ported from tripsacum_diploid.py, same session):
the original DIPLOID_CKPT (checkpoints/diploid-sim512-h3) is a plain model
(founder_affinity=False) -- crf/eval.py's evaluate_diploid never threads
ext_emb for the diploid branch, so naively scoring an affinity checkpoint
through it silently ignores the affinity signal. evaluate_diploid_with_affinity()
here is the fix: computes one genome-wide _founder_affinity vector from this
run's own data (label-free) and broadcasts it as ext_emb to every window --
the diploid equivalent of the "single real individual, no
--windows-per-individual grouping" pattern crf/eval.py already documents for
haploid mode. run_diploid_eval() auto-detects which behavior a checkpoint
needs via model.founder_affinity. Unlike tripsacum_diploid.py, NAM data is
natively K=24 (via the K25->K24 trim already done by window()), so there is
no pad_to_24 step -- the affinity vector is computed directly from the
windowed data's own 24 feature columns.

Usage:
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/nam_diploid.py list
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/nam_diploid.py one <A> <B> <depth>
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/nam_diploid.py all
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/nam_diploid.py report
    # affinity-model variants (same args, checkpoints/diploid-affinity-sim512-h3):
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/nam_diploid.py one-affinity <A> <B> <depth>
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/nam_diploid.py all-affinity
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/nam_diploid.py report-affinity
"""
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import nam_baseline as nb  # noqa: E402  (reuse constants/helpers; see module docstring)

DIPLOID_CKPT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/"
                     "checkpoints/diploid-sim512-h3/last.ckpt")
AFFINITY_CKPT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/checkpoints/"
                      "diploid-affinity-sim512-h3/d-epoch=04-val_pair_acc=0.6179.ckpt")

SCRATCH = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/nam_diploid")
RESULTS_TSV = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/nam_diploid.tsv")
RESULTS_MD = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/nam_diploid.md")
AFFINITY_RESULTS_TSV = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/"
                             "nam_diploid_affinity.tsv")
AFFINITY_RESULTS_MD = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/"
                            "nam_diploid_affinity.md")

PY = sys.executable
THREADS = nb.THREADS

# Per-haplotype read depths to sweep (combined total = 2x this).
DEPTHS = [250_000, 500_000, 1_000_000]

# Starter pairs: baseline (B73xOh43), divergent/expected-easy (B73xCML247,
# little IBD), IBD-rich/expected-hard (CML247xCML277, two tropical CMLs --
# the pair-CRF must break within-window ties the local emission alone can't).
PAIRS = [("B73", "Oh43"), ("B73", "CML247"), ("CML247", "CML277")]


def combo_name(a, b, depth):
    return f"{a}x{b}_{depth // 1000}k"


def make_subsample(read_path, n_reads, out_path):
    """Head-based subsample of the first n_reads reads (4*n_reads lines),
    same convention as nam_baseline.make_subsample -- reimplemented here since
    that one hardcodes SUBSAMPLE_READS=1M and the output filename, and we need
    a depth sweep. Cached; resumable."""
    if out_path.exists():
        return out_path
    n_lines = n_reads * 4
    cmd = f"zcat {read_path} | head -n {n_lines} > {out_path}.tmp"
    subprocess.run(cmd, shell=True, check=True)
    got_lines = sum(1 for _ in open(f"{out_path}.tmp"))
    if got_lines != n_lines:
        Path(f"{out_path}.tmp").unlink(missing_ok=True)
        raise RuntimeError(f"{read_path} has only {got_lines} lines "
                            f"({got_lines // 4:,} reads) -- fewer than the requested "
                            f"{n_reads:,} reads")
    Path(f"{out_path}.tmp").rename(out_path)
    return out_path


def combine_reads(a, path_a, b, path_b, depth, outdir):
    """Subsample each founder to `depth` reads and concatenate -> one FASTQ
    ("individual") with founderA's and founderB's reads interleaved at the
    file level (no recombination -- truth is exactly (A,B) everywhere)."""
    outdir.mkdir(parents=True, exist_ok=True)
    sub_a = make_subsample(path_a, depth, outdir / f"{a}_{depth}.fastq")
    sub_b = make_subsample(path_b, depth, outdir / f"{b}_{depth}.fastq")
    combined = outdir / "combined.fastq"
    if not combined.exists():
        with open(combined, "wb") as out_f:
            for sub in (sub_a, sub_b):
                with open(sub, "rb") as in_f:
                    out_f.write(in_f.read())
    return combined


def run_refmap_combined(name_for_labels, combined_fastq, outdir):
    """Same ropebwt3 refmap command as nam_baseline.run_refmap (whole-read
    placement, no --kmer*), on the combined FASTQ. --label-bed's founder name
    is irrelevant -- we overwrite the label columns after windowing -- so any
    valid panel name works; pass name_for_labels (founderA) for concreteness."""
    npy_path = outdir / "raw.npy"
    ps4g_path = outdir / "raw.ps4g"
    tsv_path = outdir / "raw.tsv"
    log_path = outdir / "raw.log"
    labels_path = nb.make_labels_bed(name_for_labels, outdir)

    if npy_path.exists() and tsv_path.exists():
        print(f"  [{outdir.name}] refmap output already exists, skipping run")
        return npy_path

    cmd = [str(nb.BIN), "refmap", "--ref-prefix=B73", "--max-occ=-1",
           f"--lift={nb.LIFT}", "-t", THREADS,
           f"--label-bed={labels_path}", f"--ps4g={ps4g_path}", f"--npy={npy_path}",
           str(nb.FMD), str(combined_fastq)]
    print(f"  [{outdir.name}] running: {' '.join(cmd)}")
    t0 = time.time()
    with open(tsv_path, "w") as out_f:
        proc = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True)
    log_text = proc.stderr
    log_path.write_text(log_text)
    if proc.returncode != 0:
        raise RuntimeError(f"refmap failed for {outdir.name}: {log_text[-2000:]}")
    print(f"  [{outdir.name}] refmap done in {time.time() - t0:.1f}s")
    return npy_path


def window(raw_npy, outdir, target_num_parents=None):
    """crf/ropebwt_npy_to_matrix.py, outdir-parameterized (nam_baseline.window
    hardcodes its own SCRATCH/founder path, not reusable here). Returns
    (out_path, dropped_idx or None)."""
    bins_path = outdir / "raw.npy.bins.tsv"
    gametes_path = outdir / "raw.npy.gametes.tsv"
    suffix = f"_k{target_num_parents}" if target_num_parents else "_k25"
    out_path = outdir / f"windowed{suffix}.npy"
    sidecar = outdir / f"windowed{suffix}.dropped_idx.txt"

    if out_path.exists():
        print(f"  [{outdir.name}] windowed{suffix} already exists, skipping")
        dropped_idx = int(sidecar.read_text()) if sidecar.exists() else None
        return out_path, dropped_idx

    cmd = [PY, str(nb.WINDOW_SCRIPT), f"--npy={raw_npy}", f"--bins={bins_path}",
           f"--gametes={gametes_path}", "--num-parents=25", "--window-size=512",
           f"--out={out_path}"]
    if target_num_parents:
        cmd.append(f"--target-num-parents={target_num_parents}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"windowing failed for {outdir.name}: {proc.stderr[-2000:]}")

    dropped_idx = None
    m = re.search(r"^\s*(\S+)\s+idx=(\d+)\s+hits=", proc.stdout, re.MULTILINE)
    if m:
        dropped_idx = int(m.group(2))
        sidecar.write_text(str(dropped_idx))
    return out_path, dropped_idx


def k24_index(idx_k25, dropped_idx):
    return idx_k25 - 1 if (dropped_idx is not None and dropped_idx < idx_k25) else idx_k25


def write_diploid_labels(k24_npy_path, idxA_k24, idxB_k24, out_path):
    """Overwrite the two label columns (K, K+1) with the known-by-construction
    constant truth (idxA, idxB) at every site -- the only genuinely new logic
    in this pipeline; everything else is unmodified reuse."""
    if out_path.exists():
        return out_path
    arr = np.load(k24_npy_path)
    K = arr.shape[-1] - 2
    arr = arr.copy()
    arr[:, :, K] = idxA_k24
    arr[:, :, K + 1] = idxB_k24
    np.save(out_path, arr)
    return out_path


def evaluate_diploid_with_affinity(model, ds, device, batch_size, num_workers, ext_vec):
    """Same metric computation as crf/eval.py::evaluate_diploid, but broadcasts one
    genome-wide affinity vector (this run's own data, via _founder_affinity) to every
    window -- crf/eval.py already documents and implements this exact "single real
    individual, no --windows-per-individual grouping" pattern for the HAPLOID case
    (its module docstring), but never wires ext_emb through for diploid mode at all
    (confirmed: the diploid branch always calls model(X) with no ext_emb). This is the
    diploid equivalent, ported from tripsacum_diploid.py (kept local rather than
    editing the shared crf/ library)."""
    import torch
    from torch.utils.data import DataLoader
    from python.crf.train_diploid import _dcrf_viterbi

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    pair_correct = hap_correct = total = 0
    homo_pred = 0
    ext_t = torch.tensor(ext_vec, dtype=torch.float32, device=device)  # [K,2]
    for batch in loader:
        X = batch["input_embeds"].to(device)
        h1 = batch["h1"].to(device)
        h2 = batch["h2"].to(device)
        ext_batch = ext_t.unsqueeze(0).expand(X.shape[0], -1, -1)      # [B,K,2]
        emis_p, _, c = model(X, None, ext_batch)
        pred = _dcrf_viterbi(emis_p, c, model.nsw_pair, model.stay_bonus)
        pair_true = model.pair_table[h1, h2]
        pair_correct += (pred == pair_true).sum().item()
        pred_lo, pred_hi = model.pi[pred], model.pj[pred]
        t_lo = torch.minimum(h1, h2)
        t_hi = torch.maximum(h1, h2)
        hap_correct += ((pred_lo == t_lo).sum() + (pred_hi == t_hi).sum()).item()
        homo_pred += (pred_lo == pred_hi).sum().item()
        total += pair_true.numel()
    return {"pair_acc": pair_correct / total, "hap_acc": hap_correct / (2 * total),
            "homo_pred": homo_pred / total, "n": total}


def run_diploid_eval(data_path, tag, device, ckpt_path=DIPLOID_CKPT):
    """In-process reuse of crf/train_diploid.py + crf/eval.py (mirrors how
    nam_baseline.eval_combo imports crf/train_haploid in-process rather than
    shelling out -- avoids eval.py's `from python.crf...` absolute imports
    needing a subprocess PYTHONPATH/cwd dance). Auto-detects an affinity-trained
    checkpoint (model.founder_affinity) and computes/feeds its ext_emb accordingly
    via evaluate_diploid_with_affinity; behaves exactly as before for the plain
    model. NAM data is natively K=24 (no padding needed, unlike Tripsacum's
    K18->K24 pad), so the affinity vector is computed directly from the windowed
    data's own 24 feature columns."""
    import numpy as np
    import torch
    from python.crf.train_diploid import GRITSCRFDiploid, make_diploid_splits, _founder_affinity
    from python.crf.eval import evaluate_diploid

    model = GRITSCRFDiploid.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval().to(device)
    _, _, test_ds = make_diploid_splits(str(data_path), num_parents=24,
                                         val_frac=0.0, test_frac=1.0)
    if len(test_ds) == 0:
        return dict(n=0, pair_acc=float("nan"), hap_acc=float("nan"), homo_pred=float("nan"))

    needs_affinity = getattr(model, "founder_affinity", False)
    if needs_affinity:
        feats = np.asarray(np.load(data_path))[:, :, :24]
        ext_vec = _founder_affinity(feats)  # [K,2], label-free -> safe at inference
        r = evaluate_diploid_with_affinity(model, test_ds, device, batch_size=128,
                                            num_workers=4, ext_vec=ext_vec)
    else:
        r = evaluate_diploid(model, test_ds, device, batch_size=128, num_workers=4)
    print(f"  [{tag}] affinity={needs_affinity}  pair_acc={r['pair_acc']:.4f}  "
          f"hap_acc={r['hap_acc']:.4f}  homo_pred={r['homo_pred']:.4f}  n={r['n']:,}")
    return r


RESULT_COLS = ["founderA", "founderB", "depth_per_hap", "n_placed", "n_unplaced",
               "self_cov_A_pct", "self_cov_B_pct", "het_frac", "n_sites",
               "pair_acc", "hap_acc", "homo_pred"]


def write_header_if_needed(results_tsv=RESULTS_TSV):
    results_tsv.parent.mkdir(parents=True, exist_ok=True)
    if not results_tsv.exists():
        results_tsv.write_text("\t".join(RESULT_COLS) + "\n")


def already_recorded(a, b, depth, results_tsv=RESULTS_TSV):
    if not results_tsv.exists():
        return False
    key = f"{a}\t{b}\t{depth}\t"
    with open(results_tsv) as f:
        return any(line.startswith(key) for line in f)


def run_one(founders, a, b, depth, device, force=False, ckpt_path=DIPLOID_CKPT,
            results_tsv=RESULTS_TSV):
    if not force and already_recorded(a, b, depth, results_tsv=results_tsv):
        print(f"[{a}x{b}@{depth}] already in results TSV, skipping entirely")
        return

    name = combo_name(a, b, depth)
    outdir = SCRATCH / name
    print(f"\n=== {name} ===")

    combined = combine_reads(a, founders[a], b, founders[b], depth, outdir)
    raw_npy = run_refmap_combined(a, combined, outdir)

    n_placed = n_unplaced = 0
    with open(outdir / "raw.tsv") as f:
        for line in f:
            status = line.split("\t", 3)[2]
            if status in ("PLACED", "EXACT"):
                n_placed += 1
            elif status == "UNPLACED":
                n_unplaced += 1

    if n_placed == 0:
        print(f"  [{name}] n_placed=0 -- degenerate, recording NaN metrics")
        row = dict(founderA=a, founderB=b, depth_per_hap=depth,
                    n_placed=n_placed, n_unplaced=n_unplaced,
                    self_cov_A_pct=float("nan"), self_cov_B_pct=float("nan"),
                    het_frac=float("nan"), n_sites=0, pair_acc=float("nan"),
                    hap_acc=float("nan"), homo_pred=float("nan"))
    else:
        idxA_k25 = nb.NAME_TO_IDX_K25[a]
        idxB_k25 = nb.NAME_TO_IDX_K25[b]

        k25_npy, _ = window(raw_npy, outdir, target_num_parents=None)
        k24_npy, dropped_idx = window(raw_npy, outdir, target_num_parents=24)

        if dropped_idx is not None and dropped_idx in (idxA_k25, idxB_k25):
            raise RuntimeError(
                f"{name}: K25->K24 trim dropped one of the combined founders' own "
                f"index (dropped_idx={dropped_idx}, A={idxA_k25}, B={idxB_k25}) -- "
                f"should never happen (both should be well-covered). Investigate.")

        _, self_cov_A = nb.founder_density_stats(k25_npy, idxA_k25)
        _, self_cov_B = nb.founder_density_stats(k25_npy, idxB_k25)

        idxA_k24 = k24_index(idxA_k25, dropped_idx)
        idxB_k24 = k24_index(idxB_k25, dropped_idx)
        diploid_npy = write_diploid_labels(k24_npy, idxA_k24, idxB_k24,
                                            outdir / "diploid_k24.npy")

        arr = np.load(diploid_npy, mmap_mode="r")
        K = arr.shape[-1] - 2
        het_frac = float((arr[:, :, K] != arr[:, :, K + 1]).mean())
        if het_frac != 1.0:
            raise RuntimeError(f"{name}: expected 100% het by construction, got "
                                f"{het_frac * 100:.2f}% -- label overwrite bug")

        r = run_diploid_eval(diploid_npy, name, device, ckpt_path=ckpt_path)
        row = dict(founderA=a, founderB=b, depth_per_hap=depth,
                    n_placed=n_placed, n_unplaced=n_unplaced,
                    self_cov_A_pct=self_cov_A, self_cov_B_pct=self_cov_B,
                    het_frac=het_frac, n_sites=r["n"], pair_acc=r["pair_acc"],
                    hap_acc=r["hap_acc"], homo_pred=r["homo_pred"])

    write_header_if_needed(results_tsv=results_tsv)
    if force and already_recorded(a, b, depth, results_tsv=results_tsv):
        lines = results_tsv.read_text().splitlines(keepends=True)
        key = f"{a}\t{b}\t{depth}\t"
        keep = [l for l in lines if not l.startswith(key)]
        results_tsv.write_text("".join(keep))
    with open(results_tsv, "a") as f:
        f.write("\t".join(str(row[c]) for c in RESULT_COLS) + "\n")
    print(f"[{name}] n_placed={n_placed}  self_cov_A={row['self_cov_A_pct']}  "
          f"self_cov_B={row['self_cov_B_pct']}  pair_acc={row['pair_acc']}  "
          f"hap_acc={row['hap_acc']}  homo_pred={row['homo_pred']}")


def _markdown_table(df, float_cols):
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.4f}" if (c in float_cols and isinstance(v, float)) else str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def write_report(results_tsv=RESULTS_TSV, results_md=RESULTS_MD, affinity=False):
    import pandas as pd
    df = pd.read_csv(results_tsv, sep="\t")
    df = df.sort_values(["founderA", "founderB", "depth_per_hap"]).reset_index(drop=True)
    float_cols = {"self_cov_A_pct", "self_cov_B_pct", "het_frac", "pair_acc",
                  "hap_acc", "homo_pred"}
    table_md = _markdown_table(df, float_cols)

    model_desc = ("the **affinity-conditioned** (`founder_affinity=True`, "
                  "`checkpoints/diploid-affinity-sim512-h3`) diploid GRITS-CRF, fed a "
                  "genome-wide `_founder_affinity` vector computed from this run's own "
                  "data (label-free) as `ext_emb` -- crf/eval.py's plain `evaluate_diploid` "
                  "never threads `ext_emb` for diploid mode, so this uses a local "
                  "`evaluate_diploid_with_affinity` (see `run_diploid_eval`/"
                  "`evaluate_diploid_with_affinity` in this file)"
                  if affinity else
                  "the plain (non-affinity; `founder_affinity=False`) diploid GRITS-CRF "
                  "(`checkpoints/diploid-sim512-h3/last.ckpt`, `crf/eval.py::evaluate_diploid`)")

    lines = [
        f"# Synthetic-diploid baseline (combined real founder reads, "
        f"{'affinity' if affinity else 'plain'} diploid CRF)\n",
        "Each row combines two founders' real WGS read files (`WGSReads/`) -- a "
        "`head`-based subsample of `depth_per_hap` reads from each, concatenated -- "
        "run through the same `ropebwt3 refmap` whole-read placement recipe as "
        "`nam_baseline.py` and windowed to K=24 (`crf/ropebwt_npy_to_matrix.py "
        "--window-size=512 --target-num-parents=24`). Because the two read sets are "
        "combined wholesale (no recombination), truth is exactly `(founderA, "
        "founderB)` at every site -- the label columns are overwritten with this "
        "known-by-construction pair, and `het_frac` is asserted to be 100%.\n\n"
        f"Scored with {model_desc}. These rows reuse the identical cached reads/refmap/"
        "windowed matrices as `results/nam_diploid.md` (the original plain-model run) -- "
        "only the final scoring step differs, so any difference in `pair_acc` between "
        "the two reports is attributable to the affinity conditioning alone.\n\n"
        f"{table_md}\n\n"
        "## Reference points\n\n"
        "- `diploid-sim512-h3` (held-out simulated test split, plain): pair_acc=0.6186\n"
        "- `diploid-affinity-sim512-h3` (held-out simulated test split, affinity): "
        "pair_acc=0.6179\n"
        "- `diploid-sim-on-ropebwt-oh43-k24` (homozygous real Oh43, plain): "
        "pair_acc=0.0409, homo_pred=0.0409\n"
        "- See `results/nam_diploid.md` for the plain-model numbers on these same "
        "pairs/depths.\n",
    ]
    results_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {results_md}")
    print(df.to_string(index=False))


def main():
    if len(sys.argv) < 2:
        print(__doc__ or "", file=sys.stderr)
        sys.exit(1)
    mode = sys.argv[1]
    founders = nb.discover_founders()

    if mode == "list":
        needed = {a for a, b in PAIRS} | {b for a, b in PAIRS}
        for name in sorted(needed):
            print(f"{name:<12} {'OK' if name in founders else 'MISSING'} "
                  f"{founders.get(name, '')}")
        print(f"\nPairs: {PAIRS}")
        print(f"Depths (per haplotype): {DEPTHS}")
        return

    if mode == "report":
        write_report()
        return

    if mode == "report-affinity":
        write_report(AFFINITY_RESULTS_TSV, AFFINITY_RESULTS_MD, affinity=True)
        return

    if mode in ("one", "one-affinity"):
        if len(sys.argv) < 5:
            raise SystemExit(f"usage: nam_diploid.py {mode} <FounderA> <FounderB> <depth>")
        a, b, depth = sys.argv[2], sys.argv[3], int(sys.argv[4])
        for name in (a, b):
            if name not in founders:
                raise SystemExit(f"{name!r} not found; run `list` to see options")
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = AFFINITY_CKPT if mode == "one-affinity" else DIPLOID_CKPT
        tsv = AFFINITY_RESULTS_TSV if mode == "one-affinity" else RESULTS_TSV
        write_header_if_needed(results_tsv=tsv)
        run_one(founders, a, b, depth, device, force=True, ckpt_path=ckpt, results_tsv=tsv)
    elif mode in ("all", "all-affinity"):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = AFFINITY_CKPT if mode == "all-affinity" else DIPLOID_CKPT
        tsv = AFFINITY_RESULTS_TSV if mode == "all-affinity" else RESULTS_TSV
        write_header_if_needed(results_tsv=tsv)
        for a, b in PAIRS:
            for depth in DEPTHS:
                run_one(founders, a, b, depth, device, ckpt_path=ckpt, results_tsv=tsv)
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    print(f"\nDone. Results in {AFFINITY_RESULTS_TSV if 'affinity' in mode else RESULTS_TSV}")


if __name__ == "__main__":
    main()
