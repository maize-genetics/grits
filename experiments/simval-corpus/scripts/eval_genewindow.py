#!/usr/bin/env python
"""Gene-window section of the eval suite (VCF-free): stitch the model's imputed haplotypes over gene
windows (gene +-5 kb, 1000 windows), align them to the truth assemblies with minimap2 and count
errors = mismatches + inserted + deleted + unaligned bases (experiments/seq-extract/gw_*.py, v2 rules,
collinear supplementary blocks).

Everything that does not depend on the model is cached once under results/eval_suite/_reference/genewindow/:

  windows.tsv, ril_truth.tsv   gene windows and the RIL2 mosaic parent per window (gw_windows.py,
                               gw_ril_truth.py)
  proj.tsv                     ropebwt3 extract --project-only table; grows with every model's new
                               founder intervals (only intervals not yet in it are projected)
  panel.tsv                    the 25 founders' whole-window projections (panel ceiling)
  scores/<line>.tsv            gw_align score row per (sequence md5[:16], truth line); the panel
                               sequences x 10 lines are the ceiling, and every model sequence scored once
                               is reused by later models (identical sequence + line = identical score)
  batches/<name>/              everything one alignment batch produced: seqs.fa.gz, expect.tsv (window
                               chromosome per sequence), <line>.hashes, <line>.paf, <line>.rescue.paf.
                               When gw_align's score columns change, scores/ is rebuilt from these PAFs
                               without re-aligning (rescore_cache; automatic on a header mismatch).
                               Seeded batches: panel_v2 and v6cfullfe_homo_v2 (the v2 run's PAFs).
  maize_v2_b12.isa, mmi/       ropebwt3 sampled ISA and the minimap2 asm20 indexes (symlink)

Scorer versions (VERSIONS; default DEFAULT_VERSION = v2):
  v1  whole-segment projection: a founder piece whose projection fails is all N (old scorer); panel.tsv,
      outputs in <model>/genewindow/
  v2  tile-on-failure (HaplotypeBuilder tile_on_failure=2000): a failed piece is split into 2 kb reference
      tiles, each projected on its own, and only the failed tiles are N. The panel (ceiling + random) is
      rebuilt with the same rule (panel_tile2000.tsv), outputs in <model>/genewindow_v2/. proj.tsv, the
      score cache and the batches are shared (scores are keyed by sequence hash, so v1 results are untouched).

Per model it writes results/eval_suite/<model>/genewindow[_v2]/: model_haps.tsv, per_hapwin.tsv (+ founder
choice columns), ceiling.tsv, summary.tsv (per depth x class and depth x scope), per_sample.tsv,
resources.tsv (wall seconds per step).

Error rates (summary.tsv; per-kb columns are kept but not displayed):
  err_pct_capped     100 * sum(min(errors, called)) / sum(called truth-side bases) = err_per_kb_capped / 10,
                     errors = mismatch + inserted + deleted + unaligned bp ("gene-window error %, per bp")
  event_err_pct      the same with event_errors (an indel or unaligned run counts once; "per event")
  ceil_* / t50_*     the same for the panel ceiling (event ceiling = founder with fewest event errors)
                     and for the anchor-trimmed (t50) variant
  err_x_ceiling      err_per_kb / ceil_err_per_kb (uncapped, bp); event_x_ceiling = event_err_pct /
                     ceil_event_err_pct
  rand_event_err_pct / rand_err_pct_capped
                     random-founder baseline: per (window, truth line) the exact expectation over the
                     scorable panel founders (gw_score.random_founder), pooled as sum(mean errors) /
                     sum(mean called) over the scored haplotype windows
  r2c_event_pct / r2c_bp_pct
                     % of the way from random to ceiling = 100 * (random - model) / (random - ceiling);
                     0 = random founder per window, 100 = best panel founder per window
  pct_truth_unextractable
                     % of haplotype windows whose in-panel truth founder's own panel sequence failed
                     extraction / scoring (gw_score in-panel extraction rule); excluded from model, ceiling
                     and random alike
Per event is the primary gene-window measure (per bp penalises every base of an indel).

Founder-choice decomposition (single-founder scored windows = haplotype windows with one founder and no
switch): pick_rank = 1 + number of panel founders whose own whole-window sequence has strictly fewer
errors against the truth line than the picked founder's; n_ranked = founders scored in that window.
  sf_better_pct    % of single-founder windows where a better panel founder exists (pick_rank > 1)
  pick_rank_median median pick_rank over single-founder windows
  wrongpick_rank_median  median pick_rank over the windows with a better founder (v2 analysis: ~6 of 23
                   at OUT 0.1x; here ranked among the founders scored in that window, median 20)
  wrongpick_excess_bp_median  median errors - ceiling errors in those windows
  wrongpick_err_share  % of all errors that sit in single-founder windows with a better founder
  excess_share     % of all errors that is excess over the best founder in those windows
                   (sum max(0, errors - ceiling errors) / sum errors)
"""
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
SEQX = WT / "experiments" / "seq-extract"
sys.path.insert(0, str(SEQX))
sys.path.insert(0, str(WT / "src"))
import gw_align  # noqa: E402
import gw_build  # noqa: E402
import gw_report  # noqa: E402
import gw_score  # noqa: E402
from python.seq_extract import AssemblySet, HaplotypeBuilder, TableProvider, read_bed_dir  # noqa: E402
from python.seq_extract import write_projection_queries  # noqa: E402
from python.seq_extract.haplotype import METADATA_COLUMNS, BedInterval, group_bed, metadata_row  # noqa: E402
from python.seq_extract.projection import ProjectionQuery, bare_chrom  # noqa: E402

WORK = Path(os.environ.get("GRITS_WORKDIR", "/nonexistent-GRITS_WORKDIR-unset"))
MANIFEST = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv"
GWC = WORK / "results/eval_suite/_reference/genewindow"
SEED = WORK / "scratch/gene_window_eval/v2"
MMI_DIR = WORK / "scratch/gene_window_eval/mmi"
RB3 = WORK / "scratch/seq_extract_ropebwt/ropebwt3_v3"
IDX = Path(os.path.join(os.environ.get("GRITS_REFMAP_ROOT", "/nonexistent-GRITS_REFMAP_ROOT-unset"), "rope_bwt_index_v2"))      # read only
FMD = IDX / "maizeFastaIndex_SampleContig_v2.fmd"
LIFT = IDX / "maizeFastaIndex_SampleContig_v2_s200.lift"
CLASSES = ["IDX-INBRED", "IDX-HYB", "IDX-RIL2", "MIX-HYB", "MIX-RIL2", "OUT-INBRED", "OUT-HYB", "OUT-RIL2"]
SCOPE = {"IDX": "idx", "MIX": "mix", "OUT": "out"}
INT_COLS = set(gw_score.INT)
VERSIONS = {"v1": {"tile": 0, "panel": "panel.tsv", "outdir": "genewindow"},
            "v2": {"tile": 2000, "panel": "panel_tile2000.tsv", "outdir": "genewindow_v2"}}
DEFAULT_VERSION = "v2"
PANEL_COLS = ["wid", "chrom", "start", "end", "founder", "status", "contig", "pstart", "pend", "strand", "seq_len",
              "seq_hash", "n_N", "clamped"]


def outdir(model, version=DEFAULT_VERSION, out_root=None):
    return Path(out_root or WORK / "results/eval_suite") / model / VERSIONS[version]["outdir"]


def log(*a):
    print("  [genewindow]", *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- cache
def _place(src, dst):
    """Hard link (same filesystem) or copy."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_batch_meta(bd, seqs, expect, hashes_by_line):
    import gzip
    bd.mkdir(parents=True, exist_ok=True)
    with gzip.open(bd / "seqs.fa.gz", "wt", compresslevel=3) as f:
        for h in sorted(seqs):
            f.write(f">{h}\n{seqs[h]}\n")
    with open(bd / "expect.tsv", "w") as f:
        for h in sorted(seqs):
            f.write(f"{h}\t{expect[h]}\n")
    for line, hs in hashes_by_line.items():
        (bd / f"{line}.hashes").write_text("".join(f"{h}\n" for h in sorted(hs)))


def seed(src=SEED):
    """Create the model-independent cache from a finished gw run directory (v2 layout): windows, RIL
    truth, projection table, panel stitch, and two alignment batches from its PAFs (panel_v2: the 25
    founder windows x 10 lines; v6cfullfe_homo_v2: that run's model haplotypes), then score them."""
    src = Path(src)
    (GWC / "batches").mkdir(parents=True, exist_ok=True)
    for f in ("windows.tsv", "ril_truth.tsv", "proj.tsv", "maize_v2_b12.isa"):
        if not (GWC / f).exists():
            shutil.copy2(src / f, GWC / f)
    if not (GWC / "panel.tsv").exists():
        shutil.copy2(src / "stitch/panel.tsv", GWC / "panel.tsv")
    if not (GWC / "mmi").exists():
        os.symlink(MMI_DIR, GWC / "mmi")
    for name, fa, setname in (("panel_v2", "seqs_panel.fa", "panel"), ("v6cfullfe_homo_v2", "seqs_model.fa", "model")):
        bd = GWC / "batches" / name
        if (bd / "expect.tsv").exists():
            continue
        qinfo, expect, need = {}, {}, defaultdict(set)
        if setname == "panel":
            for r in csv.DictReader(open(src / "stitch/panel.tsv"), delimiter="\t"):
                if r["status"] == "ok":
                    qinfo[r["seq_hash"]] = (int(r["seq_len"]), int(r["n_N"]))
                    expect.setdefault(r["seq_hash"], r["chrom"])
                    for line in gw_align.ALL_LINES:
                        need[line].add(r["seq_hash"])
        else:
            for r in csv.DictReader(open(src / "stitch/model_haps.tsv"), delimiter="\t"):
                qinfo[r["seq_hash"]] = (int(r["seq_len"]), int(r["n_N"]))
                expect.setdefault(r["seq_hash"], r["chrom"])
                for line in gw_align.truth_lines(r["kind"], r["sample"]):
                    need[line].add(r["seq_hash"])
        keep = {h for h, (n, nN) in qinfo.items() if n > nN}
        seqs = gw_align.read_fasta_subset(src / "stitch" / fa, keep)
        _write_batch_meta(bd, seqs, expect, {ln: hs & keep for ln, hs in need.items()})
        for line in gw_align.ALL_LINES:
            for x in ("paf", "rescue.paf"):
                f = src / "align" / f"{line}.{setname}.{x}"
                if f.exists():
                    _place(f, bd / f"{line}.{x}")
    rescore_cache()
    log(f"cache seeded from {src} -> {GWC}")


def _rescore_job(args):
    bd, line, seqs, expect = args
    rows, _ = gw_align.align_line(line, seqs, expect, bd, rescore_only=True)
    return line, rows


def rescore_cache(workers=4):
    """Rebuild scores/<line>.tsv from every batch's PAFs with the current gw_align scorer (no alignment)."""
    import gzip
    t0 = time.time()
    rows = defaultdict(dict)
    for bd in sorted((GWC / "batches").iterdir()):
        if not (bd / "expect.tsv").exists():
            continue
        expect = dict(l.rstrip("\n").split("\t") for l in open(bd / "expect.tsv"))
        seqs, name = {}, None
        with gzip.open(bd / "seqs.fa.gz", "rt") as f:
            for l in f:
                if l.startswith(">"):
                    name = l[1:].strip()
                else:
                    seqs[name] = l.strip()
        jobs = []
        for line in gw_align.ALL_LINES:
            hf = bd / f"{line}.hashes"
            if hf.exists() and (bd / f"{line}.paf").exists():
                hs = hf.read_text().split()
                jobs.append((bd, line, {h: seqs[h] for h in hs}, {h: expect[h] for h in hs}))
        with ProcessPoolExecutor(workers) as ex:
            for line, rr in ex.map(_rescore_job, jobs):
                rows[line].update(rr)
        log(f"rescored batch {bd.name}: {sum(len(j[2]) for j in jobs):,} (sequence, line) pairs")
    (GWC / "scores").mkdir(exist_ok=True)
    for line in gw_align.ALL_LINES:
        with open(GWC / "scores" / f"{line}.tsv.tmp", "w") as fo:
            fo.write("\t".join(gw_align.SCORE_COLS) + "\n")
        append_scores(line, [rows[line][h] for h in sorted(rows[line])], suffix=".tmp")
        os.replace(GWC / "scores" / f"{line}.tsv.tmp", GWC / "scores" / f"{line}.tsv")
    log(f"score cache rebuilt in {time.time() - t0:.0f}s")


def _int(r):
    for k in INT_COLS:
        if k in r:
            r[k] = int(r[k])
    return r


def load_score_cache():
    hdr = "\t".join(gw_align.SCORE_COLS)
    if any(not (GWC / "scores" / f"{l}.tsv").exists() or open(GWC / "scores" / f"{l}.tsv").readline().rstrip("\n") != hdr
           for l in gw_align.ALL_LINES):
        log("score cache missing or written by an older scorer: rescoring from the cached PAFs")
        rescore_cache()
    sc = {}
    for line in gw_align.ALL_LINES:
        with open(GWC / "scores" / f"{line}.tsv") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                sc[(r["seq_hash"], line)] = _int(r)
    return sc


def append_scores(line, rows, suffix=""):
    with open(GWC / "scores" / f"{line}.tsv{suffix}", "a") as fo:
        for r in rows:
            fo.write("\t".join(str(r[c]) for c in gw_align.SCORE_COLS) + "\n")


# ---------------------------------------------------------------- projection / panel
def project_new(new, tmp, threads):
    """Project ProjectionQuery values `new` with ropebwt3 extract --project-only and append them to proj.tsv
    (caller holds the cache lock)."""
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    write_projection_queries(tmp / "q.tsv", new)
    with open(tmp / "p.tsv", "w") as fo, open(tmp / "p.log", "w") as fe:
        rc = subprocess.run([str(RB3), "extract", f"--lift={LIFT}", "--ref-prefix=B73", "--project-only",
                             "-t", str(threads), "-I", str(GWC / "maize_v2_b12.isa"), "-f", str(tmp / "q.tsv"),
                             str(FMD)], stdout=fo, stderr=fe).returncode
    if rc:
        raise RuntimeError(f"ropebwt3 extract failed, see {tmp / 'p.log'}")
    body = open(tmp / "p.tsv").read().split("\n", 1)[1]
    with open(GWC / "proj.tsv", "a") as fo:
        fo.write(body)
    shutil.rmtree(tmp)


def _missing(queries, known, prefix, new):
    for q in queries:
        k = (bare_chrom(q.ref_chrom), q.ref_start, q.ref_end, q.assembly)
        if k not in known and k not in new:
            new[k] = ProjectionQuery(f"{prefix}_{len(new)}", *k)


def _panel_builder(wins, tile, asm=None, prov=None):
    """A HaplotypeBuilder whose BED has every window as one interval (founder irrelevant: build_interval is
    called with the founder directly)."""
    by = defaultdict(list)
    for w in wins:
        by[bare_chrom(w.chrom)].append(BedInterval(bare_chrom(w.chrom), w.start, w.end, "B73", "B73"))
    return HaplotypeBuilder(group_bed([iv for v in by.values() for iv in v]), asm, prov, "N", upper=True,
                            tile_on_failure=tile)


def ensure_panel(version, threads=8, jobs=2):
    """Build GWC/<panel> for a tiled scorer version (whole-window projection of every founder, tile on
    failure), project the missing tiles, and align+score the panel sequences the score cache lacks.
    v1's panel.tsv is the seeded file and is never rebuilt here. Caller holds the cache lock."""
    V = VERSIONS[version]
    pf = GWC / V["panel"]
    if pf.exists() or not V["tile"]:
        return pf
    t0 = time.time()
    wins = gw_build.read_windows(GWC / "windows.tsv")
    asm = AssemblySet.from_dir(gw_build.RENAMED, "{asm}_renamed.fa", fai_dir=gw_build.FAI_DIR)
    fs = sorted(asm.assemblies)
    known = set(TableProvider.from_tsv(GWC / "proj.tsv")._rows)
    new = {}
    _missing([ProjectionQuery("p", bare_chrom(w.chrom), w.start, w.end, f) for w in wins for f in fs if f != "B73"],
             known, f"panel{V['tile']}", new)
    if new:
        project_new(new.values(), GWC / f"panel_proj_tmp_{version}", threads)
        known |= set(new)
    prov = gw_build.CountingSpanCheck(TableProvider.from_tsv(GWC / "proj.tsv"))
    pb = _panel_builder(wins, V["tile"])
    new = {}
    for w in wins:
        for f in fs:
            if f == "B73":
                continue
            c = bare_chrom(w.chrom)
            if not prov.project(c, w.start, w.end, f).ok:
                _missing([ProjectionQuery("t", c, s, e, f) for s, e in pb.tiles(w.start, w.end)], known,
                         f"panel{V['tile']}t", new)
    if new:
        project_new(new.values(), GWC / f"panel_proj_tmp_{version}", threads)
    log(f"panel {version}: {len(new)} tile projections added ({time.time() - t0:.0f}s)")
    prov = gw_build.CountingSpanCheck(TableProvider.from_tsv(GWC / "proj.tsv"))
    pb = _panel_builder(wins, V["tile"], asm, prov)
    seqs, expect, rows = {}, {}, []
    for w in wins:
        for f in fs:
            if f == "B73":
                rows.append([w.wid, w.chrom, w.start, w.end, f, "ok", f"B73_{w.chrom}", w.start, w.end, "+"])
                seq = asm.fetch(f"B73_{w.chrom}", w.start, w.end, "+", upper=True)
                cl = False
            else:
                seq, segs = pb.build_interval(w.chrom, w.start, w.end, f)
                ok = [g for g in segs if g.status in ("ok", "ok_clamped")]
                cl = any(g.status == "ok_clamped" for g in segs)
                if not ok:
                    rows.append([w.wid, w.chrom, w.start, w.end, f, segs[0].status, ".", ".", ".", ".", 0, ".", 0, 0])
                    continue
                if len(segs) == 1:
                    g = segs[0]
                    rows.append([w.wid, w.chrom, w.start, w.end, f, "ok", g.contig, g.start, g.end, g.strand])
                else:                          # tiled: contig/coords of the usable tiles' span (first contig)
                    cs = [g for g in ok if g.contig == ok[0].contig]
                    rows.append([w.wid, w.chrom, w.start, w.end, f, "ok", ok[0].contig,
                                 min(g.start for g in cs), max(g.end for g in cs),
                                 ok[0].strand if len({g.strand for g in ok}) == 1 else "mixed"])
            h = gw_build.h16(seq)
            rows[-1] += [len(seq), h, seq.count("N"), int(cl)]
            if len(seq) > seq.count("N"):
                seqs[h] = seq
                expect.setdefault(h, w.chrom)
    with open(str(pf) + ".tmp", "w") as fo:
        fo.write("\t".join(PANEL_COLS) + "\n")
        for r in rows:
            fo.write("\t".join(map(str, r)) + "\n")
    # align what the score cache lacks (panel sequences are scored against every truth line)
    sc = load_score_cache()
    todo = {ln: sorted(h for h in seqs if (h, ln) not in sc) for ln in gw_align.ALL_LINES}
    bd = GWC / "batches" / f"panel_tile{V['tile']}_{time.strftime('%Y%m%d_%H%M%S')}"
    args = [(ln, {h: seqs[h] for h in hs}, {h: expect[h] for h in hs}, bd, max(1, threads // jobs))
            for ln, hs in todo.items() if hs]
    log(f"panel {version}: {sum(map(len, todo.values())):,} (sequence, line) pairs need alignment")
    if args:
        used = {h for a in args for h in a[1]}
        _write_batch_meta(bd, {h: seqs[h] for h in used}, expect, {a[0]: a[1].keys() for a in args})
        with ProcessPoolExecutor(min(jobs, len(args))) as ex:
            for ln, rr, info in ex.map(_align_job, args):
                append_scores(ln, rr)
                log(f"panel {version}: {ln} {info['n']} queries, minimap2 {info['mm2_s']:.0f}s")
    os.replace(str(pf) + ".tmp", pf)
    log(f"panel {version} -> {pf} ({time.time() - t0:.0f}s)")
    return pf


# ---------------------------------------------------------------- per model
def samples(tags, classes):
    out = []
    for r in csv.DictReader(open(MANIFEST), delimiter="\t"):
        dep = r["coverage"]
        if r["dataset_id"] in classes and dep in tags:
            tag = tags[dep]
            key = f"{tag}__{r['dataset_id']}__{r['individual']}__{dep}x"
            bed = WORK / "scratch" / f"{tag}_{SCOPE[r['dataset_id'].split('-')[0]]}_fast" / key / "bed"
            out.append((dep, r["dataset_id"], r["individual"], r["kind"], bed))
    return sorted(out, key=lambda x: (x[0], CLASSES.index(x[1]), x[2]))


def _align_job(args):
    line, seqs, expect, tmp, threads = args
    rows, info = gw_align.align_line(line, seqs, expect, tmp, threads=threads)
    return line, [rows[h] for h in sorted(rows)], info


def run_model(model, tags, classes=CLASSES, threads=8, jobs=2, out_root=None, version=DEFAULT_VERSION):
    """Gene-window eval for one model. tags: {depth string as in the manifest ('0.1', '1.0'): BED tag}.
    version: scorer version (VERSIONS). Returns the summary DataFrame (or None when no BEDs exist)."""
    import pandas as pd
    if not (GWC / "batches").is_dir():
        seed()
    V = VERSIONS[version]
    tile = V["tile"]
    out = outdir(model, version, out_root)
    out.mkdir(parents=True, exist_ok=True)
    res = []
    T0 = time.time()

    def step(name, t, n=""):
        res.append((name, n, round(time.time() - t, 1)))
        log(f"{model}: {name} {n} {time.time() - t:.0f}s")

    with open(GWC / "lock", "w") as lk:           # one gene-window run at a time: shared cache + threads
        fcntl.flock(lk, fcntl.LOCK_EX)
        t = time.time()
        wins = gw_build.read_windows(GWC / "windows.tsv")
        sams = [s for s in samples(tags, classes) if s[4].is_dir()]
        missing = [f"{s[0]}x {s[1]} {s[2]}" for s in samples(tags, classes) if not s[4].is_dir()]
        if missing:
            log(f"{model}: {len(missing)} samples without BEDs, skipped: {', '.join(missing[:6])}"
                + (" ..." if len(missing) > 6 else ""))
        if not sams:
            return None
        panel_tsv = ensure_panel(version, threads, jobs)
        beds = {s: read_bed_dir(s[4], s[2], "B73", "trim") for s in sams}
        known = set(TableProvider.from_tsv(GWC / "proj.tsv")._rows)
        new = {}
        for s in sams:
            _missing(HaplotypeBuilder(beds[s], None, None, "N").projection_queries(wins), known, model, new)
        step("queries", t, len(new))
        if new:                                   # project only founder intervals not in the cache
            t = time.time()
            project_new(new.values(), out / "proj_tmp", threads)
            step("project", t, len(new))
        if tile:                                  # second pass: tiles of the pieces whose projection failed
            t = time.time()
            prov = gw_build.CountingSpanCheck(TableProvider.from_tsv(GWC / "proj.tsv"))
            known = set(prov.base._rows)
            new = {}
            for s in sams:
                _missing(HaplotypeBuilder(beds[s], None, None, "N", tile_on_failure=tile).projection_queries(wins, prov),
                         known, f"{model}_t", new)
            if new:
                project_new(new.values(), out / "proj_tmp", threads)
            step("project_tiles", t, len(new))

        # stitch
        t = time.time()
        prov = gw_build.CountingSpanCheck(TableProvider.from_tsv(GWC / "proj.tsv"))
        asm = AssemblySet.from_dir(gw_build.RENAMED, "{asm}_renamed.fa", fai_dir=gw_build.FAI_DIR)
        seqs, qinfo, expect = {}, {}, {}
        need = defaultdict(set)
        with open(out / "model_haps.tsv", "w") as md:
            md.write("\t".join(["depth", "class", "sample", "kind"] + METADATA_COLUMNS + ["seq_hash", "n_N"]) + "\n")
            for s in sams:
                dep, cls, ind, kind, _ = s
                b = HaplotypeBuilder(beds[s], asm, prov, "N", upper=True, tile_on_failure=tile)
                lines = gw_align.truth_lines(kind, ind)
                for r1, r2 in b.build_all(wins):
                    for r in (r1, r2):
                        h = gw_build.h16(r.seq)
                        nN = r.seq.count("N")
                        if h not in qinfo:
                            qinfo[h] = (len(r.seq), nN)
                            expect[h] = r.window.chrom
                            if len(r.seq) > nN:
                                seqs[h] = r.seq
                        for ln in lines:
                            need[ln].add(h)
                        md.write("\t".join(map(str, [dep, cls, ind, kind] + metadata_row(r) + [h, nN])) + "\n")
        step("stitch", t, len(qinfo))

        # align what the cache does not have
        t = time.time()
        sc = load_score_cache()
        todo = {ln: sorted(h for h in hs if h in seqs and (h, ln) not in sc) for ln, hs in need.items()}
        n_todo = sum(map(len, todo.values()))
        n_need = sum(len([h for h in hs if h in seqs]) for hs in need.values())
        log(f"{model}: {n_todo:,} of {n_need:,} (sequence, line) pairs need alignment")
        bd = GWC / "batches" / f"{model}_{time.strftime('%Y%m%d_%H%M%S')}"
        args = [(ln, {h: seqs[h] for h in hs}, {h: expect[h] for h in hs}, bd, max(1, threads // jobs))
                for ln, hs in sorted(todo.items(), key=lambda x: -len(x[1])) if hs]
        if args:
            used = {h for a in args for h in a[1]}
            _write_batch_meta(bd, {h: seqs[h] for h in used}, expect, {a[0]: a[1].keys() for a in args})
            with ProcessPoolExecutor(min(jobs, len(args))) as ex:
                for ln, rows, info in ex.map(_align_job, args):
                    append_scores(ln, rows)
                    for r in rows:
                        sc[(r["seq_hash"], ln)] = _int(r)
                    log(f"{model}: {ln} {info['n']} queries, minimap2 {info['mm2_s']:.0f}s, rescued "
                        f"{info['rescued']}/{info['off']}")
        for ln, hs in need.items():               # all-N sequences are never aligned
            for h in hs:
                if h not in seqs:
                    sc[(h, ln)] = gw_align.all_n_row(h, ln, *qinfo[h])
        step("align", t, n_todo)

        t = time.time()
        gw_score.score(GWC / "windows.tsv", panel_tsv, out / "model_haps.tsv", sc, GWC / "ril_truth.tsv", out)
        hw = founder_choice(out / "per_hapwin.tsv", sc, panel_tsv)
        step("score", t)
    t = time.time()
    summ = write_summaries(out, hw)
    step("summarize", t)
    res.append(("total", "", round(time.time() - T0, 1)))
    with open(out / "resources.tsv", "w") as fo:
        fo.write("step\tn_items\twall_s\n")
        for r in res:
            fo.write("\t".join(map(str, r)) + "\n")
    (out / "tags.json").write_text(json.dumps(tags))
    (out / "scorer.json").write_text(json.dumps({"version": version, **V}))
    return summ


def rescore_model(model, out_root=None, version=DEFAULT_VERSION):
    """Re-run gw_score + founder choice + summaries for a model from its cached model_haps.tsv and the shared
    score cache (no projection, stitching or alignment). Use after a gw_score / summary rule change."""
    out = outdir(model, version, out_root)
    panel_tsv = GWC / VERSIONS[version]["panel"]
    t0 = time.time()
    with open(GWC / "lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        sc = load_score_cache()
        for r in csv.DictReader(open(out / "model_haps.tsv"), delimiter="\t"):
            h, qlen, nN = r["seq_hash"], int(r["seq_len"]), int(r["n_N"])
            for ln in gw_align.truth_lines(r["kind"], r["sample"]):
                if (h, ln) not in sc:
                    if qlen > nN:
                        raise RuntimeError(f"{model}: ({h}, {ln}) not in the score cache; run the full eval")
                    sc[(h, ln)] = gw_align.all_n_row(h, ln, qlen, nN)
        gw_score.score(GWC / "windows.tsv", panel_tsv, out / "model_haps.tsv", sc, GWC / "ril_truth.tsv", out)
        hw = founder_choice(out / "per_hapwin.tsv", sc, panel_tsv)
    summ = write_summaries(out, hw)
    log(f"{model}: rescored in {time.time() - t0:.0f}s")
    return summ


def write_summaries(out, hw=None):
    """summary.tsv and per_sample.tsv of a model's genewindow dir (from per_hapwin.tsv if hw is None)."""
    import pandas as pd
    out = Path(out)
    if hw is None:
        hw = pd.read_csv(out / "per_hapwin.tsv", sep="\t", dtype={"depth": str}, low_memory=False)
        hw["class"] = pd.Categorical(hw["class"], CLASSES, ordered=True)
    summ = summarize(hw)
    summ.to_csv(out / "summary.tsv", sep="\t", index=False, float_format="%.6g")
    g = hw.groupby(["depth", "class", "sample"], observed=True)
    ps = g.apply(gw_report.summarize, include_groups=False).join(g.apply(_rates, include_groups=False))
    ps.reset_index().to_csv(out / "per_sample.tsv", sep="\t", index=False, float_format="%.5g")
    return summ


def founder_choice(per_hapwin, sc, panel_tsv=GWC / "panel.tsv"):
    """Add pick_rank / n_ranked / pick_panel_errors to per_hapwin.tsv (rewritten in place) and return it
    as a DataFrame."""
    import pandas as pd
    perr = defaultdict(dict)                       # (wid, line) -> {founder: errors of its own window}
    for r in csv.DictReader(open(panel_tsv), delimiter="\t"):
        if r["status"] != "ok":
            continue
        for line in gw_align.ALL_LINES:
            s = sc.get((r["seq_hash"], line))
            if gw_score.state(s, r["chrom"]) == "scored":
                perr[(r["wid"], line)][r["founder"]] = s["errors"]
    d = pd.read_csv(per_hapwin, sep="\t", dtype={"depth": str}, low_memory=False)
    single = (d.state == "scored") & (d.n_switches == 0) & ~d.model_founders.astype(str).str.contains(",") & \
             (d.model_founders.astype(str) != ".")
    rank = np.full(len(d), np.nan)
    nr = np.full(len(d), np.nan)
    pe = np.full(len(d), np.nan)
    for i in np.flatnonzero(single.values):
        e = perr.get((d.wid.iat[i], d.truth.iat[i]))
        f = d.model_founders.iat[i]
        if e and f in e:
            rank[i] = 1 + sum(v < e[f] for v in e.values())
            nr[i] = len(e)
            pe[i] = e[f]
    d["pick_rank"], d["n_ranked"], d["pick_panel_errors"] = rank, nr, pe
    d.to_csv(per_hapwin, sep="\t", index=False)
    d["class"] = pd.Categorical(d["class"], CLASSES, ordered=True)
    return d


def _choice(d):
    import pandas as pd
    sc = d[d.state == "scored"]
    E = sc.errors.sum()
    sf = sc[sc.pick_rank.notna()]
    wrong = sf[sf.pick_rank > 1]
    ce = wrong.ceil_errors.astype(float)
    return pd.Series({
        "sf_windows": len(sf),
        "sf_better_pct": 100 * (sf.pick_rank > 1).mean() if len(sf) else np.nan,
        "pick_rank_median": sf.pick_rank.median() if len(sf) else np.nan,
        "wrongpick_rank_median": wrong.pick_rank.median() if len(wrong) else np.nan,
        "wrongpick_excess_bp_median": float(np.median(wrong.errors - ce)) if len(wrong) else np.nan,
        "n_ranked_median": sf.n_ranked.median() if len(sf) else np.nan,
        "wrongpick_err_share": 100 * wrong.errors.sum() / E if E else np.nan,
        "excess_share": 100 * np.maximum(0, wrong.errors - ce).sum() / E if E else np.nan,
        "total_errors": E,
    })


def _rates(d):
    import pandas as pd
    sc = d[d.state == "scored"]
    called = (sc.qlen - sc.query_N).astype(float)
    ce = sc[sc.ceil_state == "scored"]
    cec = ce.ceil_called.astype(float)
    ev = ce[ce.ceil_ev_founder.astype(str) != "."]
    evc = ev.ceil_ev_called.astype(float)
    an = sc[sc.t50_anchored == 1]
    at = an[an.ceil_t50_founder.astype(str) != "."]
    atc = at.ceil_t50_called.astype(float)

    def p(num, den):
        den = float(den.sum()) if hasattr(den, "sum") else float(den)
        return 100 * float(num.sum()) / den if den else np.nan
    o = {
        "err_pct": p(sc.errors, called),
        "err_pct_capped": p(np.minimum(sc.errors, called), called),
        "ceil_err_pct_capped": p(np.minimum(ce.ceil_errors.astype(float), cec), cec),
        "t50_err_pct_capped": p(np.minimum(an.t50_errors, an.t50_called), an.t50_called),
        "t50_ceil_err_pct_capped": p(np.minimum(at.ceil_t50_errors.astype(float), atc), atc),
        "event_err_pct": p(np.minimum(sc.event_errors, called), called),
        "ceil_event_err_pct": p(np.minimum(ev.ceil_ev_errors.astype(float), evc), evc),
        "t50_event_err_pct": p(np.minimum(an.t50_event_errors, an.t50_called), an.t50_called),
        "t50_ceil_event_err_pct": p(np.minimum(at.ceil_t50_event_errors.astype(float), atc), atc),
        "pct_exact_event": 100 * float((sc.event_errors == 0).mean()) if len(sc) else np.nan,
        "indel_bp_per_event": float((sc.ins.sum() + sc.dels.sum()) / max(1, sc.ins_events.sum() + sc.del_events.sum())),
        "pct_truth_unextractable": 100 * float((d.state == "truth_unextractable").mean()) if len(d) else np.nan,
    }
    if "rand_n" in sc.columns:
        rr = sc[sc.rand_n > 0]
        rc = rr.rand_called.astype(float)
        o["rand_err_pct_capped"] = p(rr.rand_err_capped.astype(float), rc)
        o["rand_event_err_pct"] = p(rr.rand_event_errors.astype(float), rc)
    else:
        o["rand_err_pct_capped"] = o["rand_event_err_pct"] = np.nan

    def r2c(model, ceil, rand):
        return 100 * (rand - model) / (rand - ceil) if rand - ceil > 0 else np.nan
    o["r2c_event_pct"] = r2c(o["event_err_pct"], o["ceil_event_err_pct"], o["rand_event_err_pct"])
    o["r2c_bp_pct"] = r2c(o["err_pct_capped"], o["ceil_err_pct_capped"], o["rand_err_pct_capped"])
    o["event_x_ceiling"] = o["event_err_pct"] / o["ceil_event_err_pct"] if o["ceil_event_err_pct"] else np.nan
    o["err_capped_x_ceiling"] = o["err_pct_capped"] / o["ceil_err_pct_capped"] if o["ceil_err_pct_capped"] else np.nan
    return pd.Series(o)


def summarize(d):
    """Rows: (depth, level=class|scope, group) with gw_report.summarize's metrics + founder choice."""
    import pandas as pd
    parts = []
    for level, key in (("class", d["class"].astype(str)), ("scope", d["class"].astype(str).str[:3])):
        g = d.assign(group=key).groupby(["depth", "group"], observed=True)
        a = g.apply(gw_report.summarize, include_groups=False)
        b = g.apply(_choice, include_groups=False)
        c = g.apply(_rates, include_groups=False)
        x = a.join(b).join(c).reset_index()
        x.insert(1, "level", level)
        parts.append(x)
    s = pd.concat(parts, ignore_index=True)
    s["err_x_ceiling"] = s.err_per_kb / s.ceil_err_per_kb
    order = {c: i for i, c in enumerate(CLASSES + ["IDX", "MIX", "OUT"])}
    return s.sort_values(["depth", "level", "group"], key=lambda c: c.map(order) if c.name == "group" else c)


HEADLINE = ["event_err_pct", "ceil_event_err_pct", "rand_event_err_pct", "r2c_event_pct", "event_x_ceiling",
            "err_pct_capped", "ceil_err_pct_capped", "rand_err_pct_capped", "r2c_bp_pct", "err_x_ceiling",
            "pct_truth_unextractable", "pct_exact", "t50_event_err_pct", "t50_err_pct_capped", "pct_uncalled_bp", "sf_better_pct", "pick_rank_median", "wrongpick_rank_median", "excess_share",
            "wrongpick_err_share"]
