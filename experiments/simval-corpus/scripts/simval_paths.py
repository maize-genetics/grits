"""
Central path/config overrides for the full-corpus affinity-model evaluation
run (see /home/zrm22/.claude/plans/swift-chasing-melody.md).

Every v1->v2 refmap-index/panel-VCF override lives here, in one place, so
nothing in the shared heldout_assembly_eval.py / nam_baseline.py / test_crf_
relatedness tree needs to be edited -- simval_eval_one.py patches these
module globals in-process after importing them.

The v2 index and panel are separately verified complete (HANDOFF.md
2026-08-08 entry): rope_bwt_index_v2/ has all four index artifacts present
(.fmd 4.96GB, .fmd.ssa 3.45GB, .fmd.len.gz, .lift 173MB), same 25 NAM
founders as v1 in the same order, only the reference coordinate system
changed (v1 B73 -> v2 B73, same chr1-10 lengths -- confirmed byte-identical
in .fmd.len.gz, so nam_baseline.LABELS_TEMPLATE needs no change).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# v1 -> v2 refmap index / panel VCF
# ---------------------------------------------------------------------------

REFMAP_ROOT = Path("/workdir/zrm22/HackathonJun2026/ropebwt_refMap")
FMD_V2 = REFMAP_ROOT / "rope_bwt_index_v2/maizeFastaIndex_SampleContig_v2.fmd"
LIFT_V2 = REFMAP_ROOT / "rope_bwt_index_v2/maizeFastaIndex_SampleContig_v2.lift"

PANEL_VCF_V2 = Path(
    "/local/workdir/zrm22/HackathonJun2026/grits_workdir/data/"
    "maize_v2_rebuild/panel/panel_25founders_v2.vcf"
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

CKPT_DIPLOID = Path(
    "/local/workdir/zrm22/HackathonJun2026/grits_workdir/checkpoints/"
    "diploid-affinity-sim512-h3/d-epoch=04-val_pair_acc=0.6179.ckpt"
)
SOURCE_K = 25
TARGET_K = 24

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

MANIFEST = Path(
    "/workdir/shared_files/grits_crf_evaluation/reads/maize/"
    "simulated_validation/manifest.tsv"
)

# ---------------------------------------------------------------------------
# Scratch / results / logs
# ---------------------------------------------------------------------------

GRITS_WORKDIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
SCRATCH_ROOT = GRITS_WORKDIR / "scratch/simval_eval"
RESULTS_DIR = GRITS_WORKDIR / "results"
RESULTS_TSV = RESULTS_DIR / "simval_results.tsv"
LOG_DIR = RESULTS_DIR / "simval_logs"

# bcftools query / sort intermediates in compare_gvcf_truth[_diploid].py can
# reach tens of GB per row at high --parallel-score; keep them off the
# system /tmp (shared across all jobs on this host) and onto /local instead.
TMPDIR = GRITS_WORKDIR / "scratch/simval_tmp"

# ---------------------------------------------------------------------------
# Fixed founder-drop index (see plan section 1 -- the K25->K24 drop the
# shared windowing script makes is data-dependent per run, which would
# corrupt the coverage-vs-accuracy comparison if left to vary rung to rung.
# Pinned once from a high-coverage reference run; None until the pilot sets
# it, at which point this file gets a one-line edit.)
# ---------------------------------------------------------------------------

FIXED_DROP_IDX = 23  # determined from the pilot reference run IDX-INBRED/B73/2.0x
# (2.0x coverage, ~28M reads -- the largest, most representative run in the
# pilot). NOT B73 itself (index 4, B73's own gamete column, which would be
# nonsensical to drop). Held fixed across every row in the full batch so the
# founder panel doesn't silently change between coverage rungs.
