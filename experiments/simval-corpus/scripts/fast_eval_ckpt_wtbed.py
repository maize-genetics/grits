"""Run fast_eval_ckpt.py with src.python.bed_io.bed taken from the sh-distfix-wt worktree.
heldout_assembly_eval.py hard-codes CRF_REPO_ROOT=test_crf_relatedness at sys.path[0], so its
`from src.python.bed_io.bed import output_collapse_bed` would otherwise load the unfixed bed.py there.
Preloading the `src` package from the worktree pins it in sys.modules."""
import runpy, sys
WT = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/sh-distfix-wt"
sys.path.insert(0, WT)
import src.python.bed_io.bed as b  # noqa: E402
assert b.__file__.startswith(WT), b.__file__
print("bed.py:", b.__file__, flush=True)
sys.path.remove(WT)
script = WT + "/experiments/simval-corpus/scripts/fast_eval_ckpt.py"
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
