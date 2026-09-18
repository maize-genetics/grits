"""Tests for ropebwt_npy_to_matrix.py's --emit-density sidecar (the real-data
counterpart to simulate_alleles.py's own --emit-density, sharing the same
compute_window_density helper -- experiments/depth-confidence-fix/). Run via
subprocess against a tiny handcrafted --anchor-dist-npy fixture so the test
exercises the actual CLI path, not a re-derived copy of its windowing logic.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "src" / "python" / "crf" / "ropebwt_npy_to_matrix.py"


def _subprocess_env():
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    return env


def _write_fixture(tmp_path, bins):
    """K=3, one row per entry of `bins` (all on contig chr1), ternary/
    distance/labels filled with arbitrary valid values (irrelevant to the
    density sidecar, which only depends on bins.tsv's bin column)."""
    K = 3
    n = len(bins)
    counts = np.ones((n, K), dtype=np.int32)
    ternary = np.zeros((n, K), dtype=np.int32)           # all "diverged"
    dist_bp = np.zeros((n, K), dtype=np.int32)            # all colinear
    gA = np.zeros((n, 1), dtype=np.int32)
    gB = np.zeros((n, 1), dtype=np.int32)
    arr = np.concatenate([counts, ternary, dist_bp, gA, gB], axis=1)  # [n, 3K+2]
    npy_path = tmp_path / "raw.npy"
    np.save(npy_path, arr)

    bins_df = pd.DataFrame({"row": np.arange(n), "contig": ["chr1"] * n, "bin": bins})
    bins_path = tmp_path / "raw.npy.bins.tsv"
    bins_df.to_csv(bins_path, sep="\t", index=False)
    return npy_path, bins_path, K


def test_emit_density_sidecar_matches_hand_computed_values(tmp_path):
    # 20 rows, window_size=5 -> 4 complete windows, no padding (this mode
    # never pads -- a trailing partial window is dropped, not padded).
    bins = [0, 0, 1, 3, 3, 3, 4, 6, 6, 7, 9, 9, 9, 9, 10, 12, 13, 13, 14, 16]
    npy_path, bins_path, K = _write_fixture(tmp_path, bins)
    out_path = tmp_path / "out.npy"

    subprocess.run(
        [sys.executable, str(SCRIPT), "--npy", str(npy_path), "--bins", str(bins_path),
         "--num-parents", str(K), "--window-size", "5", "--anchor-dist-npy",
         "--emit-density", "--out", str(out_path)],
        cwd=REPO_ROOT / "src", check=True, capture_output=True, text=True,
        env=_subprocess_env())

    density_path = tmp_path / "out.density.npy"
    assert density_path.exists()
    density = np.load(density_path)
    assert density.shape == (4, 2)
    assert density.dtype == np.int8

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from python.crf.simulate_alleles import _encode_dist

    # window 0: bins [0,0,1,3,3] -> span=3, uniq={0,1,3}=3
    # window 1: bins [3,4,6,6,7] -> span=4, uniq={3,4,6,7}=4
    # window 2: bins [9,9,9,9,10] -> span=1, uniq={9,10}=2
    # window 3: bins [12,13,13,14,16] -> span=4, uniq={12,13,14,16}=4
    expected_span = [3, 4, 1, 4]
    expected_uniq = [3, 4, 2, 4]
    for i in range(4):
        assert density[i, 0] == _encode_dist(expected_span[i]), i
        assert density[i, 1] == _encode_dist(expected_uniq[i]), i


def test_no_emit_density_flag_writes_no_sidecar(tmp_path):
    bins = list(range(10))
    npy_path, bins_path, K = _write_fixture(tmp_path, bins)
    out_path = tmp_path / "out.npy"

    subprocess.run(
        [sys.executable, str(SCRIPT), "--npy", str(npy_path), "--bins", str(bins_path),
         "--num-parents", str(K), "--window-size", "5", "--anchor-dist-npy",
         "--out", str(out_path)],
        cwd=REPO_ROOT / "src", check=True, capture_output=True, text=True,
        env=_subprocess_env())

    assert out_path.exists()
    assert not (tmp_path / "out.density.npy").exists()


def test_emit_density_without_anchor_dist_npy_raises(tmp_path):
    bins = list(range(10))
    npy_path, bins_path, K = _write_fixture(tmp_path, bins)
    out_path = tmp_path / "out.npy"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--npy", str(npy_path), "--bins", str(bins_path),
         "--num-parents", str(K), "--window-size", "5", "--emit-density",
         "--out", str(out_path)],
        cwd=REPO_ROOT / "src", capture_output=True, text=True,
        env=_subprocess_env())

    assert result.returncode != 0
    assert "--emit-density requires --anchor-dist-npy" in result.stderr
