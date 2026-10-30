"""Tests for ropebwt_npy_to_matrix.py's --emit-read-counts (the real-data
counterpart to simulate_alleles.py's own --emit-read-counts --
experiments/depth-confidence-fix/). Run via subprocess against a tiny
handcrafted --anchor-dist-npy fixture so the test exercises the actual CLI
path, not a re-derived copy of its logic.
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


def _write_fixture(tmp_path, n=10, K=3, counts=None):
    """K founders, n rows, one contig. `counts` (optional [n,K] array)
    overrides the default all-ones counts block; ternary/distance/labels
    are fixed, valid values (irrelevant to the count block itself)."""
    if counts is None:
        counts = np.ones((n, K), dtype=np.int32)
    ternary = np.zeros((n, K), dtype=np.int32)     # all "diverged"
    ternary[:, 0] = 1                               # founder 0 always MATCH
    dist_bp = np.zeros((n, K), dtype=np.int32)       # all colinear
    gA = np.zeros((n, 1), dtype=np.int32)
    gB = np.zeros((n, 1), dtype=np.int32)
    arr = np.concatenate([counts.astype(np.int32), ternary, dist_bp, gA, gB], axis=1)
    npy_path = tmp_path / "raw.npy"
    np.save(npy_path, arr)

    bins_df = pd.DataFrame({"row": np.arange(n), "contig": ["chr1"] * n, "bin": np.arange(n)})
    bins_path = tmp_path / "raw.npy.bins.tsv"
    bins_df.to_csv(bins_path, sep="\t", index=False)
    return npy_path, bins_path, K


def _run(args, tmp_path):
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + args,
        cwd=REPO_ROOT / "src", capture_output=True, text=True, env=_subprocess_env())


def test_emit_read_counts_widens_output_and_matches_source(tmp_path):
    K = 3
    counts = np.array([[5, 0, 2]] * 10, dtype=np.int32)  # constant across all 10 rows
    npy_path, bins_path, K = _write_fixture(tmp_path, n=10, K=K, counts=counts)
    out_path = tmp_path / "out.npy"

    result = _run(
        ["--npy", str(npy_path), "--bins", str(bins_path), "--num-parents", str(K),
         "--window-size", "10", "--anchor-dist-npy", "--emit-read-counts",
         "--out", str(out_path)],
        tmp_path)
    assert result.returncode == 0, result.stderr

    out = np.load(out_path)
    assert out.shape == (1, 10, 3 * K + 2)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from python.crf.simulate_alleles import _encode_dist

    count_block = out[0, :, 2 * K + 2:3 * K + 2]
    expected = _encode_dist(counts, 8.0)
    np.testing.assert_array_equal(count_block, expected)


def test_no_emit_read_counts_flag_keeps_2k2(tmp_path):
    npy_path, bins_path, K = _write_fixture(tmp_path, n=10, K=3)
    out_path = tmp_path / "out.npy"

    result = _run(
        ["--npy", str(npy_path), "--bins", str(bins_path), "--num-parents", str(K),
         "--window-size", "10", "--anchor-dist-npy", "--out", str(out_path)],
        tmp_path)
    assert result.returncode == 0, result.stderr

    out = np.load(out_path)
    assert out.shape == (1, 10, 2 * K + 2)


def test_emit_read_counts_without_anchor_dist_npy_raises(tmp_path):
    npy_path, bins_path, K = _write_fixture(tmp_path, n=10, K=3)
    out_path = tmp_path / "out.npy"

    result = _run(
        ["--npy", str(npy_path), "--bins", str(bins_path), "--num-parents", str(K),
         "--window-size", "10", "--emit-read-counts", "--out", str(out_path)],
        tmp_path)
    assert result.returncode != 0
    assert "--emit-read-counts requires --anchor-dist-npy" in result.stderr


def test_emit_read_counts_survives_drop_idx(tmp_path):
    """--drop-idx must drop the same founder from the count block too, not
    just tern/dist -- a real regression risk since counts is threaded
    through a separate code path from tern/dist."""
    K = 4
    counts = np.array([[1, 2, 3, 4]] * 10, dtype=np.int32)
    npy_path, bins_path, K = _write_fixture(tmp_path, n=10, K=K, counts=counts)
    out_path = tmp_path / "out.npy"

    result = _run(
        ["--npy", str(npy_path), "--bins", str(bins_path), "--num-parents", str(K),
         "--window-size", "10", "--anchor-dist-npy", "--emit-read-counts",
         "--drop-idx", "1", "--out", str(out_path)],
        tmp_path)
    assert result.returncode == 0, result.stderr

    out = np.load(out_path)
    K_new = K - 1
    assert out.shape == (1, 10, 3 * K_new + 2)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from python.crf.simulate_alleles import _encode_dist

    count_block = out[0, :, 2 * K_new + 2:3 * K_new + 2]
    expected = _encode_dist(np.array([[1, 3, 4]] * 10), 8.0)  # founder idx 1 dropped
    np.testing.assert_array_equal(count_block, expected)
