"""Tests for block-matching stereo (pure NumPy)."""

import sys

import numpy as np

sys.path.insert(0, "src")

from aegis.stereo import StereoRig  # noqa: E402
from aegis.stereo_match import (  # noqa: E402
    _box_sum,
    block_match_disparity,
    disparity_to_depth,
    make_synthetic_pair,
)


def test_box_sum_matches_bruteforce():
    x = np.arange(25, dtype=float).reshape(5, 5)
    out = _box_sum(x, 3)
    # centre pixel sums its full 3×3 neighbourhood: rows1-3,cols1-3 around (2,2)
    assert out[2, 2] == x[1:4, 1:4].sum()


def test_recovers_known_disparity_per_region():
    left, right, true = make_synthetic_pair(size=128, bg_disp=4, fg_disp=16, seed=0)
    disp = block_match_disparity(left, right, max_disparity=24, block_size=7)

    # Ignore a border margin (no valid match near edges).
    m = 12
    bg = disp[m:40, 70:128 - m]          # background region
    fg = disp[50:78, 50:78]              # inside the foreground square
    assert abs(np.median(bg) - 4) <= 1
    assert abs(np.median(fg) - 16) <= 1


def test_ssd_method_also_recovers_disparity():
    left, right, _ = make_synthetic_pair(size=96, bg_disp=3, fg_disp=12, seed=1)
    disp = block_match_disparity(left, right, max_disparity=20, block_size=5, method="ssd")
    assert abs(np.median(disp[40:56, 40:56]) - 12) <= 1


def test_rejects_even_block_size():
    try:
        block_match_disparity(np.zeros((8, 8)), np.zeros((8, 8)), block_size=4)
    except ValueError:
        return
    raise AssertionError("expected ValueError for even block size")


def test_disparity_to_depth_uses_geometry():
    rig = StereoRig(focal_px=800.0, baseline_m=0.06)
    disp = np.array([[16.0, 8.0], [0.0, 32.0]])
    depth = disparity_to_depth(disp, rig)
    assert np.isclose(depth[0, 0], 800 * 0.06 / 16)       # = 3.0 m
    assert depth[0, 1] > depth[0, 0]                       # smaller disparity -> farther
    assert np.isinf(depth[1, 0])                           # zero disparity -> infinite


def test_closer_object_reads_nearer_depth():
    left, right, _ = make_synthetic_pair(size=128, bg_disp=4, fg_disp=16, seed=2)
    disp = block_match_disparity(left, right, max_disparity=24, block_size=7)
    rig = StereoRig(focal_px=500.0, baseline_m=0.06)
    depth = disparity_to_depth(disp, rig)
    near = np.median(depth[55:73, 55:73])                  # foreground (large disp)
    far = np.median(depth[12:36, 80:116])                  # background (small disp)
    assert near < far
