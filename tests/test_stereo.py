"""Tests for stereo geometry (pure)."""

import sys

sys.path.insert(0, "src")

from aegis.stereo import StereoRig  # noqa: E402


def _rig():
    # focal 800 px, 6 cm baseline, image centre at (320, 240).
    return StereoRig(focal_px=800.0, baseline_m=0.06, cx=320.0, cy=240.0)


def test_depth_disparity_round_trip():
    rig = _rig()
    for depth in (1.0, 2.5, 5.0):
        assert abs(rig.depth(rig.disparity(depth)) - depth) < 1e-9


def test_closer_target_has_larger_disparity():
    rig = _rig()
    assert rig.disparity(1.0) > rig.disparity(4.0)


def test_depth_rejects_nonpositive_disparity():
    for bad in (0.0, -3.0):
        try:
            _rig().depth(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_depth_error_grows_quadratically():
    rig = _rig()
    e1 = rig.depth_error(2.0)
    e2 = rig.depth_error(4.0)
    # Doubling range ~quadruples the range uncertainty.
    assert abs(e2 / e1 - 4.0) < 1e-6


def test_pixel_to_camera_centre_is_on_axis():
    rig = _rig()
    x, y, z = rig.pixel_to_camera(320.0, 240.0, 3.0)  # principal point
    assert (round(x, 6), round(y, 6), z) == (0.0, 0.0, 3.0)
