"""Stereo geometry — range from a calibrated stereo pair.

A horizontal stereo rig recovers depth from the horizontal *disparity* (how far
a point shifts between the left and right images):

    depth Z = focal_px · baseline_m / disparity_px

Crucially, depth error grows with the **square** of distance — stereo is sharp
up close and vague far away, which is exactly why a known-size monocular range
can beat it at long range. :meth:`StereoRig.depth_error` quantifies that.

Pure math, no deps — the geometry is unit-tested before any camera exists.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StereoRig:
    focal_px: float       # focal length in pixels (from calibration)
    baseline_m: float     # distance between the two cameras
    cx: float = 0.0       # principal point (image centre)
    cy: float = 0.0

    def depth(self, disparity_px: float) -> float:
        """Range to a point given its left/right pixel disparity."""
        if disparity_px <= 0:
            raise ValueError(f"disparity must be > 0, got {disparity_px}")
        return self.focal_px * self.baseline_m / disparity_px

    def disparity(self, depth_m: float) -> float:
        """Inverse of :meth:`depth` — the disparity a target at this range shows."""
        if depth_m <= 0:
            raise ValueError(f"depth must be > 0, got {depth_m}")
        return self.focal_px * self.baseline_m / depth_m

    def depth_error(self, depth_m: float, disparity_error_px: float = 0.5) -> float:
        """Approx range uncertainty for a sub-pixel disparity error.

        ΔZ ≈ Z² / (f·b) · Δd — grows quadratically with range.
        """
        return depth_m ** 2 / (self.focal_px * self.baseline_m) * disparity_error_px

    def pixel_to_camera(self, u: float, v: float, depth_m: float) -> tuple[float, float, float]:
        """Back-project an image pixel + its depth into a 3D camera-frame point
        ``(x: right, y: down, z: forward)`` in metres."""
        x = (u - self.cx) * depth_m / self.focal_px
        y = (v - self.cy) * depth_m / self.focal_px
        return (x, y, depth_m)
