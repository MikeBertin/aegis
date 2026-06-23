"""Target selection and aim-error maths.

Deliberately free of any heavy dependencies (no torch, no cv2) so it can be
unit-tested in isolation and reused unchanged by the M2 PID controller, which
consumes :func:`aim_error` as its input signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class Detection:
    """A single object detection in pixel coordinates."""

    class_id: int
    label: str
    confidence: float
    # Bounding box as (x1, y1, x2, y2), top-left origin, pixels.
    xyxy: tuple[float, float, float, float]

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_target(
    detections: Sequence[Detection],
    frame_w: int,
    frame_h: int,
    strategy: str = "largest",
) -> Optional[Detection]:
    """Pick one detection to track from a frame's worth of candidates.

    Strategies:
        - ``"largest"``  : biggest bounding box (closest / most prominent).
        - ``"centermost"``: detection whose centroid is nearest frame centre.

    Returns ``None`` when there is nothing to track.
    """
    if not detections:
        return None

    if strategy == "largest":
        return max(detections, key=lambda d: d.area)

    if strategy == "centermost":
        cx, cy = frame_w / 2.0, frame_h / 2.0

        def dist2(d: Detection) -> float:
            px, py = d.centroid
            return (px - cx) ** 2 + (py - cy) ** 2

        return min(detections, key=dist2)

    raise ValueError(f"Unknown target-selection strategy: {strategy!r}")


def aim_error(
    centroid: tuple[float, float],
    frame_w: int,
    frame_h: int,
) -> tuple[float, float]:
    """Normalised aim error of a target centroid relative to frame centre.

    Returns ``(ex, ey)`` each in roughly ``[-1.0, 1.0]``:
        - ``ex`` > 0  → target is to the RIGHT of centre  → pan right.
        - ``ey`` > 0  → target is BELOW centre            → tilt down.

    This is the raw signal the PID loop (M2) will drive to zero. Image-Y grows
    downward, so a positive ``ey`` genuinely means "below centre".
    """
    px, py = centroid
    ex = (px - frame_w / 2.0) / (frame_w / 2.0)
    ey = (py - frame_h / 2.0) / (frame_h / 2.0)
    return ex, ey
