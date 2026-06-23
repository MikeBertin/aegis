"""HUD rendering — crosshair, target lock, detection boxes, aim error.

The only module that talks to cv2 for drawing. Kept separate so the targeting
logic stays headless and testable.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2

from .tracker import Detection, aim_error

# BGR colours
_GREY = (160, 160, 160)
_GREEN = (60, 220, 60)
_RED = (40, 40, 230)
_WHITE = (240, 240, 240)


def draw(
    frame,
    detections: Sequence[Detection],
    target: Optional[Detection],
    draw_all_boxes: bool = True,
) -> None:
    """Annotate ``frame`` in place with the full targeting HUD."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Frame-centre crosshair (where the barrel points).
    cv2.line(frame, (cx - 18, cy), (cx + 18, cy), _WHITE, 1)
    cv2.line(frame, (cx, cy - 18), (cx, cy + 18), _WHITE, 1)

    if draw_all_boxes:
        for d in detections:
            if d is target:
                continue
            x1, y1, x2, y2 = (int(v) for v in d.xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), _GREY, 1)
            cv2.putText(
                frame, f"{d.label} {d.confidence:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GREY, 1, cv2.LINE_AA,
            )

    if target is None:
        _hud(frame, "NO TARGET", _GREY)
        return

    x1, y1, x2, y2 = (int(v) for v in target.xyxy)
    tx, ty = (int(v) for v in target.centroid)
    cv2.rectangle(frame, (x1, y1), (x2, y2), _GREEN, 2)
    cv2.circle(frame, (tx, ty), 4, _RED, -1)
    # Line from barrel centre to target — the error the PID will null out.
    cv2.line(frame, (cx, cy), (tx, ty), _RED, 1)

    ex, ey = aim_error(target.centroid, w, h)
    cv2.putText(
        frame, f"{target.label} {target.confidence:.2f}", (x1, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREEN, 2, cv2.LINE_AA,
    )
    _hud(frame, f"LOCK {target.label}  err=({ex:+.2f},{ey:+.2f})", _GREEN)


def _hud(frame, text: str, colour) -> None:
    cv2.putText(
        frame, text, (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA,
    )
