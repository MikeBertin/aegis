"""Runtime configuration for the AEGIS targeting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # --- Perception ---
    model: str = "yolo11n.pt"  # nano: fast, downloads on first run
    conf: float = 0.35  # detection confidence threshold
    # COCO class names to keep, e.g. {"person", "sports ball"}. None = all classes.
    target_classes: Optional[set[str]] = None

    # --- Target selection ---
    # "largest" (most prominent) or "centermost" (nearest frame centre).
    strategy: str = "largest"

    # --- Camera / display ---
    camera: int = 0  # cv2 capture index
    width: int = 1280
    height: int = 720
    show: bool = True  # render the live window
    mirror: bool = True  # flip horizontally (natural webcam view)

    # --- Overlay ---
    draw_all_boxes: bool = True  # show non-target detections too

    def __post_init__(self) -> None:
        if not (0.0 < self.conf < 1.0):
            raise ValueError(f"conf must be in (0, 1), got {self.conf}")
        if self.strategy not in ("largest", "centermost"):
            raise ValueError(f"unknown strategy: {self.strategy!r}")
