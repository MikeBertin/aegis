"""Dataset plumbing for M4 — YOLO label format, train/val split, data.yaml.

Pure functions (no torch/cv2): the fiddly, bug-prone parts of a training
pipeline — coordinate conversion, deterministic splitting, config generation —
are isolated here and unit-tested, so the heavy training wrappers stay thin.

YOLO label format: one `.txt` per image, one line per object:
    ``<class_id> <cx> <cy> <w> <h>``  — all box values normalised to [0, 1].
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


def xyxy_to_yolo(
    xyxy: tuple[float, float, float, float], img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Pixel (x1,y1,x2,y2) -> normalised (cx,cy,w,h), clamped to [0,1]."""
    x1, y1, x2, y2 = xyxy
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = abs(x2 - x1) / img_w
    h = abs(y2 - y1) / img_h
    clamp = lambda v: 0.0 if v < 0.0 else 1.0 if v > 1.0 else v
    return clamp(cx), clamp(cy), clamp(w), clamp(h)


def yolo_to_xyxy(
    yolo: tuple[float, float, float, float], img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Inverse of :func:`xyxy_to_yolo` (normalised -> pixel corners)."""
    cx, cy, w, h = yolo
    px, py = cx * img_w, cy * img_h
    pw, ph = w * img_w, h * img_h
    return (px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2)


def label_line(class_id: int, yolo_box: tuple[float, float, float, float]) -> str:
    cx, cy, w, h = yolo_box
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


@dataclass(frozen=True)
class Split:
    train: list
    val: list


def split_dataset(items: Sequence, val_frac: float = 0.2, seed: int = 0) -> Split:
    """Deterministic shuffle + split. At least one item in each side when
    there are >= 2 items, so training never gets an empty val set."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0,1), got {val_frac}")
    items = list(items)
    random.Random(seed).shuffle(items)
    n_val = max(1, round(len(items) * val_frac)) if len(items) >= 2 else 0
    n_val = min(n_val, len(items) - 1) if len(items) >= 2 else 0
    return Split(train=items[n_val:], val=items[:n_val])


def data_yaml(dataset_path: str, class_names: Sequence[str]) -> str:
    """Render an Ultralytics ``data.yaml`` body."""
    lines = [
        f"path: {dataset_path}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(class_names)]
    return "\n".join(lines) + "\n"
