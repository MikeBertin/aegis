"""Non-maximum suppression from scratch.

A detector emits many overlapping boxes per object; NMS keeps the highest-
scoring box and discards its duplicates — every remaining box overlapping it by
more than ``iou_threshold`` — then repeats down the score-sorted list until one
box per object survives.

YOLO does this internally; this is the first-principles version, checked against
``torchvision.ops.nms`` in the tests. Reuses :func:`aegis.safety.iou`.
"""

from __future__ import annotations

import math
from typing import Sequence

from .safety import iou
from .tracker import Detection

Box = tuple[float, float, float, float]


def nms(boxes: Sequence[Box], scores: Sequence[float], iou_threshold: float = 0.5) -> list[int]:
    """Greedy NMS. Returns the kept box indices, highest score first.

    Walk boxes in descending score; keep each box that hasn't been suppressed,
    and suppress every lower-scored box overlapping it by more than the threshold.
    """
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    removed: set[int] = set()
    keep: list[int] = []
    for pos, i in enumerate(order):
        if i in removed:
            continue
        keep.append(i)
        for j in order[pos + 1:]:           # only lower-scored candidates
            if j not in removed and iou(boxes[i], boxes[j]) > iou_threshold:
                removed.add(j)
    return keep


def nms_per_class(
    detections: Sequence[Detection], iou_threshold: float = 0.5
) -> list[Detection]:
    """Class-aware NMS over Detections — a person box never suppresses an
    overlapping balloon. Returns the surviving detections."""
    by_class: dict[int, list[Detection]] = {}
    for d in detections:
        by_class.setdefault(d.class_id, []).append(d)

    kept: list[Detection] = []
    for dets in by_class.values():
        boxes = [d.xyxy for d in dets]
        scores = [d.confidence for d in dets]
        kept.extend(dets[i] for i in nms(boxes, scores, iou_threshold))
    return kept


def soft_nms(
    boxes: Sequence[Box],
    scores: Sequence[float],
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    method: str = "gaussian",
    iou_threshold: float = 0.5,
) -> list[tuple[int, float]]:
    """Soft-NMS: instead of hard-removing overlapping boxes, *decay* their
    scores by overlap (Gaussian or linear), dropping only those that fall below
    ``score_threshold``. Better for crowded scenes where true objects overlap.
    Returns ``(index, decayed_score)`` pairs, highest first."""
    s = list(scores)
    remaining = list(range(len(boxes)))
    result: list[tuple[int, float]] = []
    while remaining:
        m = max(remaining, key=lambda i: s[i])
        remaining.remove(m)
        result.append((m, s[m]))
        for j in remaining:
            ov = iou(boxes[m], boxes[j])
            if method == "gaussian":
                s[j] *= math.exp(-(ov * ov) / sigma)
            elif ov > iou_threshold:        # linear
                s[j] *= (1.0 - ov)
        remaining = [j for j in remaining if s[j] > score_threshold]
    return result
