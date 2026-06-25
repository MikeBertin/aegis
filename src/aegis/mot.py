"""SORT-style multi-target tracking.

A detector gives one frame's boxes; this turns them into persistent **tracks**
with stable IDs that survive brief occlusion and missed detections. Per frame:

    predict tracks forward → match detections by IoU → update matched →
    spawn tracks for unmatched detections → age out tracks gone too long.

A track is *tentative* until it has been seen ``min_hits`` times (rejects
one-frame false positives), then *confirmed*. Confirmed tracks coast on their
estimated velocity while unmatched, so a momentary occlusion doesn't drop the
lock or change the ID — far steadier than naive per-frame target selection.

Pure Python (reuses the α-β idea via a simple velocity smoother) — fully tested.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .safety import iou
from .tracker import Detection

Box = tuple[float, float, float, float]


def _centroid(b: Box) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


class Track:
    _next_id = 1

    def __init__(self, det: Detection, vel_smooth: float = 0.5) -> None:
        self.id = Track._next_id
        Track._next_id += 1
        cx, cy = _centroid(det.xyxy)
        self.cx, self.cy = cx, cy
        self.vx = self.vy = 0.0
        self.w = det.xyxy[2] - det.xyxy[0]
        self.h = det.xyxy[3] - det.xyxy[1]
        self.label = det.label
        self.confidence = det.confidence
        self.class_id = det.class_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.confirmed = False
        self._vs = vel_smooth

    @classmethod
    def reset_ids(cls) -> None:
        cls._next_id = 1

    def box(self) -> Box:
        return (self.cx - self.w / 2, self.cy - self.h / 2,
                self.cx + self.w / 2, self.cy + self.h / 2)

    def predicted_box(self, dt: float) -> Box:
        cx, cy = self.cx + self.vx * dt, self.cy + self.vy * dt
        return (cx - self.w / 2, cy - self.h / 2, cx + self.w / 2, cy + self.h / 2)

    def update(self, det: Detection, dt: float) -> None:
        mcx, mcy = _centroid(det.xyxy)
        if dt > 0:
            self.vx = self._vs * self.vx + (1 - self._vs) * (mcx - self.cx) / dt
            self.vy = self._vs * self.vy + (1 - self._vs) * (mcy - self.cy) / dt
        self.cx, self.cy = mcx, mcy
        self.w = det.xyxy[2] - det.xyxy[0]
        self.h = det.xyxy[3] - det.xyxy[1]
        self.label = det.label
        self.confidence = det.confidence
        self.class_id = det.class_id
        self.hits += 1
        self.age += 1
        self.time_since_update = 0

    def mark_missed(self, dt: float) -> None:
        self.cx += self.vx * dt   # coast on the estimate
        self.cy += self.vy * dt
        self.age += 1
        self.time_since_update += 1

    def as_detection(self) -> Detection:
        return Detection(self.class_id, self.label, self.confidence, self.box())


def _greedy_match(track_boxes, det_boxes, iou_threshold):
    pairs = []
    for ti, tb in enumerate(track_boxes):
        for di, db in enumerate(det_boxes):
            v = iou(tb, db)
            if v >= iou_threshold:
                pairs.append((v, ti, di))
    pairs.sort(reverse=True)
    used_t, used_d, matches = set(), set(), []
    for _, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matches.append((ti, di))
    unmatched_t = [i for i in range(len(track_boxes)) if i not in used_t]
    unmatched_d = [i for i in range(len(det_boxes)) if i not in used_d]
    return matches, unmatched_t, unmatched_d


class MultiTargetTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 10,     # frames a confirmed track may go unseen
        min_hits: int = 3,     # detections before a track is confirmed
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[Track] = []

    def update(self, detections: Sequence[Detection], dt: float = 1 / 30) -> list[Track]:
        det_boxes = [d.xyxy for d in detections]
        track_boxes = [t.predicted_box(dt) for t in self.tracks]

        matches, unmatched_t, unmatched_d = _greedy_match(
            track_boxes, det_boxes, self.iou_threshold
        )

        for ti, di in matches:
            self.tracks[ti].update(detections[di], dt)
        for ti in unmatched_t:
            self.tracks[ti].mark_missed(dt)
        for di in unmatched_d:
            self.tracks.append(Track(detections[di]))

        # Confirm and prune.
        for t in self.tracks:
            if not t.confirmed and t.hits >= self.min_hits:
                t.confirmed = True
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [t for t in self.tracks if t.confirmed and t.time_since_update == 0]


def prioritize(
    tracks: Sequence[Track], frame_w: int, frame_h: int, strategy: str = "largest"
) -> Optional[Track]:
    """Pick the primary track to engage from the confirmed set."""
    if not tracks:
        return None
    if strategy == "largest":
        return max(tracks, key=lambda t: t.w * t.h)
    if strategy == "centermost":
        cx, cy = frame_w / 2.0, frame_h / 2.0
        return min(tracks, key=lambda t: (t.cx - cx) ** 2 + (t.cy - cy) ** 2)
    if strategy == "confidence":
        return max(tracks, key=lambda t: t.confidence)
    raise ValueError(f"unknown prioritisation strategy: {strategy!r}")
