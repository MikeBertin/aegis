"""Tests for SORT-style multi-target tracking (pure)."""

import sys

sys.path.insert(0, "src")

from aegis.mot import MultiTargetTracker, Track, prioritize  # noqa: E402
from aegis.tracker import Detection  # noqa: E402


def det(label, xyxy, conf=0.9, cid=0):
    return Detection(cid, label, conf, xyxy)


def box_at(cx, cy, w=40, h=40):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def setup_function():
    Track.reset_ids()


# --- creation / confirmation ---

def test_confirms_after_min_hits():
    mot = MultiTargetTracker(min_hits=3)
    d = det("balloon", box_at(100, 100))
    assert mot.update([d]) == []          # tentative
    assert mot.update([d]) == []
    out = mot.update([d])
    assert len(out) == 1 and out[0].confirmed and out[0].label == "balloon"


def test_two_objects_get_distinct_ids():
    mot = MultiTargetTracker(min_hits=1)
    a, b = det("balloon", box_at(100, 100)), det("balloon", box_at(400, 300))
    out = mot.update([a, b])
    assert len({t.id for t in out}) == 2


# --- identity persistence ---

def test_same_object_keeps_its_id_while_moving():
    mot = MultiTargetTracker(min_hits=1, iou_threshold=0.2)
    ids = []
    for cx in (100, 118, 136, 154):       # drifting right, boxes still overlap
        out = mot.update([det("balloon", box_at(cx, 100))])
        ids.append(out[0].id)
    assert len(set(ids)) == 1             # one stable identity


# --- occlusion survival ---

def test_track_survives_brief_occlusion_same_id():
    mot = MultiTargetTracker(min_hits=2, max_age=5, iou_threshold=0.2)
    mot.update([det("balloon", box_at(100, 100))])
    out = mot.update([det("balloon", box_at(110, 100))])
    tid = out[0].id
    # Two frames with NO detections (occluded) — track coasts, stays alive.
    mot.update([])
    mot.update([])
    # Reappears near the predicted position -> same id, not a new track.
    out = mot.update([det("balloon", box_at(130, 100))])
    assert out and out[0].id == tid


def test_track_deleted_after_max_age():
    mot = MultiTargetTracker(min_hits=1, max_age=3)
    mot.update([det("balloon", box_at(100, 100))])
    for _ in range(4):                    # exceed max_age with no detections
        mot.update([])
    assert mot.tracks == []


# --- matching ---

def test_overlapping_detection_matches_not_duplicates():
    mot = MultiTargetTracker(min_hits=1, iou_threshold=0.3)
    mot.update([det("balloon", box_at(100, 100))])
    mot.update([det("balloon", box_at(104, 102))])   # heavy overlap
    assert len(mot.tracks) == 1


# --- prioritisation ---

def test_prioritise_largest():
    mot = MultiTargetTracker(min_hits=1)
    small = det("balloon", box_at(100, 100, 20, 20))
    big = det("balloon", box_at(400, 300, 80, 90))
    out = mot.update([small, big])
    p = prioritize(out, 640, 480, "largest")
    assert p.w * p.h == 80 * 90


def test_prioritise_none_when_empty():
    assert prioritize([], 640, 480) is None


def test_velocity_estimate_tracks_motion():
    mot = MultiTargetTracker(min_hits=1, iou_threshold=0.1)
    t = None
    for cx in (100, 110, 120, 130, 140):  # +10 px each step
        out = mot.update([det("balloon", box_at(cx, 100))], dt=0.1)
        if out:
            t = out[0]
    assert t is not None and t.vx > 50    # ~100 px/s
