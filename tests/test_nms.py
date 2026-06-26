"""Tests for from-scratch NMS (pure logic; torchvision check is gated)."""

import sys

import pytest

sys.path.insert(0, "src")

from aegis.nms import nms, nms_per_class, soft_nms  # noqa: E402
from aegis.tracker import Detection  # noqa: E402


def box(cx, cy, s=20):
    return (cx - s, cy - s, cx + s, cy + s)


# --- greedy nms ---

def test_collapses_duplicates_to_highest_score():
    boxes = [box(100, 100), box(102, 101), box(98, 99)]   # heavy overlap
    scores = [0.7, 0.9, 0.6]
    keep = nms(boxes, scores, 0.5)
    assert keep == [1]                                    # only the top-scorer


def test_keeps_non_overlapping_boxes():
    boxes = [box(50, 50), box(200, 200), box(400, 100)]
    keep = nms(boxes, [0.9, 0.8, 0.7], 0.5)
    assert sorted(keep) == [0, 1, 2]


def test_threshold_controls_suppression():
    boxes = [box(100, 100), box(118, 100)]                # partial overlap
    assert len(nms(boxes, [0.9, 0.8], 0.2)) == 1          # low thr -> suppress
    assert len(nms(boxes, [0.9, 0.8], 0.9)) == 2          # high thr -> keep both


def test_empty():
    assert nms([], [], 0.5) == []


# --- class-aware ---

def test_per_class_keeps_overlapping_different_classes():
    a = Detection(0, "balloon", 0.9, box(100, 100))
    b = Detection(1, "person", 0.8, box(101, 101))        # overlaps, other class
    dup = Detection(0, "balloon", 0.5, box(99, 100))      # duplicate balloon
    kept = nms_per_class([a, b, dup], 0.5)
    labels = sorted(d.label for d in kept)
    assert labels == ["balloon", "person"] and len(kept) == 2


# --- soft-nms ---

def test_soft_nms_decays_instead_of_dropping():
    boxes = [box(100, 100), box(112, 100)]                # moderate overlap
    scores = [0.9, 0.8]
    res = dict(soft_nms(boxes, scores, sigma=0.5, score_threshold=0.0))
    assert res[0] == 0.9                                  # winner unchanged
    assert 0.0 < res[1] < 0.8                             # loser decayed, not dropped


# --- equivalence to the reference implementation ---

def test_matches_torchvision_nms():
    torch = pytest.importorskip("torch")
    tv = pytest.importorskip("torchvision")
    import random

    rng = random.Random(0)
    boxes, scores = [], []
    for _ in range(40):
        cx, cy = rng.uniform(0, 200), rng.uniform(0, 200)
        s = rng.uniform(8, 30)
        boxes.append((cx - s, cy - s, cx + s, cy + s))
        scores.append(rng.random())

    ours = set(nms(boxes, scores, 0.5))
    ref = set(tv.ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5).tolist())
    assert ours == ref
