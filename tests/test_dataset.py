"""Tests for the M4 dataset plumbing (pure — no torch/cv2)."""

import sys

sys.path.insert(0, "src")

from aegis.data.dataset import (  # noqa: E402
    data_yaml,
    label_line,
    split_dataset,
    xyxy_to_yolo,
    yolo_to_xyxy,
)


# --- Coordinate conversion ---

def test_xyxy_to_yolo_centre_box():
    # A 40x40 box centred in a 200x100 image.
    cx, cy, w, h = xyxy_to_yolo((80, 30, 120, 70), 200, 100)
    assert (cx, cy) == (0.5, 0.5)
    assert (round(w, 3), round(h, 3)) == (0.2, 0.4)


def test_yolo_xyxy_round_trip():
    box = (80, 30, 120, 70)
    yolo = xyxy_to_yolo(box, 200, 100)
    back = yolo_to_xyxy(yolo, 200, 100)
    assert tuple(round(v, 3) for v in back) == tuple(float(v) for v in box)


def test_conversion_clamps_out_of_bounds():
    cx, cy, w, h = xyxy_to_yolo((-50, -50, 250, 150), 200, 100)
    assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
    assert w == 1.0 and h == 1.0


def test_label_line_format():
    line = label_line(0, (0.5, 0.5, 0.2, 0.4))
    assert line == "0 0.500000 0.500000 0.200000 0.400000"


# --- Splitting ---

def test_split_is_deterministic_for_seed():
    items = list(range(20))
    a = split_dataset(items, val_frac=0.25, seed=7)
    b = split_dataset(items, val_frac=0.25, seed=7)
    assert a.train == b.train and a.val == b.val


def test_split_proportions_and_no_overlap():
    items = list(range(20))
    s = split_dataset(items, val_frac=0.25, seed=1)
    assert len(s.val) == 5 and len(s.train) == 15
    assert set(s.train).isdisjoint(s.val)
    assert sorted(s.train + s.val) == items


def test_split_keeps_one_in_each_for_tiny_sets():
    s = split_dataset([1, 2], val_frac=0.5, seed=0)
    assert len(s.train) == 1 and len(s.val) == 1


def test_split_rejects_bad_frac():
    for bad in (0.0, 1.0, 1.5):
        try:
            split_dataset([1, 2, 3], val_frac=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for val_frac={bad}")


# --- data.yaml ---

def test_data_yaml_lists_classes_indexed():
    y = data_yaml("/abs/ds", ["balloon", "sports ball"])
    assert "path: /abs/ds" in y
    assert "train: images/train" in y
    assert "  0: balloon" in y and "  1: sports ball" in y
