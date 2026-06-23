"""Unit tests for the headless targeting maths (no torch/cv2 needed)."""

import sys

sys.path.insert(0, "src")

from aegis.tracker import Detection, aim_error, select_target  # noqa: E402


def _det(label, xyxy, conf=0.9, class_id=0):
    return Detection(class_id=class_id, label=label, confidence=conf, xyxy=xyxy)


# --- Detection geometry ---

def test_centroid_and_area():
    d = _det("person", (10, 20, 30, 60))
    assert d.centroid == (20.0, 40.0)
    assert d.area == 20.0 * 40.0


def test_area_clamps_to_zero_for_inverted_box():
    d = _det("x", (30, 60, 10, 20))  # x2<x1, y2<y1
    assert d.area == 0.0


# --- Target selection ---

def test_select_target_none_when_empty():
    assert select_target([], 100, 100, "largest") is None


def test_select_largest_picks_biggest_box():
    small = _det("ball", (0, 0, 10, 10))
    big = _det("person", (0, 0, 50, 80))
    assert select_target([small, big], 200, 200, "largest") is big


def test_select_centermost_picks_nearest_centre():
    centre = _det("a", (90, 90, 110, 110))  # centroid (100,100) == frame centre
    edge = _det("b", (0, 0, 20, 20))  # centroid (10,10)
    assert select_target([edge, centre], 200, 200, "centermost") is centre


def test_unknown_strategy_raises():
    try:
        select_target([_det("x", (0, 0, 1, 1))], 10, 10, "bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown strategy")


# --- Aim error sign conventions (the signal M2's PID consumes) ---

def test_aim_error_centre_is_zero():
    assert aim_error((100, 100), 200, 200) == (0.0, 0.0)


def test_aim_error_right_is_positive_x():
    ex, ey = aim_error((200, 100), 200, 200)
    assert ex > 0 and ey == 0.0


def test_aim_error_below_is_positive_y():
    ex, ey = aim_error((100, 200), 200, 200)
    assert ey > 0 and ex == 0.0


def test_aim_error_normalised_to_unit_at_edges():
    # Far corners map to +/-1 on each axis.
    assert aim_error((0, 0), 200, 200) == (-1.0, -1.0)
    assert aim_error((200, 200), 200, 200) == (1.0, 1.0)
