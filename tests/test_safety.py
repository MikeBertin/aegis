"""Tests for the fire-authorisation gate — the safety spine.

These encode the core promise: track all, fire inanimate only, never at a
living thing, never disarmed, never mid-slew, never with a person near the
target. If any of these regress, the build is unsafe.
"""

import sys

sys.path.insert(0, "src")

from aegis.safety import DEFAULT_FIREABLE, SafetyGate, iou  # noqa: E402
from aegis.tracker import Detection  # noqa: E402


def det(label, xyxy=(0, 0, 10, 10)):
    return Detection(class_id=0, label=label, confidence=0.9, xyxy=xyxy)


BALL = det("sports ball", (100, 100, 140, 140))


# --- IoU helper ---

def test_iou_disjoint_is_zero():
    assert iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0


def test_iou_identical_is_one():
    assert iou((0, 0, 2, 2), (0, 0, 2, 2)) == 1.0


# --- Gate refusals ---

def test_disarmed_never_fires():
    d = SafetyGate().evaluate(BALL, [BALL], locked=True, armed=False)
    assert not d.permit and d.reason == "DISARMED"


def test_no_target_refused():
    d = SafetyGate().evaluate(None, [], locked=True, armed=True)
    assert not d.permit


def test_person_target_never_fireable_even_if_allowlisted():
    # Misconfigure the allowlist to include 'person' — the hard denylist wins.
    gate = SafetyGate(fireable=frozenset(DEFAULT_FIREABLE | {"person"}))
    person = det("person", (100, 100, 140, 140))
    d = gate.evaluate(person, [person], locked=True, armed=True)
    assert not d.permit and "forbidden" in d.reason


def test_non_fireable_object_refused():
    chair = det("chair")
    d = SafetyGate().evaluate(chair, [chair], locked=True, armed=True)
    assert not d.permit and "non-fireable" in d.reason


def test_not_locked_refused():
    d = SafetyGate().evaluate(BALL, [BALL], locked=False, armed=True)
    assert not d.permit and d.reason == "not locked"


def test_person_near_target_blocks_fire():
    # A person whose (inflated) box overlaps the ball blocks the shot.
    person = det("person", (130, 130, 200, 260))  # touches the ball at (140,140)
    d = SafetyGate().evaluate(BALL, [BALL, person], locked=True, armed=True)
    assert not d.permit and "person near target" in d.reason


# --- Gate permit ---

def test_clear_shot_permitted():
    far_person = det("person", (600, 600, 700, 800))
    d = SafetyGate().evaluate(BALL, [BALL, far_person], locked=True, armed=True)
    assert d.permit and d.reason == "CLEAR"
    assert bool(d) is True  # FireDecision truthiness


def test_clear_shot_with_only_the_target():
    d = SafetyGate().evaluate(BALL, [BALL], locked=True, armed=True)
    assert d.permit
