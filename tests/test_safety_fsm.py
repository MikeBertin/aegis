"""Tests for the turret safety state machine + failsafes (pure)."""

import sys

sys.path.insert(0, "src")

from aegis.safety import SafetyGate  # noqa: E402
from aegis.safety_fsm import (  # noqa: E402
    NoFireZone,
    SafetyConfig,
    SafetyStateMachine,
    TurretState,
)
from aegis.tracker import Detection  # noqa: E402

BALL = Detection(0, "balloon", 0.95, (100, 100, 140, 140))


def _sm(**cfg):
    return SafetyStateMachine(SafetyGate(), SafetyConfig(**cfg))


def _tick(sm, t, armed=True, target=BALL, dets=(BALL,), aim=(0, 0), locked=True, percept=None):
    return sm.tick(now=t, armed_switch=armed, target=target, detections=list(dets),
                   aim=aim, locked=locked, last_perception_t=t if percept is None else percept)


# --- lifecycle ---

def test_starts_safe_and_arms():
    sm = _sm(confirm_frames=1)
    assert sm.state == TurretState.SAFE
    r = _tick(sm, 0.0)
    assert sm.state in (TurretState.ARMED, TurretState.TRACKING)
    assert r.changed


def test_disarm_returns_to_safe():
    sm = _sm(confirm_frames=1)
    _tick(sm, 0.0)
    _tick(sm, 0.1, armed=False)
    assert sm.state == TurretState.SAFE


def test_tracking_when_locked_then_armed_when_lost():
    sm = _sm(confirm_frames=1)
    _tick(sm, 0.0)
    assert sm.state == TurretState.TRACKING
    _tick(sm, 0.1, target=None, dets=())
    assert sm.state == TurretState.ARMED


# --- watchdog / fault ---

def test_perception_watchdog_trips_fault():
    sm = _sm(watchdog_timeout_s=0.5, confirm_frames=1)
    _tick(sm, 0.0)
    r = _tick(sm, 1.0, percept=0.0)  # 1 s since last perception > 0.5
    assert sm.state == TurretState.FAULT and not r.can_fire


def test_fault_latches_until_reset():
    sm = _sm(confirm_frames=1)
    sm.estop(0.0)
    assert sm.state == TurretState.FAULT
    _tick(sm, 0.1)  # ticks don't clear it
    assert sm.state == TurretState.FAULT
    sm.reset(0.2)
    assert sm.state == TurretState.SAFE


# --- temporal confirmation ---

def test_requires_consecutive_confirm_frames_before_fire():
    sm = _sm(confirm_frames=3)
    r1 = _tick(sm, 0.0); r2 = _tick(sm, 0.03)
    assert not r1.can_fire and "confirming" in r1.reason
    assert not r2.can_fire
    r3 = _tick(sm, 0.06)
    assert r3.can_fire  # third consecutive clear frame


def test_confirmation_resets_when_blocked():
    sm = _sm(confirm_frames=2)
    _tick(sm, 0.0); _tick(sm, 0.03)  # confirmed
    _tick(sm, 0.06, locked=False)    # blocked -> resets counter
    r = _tick(sm, 0.09)
    assert not r.can_fire and "confirming" in r.reason


# --- no-fire zones ---

def test_no_fire_zone_blocks_when_aim_inside():
    sm = _sm(confirm_frames=1, no_fire_zones=(NoFireZone(-10, 10, label="doorway"),))
    r_in = _tick(sm, 0.0, aim=(0, 0))
    assert not r_in.can_fire and "no-fire zone" in r_in.reason
    r_out = _tick(sm, 0.03, aim=(40, 0))
    assert r_out.can_fire


# --- rate limit + magazine ---

def test_rate_limit_between_shots():
    sm = _sm(confirm_frames=1, min_fire_interval_s=0.4)
    _tick(sm, 0.0)
    assert sm.fire(0.0)
    r = _tick(sm, 0.1)
    assert not r.can_fire and r.reason == "rate limited"
    assert _tick(sm, 0.5).can_fire


def test_magazine_empties():
    sm = _sm(confirm_frames=1, magazine=2, min_fire_interval_s=0.0)
    _tick(sm, 0.0)
    assert sm.fire(0.0) and sm.fire(0.1)
    assert not sm.fire(0.2)  # empty
    assert _tick(sm, 0.3).reason == "magazine empty"
    sm.reload()
    assert _tick(sm, 0.4).can_fire


# --- gate delegation + audit ---

def test_delegates_to_gate_person_interlock():
    person = Detection(0, "person", 0.9, (130, 130, 220, 320))  # overlaps ball
    sm = _sm(confirm_frames=1)
    r = _tick(sm, 0.0, dets=(BALL, person))
    assert not r.can_fire and "person near target" in r.reason


def test_audit_log_records_transitions_and_fire():
    sm = _sm(confirm_frames=1)
    _tick(sm, 0.0)
    sm.fire(0.0)
    kinds = [e.kind for e in sm.audit]
    assert "transition" in kinds and "fire" in kinds
