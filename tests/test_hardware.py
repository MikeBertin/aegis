"""Tests for the servo-angle mapping, mock drivers, and Turret integration —
all runnable on a laptop with no hardware attached."""

import sys

sys.path.insert(0, "src")

from aegis.hardware.base import ServoCalibration, to_servo_angle  # noqa: E402
from aegis.hardware.mock import MockServoDriver, MockTrigger  # noqa: E402
from aegis.safety import SafetyGate  # noqa: E402
from aegis.tracker import Detection  # noqa: E402
from aegis.turret import Turret  # noqa: E402


# --- Servo mapping ---

def test_zero_angle_maps_to_centre():
    cal = ServoCalibration(channel=0, center_deg=90)
    assert to_servo_angle(0.0, cal) == 90.0


def test_invert_flips_direction():
    cal = ServoCalibration(channel=0, center_deg=90, invert=True)
    assert to_servo_angle(30.0, cal) == 60.0


def test_mapping_clamps_to_servo_limits():
    cal = ServoCalibration(channel=0, center_deg=90, min_deg=45, max_deg=135)
    assert to_servo_angle(200.0, cal) == 135.0
    assert to_servo_angle(-200.0, cal) == 45.0


# --- Mock trigger interlock ---

def test_mock_trigger_requires_spinup_before_fire():
    t = MockTrigger()
    try:
        t.fire()
    except RuntimeError:
        pass
    else:
        raise AssertionError("fire before spin_up should raise")
    t.spin_up()
    t.fire(2)
    assert t.shots == 2


# --- Turret integration (with mocks) ---

def ball(xyxy=(100, 100, 140, 140)):
    return Detection(class_id=0, label="sports ball", confidence=0.9, xyxy=xyxy)


def _turret():
    return Turret(MockServoDriver(), MockTrigger(), SafetyGate())


def test_update_drives_servos():
    t = _turret()
    t.update(12.0, -7.0, ball(), [ball()], aim_err=(0.0, 0.0))
    assert t.servos.last == (12.0, -7.0)


def test_disarmed_turret_never_permits():
    t = _turret()
    d = t.update(0, 0, ball(), [ball()], aim_err=(0.0, 0.0))
    assert not d.permit
    assert t.try_fire() is False
    assert t.trigger.shots == 0


def test_arm_spins_up_and_clear_shot_fires():
    t = _turret()
    t.arm()
    assert t.trigger.spinning is True
    d = t.update(0, 0, ball(), [ball()], aim_err=(0.0, 0.0))  # centred -> locked
    assert d.permit
    assert t.try_fire() is True
    assert t.trigger.shots == 1


def test_not_locked_blocks_fire_even_when_armed():
    t = _turret()
    t.arm()
    t.update(0, 0, ball(), [ball()], aim_err=(0.5, 0.5))  # large error -> unlocked
    assert t.try_fire() is False


def test_disarm_spins_down():
    t = _turret()
    t.arm()
    t.disarm()
    assert t.trigger.spinning is False
    assert t.armed is False
