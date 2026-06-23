"""Unit tests for the PID + PanTiltController (pure, no heavy deps)."""

import sys

sys.path.insert(0, "src")

from aegis.controller import (  # noqa: E402
    AxisLimits,
    PanTiltController,
    PID,
    TUNED_GAINS,
    default_pan_tilt,
)


# --- PID term behaviour ---

def test_proportional_only_is_kp_times_error():
    pid = PID(kp=2.0, ki=0.0, kd=0.0)
    assert pid.update(3.0, dt=0.1) == 6.0


def test_zero_error_zero_output():
    pid = PID(kp=5.0, ki=1.0, kd=1.0)
    assert pid.update(0.0, dt=0.1) == 0.0


def test_integral_accumulates_over_time():
    pid = PID(kp=0.0, ki=1.0, kd=0.0)
    pid.update(2.0, dt=0.5)  # integral = 1.0 -> out 1.0
    out = pid.update(2.0, dt=0.5)  # integral = 2.0 -> out 2.0
    assert out == 2.0


def test_integral_anti_windup_clamps_accumulator():
    pid = PID(kp=0.0, ki=1.0, kd=0.0, i_limit=1.0)
    for _ in range(10):
        out = pid.update(5.0, dt=1.0)  # would blow up without the clamp
    assert out == 1.0  # ki * clamped integral (1.0)


def test_derivative_responds_to_change_not_steady_state():
    pid = PID(kp=0.0, ki=0.0, kd=2.0)
    assert pid.update(1.0, dt=0.1) == 0.0  # first sample: no kick
    # error jumps 1->2 over dt=0.1 -> de/dt=10 -> out = kd*10 = 20
    assert pid.update(2.0, dt=0.1) == 20.0


def test_output_limit_clamps():
    pid = PID(kp=1000.0, ki=0.0, kd=0.0, out_limit=50.0)
    assert pid.update(1.0, dt=0.1) == 50.0


def test_reset_clears_state():
    pid = PID(kp=0.0, ki=1.0, kd=0.0)
    pid.update(5.0, dt=1.0)
    pid.reset()
    assert pid.update(0.0, dt=1.0) == 0.0


def test_update_rejects_nonpositive_dt():
    pid = PID(1, 0, 0)
    try:
        pid.update(1.0, dt=0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for dt<=0")


# --- PanTiltController ---

def _ctrl(**kw):
    lim = AxisLimits(-90, 90, max_slew_deg_s=300)
    return PanTiltController(
        pan_pid=PID(120, 20, 10, out_limit=300, i_limit=2),
        tilt_pid=PID(120, 20, 10, out_limit=300, i_limit=2),
        pan_limits=lim, tilt_limits=AxisLimits(-45, 45, 300), **kw,
    )


def test_lost_target_holds_position():
    c = _ctrl()
    c.pan, c.tilt = 12.0, -7.0
    assert c.update(None, dt=0.033) == (12.0, -7.0)


def test_travel_limits_are_respected():
    c = _ctrl()
    # Drive hard right for many steps; pan must clamp at the +90 limit.
    for _ in range(200):
        c.update((1.0, 0.0), dt=0.033)
    assert c.pan == 90.0


def test_slew_rate_limits_single_step_motion():
    c = _ctrl()
    # One step at full error: motion <= max_slew * dt = 300 * 0.1 = 30 deg.
    c.update((1.0, 0.0), dt=0.1)
    assert 0 < c.pan <= 30.0 + 1e-9


def test_default_factory_uses_tuned_gains_and_limits():
    c = default_pan_tilt()
    assert (c.pan_pid.kp, c.pan_pid.ki, c.pan_pid.kd) == (
        TUNED_GAINS["kp"], TUNED_GAINS["ki"], TUNED_GAINS["kd"],
    )
    assert (c.pan_limits.min_deg, c.pan_limits.max_deg) == (-90.0, 90.0)
    assert (c.tilt_limits.min_deg, c.tilt_limits.max_deg) == (-45.0, 45.0)
    assert c.tilt_sign == -1  # matches the geometry convention


def test_closed_loop_converges_on_static_target():
    """Sign sanity + stability: a fixed error should drive the axis toward it
    and settle, not diverge."""
    c = _ctrl()
    # Simulate observing a target 20 deg to the right (hfov 60 -> ex=20/30).
    target_az = 20.0
    hfov = 60.0
    for _ in range(300):  # ~10 s at 30 fps
        ex = (target_az - c.pan) / (hfov / 2)
        ex = max(-1.0, min(1.0, ex))
        c.update((ex, 0.0), dt=0.033)
    assert abs(c.pan - target_az) < 0.5  # settled within half a degree
