"""PID control: turn normalised aim error into pan/tilt commands.

`tracker.aim_error()` (M1) gives a normalised ``(ex, ey)`` in ~[-1, 1] each
frame. This module drives that error to zero by commanding two servos.

Conventions (documented once, relied on everywhere):
    - A PID's output is an **angular velocity** (deg/s). The pan/tilt layer
      integrates it: ``angle += velocity * dt``. This makes the gains
      physical and frame-rate independent.
    - Mount mapping: pan increases → camera looks RIGHT; tilt increases →
      camera looks DOWN. With those, ``ex`` drives pan with +1 sign and
      ``ey`` drives tilt with +1 sign (both negative feedback). Real servos
      may be mounted inverted — flip ``pan_sign`` / ``tilt_sign`` in config,
      exactly as you would on the bench.

Pure Python, no numpy/torch/cv2, so it is unit-tested in isolation and runs
identically in the simulator (M2) and on the Jetson (M3).
"""

from __future__ import annotations

from dataclasses import dataclass


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass
class PID:
    """A velocity-output PID with anti-windup and output limiting.

    Setpoint is implicitly 0 (we drive the aim error to zero), so the value
    passed to :meth:`update` *is* the error.
    """

    kp: float
    ki: float
    kd: float
    out_limit: float = 1e9  # |output| clamp (deg/s)
    i_limit: float = 1e9    # |integral accumulator| clamp (anti-windup)

    _integral: float = 0.0
    _prev_error: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        # Integral with clamped accumulator (anti-windup).
        self._integral = _clamp(
            self._integral + error * dt, -self.i_limit, self.i_limit
        )

        # Derivative on error; no kick on the first sample.
        if self._prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        return _clamp(out, -self.out_limit, self.out_limit)


@dataclass
class AxisLimits:
    min_deg: float
    max_deg: float
    max_slew_deg_s: float  # servo can't move faster than this


class PanTiltController:
    """Two PIDs driving a pan/tilt mount, with travel + slew-rate limits."""

    def __init__(
        self,
        pan_pid: PID,
        tilt_pid: PID,
        pan_limits: AxisLimits,
        tilt_limits: AxisLimits,
        pan_sign: int = 1,
        tilt_sign: int = 1,
        pan0: float = 0.0,
        tilt0: float = 0.0,
    ) -> None:
        self.pan_pid = pan_pid
        self.tilt_pid = tilt_pid
        self.pan_limits = pan_limits
        self.tilt_limits = tilt_limits
        self.pan_sign = pan_sign
        self.tilt_sign = tilt_sign
        self.pan = pan0
        self.tilt = tilt0

    def reset(self, pan0: float = 0.0, tilt0: float = 0.0) -> None:
        self.pan, self.tilt = pan0, tilt0
        self.pan_pid.reset()
        self.tilt_pid.reset()

    def update(
        self,
        error: tuple[float, float] | None,
        dt: float,
        feedforward: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[float, float]:
        """Advance one control step.

        ``error`` is ``(ex, ey)`` from :func:`tracker.aim_error`, or ``None``
        when the target is lost — in which case the turret holds position and
        the integrators freeze (no windup while blind).

        ``feedforward`` is an optional ``(pan, tilt)`` axis velocity (deg/s)
        added to the PID output *before* the slew limit. Driving it with the
        target's estimated angular velocity cancels tracking lag — the PID then
        only corrects the small residual error (see :mod:`aegis.tracking`).
        """
        if error is None:
            return self.pan, self.tilt

        ex, ey = error
        ff_pan, ff_tilt = feedforward
        # Feedforward is an actual axis velocity, so it is added after the
        # mount-sign mapping (which applies to the PID error term only).
        v_pan = self.pan_sign * self.pan_pid.update(ex, dt) + ff_pan
        v_tilt = self.tilt_sign * self.tilt_pid.update(ey, dt) + ff_tilt

        self.pan = self._step_axis(self.pan, v_pan, dt, self.pan_limits)
        self.tilt = self._step_axis(self.tilt, v_tilt, dt, self.tilt_limits)
        return self.pan, self.tilt

    @staticmethod
    def _step_axis(angle: float, velocity: float, dt: float, lim: AxisLimits) -> float:
        velocity = _clamp(velocity, -lim.max_slew_deg_s, lim.max_slew_deg_s)
        return _clamp(angle + velocity * dt, lim.min_deg, lim.max_deg)


# Gains tuned in-sim 2026-06-23 (see sim.py). Single source of truth shared by
# the simulator and the live pipeline so they stay in lockstep.
TUNED_GAINS = dict(kp=200.0, ki=8.0, kd=14.0)


def default_pan_tilt(
    kp: float = TUNED_GAINS["kp"],
    ki: float = TUNED_GAINS["ki"],
    kd: float = TUNED_GAINS["kd"],
    max_slew_deg_s: float = 300.0,  # ~MG996R under load
    pan_range: tuple[float, float] = (-90.0, 90.0),
    tilt_range: tuple[float, float] = (-45.0, 45.0),
    tilt_sign: int = -1,  # matches "tilt+ = camera up" geometry; flip on real mount
) -> PanTiltController:
    """Build a PanTiltController with the tuned gains and sane servo limits."""
    common = dict(out_limit=max_slew_deg_s, i_limit=2.0)
    return PanTiltController(
        pan_pid=PID(kp, ki, kd, **common),
        tilt_pid=PID(kp, ki, kd, **common),
        pan_limits=AxisLimits(*pan_range, max_slew_deg_s),
        tilt_limits=AxisLimits(*tilt_range, max_slew_deg_s),
        pan_sign=1,
        tilt_sign=tilt_sign,
    )
