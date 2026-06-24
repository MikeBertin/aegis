"""Closed-loop turret simulator for tuning the M2 controller (no hardware).

Models the camera as an angular sensor: a target at world angles ``(az, el)``
seen by a turret pointing at ``(pan, tilt)`` produces the same normalised aim
error the real :func:`tracker.aim_error` would. Feed that to the
:class:`PanTiltController`, integrate, repeat — a faithful closed loop we can
run thousands of steps of in milliseconds to measure settling, overshoot and
tracking error before a single servo is wired.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from .controller import PanTiltController


# A target-motion profile: t (seconds) -> (azimuth_deg, elevation_deg).
Motion = Callable[[float], tuple[float, float]]


def step(az: float = 20.0, el: float = -10.0) -> Motion:
    """Target appears at a fixed offset and stays there (step response)."""
    return lambda t: (az, el)


def sine(amp_az=25.0, amp_el=12.0, freq_hz=0.4, phase=math.pi / 2) -> Motion:
    """Target sweeps sinusoidally — the moving-target tracking case."""
    w = 2 * math.pi * freq_hz
    return lambda t: (amp_az * math.sin(w * t), amp_el * math.sin(w * t + phase))


def ramp(rate_az=15.0, rate_el=0.0, az0=-20.0, el0=0.0) -> Motion:
    """Target crosses at constant angular velocity (deg/s)."""
    return lambda t: (az0 + rate_az * t, el0 + rate_el * t)


@dataclass
class SimResult:
    t: list[float] = field(default_factory=list)
    target_az: list[float] = field(default_factory=list)
    target_el: list[float] = field(default_factory=list)
    pan: list[float] = field(default_factory=list)
    tilt: list[float] = field(default_factory=list)
    err_az: list[float] = field(default_factory=list)  # pointing error, deg
    err_el: list[float] = field(default_factory=list)
    on_frame: list[bool] = field(default_factory=list)
    # Lead point the turret aims at (populated by run_tracking; = target if none).
    lead_az: list[float] = field(default_factory=list)
    lead_el: list[float] = field(default_factory=list)


def observe(
    az: float, el: float, pan: float, tilt: float, hfov: float, vfov: float
) -> tuple[float, float] | None:
    """Camera model: world/turret angles -> normalised aim error, or None.

    Matches tracker.aim_error sign conventions: ex>0 target right, ey>0 below.
    """
    ex = (az - pan) / (hfov / 2.0)
    ey = (tilt - el) / (vfov / 2.0)
    if abs(ex) > 1.0 or abs(ey) > 1.0:
        return None  # target outside the field of view
    return ex, ey


def run(
    controller: PanTiltController,
    motion: Motion,
    duration: float = 3.0,
    fps: float = 30.0,
    hfov: float = 60.0,
    vfov: float = 37.0,
) -> SimResult:
    dt = 1.0 / fps
    n = int(duration * fps)
    res = SimResult()

    for i in range(n):
        t = i * dt
        az, el = motion(t)
        err = observe(az, el, controller.pan, controller.tilt, hfov, vfov)
        controller.update(err, dt)

        res.t.append(t)
        res.target_az.append(az)
        res.target_el.append(el)
        res.pan.append(controller.pan)
        res.tilt.append(controller.tilt)
        res.err_az.append(az - controller.pan)
        res.err_el.append(el - controller.tilt)
        res.on_frame.append(err is not None)
    return res


def run_tracking(
    tracker,
    motion: Motion,
    duration: float = 4.0,
    fps: float = 30.0,
    hfov: float = 60.0,
    vfov: float = 37.0,
) -> SimResult:
    """Closed loop driven by a TargetTracker (feedforward + lead, M2.5)."""
    dt = 1.0 / fps
    n = int(duration * fps)
    res = SimResult()
    tracker.hfov, tracker.vfov = hfov, vfov

    for i in range(n):
        t = i * dt
        az, el = motion(t)
        err = observe(az, el, tracker.c.pan, tracker.c.tilt, hfov, vfov)
        out = tracker.step(err, dt)

        res.t.append(t)
        res.target_az.append(az)
        res.target_el.append(el)
        res.pan.append(out.pan)
        res.tilt.append(out.tilt)
        res.err_az.append(az - out.pan)
        res.err_el.append(el - out.tilt)
        res.on_frame.append(err is not None)
        res.lead_az.append(out.lead_az if out.has_target else az)
        res.lead_el.append(out.lead_el if out.has_target else el)
    return res


# --- Metrics -------------------------------------------------------------

def _rms(xs: list[float]) -> float:
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else 0.0


def step_metrics(res: SimResult, tol_deg: float = 0.5) -> dict[str, float]:
    """Settling time, overshoot and steady-state error for a step response."""
    final_az = res.target_az[-1]
    # Settling time: last moment the pan error leaves the tolerance band.
    settle_t = 0.0
    for t, pan in zip(res.t, res.pan):
        if abs(final_az - pan) > tol_deg:
            settle_t = t
    # Overshoot relative to the step size (azimuth axis).
    step_size = final_az - res.pan[0]
    if abs(step_size) > 1e-9:
        peak = max(res.pan, key=lambda p: (p - res.pan[0]) / step_size)
        overshoot = max(0.0, ((peak - res.pan[0]) / step_size - 1.0) * 100.0)
    else:
        overshoot = 0.0
    return {
        "settle_s": round(settle_t, 3),
        "overshoot_pct": round(overshoot, 1),
        "steady_err_deg": round(abs(final_az - res.pan[-1]), 3),
    }


def tracking_metrics(res: SimResult) -> dict[str, float]:
    """RMS / peak pointing error and time-on-target for a moving target."""
    err = [math.hypot(a, e) for a, e in zip(res.err_az, res.err_el)]
    return {
        "rms_err_deg": round(_rms(err), 3),
        "max_err_deg": round(max(err), 3),
        "on_frame_pct": round(100.0 * sum(res.on_frame) / len(res.on_frame), 1),
    }
