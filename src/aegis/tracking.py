"""Predictive tracking — feedforward + lead on top of the PID loop (M2.5).

Pure feedback (PID) always trails a moving target: it needs a position error to
generate the velocity to keep up. This module removes that lag and lets the
turret aim *ahead* of the target, the way a gun director does:

    1. Reconstruct the target's absolute angle from the aim error + the turret's
       current pointing angle.
    2. Smooth it through an α-β filter -> position + velocity estimate.
    3. **Lead:** aim point = position + velocity · lead_time (dart time-of-flight).
    4. **Feedforward:** add the target's velocity straight to the servo command,
       so the PID only trims the residual.

With feedforward on and lead_time = 0 the crosshair sits *on* the moving target;
with lead_time > 0 it sits ahead of it by the predicted travel — required to
hit a moving target with a projectile that has flight time.

Pure Python — same code in the simulator, the live pipeline, and the Jetson.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .ballistics import FireSolution, firing_solution
from .controller import PanTiltController
from .estimator import Estimator3D, TargetEstimator


@dataclass
class TrackOutput:
    pan: float
    tilt: float
    # Smoothed target state (absolute angles, deg / deg-per-s).
    target_az: float = 0.0
    target_el: float = 0.0
    vel_az: float = 0.0
    vel_el: float = 0.0
    # Where we're actually aiming (the lead point), for visualisation.
    lead_az: float = 0.0
    lead_el: float = 0.0
    has_target: bool = False


class TargetTracker:
    """Wraps a PanTiltController with α-β estimation, feedforward and lead."""

    def __init__(
        self,
        controller: PanTiltController,
        estimator: Optional[TargetEstimator] = None,
        hfov: float = 60.0,
        vfov: float = 37.0,
        lead_time: float = 0.0,     # seconds to lead (≈ dart time-of-flight)
        ff_gain: float = 1.0,       # 1.0 = full velocity feedforward; 0 = off
    ) -> None:
        self.c = controller
        self.est = estimator or TargetEstimator()
        self.hfov = hfov
        self.vfov = vfov
        self.lead_time = lead_time
        self.ff_gain = ff_gain

    def reset(self) -> None:
        self.est.reset()

    def step(self, aim_error: Optional[tuple[float, float]], dt: float) -> TrackOutput:
        pan, tilt = self.c.pan, self.c.tilt
        if aim_error is None:
            self.c.update(None, dt)
            self.est.reset()  # stale velocity is worse than none
            return TrackOutput(self.c.pan, self.c.tilt, has_target=False)

        ex, ey = aim_error
        # 1. Absolute target angle (inverse of simulator.observe / aim_error).
        az = pan + ex * (self.hfov / 2.0)
        el = tilt - ey * (self.vfov / 2.0)

        # 2. Smooth -> position + velocity.
        (az_s, el_s), (az_v, el_v) = self.est.update(az, el, dt)

        # 3. Lead point.
        lead_az = az_s + az_v * self.lead_time
        lead_el = el_s + el_v * self.lead_time

        # 4. Drive the PID toward the lead point, with velocity feedforward.
        ex_lead = (lead_az - pan) / (self.hfov / 2.0)
        ey_lead = (tilt - lead_el) / (self.vfov / 2.0)
        ff = (self.ff_gain * az_v, self.ff_gain * el_v)
        self.c.update((ex_lead, ey_lead), dt, feedforward=ff)

        return TrackOutput(
            pan=self.c.pan, tilt=self.c.tilt,
            target_az=az_s, target_el=el_s, vel_az=az_v, vel_el=el_v,
            lead_az=lead_az, lead_el=lead_el, has_target=True,
        )


@dataclass
class FireControlOutput:
    pan: float
    tilt: float
    solution: Optional[FireSolution] = None
    target_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    has_target: bool = False


def _angles_to_unit(az_deg: float, el_deg: float) -> tuple[float, float, float]:
    """Bearing (deg) -> unit direction in the turret frame (x right, y up, z fwd)."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    return (math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e))


class FireControlTracker:
    """Full chain: bearing + stereo range -> 3D target state -> ballistic firing
    solution -> servo command. Unlike :class:`TargetTracker` (which uses a fixed
    lead time), the lead and gravity hold-over here are *computed* from the
    target's range, the dart's speed/drag and gravity.
    """

    def __init__(
        self,
        controller: PanTiltController,
        dart,                                   # muzzle speed or DartModel
        estimator=None,                          # Estimator3D or Estimator3DCA (accel-aware)
        hfov: float = 60.0,
        vfov: float = 37.0,
        gravity: bool = True,
        ff_gain: float = 1.0,
        latency_s: float = 0.0,                  # pipeline delay to compensate
        refine: int = 0,                         # numerical solver polish passes
    ) -> None:
        self.c = controller
        self.dart = dart
        self.est = estimator or Estimator3D()
        self.hfov = hfov
        self.vfov = vfov
        self.gravity = gravity
        self.ff_gain = ff_gain
        self.latency_s = latency_s
        self.refine = refine
        self._last_aim: Optional[tuple[float, float]] = None

    def reset(self) -> None:
        self.est.reset()
        self._last_aim = None

    def step(
        self,
        aim_error: Optional[tuple[float, float]],
        range_m: Optional[float],
        dt: float,
    ) -> FireControlOutput:
        pan, tilt = self.c.pan, self.c.tilt
        if aim_error is None or range_m is None:
            self.c.update(None, dt)
            self.reset()
            return FireControlOutput(self.c.pan, self.c.tilt, has_target=False)

        ex, ey = aim_error
        # 1. Absolute target bearing, then 3D position from bearing + range.
        az = pan + ex * (self.hfov / 2.0)
        el = tilt - ey * (self.vfov / 2.0)
        p_meas = tuple(c * range_m for c in _angles_to_unit(az, el))

        # 2. Smooth -> 3D position + velocity (+ acceleration if a CA estimator).
        est_out = self.est.update(p_meas, dt)
        if len(est_out) == 3:
            p, v, a = est_out
        else:
            p, v = est_out
            a = (0.0, 0.0, 0.0)

        # 3. Ballistic firing solution — computed lead + gravity hold-over,
        #    latency-compensated, accel-aware, optionally numerically refined.
        sol = firing_solution(
            p, v, self.dart, self.gravity,
            latency=self.latency_s, accel=a, refine=self.refine,
        )
        if not sol.ok:
            self.c.update((ex, ey), dt)  # fall back to centring on the target
            return FireControlOutput(self.c.pan, self.c.tilt, sol, p, v, True)

        # 4. Drive the turret toward the firing solution, with feedforward on the
        #    aim point's angular velocity (finite-difference of the solution).
        ff = (0.0, 0.0)
        if self._last_aim is not None:
            ff = (
                self.ff_gain * (sol.aim_az - self._last_aim[0]) / dt,
                self.ff_gain * (sol.aim_el - self._last_aim[1]) / dt,
            )
        self._last_aim = (sol.aim_az, sol.aim_el)

        err = ((sol.aim_az - pan) / (self.hfov / 2.0),
               (tilt - sol.aim_el) / (self.vfov / 2.0))
        self.c.update(err, dt, feedforward=ff)
        return FireControlOutput(self.c.pan, self.c.tilt, sol, p, v, True)
