"""Tests for feedforward + lead tracking (M2.5)."""

import math
import sys

sys.path.insert(0, "src")

from aegis import simulator as sim  # noqa: E402
from aegis.controller import default_pan_tilt  # noqa: E402
from aegis.estimator import TargetEstimator  # noqa: E402
from aegis.tracking import TargetTracker  # noqa: E402


def _tracker(lead_time=0.0, ff_gain=1.0):
    # Simulator convention: tilt+ = camera up, so tilt_sign = -1.
    ctrl = default_pan_tilt(tilt_sign=-1)
    return TargetTracker(ctrl, TargetEstimator(), lead_time=lead_time, ff_gain=ff_gain)


def _steady_rms(res, frac=0.4):
    """RMS pointing error over the tail (excludes the acquisition transient)."""
    k = int(len(res.t) * frac)
    e = [math.hypot(a, b) for a, b in zip(res.err_az[k:], res.err_el[k:])]
    return math.sqrt(sum(x * x for x in e) / len(e))


def test_feedforward_cuts_tracking_lag():
    """Velocity feedforward should dramatically reduce steady-state lag on a
    moving target vs the plain PID feedback loop."""
    motion = sim.sine(amp_az=26.0, amp_el=13.0, freq_hz=0.5)

    plain_rms = _steady_rms(sim.run(default_pan_tilt(tilt_sign=-1), motion, duration=5.0))
    tracked_rms = _steady_rms(sim.run_tracking(_tracker(), motion, duration=5.0))

    assert tracked_rms < plain_rms * 0.5   # at least halved
    assert tracked_rms < 2.5               # and genuinely small


def test_lead_aims_ahead_of_target():
    """With a positive lead time on a rightward-moving target, the lead point
    (and the turret) should sit ahead of the current target in azimuth."""
    motion = sim.ramp(rate_az=18.0, az0=0.0)  # moving +az
    res = sim.run_tracking(_tracker(lead_time=0.2), motion, duration=3.0)

    # After settling, the aim leads the current target position.
    i = int(len(res.t) * 0.8)
    assert res.lead_az[i] > res.target_az[i] + 1.0   # lead point is ahead
    assert res.pan[i] > res.target_az[i]             # turret points ahead too


def test_zero_lead_sits_on_target():
    motion = sim.ramp(rate_az=18.0, az0=0.0)
    res = sim.run_tracking(_tracker(lead_time=0.0), motion, duration=3.0)
    i = int(len(res.t) * 0.8)
    # No lead: turret sits essentially on the moving target (small residual).
    assert abs(res.pan[i] - res.target_az[i]) < 1.0


def test_lost_target_holds_and_resets():
    ctrl = default_pan_tilt(tilt_sign=-1)
    tr = TargetTracker(ctrl, TargetEstimator(0.6))
    tr.step((0.3, 0.1), dt=0.033)            # acquire
    out = tr.step(None, dt=0.033)            # lost
    assert out.has_target is False
    assert tr.est.az.x is None               # estimator reset on loss
