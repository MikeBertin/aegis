"""End-to-end fire-control tracking: bearing + range -> 3D state -> firing
solution -> servo command, proven by firing a dart from the commanded aim."""

import math
import sys

sys.path.insert(0, "src")

from aegis.ballistics import DartModel, simulate_shot  # noqa: E402
from aegis.controller import default_pan_tilt  # noqa: E402
from aegis.estimator import Estimator3D, Estimator3DCA  # noqa: E402
from aegis.tracking import FireControlTracker  # noqa: E402

HFOV, VFOV = 60.0, 37.0


def _bearing(p):
    rng = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
    az = math.degrees(math.atan2(p[0], p[2]))
    el = math.degrees(math.atan2(p[1], math.hypot(p[0], p[2])))
    return az, el, rng


def _run(dart, p0, vel, gravity, seconds=2.5, fps=60, tracker=None, accel=(0, 0, 0)):
    ctrl = default_pan_tilt(tilt_sign=-1)
    fct = tracker or FireControlTracker(ctrl, dart, Estimator3D(0.7), gravity=gravity)
    fct.c = ctrl
    dt = 1.0 / fps
    last_p = p0
    for i in range(int(seconds * fps)):
        t = i * dt
        p = tuple(p0[k] + vel[k] * t + 0.5 * accel[k] * t * t for k in range(3))
        az, el, rng = _bearing(p)
        ex = (az - ctrl.pan) / (HFOV / 2)
        ey = (ctrl.tilt - el) / (VFOV / 2)
        if abs(ex) > 1 or abs(ey) > 1:
            fct.step(None, None, dt)
        else:
            fct.step((ex, ey), rng, dt)
        last_p = p
    return ctrl, last_p


def test_firecontrol_aim_hits_moving_target_with_gravity_and_drag():
    dart = DartModel(20.0, drag_k=0.05)
    p0, vel = (-0.6, 0.15, 3.0), (0.6, 0.0, 0.0)   # crossing, stays in frame
    ctrl, last_p = _run(dart, p0, vel, gravity=True)
    hit, _, close = simulate_shot(ctrl.pan, ctrl.tilt, dart, last_p, vel, gravity=True)
    assert hit, f"commanded aim missed by {close*100:.0f} cm"


def test_firecontrol_leads_a_crossing_target():
    # The turret should point ahead of the current bearing (lead), not at it.
    dart = DartModel(20.0, 0.05)
    p0, vel = (-0.6, 0.0, 3.0), (0.8, 0.0, 0.0)
    ctrl, last_p = _run(dart, p0, vel, gravity=False, seconds=2.0)
    az_now, _, _ = _bearing(last_p)
    assert ctrl.pan > az_now + 0.5   # leading into the direction of travel


def test_firecontrol_holds_over_for_gravity():
    # With gravity, a shot needs hold-over; without it the same aim flies high.
    dart = DartModel(18.0, 0.04)
    p0, vel = (0.0, 0.0, 4.0), (0.0, 0.0, 0.0)     # stationary, straight ahead
    ctrl, last_p = _run(dart, p0, vel, gravity=True, seconds=2.0)
    _, el_now, _ = _bearing(last_p)
    assert ctrl.tilt > el_now + 0.3  # aims above the target


def test_accel_aware_tracker_hits_a_maneuvering_target():
    # Constant-acceleration estimator + numerical refine on an accelerating target.
    dart = DartModel(22.0, 0.05)
    ctrl = default_pan_tilt(tilt_sign=-1)
    fct = FireControlTracker(
        ctrl, dart, Estimator3DCA(0.6, 0.5, 0.2),
        gravity=True, latency_s=0.03, refine=3,
    )
    p0, vel, acc = (-0.5, 0.0, 3.0), (0.4, 0.0, 0.0), (0.5, 0.0, 0.0)
    ctrl, last_p = _run(dart, p0, vel, gravity=True, seconds=2.0, tracker=fct, accel=acc)
    # Fire from the commanded aim; the target keeps accelerating -> must still hit.
    v_now = tuple(vel[k] + acc[k] * 2.0 for k in range(3))
    hit, _, close = simulate_shot(
        ctrl.pan, ctrl.tilt, dart, last_p, v_now, gravity=True, accel=acc
    )
    assert hit, f"missed by {close*100:.0f} cm"
