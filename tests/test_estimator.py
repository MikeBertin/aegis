"""Tests for the α-β target estimator (pure)."""

import sys

sys.path.insert(0, "src")

from aegis.estimator import AlphaBeta, TargetEstimator  # noqa: E402


def test_rejects_bad_gains():
    for a, b in [(0.0, 0.1), (1.0, 0.1), (0.5, 0.0), (0.5, 2.0)]:
        try:
            AlphaBeta(a, b)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for alpha={a} beta={b}")


def test_first_update_initialises_to_measurement():
    f = AlphaBeta(0.5, 0.1)
    x, v = f.update(10.0, dt=0.1)
    assert x == 10.0 and v == 0.0


def test_tracks_constant_velocity():
    # Feed a clean ramp at 12 deg/s; velocity estimate should converge to it.
    f = AlphaBeta.critically_damped(0.6)
    dt, vel = 1 / 30, 12.0
    x = 0.0
    for _ in range(120):
        x += vel * dt
        est_x, est_v = f.update(x, dt)
    assert abs(est_v - vel) < 0.5          # velocity locked on
    assert abs(est_x - x) < 0.5            # position tracks


def test_critically_damped_beta_formula():
    f = AlphaBeta.critically_damped(0.5)
    # beta = 2(2-a) - 4*sqrt(1-a) = 3 - 4*sqrt(0.5)
    assert abs(f.beta - (3.0 - 4.0 * 0.5 ** 0.5)) < 1e-9


def test_reset_clears_state():
    f = AlphaBeta(0.5, 0.1)
    f.update(5.0, 0.1)
    f.reset()
    x, v = f.update(9.0, 0.1)
    assert x == 9.0 and v == 0.0


def test_rejects_nonpositive_dt():
    f = AlphaBeta(0.5, 0.1)
    f.update(1.0, 0.1)
    try:
        f.update(2.0, 0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for dt<=0")


def test_two_axis_estimator_independent():
    est = TargetEstimator(alpha=0.6)
    (az, el), (av, ev) = est.update(3.0, -4.0, dt=0.1)
    assert (az, el) == (3.0, -4.0) and (av, ev) == (0.0, 0.0)
