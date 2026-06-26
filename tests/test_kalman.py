"""Tests for the from-scratch Kalman filter (pure NumPy)."""

import sys

import numpy as np

sys.path.insert(0, "src")

from aegis.kalman import CVKalman1D, KalmanFilter, TargetKalman3D  # noqa: E402


def _cv_kf(dt=0.1, q=1.0, r=1.0):
    F = [[1, dt], [0, 1]]
    H = [[1, 0]]
    Q = q * np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]])
    return KalmanFilter(F, H, Q, [[r]], x0=[0, 0], P0=np.eye(2) * 10)


# --- general filter ---

def test_tracks_constant_velocity():
    kf = _cv_kf(dt=0.1, q=0.01, r=0.5)
    x, vel, dt = 0.0, 5.0, 0.1
    rng = np.random.RandomState(0)
    for _ in range(200):
        x += vel * dt
        kf.step(x + rng.randn() * 0.3)   # noisy position measurement
    pos, v = kf.x.ravel()
    assert abs(v - vel) < 0.5            # velocity locked on
    assert abs(pos - x) < 0.5


def test_covariance_shrinks_with_measurements():
    kf = _cv_kf()
    p0 = np.trace(kf.P)
    for _ in range(10):
        kf.step(1.0)
    assert np.trace(kf.P) < p0           # uncertainty falls as data arrives


def test_covariance_grows_while_coasting():
    kf = _cv_kf()
    for _ in range(10):
        kf.step(1.0)
    settled = np.trace(kf.P)
    for _ in range(5):
        kf.predict()                     # predict-only: no measurements
    assert np.trace(kf.P) > settled


def test_smooths_noisy_measurements():
    # The filtered estimate should be closer to truth than the raw measurements.
    kf = _cv_kf(dt=0.1, q=0.01, r=1.0)
    rng = np.random.RandomState(1)
    x, vel, dt = 0.0, 2.0, 0.1
    raw_err, filt_err, n = 0.0, 0.0, 150
    for _ in range(n):
        x += vel * dt
        z = x + rng.randn() * 1.0
        est = kf.step(z)[0]
        raw_err += (z - x) ** 2
        filt_err += (est - x) ** 2
    assert filt_err < raw_err            # smoothing reduces error


# --- CV 1D + 3D target tracker ---

def test_cv1d_initialises_to_first_measurement():
    k = CVKalman1D()
    pos, vel = k.update(7.0, 0.1)
    assert pos == 7.0 and vel == 0.0


def test_target_kalman3d_is_estimator_dropin():
    est = TargetKalman3D()
    p, v = est.update((1.0, 2.0, 3.0), 0.1)        # returns (pos, vel) like Estimator3D
    assert p == (1.0, 2.0, 3.0) and v == (0.0, 0.0, 0.0)


def test_target_kalman3d_uncertainty_falls_with_data():
    est = TargetKalman3D(meas_var=0.05)
    est.update((0, 0, 3.0), 0.1)
    s0 = est.position_std()
    for i in range(20):
        est.update((0, 0, 3.0 + 0.1 * i), 0.1)
    s1 = est.position_std()
    assert all(b < a for a, b in zip(s0, s1)) or s1[2] < s0[2]
