"""A Kalman filter from scratch.

Our α-β / α-β-γ trackers are *fixed-gain* approximations of a Kalman filter.
This is the real thing: it carries a **covariance** ``P`` and computes the
optimal gain ``K`` each step from how much it trusts its prediction vs the
measurement. The bonus over α-β is that ``P`` *is* the estimate's uncertainty —
it shrinks as measurements arrive and grows while coasting blind, which ties
straight into the stereo depth-error story (range variance feeds ``R``).

:class:`KalmanFilter` is the general matrix form; :class:`TargetKalman3D` is a
constant-velocity tracker that is a **drop-in for** :class:`~aegis.estimator.Estimator3D`
(returns ``(position, velocity)``), so the fire-control loop can use it unchanged.

Pure NumPy — fully tested, no torch.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """General linear Kalman filter.

    x' = F x            (predict)        P' = F P Fᵀ + Q
    y  = z - H x        (innovation)     S  = H P Hᵀ + R
    K  = P Hᵀ S⁻¹       (gain)           x  = x + K y ; P = (I - K H) P
    """

    def __init__(self, F, H, Q, R, x0, P0):
        self.F = np.atleast_2d(np.asarray(F, float))
        self.H = np.atleast_2d(np.asarray(H, float))
        self.Q = np.atleast_2d(np.asarray(Q, float))
        self.R = np.atleast_2d(np.asarray(R, float))
        self.x = np.asarray(x0, float).reshape(-1, 1)
        self.P = np.asarray(P0, float)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.ravel()

    def update(self, z):
        z = np.asarray(z, float).reshape(-1, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        ident = np.eye(self.P.shape[0])
        self.P = (ident - K @ self.H) @ self.P
        return self.x.ravel()

    def step(self, z):
        self.predict()
        return self.update(z)


class CVKalman1D:
    """Constant-velocity Kalman filter for one axis (state = [position, velocity]).

    ``dt`` is supplied per update (variable frame rate), so F and the discrete
    white-noise-acceleration Q are rebuilt each step.
    """

    def __init__(self, process_var: float = 4.0, meas_var: float = 0.02, p0: float = 10.0):
        self.q = process_var
        self.R = np.array([[meas_var]])
        self.H = np.array([[1.0, 0.0]])
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * p0
        self._init = False

    def update(self, z: float, dt: float):
        if not self._init:               # initialise on first measurement
            self.x[0, 0] = z
            self._init = True
            return self.x[0, 0], self.x[1, 0]
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        # predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        # update
        y = np.array([[z]]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0, 0], self.x[1, 0]

    @property
    def pos_var(self) -> float:
        return float(self.P[0, 0])

    def reset(self) -> None:
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 10.0
        self._init = False


class TargetKalman3D:
    """3D constant-velocity target tracker — a drop-in for Estimator3D, but with
    an uncertainty estimate. ``update`` returns ``(position, velocity)``."""

    def __init__(self, process_var: float = 4.0, meas_var: float = 0.02):
        self.k = [CVKalman1D(process_var, meas_var) for _ in range(3)]

    def reset(self) -> None:
        for k in self.k:
            k.reset()

    def update(self, p, dt):
        px, vx = self.k[0].update(p[0], dt)
        py, vy = self.k[1].update(p[1], dt)
        pz, vz = self.k[2].update(p[2], dt)
        return (px, py, pz), (vx, vy, vz)

    def position_std(self) -> tuple[float, float, float]:
        return tuple(self.k[i].pos_var ** 0.5 for i in range(3))
