"""Target state estimation — α-β filter for smoothed position + velocity.

Feedforward and lead need the target's *velocity*, and a raw frame-to-frame
difference of noisy detections is far too jittery to feed into servos. An α-β
filter (a fixed-gain constant-velocity tracker — the lightweight cousin of a
Kalman filter) gives a smoothed position and velocity in one cheap recursive
step.

Pure Python, no deps — unit-tested and identical in sim and on the Jetson.
"""

from __future__ import annotations

import math


class AlphaBeta:
    """Scalar α-β tracker.

    Predict: ``x += v·dt`` ; correct with residual ``r = z - x_pred``:
        ``x = x_pred + α·r`` , ``v = v + (β/dt)·r``.

    α governs position smoothing, β governs velocity smoothing. Stable for
    ``0 < α < 1`` and ``0 < β < 2``.
    """

    def __init__(self, alpha: float, beta: float) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        if not 0.0 < beta < 2.0:
            raise ValueError(f"beta must be in (0,2), got {beta}")
        self.alpha = alpha
        self.beta = beta
        self.x: float | None = None
        self.v: float = 0.0

    @classmethod
    def critically_damped(cls, alpha: float) -> "AlphaBeta":
        """β chosen for a critically-damped response (no ringing) given α."""
        beta = 2.0 * (2.0 - alpha) - 4.0 * math.sqrt(1.0 - alpha)
        return cls(alpha, beta)

    def reset(self) -> None:
        self.x = None
        self.v = 0.0

    def update(self, z: float, dt: float) -> tuple[float, float]:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if self.x is None:  # initialise on first measurement
            self.x = z
            self.v = 0.0
            return self.x, self.v
        x_pred = self.x + self.v * dt
        r = z - x_pred
        self.x = x_pred + self.alpha * r
        self.v = self.v + (self.beta / dt) * r
        return self.x, self.v


class TargetEstimator:
    """Two α-β trackers — one per axis — for an angular target (az, el)."""

    def __init__(self, alpha: float = 0.7) -> None:
        # alpha 0.7: near-best tracking on the sim while staying tolerant of
        # noisy real detections (lower = smoother/laggier, higher = snappier/jitterier).
        self.az = AlphaBeta.critically_damped(alpha)
        self.el = AlphaBeta.critically_damped(alpha)

    def reset(self) -> None:
        self.az.reset()
        self.el.reset()

    def update(
        self, az: float, el: float, dt: float
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ``((az, el) position, (az_v, el_v) velocity)`` — all smoothed."""
        ax, av = self.az.update(az, dt)
        ex, ev = self.el.update(el, dt)
        return (ax, ex), (av, ev)


class AlphaBetaGamma:
    """Scalar α-β-γ tracker — a constant-*acceleration* model.

    Adds an acceleration state on top of α-β, so it follows a *maneuvering*
    target (one that is speeding up or turning) instead of assuming constant
    velocity. Predict ``x += v·dt + ½a·dt²``; correct with residual ``r``:
        ``x += α·r`` , ``v += (β/dt)·r`` , ``a += (2γ/dt²)·r``.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.4, gamma: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        if not 0.0 < beta <= 2.0:
            raise ValueError(f"beta must be in (0,2], got {beta}")
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0,1], got {gamma}")
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.x: float | None = None
        self.v: float = 0.0
        self.a: float = 0.0

    def reset(self) -> None:
        self.x = None
        self.v = 0.0
        self.a = 0.0

    def update(self, z: float, dt: float) -> tuple[float, float, float]:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if self.x is None:
            self.x = z
            return self.x, 0.0, 0.0
        x_pred = self.x + self.v * dt + 0.5 * self.a * dt * dt
        v_pred = self.v + self.a * dt
        r = z - x_pred
        self.x = x_pred + self.alpha * r
        self.v = v_pred + (self.beta / dt) * r
        self.a = self.a + (2.0 * self.gamma / (dt * dt)) * r
        return self.x, self.v, self.a


class Estimator3D:
    """Three α-β trackers — one per axis — for a 3D point (x, y, z).

    Used by fire-control: stereo range + bearing gives the target's 3D
    position each frame; this smooths it into position + velocity for the
    ballistic intercept solver.
    """

    def __init__(self, alpha: float = 0.7) -> None:
        self.x = AlphaBeta.critically_damped(alpha)
        self.y = AlphaBeta.critically_damped(alpha)
        self.z = AlphaBeta.critically_damped(alpha)

    def reset(self) -> None:
        for f in (self.x, self.y, self.z):
            f.reset()

    def update(
        self, p: tuple[float, float, float], dt: float
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        px, vx = self.x.update(p[0], dt)
        py, vy = self.y.update(p[1], dt)
        pz, vz = self.z.update(p[2], dt)
        return (px, py, pz), (vx, vy, vz)


class Estimator3DCA:
    """Three α-β-γ trackers — 3D constant-acceleration estimate for a maneuvering
    target. Returns smoothed ``(position, velocity, acceleration)``."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.4, gamma: float = 0.1) -> None:
        self.x = AlphaBetaGamma(alpha, beta, gamma)
        self.y = AlphaBetaGamma(alpha, beta, gamma)
        self.z = AlphaBetaGamma(alpha, beta, gamma)

    def reset(self) -> None:
        for f in (self.x, self.y, self.z):
            f.reset()

    def update(self, p, dt):
        px, vx, ax = self.x.update(p[0], dt)
        py, vy, ay = self.y.update(p[1], dt)
        pz, vz, az = self.z.update(p[2], dt)
        return (px, py, pz), (vx, vy, vz), (ax, ay, az)
