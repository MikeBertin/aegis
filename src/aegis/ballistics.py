"""Ballistic fire-control — intercept solution for a moving target + gravity.

Once stereo (or known-size monocular) gives the target's 3D position and
velocity, and the dart's muzzle speed is known, we can solve the *real* fire-
control problem: in which direction, and how far ahead, must the turret aim so
that the dart and the target arrive at the same point at the same time —
accounting for gravity drop.

This is the classic moving-interceptor problem. It is implicit (time-of-flight
depends on the intercept range, which depends on time-of-flight), so it is
solved by iteration seeded with the closed-form no-gravity quadratic.

Frame: ``(x: right, y: up, z: forward)``, turret at the origin, metres.
Pure math, no deps — validated by a hit/miss shot simulation in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.81  # m/s², gravity (pulls -y)

Vec3 = tuple[float, float, float]


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


def _azimuth(v: Vec3) -> float:
    return math.degrees(math.atan2(v[0], v[2]))          # left/right of forward


def _elevation(v: Vec3) -> float:
    return math.degrees(math.atan2(v[1], math.hypot(v[0], v[2])))  # up/down


@dataclass(frozen=True)
class FireSolution:
    ok: bool
    tof: float                 # time of flight, s
    aim_az: float              # commanded azimuth, deg
    aim_el: float              # commanded elevation, deg (includes gravity hold-over)
    lead_deg: float            # horizontal lead vs aiming straight at the target
    holdover_deg: float        # vertical aim-up vs the target
    intercept: Vec3            # where dart and target meet


def _smallest_positive_root(a: float, b: float, c: float) -> float | None:
    """Smallest positive root of a t² + b t + c = 0, or None."""
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return None
        t = -c / b
        return t if t > 0 else None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    roots = sorted(((-b - sq) / (2 * a), (-b + sq) / (2 * a)))
    for t in roots:
        if t > 1e-9:
            return t
    return None


def intercept_time(
    p: Vec3, v: Vec3, dart_speed: float, gravity: bool = True, iters: int = 12
) -> float | None:
    """Time of flight to intercept, or None if unreachable (target too fast)."""
    # Seed with the closed-form no-gravity solution:
    #   |p + v·t| = s·t  ->  (v·v - s²)t² + 2(p·v)t + p·p = 0
    a = _dot(v, v) - dart_speed * dart_speed
    b = 2 * _dot(p, v)
    c = _dot(p, p)
    t = _smallest_positive_root(a, b, c)
    if t is None or not gravity:
        return t
    # Refine including gravity: the dart must cover the straight-line launch
    # displacement L(t) = (p + v·t) + (0, ½g t², 0) in time t at speed s.
    for _ in range(iters):
        intercept = _add(p, _scale(v, t))
        launch = (intercept[0], intercept[1] + 0.5 * G * t * t, intercept[2])
        t = _norm(launch) / dart_speed
    return t


def firing_solution(
    p: Vec3, v: Vec3, dart_speed: float, gravity: bool = True
) -> FireSolution:
    """Full fire-control solution for a target at ``p`` moving at ``v``."""
    t = intercept_time(p, v, dart_speed, gravity)
    if t is None or t <= 0:
        return FireSolution(False, 0, 0, 0, 0, 0, (0, 0, 0))

    intercept = _add(p, _scale(v, t))
    launch = (
        intercept[0],
        intercept[1] + (0.5 * G * t * t if gravity else 0.0),
        intercept[2],
    )
    return FireSolution(
        ok=True,
        tof=t,
        aim_az=_azimuth(launch),
        aim_el=_elevation(launch),
        lead_deg=_azimuth(launch) - _azimuth(p),
        holdover_deg=_elevation(launch) - _elevation(p),
        intercept=intercept,
    )


def simulate_shot(
    aim_az: float,
    aim_el: float,
    dart_speed: float,
    p0: Vec3,
    v: Vec3,
    gravity: bool = True,
    hit_radius: float = 0.14,   # ~balloon radius
    dt: float = 0.002,
    t_max: float = 3.0,
) -> tuple[bool, float, float]:
    """Fire a dart along ``(aim_az, aim_el)`` and fly both dart and target
    forward. Returns ``(hit, time, closest_approach_m)``. Used to *prove* a
    firing solution actually connects (and that naive aim doesn't)."""
    az, el = math.radians(aim_az), math.radians(aim_el)
    direction = (
        math.sin(az) * math.cos(el),
        math.sin(el),
        math.cos(az) * math.cos(el),
    )
    dart = (0.0, 0.0, 0.0)
    dvel = _scale(direction, dart_speed)
    target = p0
    t, closest = 0.0, float("inf")
    while t < t_max:
        dart = _add(dart, _scale(dvel, dt))
        if gravity:
            dvel = (dvel[0], dvel[1] - G * dt, dvel[2])
        target = _add(target, _scale(v, dt))
        d = _norm(_sub(dart, target))
        closest = min(closest, d)
        if d <= hit_radius:
            return True, t, closest
        if dart[2] > target[2] + 0.5:  # dart has flown past the target plane
            break
        t += dt
    return False, t, closest
