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


@dataclass(frozen=True)
class DartModel:
    """Dart with optional quadratic aerodynamic drag.

    Foam darts are light and high-drag: speed decays with distance as
    ``v(s) = v0·e^(-k·s)`` (from dv/ds = -k·v). That stretches the time-of-
    flight to ``t(L) = (e^(k·L) - 1) / (k·v0)`` instead of ``L/v0`` — so drag
    *increases* both the required lead and the gravity hold-over. ``drag_k = 0``
    recovers the ideal constant-speed dart.
    """

    muzzle_speed: float
    drag_k: float = 0.0   # quadratic drag coefficient, 1/m

    def time_of_flight(self, distance: float) -> float:
        if self.drag_k <= 1e-9:
            return distance / self.muzzle_speed
        return (math.exp(self.drag_k * distance) - 1.0) / (self.drag_k * self.muzzle_speed)

    def speed_at(self, distance: float) -> float:
        return self.muzzle_speed * math.exp(-self.drag_k * distance)


def _as_dart(dart) -> DartModel:
    """Accept a bare muzzle speed (float) or a full DartModel."""
    return dart if isinstance(dart, DartModel) else DartModel(float(dart), 0.0)


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
    p: Vec3, v: Vec3, dart, gravity: bool = True, iters: int = 16
) -> float | None:
    """Time of flight to intercept, or None if unreachable (target too fast).

    ``dart`` is a muzzle speed (float) or a :class:`DartModel` (with drag).
    """
    dm = _as_dart(dart)
    # Seed with the closed-form no-gravity, no-drag solution:
    #   |p + v·t| = s·t  ->  (v·v - s²)t² + 2(p·v)t + p·p = 0
    a = _dot(v, v) - dm.muzzle_speed * dm.muzzle_speed
    b = 2 * _dot(p, v)
    c = _dot(p, p)
    t = _smallest_positive_root(a, b, c)
    if t is None or (not gravity and dm.drag_k <= 1e-9):
        return t
    # Refine: the dart must cover the straight-line launch displacement
    # L(t) = (p + v·t) + (0, ½g t², 0); with drag the time to cover |L| is
    # dm.time_of_flight(|L|) rather than |L|/speed.
    for _ in range(iters):
        intercept = _add(p, _scale(v, t))
        launch = (intercept[0], intercept[1] + (0.5 * G * t * t if gravity else 0.0), intercept[2])
        t = dm.time_of_flight(_norm(launch))
    return t


def firing_solution(
    p: Vec3,
    v: Vec3,
    dart,
    gravity: bool = True,
    latency: float = 0.0,
    accel: Vec3 = (0.0, 0.0, 0.0),
    refine: int = 0,
) -> FireSolution:
    """Full fire-control solution for a target at ``p`` moving at ``v``.

    ``dart``   — muzzle speed (float) or a :class:`DartModel` (with drag).
    ``latency``— pipeline delay (inference + actuation) before the dart launches;
                 the target is predicted forward by it, so the lead also covers
                 system latency, not just dart flight time.
    ``accel``  — target acceleration, used over the latency window and (via
                 ``refine``) the flight for a maneuvering target.
    ``refine`` — numerical polish passes: fly the shot and null the closest-
                 approach miss. Closes the heavy-drag / acceleration gap the
                 closed-form seed leaves.
    """
    # Advance the target through the pipeline latency before launch.
    p0 = _add(p, _add(_scale(v, latency), _scale(accel, 0.5 * latency * latency)))
    v0 = _add(v, _scale(accel, latency))

    t = intercept_time(p0, v0, dart, gravity)
    if t is None or t <= 0:
        return FireSolution(False, 0, 0, 0, 0, 0, (0, 0, 0))

    intercept = _add(p0, _scale(v0, t))
    launch = (
        intercept[0],
        intercept[1] + (0.5 * G * t * t if gravity else 0.0),
        intercept[2],
    )
    aim_az, aim_el = _azimuth(launch), _elevation(launch)

    if refine > 0:
        aim_az, aim_el = _refine_aim(
            aim_az, aim_el, _as_dart(dart), p0, v0, accel, gravity, refine
        )

    return FireSolution(
        ok=True,
        tof=t,
        aim_az=aim_az,
        aim_el=aim_el,
        lead_deg=aim_az - _azimuth(p),
        holdover_deg=aim_el - _elevation(p),
        intercept=intercept,
    )


def _fly(
    aim_az, aim_el, dm, p0, v, accel, gravity, dt=0.002, t_max=3.0
):
    """Integrate dart (gravity + drag) and target (v + accel); return the
    closest-approach distance, time, and the dart/target positions there."""
    az, el = math.radians(aim_az), math.radians(aim_el)
    direction = (math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el))
    dpos = (0.0, 0.0, 0.0)
    dvel = _scale(direction, dm.muzzle_speed)
    tpos, tvel = p0, v
    t, best = 0.0, (float("inf"), 0.0, dpos, tpos)
    while t < t_max:
        dpos = _add(dpos, _scale(dvel, dt))
        speed = _norm(dvel)
        dvel = (
            dvel[0] + (-dm.drag_k * speed * dvel[0]) * dt,
            dvel[1] + ((-G if gravity else 0.0) - dm.drag_k * speed * dvel[1]) * dt,
            dvel[2] + (-dm.drag_k * speed * dvel[2]) * dt,
        )
        tvel = _add(tvel, _scale(accel, dt))
        tpos = _add(tpos, _scale(tvel, dt))
        d = _norm(_sub(dpos, tpos))
        if d < best[0]:
            best = (d, t, dpos, tpos)
        if dpos[2] > tpos[2] + 0.5:
            break
        t += dt
    return best


def _refine_aim(aim_az, aim_el, dm, p0, v, accel, gravity, iters):
    """Fixed-point polish: nudge the aim by the angular miss at closest approach
    until the dart passes through the (moving, accelerating, dragging) target."""
    for _ in range(iters):
        _, _, dart_ca, tgt_ca = _fly(aim_az, aim_el, dm, p0, v, accel, gravity)
        aim_az += _azimuth(tgt_ca) - _azimuth(dart_ca)
        aim_el += _elevation(tgt_ca) - _elevation(dart_ca)
    return aim_az, aim_el


def simulate_shot(
    aim_az: float,
    aim_el: float,
    dart,
    p0: Vec3,
    v: Vec3,
    gravity: bool = True,
    accel: Vec3 = (0.0, 0.0, 0.0),
    hit_radius: float = 0.14,   # ~balloon radius
    dt: float = 0.002,
    t_max: float = 3.0,
) -> tuple[bool, float, float]:
    """Fire a dart along ``(aim_az, aim_el)`` and fly both dart (gravity + drag)
    and target (``v`` + ``accel``) forward. Returns ``(hit, time, closest_m)`` —
    the *truth* model that proves a firing solution connects (and naive doesn't)."""
    dm = _as_dart(dart)
    closest, t_ca, dart_ca, tgt_ca = _fly(
        aim_az, aim_el, dm, p0, v, accel, gravity, dt=dt, t_max=t_max
    )
    return (closest <= hit_radius, t_ca, closest)
