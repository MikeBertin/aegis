"""Tests for the ballistic fire-control solver (pure).

The headline tests *prove* the solution by flying the dart: the firing solution
must hit a moving / dropping target where naive aim misses.
"""

import sys

sys.path.insert(0, "src")

from aegis.ballistics import (  # noqa: E402
    firing_solution,
    intercept_time,
    simulate_shot,
)


# Target 3 m forward, 1.5 m up (turret at origin). Dart 20 m/s.
P = (0.0, 1.5, 3.0)
S = 20.0


def test_stationary_no_gravity_aims_straight_tof_is_range_over_speed():
    sol = firing_solution(P, (0, 0, 0), S, gravity=False)
    assert sol.ok
    assert abs(sol.lead_deg) < 1e-6                 # no lead for a still target
    assert abs(sol.tof - (P[0] ** 2 + P[1] ** 2 + P[2] ** 2) ** 0.5 / S) < 1e-6


def test_moving_target_solution_hits_naive_misses():
    v = (4.0, 0.0, 0.0)  # crossing right at 4 m/s
    sol = firing_solution(P, v, S, gravity=False)
    assert sol.ok and sol.lead_deg > 1.0           # leads into the motion

    hit, _, _ = simulate_shot(sol.aim_az, sol.aim_el, S, P, v, gravity=False)
    assert hit

    # Aiming straight at where the target *is* misses a crossing target.
    naive_hit, _, miss = simulate_shot(
        0.0, _el_to(P), S, P, v, gravity=False
    )
    assert not naive_hit and miss > 0.14


def test_gravity_requires_holdover_and_still_hits():
    sol = firing_solution(P, (0, 0, 0), S, gravity=True)
    assert sol.ok
    assert sol.holdover_deg > 0.5                   # must aim above the target
    hit, _, _ = simulate_shot(sol.aim_az, sol.aim_el, S, P, (0, 0, 0), gravity=True)
    assert hit


def test_gravity_naive_aim_falls_short_low():
    # Aiming straight at the target (no hold-over) drops below it. Use 5 m, where
    # the ~33 cm drop clearly exceeds the hit radius (at 3 m the drop ~= radius).
    far = (0.0, 1.5, 5.0)
    naive_hit, _, miss = simulate_shot(0.0, _el_to(far), S, far, (0, 0, 0), gravity=True)
    assert not naive_hit and miss > 0.14


def test_unreachable_when_target_outruns_dart():
    # Target receding faster than the dart can never be caught.
    assert intercept_time((0, 0, 3.0), (0, 0, 50.0), 20.0) is None


def _el_to(p):
    import math
    return math.degrees(math.atan2(p[1], math.hypot(p[0], p[2])))
