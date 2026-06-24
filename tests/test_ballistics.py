"""Tests for the ballistic fire-control solver (pure).

The headline tests *prove* the solution by flying the dart: the firing solution
must hit a moving / dropping target where naive aim misses.
"""

import sys

sys.path.insert(0, "src")

from aegis.ballistics import (  # noqa: E402
    DartModel,
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


# --- Drag model ---

def test_drag_slows_dart_and_stretches_time_of_flight():
    dart = DartModel(20.0, drag_k=0.1)
    assert dart.speed_at(5.0) < 20.0                       # decelerates with range
    assert dart.time_of_flight(5.0) > 5.0 / 20.0           # slower => longer TOF
    assert DartModel(20.0, 0.0).time_of_flight(5.0) == 5.0 / 20.0  # k=0 is ideal


def test_drag_increases_lead_and_holdover_and_still_hits():
    p, v = (0.0, 0.0, 5.0), (3.0, 0.0, 0.0)
    ideal = firing_solution(p, v, DartModel(20.0, 0.0), gravity=True)
    dragged = firing_solution(p, v, DartModel(20.0, 0.07), gravity=True)

    assert dragged.tof > ideal.tof
    assert dragged.lead_deg > ideal.lead_deg               # must lead more
    assert dragged.holdover_deg > ideal.holdover_deg       # and hold over more

    # The dragged solution must hit when the same drag is flown (consistent model).
    hit, _, _ = simulate_shot(
        dragged.aim_az, dragged.aim_el, DartModel(20.0, 0.07), p, v, gravity=True
    )
    assert hit


def test_ignoring_drag_undershoots_a_dragging_dart():
    # Solve assuming no drag, but fly with drag -> falls short (under-led/low).
    p, v = (0.0, 0.0, 5.0), (3.0, 0.0, 0.0)
    ideal = firing_solution(p, v, 20.0, gravity=True)      # solved drag-free
    hit, _, miss = simulate_shot(
        ideal.aim_az, ideal.aim_el, DartModel(20.0, 0.1), p, v, gravity=True
    )
    assert not hit and miss > 0.14


def _el_to(p):
    import math
    return math.degrees(math.atan2(p[1], math.hypot(p[0], p[2])))
