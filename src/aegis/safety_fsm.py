"""Turret safety state machine + operational failsafes.

Wraps :class:`~aegis.safety.SafetyGate` (the *track-all, fire-inanimate-only*
policy) with the lifecycle and failsafes a real weapon system needs:

    SAFE → ARMED → TRACKING → FIRING        (normal flow)
      ↑________________________│
    any state ──(fault)──► FAULT ──(reset)──► SAFE

On top of the gate, firing also requires:
  • **perception watchdog** — stale vision (lost camera/inference) trips FAULT;
  • **temporal confirmation** — N consecutive CLEAR frames before a shot (kills
    single-frame false positives);
  • **no-fire zones** — angular keep-out sectors (never fire toward a doorway);
  • **rate limit + magazine** — min interval between shots, finite ammo.

Every transition and shot is recorded in an audit log. Pure + fully tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

from .safety import FireDecision, SafetyGate
from .tracker import Detection


class TurretState(str, Enum):
    SAFE = "SAFE"          # disarmed, flywheels off
    ARMED = "ARMED"        # armed, no fireable lock
    TRACKING = "TRACKING"  # armed + locked on a fireable, clear target
    FIRING = "FIRING"      # a shot was just taken (transient)
    FAULT = "FAULT"        # failsafe tripped — latches until reset()


@dataclass(frozen=True)
class NoFireZone:
    """Angular keep-out sector (degrees). Firing is blocked while the aim is inside."""
    az_min: float
    az_max: float
    el_min: float = -90.0
    el_max: float = 90.0
    label: str = "zone"

    def contains(self, az: float, el: float) -> bool:
        return self.az_min <= az <= self.az_max and self.el_min <= el <= self.el_max


@dataclass(frozen=True)
class SafetyConfig:
    watchdog_timeout_s: float = 0.5
    confirm_frames: int = 5
    min_fire_interval_s: float = 0.4
    magazine: int = 6
    no_fire_zones: tuple[NoFireZone, ...] = ()


@dataclass(frozen=True)
class AuditEntry:
    t: float
    kind: str     # "transition" | "fire" | "fault"
    detail: str


@dataclass(frozen=True)
class TickResult:
    state: TurretState
    can_fire: bool
    reason: str
    changed: bool   # did the state transition this tick?


class SafetyStateMachine:
    def __init__(
        self,
        gate: Optional[SafetyGate] = None,
        config: Optional[SafetyConfig] = None,
    ) -> None:
        self.gate = gate or SafetyGate()
        self.cfg = config or SafetyConfig()
        self.state = TurretState.SAFE
        self.shots = 0
        self._confirm = 0
        self._last_fire_t = -1e9
        self.audit: list[AuditEntry] = []

    # --- transitions ---

    def _to(self, state: TurretState, t: float, reason: str) -> bool:
        if state != self.state:
            self.audit.append(
                AuditEntry(t, "transition", f"{self.state.value}->{state.value}: {reason}")
            )
            self.state = state
            return True
        return False

    def estop(self, t: float, reason: str = "e-stop") -> bool:
        """Force FAULT (latches). Use for comms loss, manual e-stop, etc."""
        self._confirm = 0
        changed = self._to(TurretState.FAULT, t, reason)
        if changed:
            self.audit.append(AuditEntry(t, "fault", reason))
        return changed

    def reset(self, t: float) -> bool:
        """Clear a FAULT back to SAFE (the only way out of FAULT)."""
        if self.state == TurretState.FAULT:
            return self._to(TurretState.SAFE, t, "reset")
        return False

    # --- per-frame update ---

    def tick(
        self,
        *,
        now: float,
        armed_switch: bool,
        target: Optional[Detection],
        detections: Sequence[Detection],
        aim: tuple[float, float],
        locked: bool,
        last_perception_t: float,
    ) -> TickResult:
        if self.state == TurretState.FAULT:
            return TickResult(self.state, False, "FAULT (reset required)", False)

        # Perception watchdog — stale vision is a failsafe condition.
        if now - last_perception_t > self.cfg.watchdog_timeout_s:
            changed = self.estop(now, "perception watchdog timeout")
            return TickResult(self.state, False, "watchdog → FAULT", changed)

        if not armed_switch:
            self._confirm = 0
            changed = self._to(TurretState.SAFE, now, "disarmed")
            return TickResult(self.state, False, "DISARMED", changed)

        changed = False
        if self.state == TurretState.SAFE:
            changed = self._to(TurretState.ARMED, now, "armed")

        decision = self.gate.evaluate(target, detections, locked=locked, armed=True)

        # Temporal confirmation: count consecutive gate-CLEAR frames.
        self._confirm = min(self._confirm + 1, self.cfg.confirm_frames) if decision.permit else 0

        # State: TRACKING when we hold a clear, locked, fireable target.
        if target is not None and locked and decision.permit:
            changed = self._to(TurretState.TRACKING, now, "target locked") or changed
        elif self.state in (TurretState.TRACKING, TurretState.FIRING):
            changed = self._to(TurretState.ARMED, now, "target lost/blocked") or changed

        reason = self._fire_reason(decision, aim, now)
        return TickResult(self.state, reason == "CLEAR", reason, changed)

    def _fire_reason(self, decision: FireDecision, aim: tuple[float, float], now: float) -> str:
        if not decision.permit:
            return decision.reason
        if self._confirm < self.cfg.confirm_frames:
            return f"confirming ({self._confirm}/{self.cfg.confirm_frames})"
        az, el = aim
        for z in self.cfg.no_fire_zones:
            if z.contains(az, el):
                return f"no-fire zone: {z.label}"
        if self.shots >= self.cfg.magazine:
            return "magazine empty"
        if now - self._last_fire_t < self.cfg.min_fire_interval_s:
            return "rate limited"
        return "CLEAR"

    # --- firing ---

    def fire(self, now: float) -> bool:
        """Record a shot if rate limit + magazine allow. Returns whether it fired.
        Call only after a tick reported ``can_fire`` (re-guarded here in depth)."""
        if self.shots >= self.cfg.magazine:
            return False
        if now - self._last_fire_t < self.cfg.min_fire_interval_s:
            return False
        self.shots += 1
        self._last_fire_t = now
        self.audit.append(AuditEntry(now, "fire", f"shot {self.shots}/{self.cfg.magazine}"))
        self._to(TurretState.FIRING, now, "fire")
        return True

    def reload(self) -> None:
        self.shots = 0
