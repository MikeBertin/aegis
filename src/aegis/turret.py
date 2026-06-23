"""Turret integration — the M3 actuation layer.

Ties together the pieces that already exist and are tested independently:
    controller (M2)  -> pan/tilt angles -> ServoDriver
    detections (M1)  -> SafetyGate      -> gated Trigger

Driver-agnostic: hand it mocks (laptop) or the real PCA9685/Nerf drivers
(Jetson) — the logic is identical. Firing is never automatic: a shot requires
(1) the turret ARMED and (2) an explicit :meth:`try_fire` call, and even then
only proceeds if the SafetyGate permits. Human-in-the-loop by construction.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .hardware.base import ServoDriver, Trigger
from .safety import FireDecision, SafetyGate
from .tracker import Detection


class Turret:
    def __init__(
        self,
        servos: ServoDriver,
        trigger: Trigger,
        gate: SafetyGate,
        lock_tol: float = 0.08,  # |aim error| below which we count as "locked"
    ) -> None:
        self.servos = servos
        self.trigger = trigger
        self.gate = gate
        self.lock_tol = lock_tol
        self.armed = False
        self.last_decision = FireDecision(False, "DISARMED")

    # --- Arming (the human-in-the-loop switch) ---

    def arm(self) -> None:
        """Arm and spin up the flywheels. Nothing fires from arming alone."""
        self.armed = True
        self.trigger.spin_up()

    def disarm(self) -> None:
        self.armed = False
        self.trigger.spin_down()
        self.last_decision = FireDecision(False, "DISARMED")

    # --- Per-frame update ---

    def update(
        self,
        pan_deg: float,
        tilt_deg: float,
        target: Optional[Detection],
        detections: Sequence[Detection],
        aim_err: Optional[tuple[float, float]],
    ) -> FireDecision:
        """Drive the servos and (re)compute the fire decision. Never fires."""
        self.servos.set_angles(pan_deg, tilt_deg)
        locked = (
            aim_err is not None
            and abs(aim_err[0]) <= self.lock_tol
            and abs(aim_err[1]) <= self.lock_tol
        )
        self.last_decision = self.gate.evaluate(
            target, detections, locked=locked, armed=self.armed
        )
        return self.last_decision

    # --- Firing (explicit, re-gated) ---

    def try_fire(self, darts: int = 1) -> bool:
        """Fire iff the most recent decision permits. Returns whether it fired."""
        if self.last_decision.permit:
            self.trigger.fire(darts)
            return True
        return False

    def close(self) -> None:
        self.disarm()
        self.servos.close()
        self.trigger.close()
