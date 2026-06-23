"""In-memory mock drivers — let the full M3 actuation + safety loop run and be
tested on a laptop with no hardware attached."""

from __future__ import annotations

from .base import ServoDriver, Trigger


class MockServoDriver(ServoDriver):
    def __init__(self) -> None:
        self.last: tuple[float, float] = (0.0, 0.0)
        self.history: list[tuple[float, float]] = []

    def set_angles(self, pan_deg: float, tilt_deg: float) -> None:
        self.last = (pan_deg, tilt_deg)
        self.history.append(self.last)


class MockTrigger(Trigger):
    def __init__(self) -> None:
        self.spinning = False
        self.shots = 0
        self.events: list[str] = []

    def spin_up(self) -> None:
        self.spinning = True
        self.events.append("spin_up")

    def spin_down(self) -> None:
        self.spinning = False
        self.events.append("spin_down")

    def fire(self, darts: int = 1) -> None:
        # Mirror the real interlock: flywheels must be spinning to fire.
        if not self.spinning:
            raise RuntimeError("fire() called before spin_up()")
        self.shots += darts
        self.events.append(f"fire:{darts}")
