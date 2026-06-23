"""Hardware abstraction: driver interfaces + pure servo-angle mapping.

The controller speaks in axis angles (pan in [-90, 90], tilt in [-45, 45], 0 =
centre). Real servos want a 0–180 command with per-axis centre, travel limits
and possible inversion. :func:`to_servo_angle` does that mapping — pure and
tested, so the geometry is correct before any hardware is connected.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ServoCalibration:
    channel: int
    center_deg: float = 90.0      # servo command that points the axis at 0
    min_deg: float = 0.0          # servo travel limits (protect the gimbal)
    max_deg: float = 180.0
    invert: bool = False          # flip if the servo is mounted reversed
    deg_per_unit: float = 1.0     # gear ratio: controller-deg -> servo-deg


def to_servo_angle(angle_deg: float, cal: ServoCalibration) -> float:
    """Map a controller axis angle to a clamped servo command angle."""
    a = -angle_deg if cal.invert else angle_deg
    servo = cal.center_deg + a * cal.deg_per_unit
    return max(cal.min_deg, min(cal.max_deg, servo))


class ServoDriver(ABC):
    """Drives the pan/tilt gimbal."""

    @abstractmethod
    def set_angles(self, pan_deg: float, tilt_deg: float) -> None:
        ...

    def close(self) -> None:  # optional cleanup
        pass


class Trigger(ABC):
    """Drives the flywheel motors and the dart-push trigger."""

    @abstractmethod
    def spin_up(self) -> None:
        ...

    @abstractmethod
    def spin_down(self) -> None:
        ...

    @abstractmethod
    def fire(self, darts: int = 1) -> None:
        ...

    def close(self) -> None:
        pass
