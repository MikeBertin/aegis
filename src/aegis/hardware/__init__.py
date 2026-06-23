"""Hardware drivers for the AEGIS turret.

Import-safe: only the base interfaces and mocks are exported eagerly. The real
drivers (`pca9685`, `nerf`) import their hardware libraries lazily inside their
constructors, so importing this package never requires adafruit/Jetson libs on
a dev machine. Import the real drivers explicitly when on the Jetson:

    from aegis.hardware.pca9685 import PCA9685ServoDriver
    from aegis.hardware.nerf import NerfTrigger
"""

from .base import ServoCalibration, ServoDriver, Trigger, to_servo_angle
from .mock import MockServoDriver, MockTrigger

__all__ = [
    "ServoCalibration",
    "ServoDriver",
    "Trigger",
    "to_servo_angle",
    "MockServoDriver",
    "MockTrigger",
]
