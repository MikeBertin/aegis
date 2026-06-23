"""Real Nerf trigger: flywheel motors via a GPIO-switched relay/MOSFET, dart
push via a servo on a spare PCA9685 channel.

UNTESTED ON HARDWARE — M3. Lazy imports throughout.

Wiring:
    Flywheels: motor pack -> MOSFET/relay -> switched from `flywheel_pin`
    (BCM/Jetson header pin). Motors powered from their own pack, common ground.
    Trigger: a servo on `trigger_channel` pushes the dart-advance/trigger; it
    sweeps `rest_deg` -> `push_deg` -> `rest_deg` per shot.

    SAFETY: the flywheel pin must default LOW (off) at boot and on close().
"""

from __future__ import annotations

import time

from .base import ServoDriver, Trigger


class NerfTrigger(Trigger):
    def __init__(
        self,
        servos: "ServoDriverWithChannel",
        trigger_channel: int,
        flywheel_pin: int,
        rest_deg: float = 0.0,
        push_deg: float = 45.0,
        spinup_s: float = 0.5,
        push_s: float = 0.18,
    ) -> None:
        import Jetson.GPIO as GPIO  # lazy: hardware only

        self._gpio = GPIO
        self._servos = servos
        self._ch = trigger_channel
        self._pin = flywheel_pin
        self._rest, self._push = rest_deg, push_deg
        self._spinup_s, self._push_s = spinup_s, push_s

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(flywheel_pin, GPIO.OUT, initial=GPIO.LOW)  # default OFF
        self._servos.set_channel_angle(self._ch, self._rest)

    def spin_up(self) -> None:
        self._gpio.output(self._pin, self._gpio.HIGH)
        time.sleep(self._spinup_s)  # let flywheels reach speed before firing

    def spin_down(self) -> None:
        self._gpio.output(self._pin, self._gpio.LOW)

    def fire(self, darts: int = 1) -> None:
        for _ in range(darts):
            self._servos.set_channel_angle(self._ch, self._push)
            time.sleep(self._push_s)
            self._servos.set_channel_angle(self._ch, self._rest)
            time.sleep(self._push_s)

    def close(self) -> None:
        self.spin_down()
        self._gpio.cleanup(self._pin)


class ServoDriverWithChannel(ServoDriver):
    """Marker protocol: a servo driver that can also set an arbitrary channel
    (so the trigger servo can share the PCA9685). Implemented on the real
    driver in M3; kept here to document the contract."""

    def set_channel_angle(self, channel: int, angle_deg: float) -> None:  # pragma: no cover
        raise NotImplementedError
