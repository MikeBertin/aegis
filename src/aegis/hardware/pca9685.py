"""Real pan/tilt servo driver via an Adafruit PCA9685 (I2C, 16-channel PWM).

UNTESTED ON HARDWARE — wiring/calibration to be validated in M3 once the
gimbal arrives. Heavy libs are imported lazily so importing this module never
requires them on a dev machine.

Wiring (Jetson Orin Nano):
    PCA9685 VCC->3V3, GND->GND, SDA->pin 3, SCL->pin 5, V+ -> external 5–6V
    servo supply (NOT the Jetson). Pan servo on `pan_cal.channel`, tilt on
    `tilt_cal.channel`. Common ground between Jetson, PCA9685 and servo supply.

Install on the Jetson:
    pip install adafruit-circuitpython-servokit
"""

from __future__ import annotations

from .base import ServoCalibration, ServoDriver, to_servo_angle


class PCA9685ServoDriver(ServoDriver):
    def __init__(
        self,
        pan_cal: ServoCalibration,
        tilt_cal: ServoCalibration,
        i2c_address: int = 0x40,
        freq_hz: int = 50,
    ) -> None:
        from adafruit_servokit import ServoKit  # lazy: hardware only

        self._kit = ServoKit(channels=16, address=i2c_address, frequency=freq_hz)
        self._pan_cal = pan_cal
        self._tilt_cal = tilt_cal
        # Centre on startup so we begin from a known pose.
        self.set_angles(0.0, 0.0)

    def set_angles(self, pan_deg: float, tilt_deg: float) -> None:
        self._kit.servo[self._pan_cal.channel].angle = to_servo_angle(
            pan_deg, self._pan_cal
        )
        self._kit.servo[self._tilt_cal.channel].angle = to_servo_angle(
            tilt_deg, self._tilt_cal
        )

    def set_channel_angle(self, channel: int, angle_deg: float) -> None:
        """Set a raw channel angle — lets the Nerf trigger servo share this board."""
        self._kit.servo[channel].angle = angle_deg
