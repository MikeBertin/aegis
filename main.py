#!/usr/bin/env python3
"""AEGIS M1 entry point — live webcam target tracking.

Examples:
    python main.py                          # track the largest object
    python main.py --classes person          # only track people
    python main.py --classes "sports ball" --strategy centermost
    python main.py --camera 1 --conf 0.5
    python main.py --classes "sports ball" --turret mock   # full safety+fire loop (no hardware)
    python main.py --classes "sports ball" --turret real   # on the Jetson with servos
"""

from __future__ import annotations

import argparse
import sys

# Allow `python main.py` from the project root without installing the package.
sys.path.insert(0, "src")

from aegis.config import Config  # noqa: E402
from aegis.pipeline import run  # noqa: E402
from aegis.safety import SafetyGate  # noqa: E402
from aegis.turret import Turret  # noqa: E402


def build_turret(kind: str):
    """Construct a Turret with mock (laptop) or real (Jetson) drivers."""
    if kind == "off":
        return None
    gate = SafetyGate()
    if kind == "mock":
        from aegis.hardware.mock import MockServoDriver, MockTrigger
        return Turret(MockServoDriver(), MockTrigger(), gate)
    if kind == "real":
        # Imports adafruit/Jetson libs lazily — only works on the wired Jetson.
        from aegis.hardware.base import ServoCalibration
        from aegis.hardware.pca9685 import PCA9685ServoDriver
        from aegis.hardware.nerf import NerfTrigger
        servos = PCA9685ServoDriver(
            pan_cal=ServoCalibration(channel=0, min_deg=0, max_deg=180),
            tilt_cal=ServoCalibration(channel=1, min_deg=45, max_deg=135, invert=True),
        )
        trigger = NerfTrigger(servos, trigger_channel=2, flywheel_pin=18)
        return Turret(servos, trigger, gate)
    raise ValueError(f"unknown turret kind: {kind!r}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="AEGIS M1 — live target tracking")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO weights")
    p.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    p.add_argument(
        "--classes", nargs="*", default=None,
        help="COCO class names to track, e.g. person 'sports ball' (default: all)",
    )
    p.add_argument(
        "--strategy", choices=("largest", "centermost"), default="largest",
        help="which detection to lock when several match",
    )
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-mirror", action="store_true", help="do not flip the image")
    p.add_argument(
        "--turret", choices=("off", "mock", "real"), default="off",
        help="off: view only; mock: full safety+fire loop, no hardware; real: Jetson servos",
    )
    args = p.parse_args()

    cfg = Config(
        model=args.model,
        conf=args.conf,
        target_classes=set(args.classes) if args.classes else None,
        strategy=args.strategy,
        camera=args.camera,
        width=args.width,
        height=args.height,
        mirror=not args.no_mirror,
    )
    return cfg, args.turret


if __name__ == "__main__":
    config, turret_kind = parse_args()
    run(config, build_turret(turret_kind))
