#!/usr/bin/env python3
"""AEGIS M1 entry point — live webcam target tracking.

Examples:
    python main.py                          # track the largest object
    python main.py --classes person          # only track people
    python main.py --classes "sports ball" --strategy centermost
    python main.py --camera 1 --conf 0.5
"""

from __future__ import annotations

import argparse
import sys

# Allow `python main.py` from the project root without installing the package.
sys.path.insert(0, "src")

from aegis.config import Config  # noqa: E402
from aegis.pipeline import run  # noqa: E402


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
    args = p.parse_args()

    return Config(
        model=args.model,
        conf=args.conf,
        target_classes=set(args.classes) if args.classes else None,
        strategy=args.strategy,
        camera=args.camera,
        width=args.width,
        height=args.height,
        mirror=not args.no_mirror,
    )


if __name__ == "__main__":
    run(parse_args())
