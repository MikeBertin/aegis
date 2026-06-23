#!/usr/bin/env python3
"""AEGIS M4 — export trained weights for edge deployment.

ONNX runs anywhere (export + sanity-check it on the laptop). The TensorRT
'engine' is GPU/JetPack-specific and must be built ON the Jetson — see the
note printed for that path.

Examples:
    python export.py runs/detect/aegis/weights/best.pt --format onnx
    # On the Jetson, for the deployed model:
    python export.py best.pt --format engine --half
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")


def main() -> None:
    p = argparse.ArgumentParser(description="AEGIS M4 — export detector")
    p.add_argument("weights", help="path to trained .pt weights")
    p.add_argument("--format", default="onnx", choices=("onnx", "engine"),
                   help="onnx (portable) or engine (TensorRT, Jetson-only)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", help="FP16 (TensorRT)")
    p.add_argument("--int8", action="store_true", help="INT8 quantise (TensorRT)")
    args = p.parse_args()

    if args.format == "engine":
        print("NOTE: TensorRT .engine files are device-specific — build this ON "
              "the Jetson Orin Nano, not on the laptop. INT8 needs a calibration "
              "set from your real captures for best accuracy.")

    from ultralytics import YOLO
    model = YOLO(args.weights)
    path = model.export(
        format=args.format, imgsz=args.imgsz, half=args.half, int8=args.int8
    )
    print(f"Exported -> {path}")


if __name__ == "__main__":
    main()
