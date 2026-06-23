#!/usr/bin/env python3
"""AEGIS M4 — fine-tune a YOLOv11 detector on a custom dataset.

Thin wrapper around Ultralytics training. The interesting/bug-prone logic
(label format, splitting, data.yaml) lives in aegis.data and is unit-tested.

Examples:
    # Smoke-test the whole pipeline on synthetic balloons (CPU, seconds):
    python train.py --synthetic 12 --epochs 1 --imgsz 160

    # Train on a real dataset you built with capture.py + labelling:
    python train.py --data datasets/balloons/data.yaml --epochs 100
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, "src")


def main() -> None:
    p = argparse.ArgumentParser(description="AEGIS M4 — train custom detector")
    p.add_argument("--data", help="path to an existing data.yaml")
    p.add_argument("--synthetic", type=int, metavar="N",
                   help="build N synthetic balloon samples and train on them")
    p.add_argument("--base", default="yolo11n.pt", help="weights to fine-tune from")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="'cpu', '0', etc. (auto if unset)")
    p.add_argument("--name", default="aegis", help="run name under runs/")
    args = p.parse_args()

    data = args.data
    if args.synthetic:
        from aegis.data.build import build_dataset
        from aegis.data import synth
        print(f"Generating {args.synthetic} synthetic balloon samples...")
        samples = synth.make_samples(args.synthetic)
        data = build_dataset("datasets/synthetic", samples, synth.CLASS_NAMES)
        print(f"  wrote dataset -> {data}")
    if not data:
        p.error("provide --data <data.yaml> or --synthetic N")

    from ultralytics import YOLO
    model = YOLO(args.base)
    model.train(
        data=data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, name=args.name,
        # Absolute path keeps output inside the repo's gitignored runs/ dir,
        # overriding Ultralytics' global runs_dir setting.
        project=os.path.abspath("runs"),
    )
    print(f"Done. Best weights under runs/{args.name}/weights/best.pt")
    print(f"Use:  python main.py --model runs/{args.name}/weights/best.pt "
          "--classes balloon --turret mock")


if __name__ == "__main__":
    main()
