#!/usr/bin/env python3
"""AEGIS M4 — capture training images from the webcam.

Saves frames to datasets/<name>/raw/ for labelling. Two modes: press SPACE to
grab single frames, or --interval to grab automatically every N seconds. Label
the saved images (e.g. in Roboflow / labelImg / CVAT) into YOLO txt format,
then point train.py at the resulting data.yaml.

Examples:
    python capture.py --name balloons              # SPACE to grab, q to quit
    python capture.py --name balloons --interval 1 # auto every 1s
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, "src")


def main() -> None:
    p = argparse.ArgumentParser(description="AEGIS M4 — capture training frames")
    p.add_argument("--name", default="capture", help="dataset name under datasets/")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--interval", type=float, default=None,
                   help="auto-capture every N seconds (else SPACE to grab)")
    args = p.parse_args()

    out = os.path.join("datasets", args.name, "raw")
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    print(f"Saving to {out}/  —  SPACE grab, q quit"
          + (f"  (auto every {args.interval}s)" if args.interval else ""))
    n, last = 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            view = frame.copy()
            cv2.putText(view, f"captured: {n}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 220, 60), 2)
            cv2.imshow("AEGIS capture", view)
            key = cv2.waitKey(1) & 0xFF

            grab = key == ord(" ")
            if args.interval and time.time() - last >= args.interval:
                grab, last = True, time.time()
            if grab:
                path = os.path.join(out, f"{args.name}_{int(time.time()*1000)}.jpg")
                cv2.imwrite(path, frame)
                n += 1
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Captured {n} frames -> {out}")


if __name__ == "__main__":
    main()
