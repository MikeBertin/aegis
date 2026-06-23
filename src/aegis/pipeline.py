"""The live targeting loop: camera -> detect -> select -> aim error ->
controller -> overlay, with optional turret actuation.

M1 proves perception + tracking on screen. M2's controller runs live (commanded
pan/tilt on the HUD). If a Turret (M3) is supplied, the loop also drives the
servos and evaluates the safety-gated fire decision — 'a' arms/disarms, 'f'
fires (only if the SafetyGate permits).
"""

from __future__ import annotations

import time
from typing import Optional

import cv2

from .config import Config
from .controller import default_pan_tilt
from .detector import Detector
from .tracker import aim_error, select_target
from .turret import Turret


def run(cfg: Config, turret: Optional[Turret] = None) -> None:
    detector = Detector(
        model=cfg.model, conf=cfg.conf, target_classes=cfg.target_classes
    )
    # M2 controller, running live: turns each frame's aim error into the
    # pan/tilt angles M3 will command on the real servos.
    controller = default_pan_tilt()
    t_loop = time.time()

    cap = cv2.VideoCapture(cfg.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cfg.camera}")

    from . import overlay  # imported here so headless runs never need cv2 GUI

    keys = "'q' quit" + ("  |  'a' arm/disarm  'f' fire" if turret else "")
    print(f"AEGIS live targeting. {keys}.")
    fps, t_prev = 0.0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; stopping.")
                break
            if cfg.mirror:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            detections = detector.detect(frame)
            target = select_target(detections, w, h, cfg.strategy)

            # M2: drive the controller from the aim error (or hold if no target).
            now_loop = time.time()
            ctl_dt = max(1e-3, now_loop - t_loop)
            t_loop = now_loop
            err = aim_error(target.centroid, w, h) if target is not None else None
            pan, tilt = controller.update(err, ctl_dt)

            # M3: drive servos + evaluate the safety-gated fire decision.
            decision = None
            if turret is not None:
                decision = turret.update(pan, tilt, target, detections, err)

            # Smoothed FPS.
            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            if cfg.show:
                overlay.draw(frame, detections, target, cfg.draw_all_boxes)
                cv2.putText(
                    frame, f"{fps:4.1f} FPS", (w - 130, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, f"servo cmd  pan={pan:+6.1f}  tilt={tilt:+6.1f}",
                    (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 60), 2, cv2.LINE_AA,
                )
                if turret is not None and decision is not None:
                    _draw_safety(frame, turret, decision)
                cv2.imshow("AEGIS", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or Esc
                    break
                if turret is not None:
                    if key == ord("a"):
                        turret.disarm() if turret.armed else turret.arm()
                    elif key == ord("f"):
                        turret.try_fire()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if turret is not None:
            turret.close()


def _draw_safety(frame, turret: Turret, decision) -> None:
    h, w = frame.shape[:2]
    if not turret.armed:
        colour, state = (160, 160, 160), "SAFE"
    elif decision.permit:
        colour, state = (40, 40, 230), "ARMED — CLEAR TO FIRE"
    else:
        colour, state = (40, 170, 230), "ARMED"
    cv2.putText(
        frame, f"{state}  [{decision.reason}]", (12, h - 44),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA,
    )
