# STATUS — Project AEGIS

State: 🟢 ACTIVE
Phase: 3 — BUILD (M1–M4 software all done; hardware-gated)
Last action: 2026-06-26 — NMS from scratch (nms.py — greedy + class-aware nms_per_class + soft_nms), reusing safety.iou. Verified our greedy NMS matches torchvision.ops.nms exactly on random boxes. Added to the README "first principles" table. Recent from-scratch run: backprop+Adam (zero-autograd CNN training), Kalman filter, now NMS. 139 tests green.
Next action: Decide when to make the repo public + enable GitHub Pages (unlocks the live demo URL). To unblock the physical build, ORDER the M3 kit (~£425, docs/HARDWARE.md). M4 real-data path (capture→label→train) can start now on the laptop.
Blocked by: Hardware not ordered (M3 physical). All M1–M4 software is built & tested; remaining work is real-world: servos, real dataset, on-Jetson TensorRT.
Next milestone: M3 physical bring-up (gimbal tracks under real servos, darts out) — needs hardware
Outcome: —

> *Update MEMORY.md when significant milestones are reached or the project closes.*
