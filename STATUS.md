# STATUS — Project AEGIS

State: 🟢 ACTIVE
Phase: 3 — BUILD (M1–M4 software all done; hardware-gated)
Last action: 2026-06-25 — Added three features (10 commits total): (1) safety_fsm.py — turret state machine SAFE/ARMED/TRACKING/FIRING/FAULT + failsafes (perception watchdog, temporal confirmation, no-fire zones, rate limit, magazine, audit log); (2) sharper fire-control — latency compensation + α-β-γ constant-accel Kalman (estimator.py) + numerical refine solver (closes heavy-drag gap: 16cm miss→1cm hit); (3) mot.py — SORT multi-target tracking (stable IDs, occlusion survival, prioritise). All pure + tested. 106 tests green.
Next action: Decide when to make the repo public + enable GitHub Pages (unlocks the live demo URL). To unblock the physical build, ORDER the M3 kit (~£425, docs/HARDWARE.md). M4 real-data path (capture→label→train) can start now on the laptop.
Blocked by: Hardware not ordered (M3 physical). All M1–M4 software is built & tested; remaining work is real-world: servos, real dataset, on-Jetson TensorRT.
Next milestone: M3 physical bring-up (gimbal tracks under real servos, darts out) — needs hardware
Outcome: —

> *Update MEMORY.md when significant milestones are reached or the project closes.*
