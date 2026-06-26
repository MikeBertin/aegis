# STATUS — Project AEGIS

State: 🟢 ACTIVE
Phase: 3 — BUILD (M1–M4 software all done; hardware-gated)
Last action: 2026-06-26 — Hungarian algorithm from scratch (assignment.py — O(n³) potential method, handles rectangular). Verified vs scipy.optimize.linear_sum_assignment + the greedy-vs-optimal 2×2 example. Wired into mot.py as the matcher (optimal 1-IoU assignment, replacing greedy). Recent from-scratch run: backprop, Kalman, NMS, block-matching stereo, now Hungarian. From-scratch list complete. 152 tests green.
Next action: Decide when to make the repo public + enable GitHub Pages (unlocks the live demo URL). To unblock the physical build, ORDER the M3 kit (~£425, docs/HARDWARE.md). M4 real-data path (capture→label→train) can start now on the laptop.
Blocked by: Hardware not ordered (M3 physical). All M1–M4 software is built & tested; remaining work is real-world: servos, real dataset, on-Jetson TensorRT.
Next milestone: M3 physical bring-up (gimbal tracks under real servos, darts out) — needs hardware
Outcome: —

> *Update MEMORY.md when significant milestones are reached or the project closes.*
