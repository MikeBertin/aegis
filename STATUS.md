# STATUS — Project AEGIS

State: 🟢 ACTIVE
Phase: 3 — BUILD (M1–M4 software all done; hardware-gated)
Last action: 2026-06-26 — Block-matching stereo from scratch (stereo_match.py): SAD/SSD block_match_disparity (vectorised cost volume + own box-sum), disparity_to_depth via stereo geometry, make_synthetic_pair. Recovers known disparity per region (validated on synthetic pair); the actual capability gain (we had geometry, now compute disparity from an image pair). README gallery image (tools/stereo_viz.py). Recent from-scratch run: backprop, Kalman, NMS, now block-matching stereo. 145 tests green.
Next action: Decide when to make the repo public + enable GitHub Pages (unlocks the live demo URL). To unblock the physical build, ORDER the M3 kit (~£425, docs/HARDWARE.md). M4 real-data path (capture→label→train) can start now on the laptop.
Blocked by: Hardware not ordered (M3 physical). All M1–M4 software is built & tested; remaining work is real-world: servos, real dataset, on-Jetson TensorRT.
Next milestone: M3 physical bring-up (gimbal tracks under real servos, darts out) — needs hardware
Outcome: —

> *Update MEMORY.md when significant milestones are reached or the project closes.*
