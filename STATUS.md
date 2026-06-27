# STATUS — Project AEGIS

State: 🟢 ACTIVE
Phase: 3 — BUILD (M1–M4 software all done; hardware-gated)
Last action: 2026-06-27 — Built a FOURTH interactive demo page (docs/site/algorithms.html — ④ Algorithms): a gallery of tonight's from-scratch builds running their real modules via Pyodide+NumPy — Kalman (noisy-track smoothing, raw 1.83→KF 0.78), Hungarian (editable cost matrix, optimal vs greedy), NMS (overlapping boxes + IoU slider), block-matching stereo (disparity map + sliders). Added the 4 modules to build_site.py bundle; linked ④ in the evonav on all pages. Verified all 4 panels in-browser (correct outputs, no console errors). 152 tests green.
Next action: Decide when to make the repo public + enable GitHub Pages (unlocks the live demo URL). To unblock the physical build, ORDER the M3 kit (~£425, docs/HARDWARE.md). M4 real-data path (capture→label→train) can start now on the laptop.
Blocked by: Hardware not ordered (M3 physical). All M1–M4 software is built & tested; remaining work is real-world: servos, real dataset, on-Jetson TensorRT.
Next milestone: M3 physical bring-up (gimbal tracks under real servos, darts out) — needs hardware
Outcome: —

> *Update MEMORY.md when significant milestones are reached or the project closes.*
