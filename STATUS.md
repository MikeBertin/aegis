# STATUS — Project AEGIS

State: 🟢 ACTIVE
Phase: 3 — BUILD (M1–M4 software all done; hardware-gated)
Last action: 2026-06-24 — M2.6 stereo ranging + ballistic fire-control: stereo.py (range from disparity, quadratic error) + ballistics.py (intercept solver with gravity hold-over, validated by hit/miss shot sim). Built a SECOND interactive demo (docs/site/firecontrol.html) showing stereo ranging + the solver, linked to demo 1 by an evolution nav (existing demo untouched). Verified in browser (solution HITs, naive MISSes). New fire-control GIF. Stereo added as optional kit in HARDWARE.md. 72 tests green.
Next action: Decide when to make the repo public + enable GitHub Pages (unlocks the live demo URL). To unblock the physical build, ORDER the M3 kit (~£425, docs/HARDWARE.md). M4 real-data path (capture→label→train) can start now on the laptop.
Blocked by: Hardware not ordered (M3 physical). All M1–M4 software is built & tested; remaining work is real-world: servos, real dataset, on-Jetson TensorRT.
Next milestone: M3 physical bring-up (gimbal tracks under real servos, darts out) — needs hardware
Outcome: —

> *Update MEMORY.md when significant milestones are reached or the project closes.*
