# STATUS — Project AEGIS

State: 🟢 ACTIVE
Phase: 3 — BUILD (M1–M4 software all done; hardware-gated)
Last action: 2026-06-24 — Extended M2.6: (1) foam-dart DRAG model in ballistics.py (v=v0·e^-ks → longer TOF → more lead + hold-over; demo gained a drag slider showing speed-at-range), (2) wired the firing solution into the live loop via FireControlTracker (bearing+stereo range → 3D α-β estimate → firing_solution → servo command), proven by a shot-from-commanded-aim HIT test incl. gravity+drag. Both verified in browser/tests. 78 tests green.
Next action: Decide when to make the repo public + enable GitHub Pages (unlocks the live demo URL). To unblock the physical build, ORDER the M3 kit (~£425, docs/HARDWARE.md). M4 real-data path (capture→label→train) can start now on the laptop.
Blocked by: Hardware not ordered (M3 physical). All M1–M4 software is built & tested; remaining work is real-world: servos, real dataset, on-Jetson TensorRT.
Next milestone: M3 physical bring-up (gimbal tracks under real servos, darts out) — needs hardware
Outcome: —

> *Update MEMORY.md when significant milestones are reached or the project closes.*
