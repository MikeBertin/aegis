# STATUS — Project AEGIS

State: 🟢 ACTIVE
Phase: 3 — BUILD (M1–M4 software all done; hardware-gated)
Last action: 2026-06-24 — Portfolio glow-up: 3 README GIFs (from real sim), Mermaid-diagram README hero, and an interactive Pyodide demo site (docs/site/) running the real controller/simulator/safety code in-browser. Verified working in the preview browser (Pyodide boots clean, all safety paths + live PID tuning compute correctly). Built to run locally; publish via Pages when repo goes public. 51 tests still green.
Next action: Decide when to make the repo public + enable GitHub Pages (unlocks the live demo URL). To unblock the physical build, ORDER the M3 kit (~£425, docs/HARDWARE.md). M4 real-data path (capture→label→train) can start now on the laptop.
Blocked by: Hardware not ordered (M3 physical). All M1–M4 software is built & tested; remaining work is real-world: servos, real dataset, on-Jetson TensorRT.
Next milestone: M3 physical bring-up (gimbal tracks under real servos, darts out) — needs hardware
Outcome: —

> *Update MEMORY.md when significant milestones are reached or the project closes.*
