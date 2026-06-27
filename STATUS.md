# STATUS — Project AEGIS

State: 🟢 ACTIVE
Phase: 3 — BUILD (M1–M4 software all done; hardware-gated)
Last action: 2026-06-27 — Pages deploy confirmed live, then RESTYLED the demos to the chiron/empedocles house style (option A): new card-grid landing (index.html) with live canvas teasers + per-card accents; the 4 demos moved into subfolders (control/, firecontrol/, vision/, algorithms/), each restyled with a house-style nav, gradient hero, dark palette + per-page accent; added OG/Twitter meta + a generated og.png (1200×630). All JS/Pyodide logic untouched (scripts now referenced as ../*.js). Verified all 5 pages in the preview browser: Pyodide boots, every panel computes, zero console errors. README demo section updated to the new subfolder paths.
Last action (later 2026-06-27): added chiron-style **hover-to-explain popovers** + a **"things to notice"** card grid to all four demo pages (control was the template; then fire-control, vision, algorithms). Per-page jargon glossed (PID/integrator/feedforward · disparity/lead/hold-over/drag · kernel/feature-map/relu/pool/fc/backprop · Kalman Q-R/IoU/Hungarian/block-size), 4 notice cards each. Verified all pages in-browser: every popover term resolves, demos compute, zero console errors. Restyle commit fd72654 pushed + Pages redeployed (18s, success).
Next action: Commit + push the explainers (this batch). To unblock the physical build, ORDER the M3 kit (~£425, docs/HARDWARE.md). M4 real-data path (capture→label→train) can start now on the laptop.
Blocked by: Hardware not ordered (M3 physical). All M1–M4 software is built & tested; remaining work is real-world: servos, real dataset, on-Jetson TensorRT.
Next milestone: M3 physical bring-up (gimbal tracks under real servos, darts out) — needs hardware
Outcome: —

> *Update MEMORY.md when significant milestones are reached or the project closes.*
