# Log — Project AEGIS

Decisions, progress notes, session diary. Most recent first.

> *Entry format: `## YYYY-MM-DD — [what changed] | [why] | [next]`*

---

## 2026-06-23 — M1 built & verified | perception + targeting loop working end-to-end | next: live webcam run + M2 PID
- **Repo structure** (house style: `requirements.txt` + `src/` pkg + `main.py` + `tests/`):
  - `src/aegis/tracker.py` — headless target-selection + aim-error maths, **zero heavy deps** so it's unit-tested and reusable by M2's PID.
  - `detector.py` (lazy-torch YOLO→Detection adapter), `overlay.py` (cv2 HUD), `pipeline.py` (camera→detect→select→error→draw loop), `config.py`, `main.py` (CLI).
  - `pipeline.py` already exposes the per-frame `aim_error` as the explicit M2 hook.
- **Env:** pinned to a Python **3.13** venv (3.14 too new for torch wheels). Installed torch 2.12.1, ultralytics 8.4.75 (YOLOv11), opencv 4.13.
- **Verification:** 10/10 headless tracker tests green. End-to-end inference confirmed on the YOLO demo image — detected 1 bus + 4 people, filtered to people, locked largest at **conf 0.88**, produced `aim_error=(-0.65,+0.21)`. Noise-frame test returns 0 detections as expected.
- **Not yet done:** literal live webcam capture (needs Mike's physical camera + display — `python main.py`).
- **Next:** Mike runs the live window to eyeball tracking; then start **M2** (PID controller consuming `aim_error` → pan/tilt servo angles).

## 2026-06-23 — Project created & scoped | scaffolded from template, all design forks resolved | next: build M1 software loop
- Scaffolded from `_template/`. Phase 1 — IDEAS, but design forks resolved up front so we can start building.
- **Idea:** CV turret — CNN detects/tracks a target, pan/tilt gimbal aims via PID, Nerf flywheel gun fires. Purposes: (a) fun, (b) learn the perception→control→actuation pipeline, (c) portfolio depth.
- **Codename:** AEGIS — standalone project, own GitHub repo. Cross-linked to DAEDALUS (DAEDALUS-class robotics unit; Jetson reusable across fleet) and SENTINEL (shared CV stack).
- **Compute decision:** Jetson Orin Nano over Pi 5 + Coral. Coral runs only int8 TFLite → quantise/Edge-TPU-compile tax on every custom model. Jetson runs torch/YOLO natively, more headroom, better portfolio story (CUDA/TensorRT). ~£160 more, justified by reasons (b) + (c).
- **Safety framing:** Track all (incl. people — the CV showcase), fire on inanimate targets only, human-in-the-loop arming (physical switch + software interlocks). Safety architecture documented as a feature, not a footnote. This keeps the public repo defensible and makes a better demo.
- **Staged milestones:** M1 software loop (£0) → M2 pan/tilt + laser (no projectile) → M3 Nerf + safety gating → M4 custom-trained detector deployed on Jetson.
- **Deferred ("cross when we get there"):** PID tuning, M4 dataset collection/labelling plan, trigger mechanics.
- **Gate — IDEAS → "Worth pursuing?":** ✅ Mike signed off 2026-06-23. Proceeding to build M1.
