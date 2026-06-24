# Log — Project AEGIS

Decisions, progress notes, session diary. Most recent first.

> *Entry format: `## YYYY-MM-DD — [what changed] | [why] | [next]`*

---

## 2026-06-24 — Portfolio glow-up: GIFs + Mermaid README + interactive Pyodide demo | verified in-browser | next: publish when ready
- **Interactive demo site** (`docs/site/`) — a static page that runs the **real** `controller.py`/`simulator.py`/`safety.py` in-browser via **Pyodide** (no logic duplication). `tools/build_site.py` bundles those 4 pure modules into `aegis_modules.js` (zero-drift, regenerated from src). Two panels: ① PID tuner (live sliders → animated turret viz + response plot + metrics), ② safety-gate playground (drag a person box → real `SafetyGate.evaluate()` flips CLEAR/BLOCKED). Verified end-to-end with the preview browser: Pyodide boots clean (no console errors), sim metrics + all safety paths (clear / interlock / forbidden / disarmed / non-fireable) compute correctly, sliders re-run the sim live. Built to run locally (`cd docs/site && python -m http.server`); publish via GitHub Pages when repo goes public.
- **README GIFs** (`docs/media/`, `tools/make_gifs.py`) — pid_step, turret_track, safety_gate, all generated headlessly from the real sim. ~160–300KB each.
- **README rewrite** — hero with badges, demo gallery, Mermaid architecture flowchart + safety decision-tree, milestone table, design notes. Portfolio-grade.
- **Decisions:** kept repo private for now (build-to-run-locally); Pyodide over a JS port (real code, no drift); shipped site+GIFs+Mermaid, deferred CI/LICENSE/build-guide.
- `.claude/` gitignored (machine-specific preview config).

## 2026-06-24 — M4 custom-detector pipeline built & smoke-tested | dataset→train→export proven on CPU | next: real data / hardware
- **`aegis.data` (pure, tested):** `dataset.py` — YOLO label format (`xyxy_to_yolo`/inverse with clamping, `label_line`), deterministic `split_dataset` (keeps ≥1 in val), `data_yaml` generator. `build.py` — writes the Ultralytics tree (images/labels/{train,val} + data.yaml) from annotated samples. `synth.py` — synthetic "balloon" generator (coloured ellipse + highlight + string on noise) to validate plumbing with zero real data.
- **CLIs (thin Ultralytics wrappers):** `train.py` (fine-tune yolo11n; `--synthetic N` for smoke-test; output pinned to repo-local gitignored `runs/` via absolute project path — global ultralytics runs_dir was leaking to workspace), `export.py` (ONNX on laptop, TensorRT engine on Jetson w/ FP16/INT8 notes), `capture.py` (webcam grab → datasets/<name>/raw for labelling).
- **Verified end-to-end on CPU:** synthetic samples → build YOLO tree → 1-epoch YOLOv11 fine-tune → ONNX export. Whole chain runs. 51 headless tests (+9 dataset: coord round-trip, clamp, split determinism/proportions/edge-cases, data.yaml).
- **Safety tie-in:** `balloon` added to `SafetyGate` DEFAULT_FIREABLE — a custom-trained class becomes fireable while people/animals stay on the hard denylist. Closes targeting→safety loop.
- **Cleanup:** removed stray ultralytics output that had leaked to `workspace/runs`; training now stays inside `aegis/runs` (gitignored). datasets/ + runs/ excluded from repo.
- **Docs:** `docs/M4-TRAINING.md` — full capture→label→train→deploy workflow, why edge optimisation (ONNX→TensorRT, FP16/INT8 calibration) is the interesting part.
- **Remaining (real-world):** collect+label a real balloon dataset, train properly, build the TensorRT engine on the Jetson.

## 2026-06-23 — M3 actuation + safety layer built (mock-tested) | hardware-gated | next: order kit
- **`safety.py` — the spine.** `SafetyGate.evaluate()` enforces track-all/fire-inanimate-only in code: must be armed → target on inanimate allowlist → HARD denylist (people/animals) overrides everything → must be locked (no mid-slew fire) → person/animal-near-target interlock. Pure, defence-in-depth. End-to-end demo: only the clear inanimate shot fired; disarmed/unlocked/person-overlap/person-target all refused.
- **`hardware/` package.** `base.py` (ServoDriver/Trigger ABCs + pure `to_servo_angle` mapping with centre/limits/invert), `mock.py` (laptop drivers that record + enforce spin-up-before-fire), lazy real stubs `pca9685.py` (PCA9685ServoDriver) + `nerf.py` (NerfTrigger: GPIO flywheel relay + trigger servo). Import-safe — no adafruit/Jetson libs needed on dev machine.
- **`turret.py`** ties it together: controller angles → servos, detections → gate → gated trigger. Firing needs arm switch AND explicit fire AND CLEAR decision (never automatic).
- **Pipeline + CLI:** `python main.py --turret mock|real`. 'a' arm/disarm, 'f' fire, safety HUD shows ARMED/SAFE + reason. Mock runs the full loop today.
- **Tests:** 42 green (+20: safety refusals/permit/iou, servo mapping, mock interlock, turret arm/fire/disarm/lock).
- **BOM (b):** `docs/HARDWARE.md` — concrete M3 parts list (~£425), wiring diagram matching `hardware/`, safety wiring rules, bring-up order. Workspace `HARDWARE-BUDGET.md` updated to match.
- **Blocked on:** physical kit not ordered. All software through "drive real servos" is done & tested.

## 2026-06-23 — M2 built & tuned in sim | PID control loop validated before any servo | next: M3 hardware
- **`controller.py`** — velocity-output PID (anti-windup, output limit, no derivative kick) + `PanTiltController` (per-axis travel + slew-rate limits, configurable axis signs, holds + freezes integrators on target-loss). Pure Python, unit-tested. Output = the exact pan/tilt angles M3 sends to servos.
- **`simulator.py`** — closed-loop camera/target model (`observe()` mirrors `tracker.aim_error` conventions) + step/sine/ramp motion profiles + metrics (settle time, overshoot, steady-state, RMS/peak tracking error, on-frame %). Lets us tune thousands of steps in ms.
- **`sim.py`** — CLI tuner: run scenarios, print metrics, save matplotlib response plots to `runs/` (gitignored).
- **Bug found & fixed in sim:** elevation feedback ran away (step/sine diverged, ramp fine) — a mount-axis **sign inversion**: sim models tilt+ = camera-up while the controller default assumes tilt+ = down. Set `tilt_sign=-1` for the sim convention. This is exactly the bench bug `tilt_sign` exists to absorb; the asymmetry (ramp ok, step/sine broken) was the tell.
- **Tuning insight:** plant is an integrator (velocity→angle) ⇒ **P alone gives zero steady-state error to a step**, so high Ki was *hurting* moving-target tracking. Dropped Ki 28→8. Final **Kp=200, Ki=8, Kd=14**: step settle 0.67s, ~1% overshoot, 0.16deg steady; sine 4.6deg RMS; ramp 3.0deg RMS; all 100% on-frame.
- **Single source of truth:** `controller.default_pan_tilt()` factory holds the tuned gains; both `sim.py` and the live `pipeline.py` use it. Pipeline now runs the controller live and shows commanded `pan`/`tilt` on the HUD — connects M1↔M2 visually.
- **Tests:** 22 green (10 tracker + 12 controller incl. PID terms, anti-windup, slew/travel limits, lost-target hold, closed-loop convergence, factory gains).
- **Not done here:** servos (M3 — needs hardware ordered). Live webcam eyeball still pending on Mike's machine.

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
