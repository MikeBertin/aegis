# Log — Project AEGIS

Decisions, progress notes, session diary. Most recent first.

> *Entry format: `## YYYY-MM-DD — [what changed] | [why] | [next]`*

---

## 2026-06-25 — Three features: safety FSM, sharper fire-control, multi-target tracking | 3 commits, 106 tests
- **Safety state machine** (`safety_fsm.py`): SAFE/ARMED/TRACKING/FIRING/latching-FAULT over the existing SafetyGate. Failsafes: perception watchdog (stale vision→FAULT), temporal confirmation (N consecutive CLEAR frames before a shot), angular no-fire zones, rate limit, magazine count, full audit log. +12 tests.
- **Sharper fire-control** (3 upgrades): (a) **latency compensation** — `firing_solution(latency=)` predicts target through pipeline delay before launch (100ms→+3.5° lead at 5m); (b) **α-β-γ constant-accel filter** (`estimator.AlphaBetaGamma`/`Estimator3DCA`) for maneuvering targets; (c) **numerical solver** (`firing_solution(refine=)`) flies the shot + nulls closest-approach miss — closed the heavy-drag gap (k=0.15@5m: 16cm miss→1cm hit). `simulate_shot`/`firing_solution` now take `accel`; `FireControlTracker` gains `latency_s`+`refine` and auto-uses accel from a CA estimator. +7 tests; proven hits an accelerating target end-to-end.
- **Multi-target tracking** (`mot.py`): SORT-style — predict→greedy-IoU-match→update→age-out; tentative→confirmed (min_hits rejects 1-frame FPs); confirmed tracks coast on velocity through occlusion keeping the same ID; `prioritize()` (largest/centermost/confidence). +9 tests (identity persistence, occlusion survival, deletion, dedup, velocity).
- **Decisions:** user picked these 3 of 4 options (deferred CI/LICENSE/public). Kept all pure-testable; didn't re-plumb the live pipeline (library modules + tests + README are the deliverable). README: 3 milestone rows (M2.7/MOT/M3.1) + capability paragraphs + a Mermaid FSM state diagram.

## 2026-06-24 — M2.6 extended: dart drag + fire-control wired into tracking | both verified
- **Drag model** (`ballistics.DartModel`): quadratic drag, speed decays `v=v0·e^(-k·s)`, so `time_of_flight(L)=(e^(kL)-1)/(k·v0)` instead of L/v0. `firing_solution`/`intercept_time`/`simulate_shot` now accept a muzzle speed OR a DartModel (back-compat). Drag → longer TOF → more lead AND more hold-over. Verified: 5m crossing, k=0.07 → 20m/s decays to 14.1m/s, tof 253→307ms, lead 8.6→10.4°, hold 3.6→5.2°, still HIT. Honest limit: the decoupled closed-form approx misses at extreme drag (k=0.15 → 16cm vs 14cm radius) — documented; a full numerical solver would close it.
- **Wired into tracking** (`tracking.FireControlTracker` + `estimator.Estimator3D`): bearing (aim_error+turret pose) + stereo range → 3D target position → 3-axis α-β → firing_solution → command the controller toward the solution aim (lead+holdover) with feedforward on the aim's angular velocity. Proven end-to-end: after settling, a dart fired from the commanded (pan,tilt) HITs the moving target *with gravity and drag*; also tests for leads-crossing-target and holds-over-for-gravity.
- **Demo:** added a dart-drag slider to firecontrol.html (+ speed-at-range readout). Verified in preview browser (no console errors; drag grows tof/lead/holdover live; HIT vs naive MISS). Cache-bust ?v=4, bundle rebuilt (still 9 modules).
- **Tests:** 78 green (+6: 3 drag in test_ballistics, 3 fire-control tracking in test_firecontrol).

## 2026-06-24 — M2.6 stereo ranging + ballistic fire-control | computed lead from range + dart speed + gravity | 2nd demo for project evolution
- **Why:** Mike asked about computing lead from known muzzle velocity + range (vs the fixed lead_time slider). Chose stereo for range (BOM updated); the engineering payoff is the real fire-control solver.
- **`stereo.py`** — `StereoRig`: `depth = focal·baseline/disparity`, inverse, `depth_error` (∝ range² — sharp near, vague far), pixel→camera back-projection. Pure, tested.
- **`ballistics.py`** — the moving-interceptor solver. `intercept_time` seeds with the closed-form no-gravity quadratic then iterates with gravity (implicit: TOF depends on intercept range depends on TOF). `firing_solution` → aim az/el, lead, gravity hold-over, intercept point. `simulate_shot` flies the dart to **prove** hits. Frame (x:right, y:up, z:forward).
- **Validated:** at 3m, target crossing 4m/s, 20m/s dart, gravity → solution HIT (lead ~13°, holdover ~1.6°, closest 13cm) vs naive aim MISS (68cm). Stereo: 16px disparity at 3m, range error ±9cm@3m vs ±38cm@6m.
- **Second demo (project evolution):** `docs/site/firecontrol.html` + `firecontrol.js` (Pyodide running ballistics.py + stereo.py): stereo-ranging panel (disparity→depth + error growth) + fire-control solver (top-down lead + side-on gravity arc, animated solution vs naive darts, HIT/MISS verdict). **Existing demo left intact**; both linked by an `.evonav` bar (① Control & Safety, ② Stereo Fire-Control). Verified in preview browser (no console errors; solution HITs, naive MISSes across ranges/speeds). Bundle now 9 modules; cache-bust bumped to ?v=3.
- **GIF:** `firecontrol.gif` — top-down solution-hits-vs-naive-misses.
- **Honest caveats (in discussion + docs):** foam-dart drag makes constant-velocity optimistic; monocular known-size ranging needs no extra HW and can beat stereo at range; real Nerf spread caps physical precision — value is the solver + sim validation.
- **Tests:** 72 green (+10: stereo round-trip/error-growth/back-projection, ballistics stationary/moving-hits-naive-misses/gravity-holdover/unreachable).

## 2026-06-24 — M2.5 predictive tracking: feedforward + lead | crosshair no longer lags / now leads | verified in demo
- **Why:** the demo crosshair visibly lagged the moving target — fundamental to pure feedback (PID needs a position error to generate keep-up velocity). Feedback can't lead; feedforward can.
- **`estimator.py`** — α-β filter (fixed-gain constant-velocity tracker): smoothed position + velocity in one recursive step, critically-damped β from α. `TargetEstimator` runs one per axis. Default α=0.7 (responsiveness vs noise tolerance). Pure, tested.
- **`controller.py`** — `PanTiltController.update` gains an optional `feedforward=(pan,tilt)` axis-velocity term added after the mount-sign mapping, before the slew clamp. Backward-compatible (default 0).
- **`tracking.py`** — `TargetTracker`: reconstructs absolute target angle from aim_error + turret pose → α-β smooth → **lead** point = pos + vel·lead_time → drives PID toward the lead point **with velocity feedforward**. `simulator.run_tracking` drives it; SimResult gains lead_az/lead_el.
- **Result (sim):** steady-state tracking lag on a sine cut ~68% (5.8°→1.9° RMS). On a ramp, aim leads by exactly velocity×lead_time (verified: 0.37° aim-track error, ~4° lead). Lead on a fast sine is intentionally imperfect (predicting far ahead of an oscillating target is hard) — clean on constant-velocity, as expected.
- **Demo:** feedforward toggle + lead-time slider + amber lead-pip + lag/aim-track metrics. Verified live in preview browser (lag 5.82°→1.87° with ff on; aim leads ahead with lead>0; no console errors). Added cache-bust ?v= on scripts. Rebuilt bundle (now 7 modules incl. estimator/tracking).
- **GIF:** `feedforward.gif` — plain PID (grey, lags) vs feedforward+lead (green, leads), same target.
- **Tests:** 62 green (+11: α-β init/convergence/critically-damped/reset, feedforward lag-cut, lead-ahead, zero-lead-on-target, lost-target reset).

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
