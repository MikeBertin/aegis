# 🛡️ Project AEGIS — CV-Targeting Turret

> *A computer-vision turret that tracks anything and fires only on inanimate targets — with the safety architecture as a first-class feature, not an afterthought.*

## What This Is
AEGIS is a pan/tilt turret that uses a convolutional neural net to detect and track objects in real time, drives the gimbal with a PID control loop, and fires a Nerf flywheel gun at inanimate showcase targets under explicit human-in-the-loop arming. It is built on a Jetson Orin Nano running YOLO natively on-device.

Three reasons it exists: (a) it's a genuinely fun build, (b) it's the cleanest way to learn the full perception → control → actuation pipeline end-to-end, and (c) it's a portfolio piece that speaks the industry's language — custom-trained CNN, edge-GPU deployment, real-time control.

## Goal & Why It Matters
**Success = a turret that reliably acquires a target, tracks it smoothly, and fires on an inanimate target only when a human has armed it — running a custom-trained detector on the Jetson.**

The deeper point: the perception-to-actuation loop is the foundation of every physical robot. AEGIS is a small, self-contained vehicle for learning it for real — and a standalone showcase of the DAEDALUS robotics capability.

## Scope & Safety Principle
- **Track all** — including people; tracking is the CV showcase.
- **Fire inanimate only** — balloons, objects, moving targets on a track. The system never fires at a person.
- **Human-in-the-loop arming** — a physical arm switch plus software interlocks (e.g. no-fire while a person bounding-box overlaps the target). The safety design is documented as a feature.

## Milestones
| # | Milestone | Proves | Target |
|---|-----------|--------|--------|
| M1 | ✅ Webcam + Jetson/laptop + pretrained YOLO → on-screen crosshair tracks target | Perception + targeting loop (£0 hardware) | Done 2026-06-23 |
| M2 | ◑ PID control loop tuned in simulation (servos pending hardware) | Control loop, validated in-sim | Sim done 2026-06-23 |
| M3 | ◑ Actuation layer + safety gate built (mock-tested); awaiting hardware | Actuation + safety architecture | Software done 2026-06-23 |
| M4 | ◑ Dataset→train→export pipeline built & smoke-tested (real data + Jetson TensorRT pending) | Dataset → train → edge-deploy pipeline | Pipeline done 2026-06-24 |

## Tech Stack
| Layer | Choice |
|-------|--------|
| Compute | Jetson Orin Nano (CUDA, runs PyTorch/YOLO natively; TensorRT for optimisation) |
| Perception | YOLOv11 (pretrained → custom fine-tune), OpenCV |
| Control | PID loop (pixel error → pan/tilt angles) |
| Actuation | 2× servos (MG996R) + PCA9685 driver on pan/tilt gimbal; servo trigger on Nerf flywheel gun |
| Camera | Pi Camera v3 / USB webcam |
| Language | Python (vision/control); firmware as needed |

## Quickstart (M1 — live tracking, no hardware)
```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pulls YOLOv11 + torch + opencv
python main.py                           # track the largest object on cam 0
python main.py --classes person          # only track people
python main.py --classes "sports ball" --strategy centermost
pytest                                   # headless targeting-maths tests
```
On first run, Ultralytics auto-downloads the `yolo11n.pt` weights. Press `q`/`Esc` to quit.

## M2 — control loop (simulated)
The PID controller (`controller.py`) consumes the M1 `aim_error` and outputs
pan/tilt servo commands. It's tuned and validated against a closed-loop
simulator (`simulator.py`) — no servos needed yet:
```bash
python sim.py            # run step/sine/ramp scenarios, print metrics
python sim.py --plot     # also save response plots to runs/
python sim.py --kp 220 --ki 10 --kd 16   # try your own gains
```
**Tuned performance** (Kp=200, Ki=8, Kd=14): 20° step acquired in **0.67 s**,
~1% overshoot, 0.16° steady-state; moving-target tracking 4.6° RMS (sine) /
3.0° RMS (ramp); 100% on-frame. The live pipeline already runs this controller
and shows the commanded `pan`/`tilt` on the HUD — those are the exact angles M3
will send to the servos.

> Design note: the plant is an integrator (velocity→angle), so P alone gives
> zero steady-state error to a step; Ki is kept small (rejects tilt gravity-sag)
> and Kd damps the acquisition overshoot. `tilt_sign` absorbs the mount-axis
> inversion you hit on the real bench.

## M3 — actuation + safety (built, awaiting hardware)
The actuation layer and the **safety gate** are written and fully mock-tested,
so the whole arm → track → gated-fire loop runs on a laptop today:
```bash
python main.py --classes "sports ball" --turret mock   # full loop, no hardware
#   'a' arm/disarm   'f' fire (only if the gate says CLEAR)   'q' quit
python main.py --classes "sports ball" --turret real   # on the wired Jetson
```
**Safety policy is enforced in code, not just documented** (`safety.py`):
1. must be ARMED (human-in-the-loop), 2. target on the inanimate allowlist,
3. a HARD denylist (people/animals) overrides everything, 4. must be LOCKED
(no firing mid-slew), 5. interlock — no fire if a person/animal is near the
target. Firing is never automatic: it needs the arm switch **and** an explicit
fire action **and** a `CLEAR` decision. Drivers are swappable — `MockServoDriver`
/`MockTrigger` on a laptop, `PCA9685ServoDriver`/`NerfTrigger` on the Jetson
(same `Turret` logic). Bill of materials + wiring: [docs/HARDWARE.md](docs/HARDWARE.md).

## M4 — custom detector (pipeline built, real data pending)
Train a bespoke detector ("balloon" — the fireable showcase target) and deploy
it optimised on the Jetson. The whole **dataset → train → export** chain runs on
a laptop; only the final TensorRT engine build is Jetson-only. Smoke-tested
end-to-end on synthetic data (build → 1-epoch train → ONNX export, CPU):
```bash
python train.py --synthetic 16 --epochs 1 --imgsz 160 --device cpu --name smoke
python export.py runs/smoke/weights/best.pt --format onnx --imgsz 160
# Real flow: capture.py -> label (Roboflow/CVAT) -> train.py --data ... -> export.py
```
The fiddly bits (YOLO label conversion, deterministic split, data.yaml) live in
`aegis.data` and are unit-tested; `train.py`/`export.py` are thin Ultralytics
wrappers. The trained class `balloon` is on the SafetyGate fireable allowlist,
so a custom detector closes the targeting→safety loop. Full workflow:
[docs/M4-TRAINING.md](docs/M4-TRAINING.md).

## Repo Layout
```
aegis/
├── main.py                  # CLI: live tracking (M1) + controller (M2) + turret (M3)
├── sim.py                   # CLI: M2 control simulator & PID tuner
├── capture.py train.py export.py   # M4: dataset capture, training, edge export
├── docs/                    # HARDWARE.md (M3 BOM) + M4-TRAINING.md
├── src/aegis/
│   ├── config.py            # runtime Config dataclass
│   ├── tracker.py           # target selection + aim-error maths (no torch/cv2 — tested)
│   ├── controller.py        # PID + PanTiltController + tuned factory (no heavy deps — tested)
│   ├── simulator.py         # closed-loop camera/target model + tracking metrics
│   ├── safety.py            # SafetyGate fire-authorisation logic (pure — tested)
│   ├── turret.py            # M3 integration: controller+servos+trigger+gate
│   ├── detector.py          # Ultralytics YOLO -> Detection adapter (lazy torch import)
│   ├── overlay.py           # cv2 HUD: crosshair, lock box, error line
│   ├── pipeline.py          # camera->detect->select->controller->turret->draw loop
│   ├── hardware/            # base ABCs + servo mapping, mock drivers, PCA9685/Nerf (lazy)
│   └── data/                # M4: YOLO label/split/data.yaml (pure), dataset builder, synth
└── tests/                   # tracker, controller, safety, hardware, dataset — 51 pure tests
```
The architecture is deliberately split so the **targeting and control maths are
headless and testable**, and the same `controller.py` runs unchanged in the
simulator, the live pipeline, and (M3) on the Jetson driving real servos.

## Key Files
| File | Purpose |
|------|---------|
| `README.md` | This file |
| `STATUS.md` | Current state and next action |
| `log.md` | Decisions, progress, session notes |

## Notes
- **Standalone project, own GitHub repo** (private by default; public is the portfolio goal). Cross-linked to **DAEDALUS** (this is a DAEDALUS-class robotics unit and the Jetson is reusable across the fleet) and **SENTINEL** (shared CV stack — detection, tracking).
- **Jetson over Pi 5 + Coral** was a deliberate call: Coral runs only int8 TFLite and would impose a quantise→Edge-TPU-compile tax on every custom model. The Jetson runs torch/YOLO natively (no conversion wall), has far more headroom, and is the brain worth owning for future DAEDALUS units. Costs ~£160 more — justified by reasons (b) and (c).
- M1–M2 need almost no hardware; laptop + webcam gets the vision and control loops working before any kit is ordered.
- Hardware tracked in `projects/HARDWARE-BUDGET.md`.
- Bridges to cross when we reach them: PID tuning, the M4 dataset collection/labelling plan, and trigger mechanics.

> *Any quotes used in this project must be real, sourced attributions. No invented quotes.*
