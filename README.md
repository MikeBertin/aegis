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
| M1 | Webcam + Jetson/laptop + pretrained YOLO → on-screen crosshair tracks target | Perception + targeting loop (£0 hardware) | TBD |
| M2 | Pan/tilt rig + servos track target with laser pointer (no projectile) | PID control loop, physical | TBD |
| M3 | Nerf flywheel gun + servo trigger + full safety gating | Actuation + safety architecture | TBD |
| M4 | Custom-trained detector, TensorRT-optimised, deployed on Jetson | Dataset → train → edge-deploy pipeline | TBD |

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

## Repo Layout
```
aegis/
├── main.py                  # CLI entry — builds Config, runs the loop
├── requirements.txt
├── src/aegis/
│   ├── config.py            # runtime Config dataclass
│   ├── tracker.py           # target selection + aim-error maths (no torch/cv2 — unit-tested)
│   ├── detector.py          # Ultralytics YOLO -> Detection adapter (lazy torch import)
│   ├── overlay.py           # cv2 HUD: crosshair, lock box, error line
│   └── pipeline.py          # the M1 camera->detect->select->error->draw loop
└── tests/test_tracker.py    # pure-logic tests, no heavy deps
```
The architecture is deliberately split so the **targeting maths is headless and testable**, and `pipeline.py`'s per-frame `aim_error(...)` is the exact signal M2's PID controller will turn into pan/tilt servo commands.

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
