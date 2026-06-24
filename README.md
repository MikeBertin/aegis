# 🛡️ AEGIS — CV-Targeting Turret

> *A computer-vision turret that tracks anything and fires only on inanimate targets — with the safety architecture as a first-class, testable feature, not an afterthought.*

![milestones](https://img.shields.io/badge/software-M1--M4_complete-3fb950)
![tests](https://img.shields.io/badge/tests-62_passing-3fb950)
![python](https://img.shields.io/badge/python-3.13-3776ab)
![model](https://img.shields.io/badge/detector-YOLOv11-blue)
![edge](https://img.shields.io/badge/edge-Jetson_Orin_Nano-76b900)
![control](https://img.shields.io/badge/control-PID_tuned_in_sim-blue)

AEGIS detects and tracks objects in real time (CNN), aims a pan/tilt gimbal with a tuned PID loop, and fires a Nerf flywheel gun at inanimate showcase targets **only under explicit human-in-the-loop arming and a safety gate enforced in code**. Built to run a custom-trained detector on a Jetson Orin Nano.

Three reasons it exists: **(a)** it's a genuinely fun build, **(b)** it's the cleanest way to learn the full perception → control → actuation pipeline end-to-end, **(c)** it's a portfolio piece that speaks the industry's language — custom-trained CNN, edge-GPU deployment, real-time control, safety engineering.

---

## See it move

| Control loop tracking a moving target | Safety gate: track all, fire inanimate only |
|:--:|:--:|
| ![tracking](docs/media/turret_track.gif) | ![safety](docs/media/safety_gate.gif) |
| The PID drives the aim crosshair (green) to chase the target (red). | Drag a *person* near the target and the gate flips **CLEAR → BLOCKED** live. |

| PID step response (tuned in simulation) | Feedforward + lead — aim *ahead* of the target |
|:--:|:--:|
| ![step](docs/media/pid_step.gif) | ![feedforward](docs/media/feedforward.gif) |
| 22° step acquired in **0.67 s**, ~1% overshoot, 0.16° steady-state. | Plain PID trails the target; α-β velocity feedforward + lead puts the aim **in front** of it. |

> **▶ Interactive demo** — tune the PID gains, drive the turret sim, and play with the safety gate live in your browser. The page runs the **actual** `controller.py` / `simulator.py` / `safety.py` via Pyodide — the same code that flies the turret. See [Running the demo](#-interactive-demo).

---

## How it works

```mermaid
flowchart LR
    subgraph P["M1 · Perception"]
      cam([Camera]) --> det["YOLOv11<br/>detector"] --> trk["target select<br/>+ aim error"]
    end
    subgraph C["M2 · Control"]
      trk --> pid["PID<br/>pan/tilt"]
    end
    subgraph A["M3 · Actuation + Safety"]
      pid --> servos[["pan/tilt<br/>servos"]]
      trk --> gate{"Safety<br/>Gate"}
      gate -->|CLEAR| trig[["flywheel<br/>+ trigger"]]
      gate -->|BLOCKED| hold["hold fire"]
    end
    m4["M4 · custom-trained<br/>detector (balloon)"] -.->|deploys to| det
```

Each frame: the detector finds objects → the tracker picks one and computes a normalised **aim error** → the PID turns that error into pan/tilt angles → the turret drives the servos and asks the **safety gate** whether it may fire. The architecture is deliberately split so the **perception, control and safety maths are headless and fully unit-tested**; the same `controller.py` runs unchanged in the simulator, the live pipeline, and (M3) on the Jetson.

**Predictive tracking (M2.5).** Pure feedback always trails a moving target — it needs a position error to generate the velocity to keep up. `tracking.py` removes that lag the way a gun director does: an **α-β filter** (`estimator.py`) smooths the target's position and velocity; the velocity is fed *forward* straight to the servos (the PID only trims the residual), and the aim is **led** ahead by the dart's flight-time so a moving target can actually be hit. In sim this cuts steady-state tracking lag by **~68%** (5.8° → 1.9° RMS), and on a constant-velocity target the aim leads by exactly `velocity × lead_time`.

## Safety model — *enforced in code, not just documented*

Firing is gated by [`safety.py`](src/aegis/safety.py). It is defence-in-depth, so no single misconfiguration can authorise a shot at a living thing:

```mermaid
flowchart TD
    A([fire request]) --> B{armed?}
    B -->|no| X1[BLOCKED · disarmed]
    B -->|yes| C{"target inanimate<br/>& on allowlist?"}
    C -->|no| X2[BLOCKED · non-fireable / forbidden]
    C -->|yes| D{"locked on?<br/>aim error small"}
    D -->|no| X3[BLOCKED · not locked]
    D -->|yes| E{"person / animal<br/>near target?"}
    E -->|yes| X4[BLOCKED · interlock]
    E -->|no| OK([✅ CLEAR TO FIRE])
```

People and animals are on a **hard denylist that overrides everything** — they can never be a target even if mistakenly added to the allowlist. Firing additionally requires a physical arm switch **and** an explicit fire action. *Track all, fire inanimate only* is a property of the code, demonstrated by 9 dedicated tests and the safety-gate demo above.

## Milestones

| # | Milestone | State |
|---|-----------|-------|
| **M1** | Perception + targeting loop (YOLOv11 → aim error) | ✅ done |
| **M2** | PID pan/tilt control, tuned in closed-loop simulation | ✅ done (sim) |
| **M2.5** | Predictive tracking — α-β velocity feedforward + target lead | ✅ done (sim) |
| **M3** | Actuation layer + safety gate (mock-tested; real drivers stubbed) | ✅ software done · ⏳ hardware |
| **M4** | Custom detector: dataset → train → ONNX/TensorRT export | ✅ pipeline done · ⏳ real data |

Remaining work is real-world, not code: order the kit ([docs/HARDWARE.md](docs/HARDWARE.md)), capture+label a real dataset, build the TensorRT engine on the Jetson.

---

## Quickstart

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # YOLOv11 + torch + opencv

python main.py                            # M1+M2: live tracking + commanded servo angles
python main.py --classes "sports ball" --turret mock   # M3: full safety + fire loop, no hardware
python sim.py --plot                      # M2: tune the PID, save response plots
python train.py --synthetic 16 --epochs 1 --device cpu # M4: smoke-test the training pipeline
pytest                                     # 62 headless tests, no GPU needed
```
In the live window: `a` arm/disarm · `f` fire (only if the gate says CLEAR) · `q` quit.

## ▶ Interactive demo

A static page that runs the real control/safety code in-browser via Pyodide. Built to run locally:
```bash
cd docs/site && python -m http.server 8000   # then open http://localhost:8000
```
Two panels: a **PID tuner** (live response + animated turret viz, with a **feedforward toggle and lead-time slider** — watch the lag collapse and the aim move ahead of the target) and the **safety-gate playground** (drag a person near the target → the gate flips CLEAR/BLOCKED live). (To publish it as a live URL, make the repo public and enable GitHub Pages on `docs/site`.)

## Repo layout

```
aegis/
├── main.py                  # CLI: live tracking (M1) + controller (M2) + turret (M3)
├── sim.py                   # CLI: M2 control simulator & PID tuner
├── capture.py train.py export.py   # M4: dataset capture, training, edge export
├── docs/
│   ├── media/               # README GIFs (generated by tools/make_gifs.py)
│   ├── site/                # interactive Pyodide demo
│   ├── HARDWARE.md          # M3 bill of materials + wiring + bring-up
│   └── M4-TRAINING.md       # dataset -> train -> deploy workflow
├── src/aegis/
│   ├── tracker.py           # target select + aim-error maths (pure — tested)
│   ├── controller.py        # PID + PanTiltController + tuned factory (pure — tested)
│   ├── estimator.py         # α-β velocity filter (pure — tested)
│   ├── tracking.py          # M2.5 feedforward + lead orchestrator (pure — tested)
│   ├── simulator.py         # closed-loop camera/target model + tracking metrics
│   ├── safety.py            # SafetyGate fire-authorisation logic (pure — tested)
│   ├── turret.py            # M3 integration: controller + servos + trigger + gate
│   ├── config.py detector.py overlay.py pipeline.py   # config, YOLO adapter, HUD, loop
│   ├── hardware/            # driver ABCs + servo mapping, mocks, PCA9685/Nerf (lazy)
│   └── data/                # M4: YOLO label/split/data.yaml (pure), builder, synth
└── tests/                   # tracker, controller, safety, hardware, dataset — 62 pure tests
```

## Design notes

- **Jetson over Pi 5 + Coral** — Coral runs only int8 TFLite (a quantise→Edge-TPU-compile tax on every custom model); the Jetson runs torch/YOLO natively and is reusable across a robotics fleet. See [docs/HARDWARE.md](docs/HARDWARE.md).
- **The plant is an integrator** (velocity→angle), so P alone gives zero steady-state error to a step — Ki is kept small (it *hurts* moving-target tracking), Kd damps the acquisition overshoot. Tuned in-sim: `Kp=200, Ki=8, Kd=14`.
- **Feedback can't lead — feedforward can.** A PID only reacts to current error, so it structurally lags a moving target. An α-β filter estimates target velocity; feeding it forward cancels the lag, and projecting it forward by the dart's flight-time gives the lead needed to hit a moving target. The α gain trades responsiveness against noise tolerance on real detections.
- **Testable-core discipline** — every bug-prone bit (aim-error sign conventions, PID anti-windup, the safety policy, YOLO label conversion) is pure Python with no torch/cv2, so 51 tests run in milliseconds with no GPU.

> *Built for fun, learning, and portfolio depth. Any quotes used in project docs must be real, sourced attributions — no invented quotes.*
