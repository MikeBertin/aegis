# AEGIS — Hardware (M3 Bill of Materials)

Everything needed to take AEGIS from "tracks on screen + commands servos in
software" to a physical pan/tilt turret that fires under the safety gate.
Prices are indicative UK retail (mid-2026), incl. VAT.

## M3 — Core build (order to start the physical milestone)

| # | Item | Suggested part | ~£ | Purpose |
|---|------|----------------|----|---------|
| 1 | **Edge compute** | NVIDIA Jetson Orin Nano Dev Kit (8GB) | 250 | Runs YOLO + controller on-board. The brain. |
| 2 | NVMe SSD 256GB (M.2 2280) | WD Blue SN580 / Kingston NV2 | 25 | Jetson OS + models (faster than SD). |
| 3 | Active cooler + 5V/4A PSU | (often bundled with dev kit) | 20 | Orin Nano needs the fan; barrel-jack PSU. |
| 4 | **Pan/tilt gimbal** | 2-axis aluminium bracket for MG996R | 15 | The aiming mechanism. |
| 5 | **Servos ×2** | MG996R (metal gear, ~10 kg·cm) | 12 | Pan + tilt drive. |
| 6 | **Servo driver** | Adafruit PCA9685 16-ch PWM (I2C) | 6 | Drives servos off-CPU; spare channels for the trigger. |
| 7 | **Servo power** | 5–6V 3A UBEC / supply | 10 | Servos must NOT be powered from the Jetson. |
| 8 | **Camera** | USB UVC webcam 1080p (or Pi Cam v3 + CSI) | 25 | The eye. USB is simplest on Jetson. |
| 9 | **Nerf blaster** | Flywheel/automatic (e.g. Nerf Elite motorised) | 30 | The actuator. Flywheel models mod cleanly. |
| 10 | Relay/MOSFET module | IRLB8721 logic-level MOSFET or 1-ch relay | 5 | Switch the flywheel motors from a GPIO pin. |
| 11 | Trigger servo | SG90 (9g) | 3 | Pushes the dart-advance/trigger. Shares the PCA9685. |
| 12 | Jumper wires + breadboard + dupont | assorted kit | 8 | Prototyping the wiring. |
| 13 | Base/mount | tripod plate or printed base | 12 | Stable platform; keeps the muzzle pointed safely. |
| 14 | **Arm switch** | latching SPST toggle + LED | 4 | The physical human-in-the-loop arm. Wired to a GPIO. |
| | | **M3 subtotal** | **~£425** | |

## M4 — Custom model (later; no new turret hardware)
Mostly compute/time, not parts. The Jetson trains/quantises; optional extras:

| Item | ~£ | Purpose |
|------|----|---------|
| Foam balls / balloons + simple target rig | 10 | Fireable targets to collect a dataset of. |
| (Optional) USB SSD for dataset/runs | 30 | If 256GB fills with training data. |

## Wiring summary (matches `src/aegis/hardware/`)
```
Jetson Orin Nano
 ├─ I2C (pin 3 SDA / pin 5 SCL) ── PCA9685 ── ch0 pan servo
 │                                          ── ch1 tilt servo
 │                                          ── ch2 trigger servo (SG90)
 ├─ GPIO pin 18 ───────────────── MOSFET/relay ── flywheel motors (own pack)
 ├─ GPIO (arm switch pin) ──────── latching toggle ── 3V3 / GND
 └─ USB ───────────────────────── webcam

Power: Jetson on its own 5V/4A PSU. Servos on a separate 5–6V 3A supply.
Flywheels on the blaster's own battery pack. ALL grounds common.
```

## Safety wiring rules (non-negotiable — enforced in code AND hardware)
- The flywheel GPIO pin **defaults LOW (off)** at boot and on `close()` — see
  `hardware/nerf.py`.
- Firing requires the **physical arm switch** AND a software-gate `CLEAR`
  decision AND an explicit fire action (`SafetyGate` in `safety.py`).
- During bring-up, **remove the darts** until the servo travel limits and
  `tilt`/`pan` calibration are confirmed. Point the muzzle at a backstop.
- Never point at people during testing — the no-fire interlock blocks firing
  when a person overlaps the target, but bring-up happens with an empty magazine.

## Bring-up order (M3)
1. Flash JetPack to the NVMe, boot, `pip install -r requirements.txt` +
   `adafruit-circuitpython-servokit`.
2. Wire PCA9685 + servos only. Run `python main.py --turret real` with the
   **blaster unplugged** — confirm the gimbal tracks and stays within limits.
3. Calibrate `ServoCalibration` (centre, min/max, invert) per axis in `main.py`.
4. Add the flywheel relay + trigger servo. Test `arm`/`fire` with an **empty**
   magazine.
5. Load darts, fire at an inanimate target, verify the safety HUD/interlocks.
