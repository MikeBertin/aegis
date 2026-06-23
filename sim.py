#!/usr/bin/env python3
"""AEGIS M2 — closed-loop control simulator & PID tuner (no hardware).

Runs the PanTiltController against simulated targets, prints tracking metrics,
and (optionally) saves response plots.

Examples:
    python sim.py                       # run all scenarios, print metrics
    python sim.py --plot                # also save PNGs to runs/
    python sim.py --kp 160 --ki 30 --kd 14 --plot   # try different gains
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from aegis.controller import TUNED_GAINS, default_pan_tilt  # noqa: E402
from aegis import simulator as sim  # noqa: E402

# Tuned gains live in controller.TUNED_GAINS (single source of truth, shared
# with the live pipeline). Performance at these gains: step settle 0.67s, ~1%
# overshoot, 0.16 deg steady; sine 4.6 deg RMS; ramp 3.0 deg RMS; 100% on-frame.
DEFAULTS = dict(TUNED_GAINS)
MAX_SLEW = 300.0  # deg/s — ~MG996R loaded


def make_controller(kp: float, ki: float, kd: float):
    # Simulator models angles naturally (tilt+ = camera up) -> tilt_sign=-1.
    return default_pan_tilt(kp=kp, ki=ki, kd=kd, max_slew_deg_s=MAX_SLEW)


SCENARIOS = {
    "step":  (sim.step(az=20.0, el=-10.0), sim.step_metrics, 3.0),
    "sine":  (sim.sine(amp_az=25.0, amp_el=12.0, freq_hz=0.4), sim.tracking_metrics, 5.0),
    "ramp":  (sim.ramp(rate_az=15.0, az0=-20.0), sim.tracking_metrics, 4.0),
}


def main() -> None:
    p = argparse.ArgumentParser(description="AEGIS M2 control simulator")
    p.add_argument("--kp", type=float, default=DEFAULTS["kp"])
    p.add_argument("--ki", type=float, default=DEFAULTS["ki"])
    p.add_argument("--kd", type=float, default=DEFAULTS["kd"])
    p.add_argument("--plot", action="store_true", help="save response PNGs to runs/")
    p.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    args = p.parse_args()

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    print(f"gains: Kp={args.kp} Ki={args.ki} Kd={args.kd}  slew<={MAX_SLEW}deg/s\n")

    results = {}
    for name in names:
        motion, metric_fn, duration = SCENARIOS[name]
        ctrl = make_controller(args.kp, args.ki, args.kd)
        res = sim.run(ctrl, motion, duration=duration)
        metrics = metric_fn(res)
        results[name] = res
        print(f"[{name:5}] " + "  ".join(f"{k}={v}" for k, v in metrics.items()))

    if args.plot:
        _plot(results)


def _plot(results: dict) -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("runs", exist_ok=True)
    for name, res in results.items():
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax[0].plot(res.t, res.target_az, "r--", label="target az")
        ax[0].plot(res.t, res.pan, "b", label="pan")
        ax[0].set_ylabel("azimuth (deg)"); ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].plot(res.t, res.target_el, "r--", label="target el")
        ax[1].plot(res.t, res.tilt, "g", label="tilt")
        ax[1].set_ylabel("elevation (deg)"); ax[1].set_xlabel("time (s)")
        ax[1].legend(); ax[1].grid(alpha=0.3)
        fig.suptitle(f"AEGIS M2 — {name} response")
        path = f"runs/m2_{name}.png"
        fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
