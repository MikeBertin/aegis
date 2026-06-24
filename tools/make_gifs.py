#!/usr/bin/env python3
"""Generate README animations from the REAL AEGIS sim/controller/safety code.

Produces three compact GIFs in docs/media/:
    pid_step.gif      — step response settling (M2)
    turret_track.gif  — the aim crosshair chasing a moving target (M2)
    safety_gate.gif   — the fire decision flipping as a person nears the target (M3)

Run: python tools/make_gifs.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

sys.path.insert(0, "src")

from aegis import simulator as sim  # noqa: E402
from aegis.controller import default_pan_tilt  # noqa: E402
from aegis.safety import SafetyGate  # noqa: E402
from aegis.tracker import Detection, aim_error  # noqa: E402

OUT = "docs/media"
BG = "#0d1117"
FG = "#c9d1d9"
GREEN = "#3fb950"
RED = "#f85149"
AMBER = "#d29922"
GREY = "#6e7681"
BLUE = "#58a6ff"
FPS = 25


def _style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GREY)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


# --------------------------------------------------------------------------
def gif_pid_step() -> None:
    ctrl = default_pan_tilt()
    res = sim.run(ctrl, sim.step(az=22.0, el=-12.0), duration=2.5, fps=FPS)
    n = len(res.t)

    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=100)
    fig.patch.set_facecolor(BG)
    _style(ax)
    ax.set_title("M2 · PID step response  (Kp=200 Ki=8 Kd=14)", fontsize=10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("azimuth (deg)")
    ax.set_xlim(0, res.t[-1])
    ax.set_ylim(-3, 27)
    ax.axhline(res.target_az[-1], color=RED, ls="--", lw=1.2, label="target")
    (line,) = ax.plot([], [], color=GREEN, lw=2.2, label="turret pan")
    (head,) = ax.plot([], [], "o", color=GREEN, ms=6)
    txt = ax.text(0.97, 0.08, "", transform=ax.transAxes, ha="right",
                  color=FG, fontsize=9, family="monospace")
    ax.legend(loc="lower right", facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=8)

    def frame(i):
        line.set_data(res.t[:i + 1], res.pan[:i + 1])
        head.set_data([res.t[i]], [res.pan[i]])
        err = res.target_az[i] - res.pan[i]
        txt.set_text(f"err={err:+5.1f} deg")
        return line, head, txt

    anim = FuncAnimation(fig, frame, frames=n, interval=1000 / FPS, blit=True)
    fig.tight_layout()
    anim.save(f"{OUT}/pid_step.gif", writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print("  wrote pid_step.gif")


# --------------------------------------------------------------------------
def gif_turret_track() -> None:
    """Aim crosshair (green) chasing a moving target (red) in the angular plane."""
    ctrl = default_pan_tilt()
    motion = sim.sine(amp_az=26.0, amp_el=13.0, freq_hz=0.5)
    res = sim.run(ctrl, motion, duration=4.0, fps=FPS)
    n = len(res.t)

    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=100)
    fig.patch.set_facecolor(BG)
    _style(ax)
    ax.set_title("M2 · turret tracking a moving target", fontsize=10)
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("elevation (deg)")
    ax.set_xlim(-32, 32)
    ax.set_ylim(-20, 20)
    ax.axhline(0, color=GREY, lw=0.6)
    ax.axvline(0, color=GREY, lw=0.6)

    (tgt,) = ax.plot([], [], "o", color=RED, ms=11, label="target")
    (trail,) = ax.plot([], [], "-", color=RED, lw=1, alpha=0.35)
    (aim_h,) = ax.plot([], [], color=GREEN, lw=1.4)
    (aim_v,) = ax.plot([], [], color=GREEN, lw=1.4)
    (link,) = ax.plot([], [], color=AMBER, lw=1, alpha=0.8)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=8)

    def frame(i):
        ta, te = res.target_az[i], res.target_el[i]
        pa, pe = res.pan[i], res.tilt[i]
        tgt.set_data([ta], [te])
        trail.set_data(res.target_az[:i + 1], res.target_el[:i + 1])
        aim_h.set_data([pa - 2.5, pa + 2.5], [pe, pe])
        aim_v.set_data([pa, pa], [pe - 2.0, pe + 2.0])
        link.set_data([pa, ta], [pe, te])
        return tgt, trail, aim_h, aim_v, link

    anim = FuncAnimation(fig, frame, frames=n, interval=1000 / FPS, blit=True)
    fig.tight_layout()
    anim.save(f"{OUT}/turret_track.gif", writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print("  wrote turret_track.gif")


# --------------------------------------------------------------------------
def gif_safety_gate() -> None:
    """A person box drifts toward the locked target; the gate flips CLEAR->BLOCKED."""
    gate = SafetyGate()
    W, H = 320, 200
    ball = Detection(0, "balloon", 0.95, (140, 80, 180, 130))  # locked, centred-ish
    frames = 90

    fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("#010409")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # image coords (y down)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("M3 · safety gate  (track all · fire inanimate only)", fontsize=10, color=FG)

    ball_box = Rectangle((140, 80), 40, 50, fill=False, ec=GREEN, lw=2)
    person_box = Rectangle((0, 0), 60, 150, fill=False, ec=BLUE, lw=2)
    ax.add_patch(ball_box)
    ax.add_patch(person_box)
    ax.plot(160, 105, "+", color=GREEN, ms=10)  # lock crosshair
    ax.text(184, 95, "balloon", color=GREEN, fontsize=8)
    hud = ax.text(8, 18, "", color=FG, fontsize=10, family="monospace", weight="bold")
    plabel = ax.text(0, 0, "person", color=BLUE, fontsize=8)

    def frame(i):
        # Person enters from the right, drifts left toward the balloon, then leaves.
        t = i / frames
        px = 300 - 230 * (1 - abs(2 * t - 1))  # in then out
        person = Detection(0, "person", 0.9, (px, 40, px + 60, 190))
        person_box.set_x(px)
        person_box.set_y(40)
        plabel.set_position((px, 32))

        d = gate.evaluate(ball, [ball, person], locked=True, armed=True)
        if d.permit:
            hud.set_text("ARMED — CLEAR TO FIRE")
            hud.set_color(RED)
            ball_box.set_edgecolor(RED)
        else:
            hud.set_text(f"BLOCKED — {d.reason}")
            hud.set_color(AMBER)
            ball_box.set_edgecolor(GREEN)
        return person_box, plabel, hud, ball_box

    anim = FuncAnimation(fig, frame, frames=frames, interval=1000 / FPS, blit=False)
    fig.tight_layout()
    anim.save(f"{OUT}/safety_gate.gif", writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print("  wrote safety_gate.gif")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("Generating README GIFs from the real sim/controller/safety code...")
    gif_pid_step()
    gif_turret_track()
    gif_safety_gate()
    print("Done -> docs/media/")
