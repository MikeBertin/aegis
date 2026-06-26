#!/usr/bin/env python3
"""Visualise block-matching stereo for the README.

Synthesises a rectified stereo pair (a foreground square closer than the
background), runs our from-scratch block matcher, and shows left / right /
ground-truth disparity / recovered disparity. Saves docs/media/stereo_blockmatch.png.

Run: python tools/stereo_viz.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aegis.stereo_match import block_match_disparity, make_synthetic_pair  # noqa: E402

BG = "#0d1117"
FG = "#c9d1d9"


def main() -> None:
    left, right, true = make_synthetic_pair(size=140, bg_disp=4, fg_disp=18, seed=3)
    disp = block_match_disparity(left, right, max_disparity=26, block_size=7)

    fig, ax = plt.subplots(1, 4, figsize=(9.4, 2.7), dpi=100)
    fig.patch.set_facecolor(BG)
    panels = [
        ("left image", left, "gray"),
        ("right image", right, "gray"),
        ("true disparity", true, "magma"),
        ("recovered (block match)", disp, "magma"),
    ]
    for a, (title, img, cmap) in zip(ax, panels):
        a.imshow(img, cmap=cmap)
        a.set_title(title, color=FG, fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color("#30363d")

    fig.text(0.5, 0.02,
             "block matching slides a window along each row to find the per-pixel shift → depth",
             color=FG, fontsize=9, ha="center")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig("docs/media/stereo_blockmatch.png", dpi=110, facecolor=BG, bbox_inches="tight")
    print("wrote docs/media/stereo_blockmatch.png")


if __name__ == "__main__":
    main()
