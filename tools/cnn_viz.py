#!/usr/bin/env python3
"""Visualise the from-scratch CNN discriminator for the README.

Left: a grid of synthetic patches (the designated red balloon vs distractors).
Right: the validation-accuracy curve as our own conv net learns to tell them
apart. Saves docs/media/cnn_discriminator.png.

Run: python tools/cnn_viz.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG = "#0d1117"
FG = "#c9d1d9"
GREEN = "#3fb950"
RED = "#f85149"


def main() -> None:
    import torch
    from torch import nn, optim

    from aegis.cnn.model import build_net
    from aegis.cnn.patches import make_dataset, make_patch
    import random

    # --- train, recording val accuracy per epoch ---
    Xtr, ytr = make_dataset(1500, seed=1)
    Xva, yva = make_dataset(400, seed=99)
    Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr)
    Xva, yva = torch.tensor(Xva), torch.tensor(yva)
    net = build_net()
    opt = optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    accs = []
    for _ in range(12):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad(); loss_fn(net(Xtr[idx]), ytr[idx]).backward(); opt.step()
        with torch.no_grad():
            accs.append((net(Xva).argmax(1) == yva).float().mean().item())

    # --- figure ---
    fig = plt.figure(figsize=(8.6, 3.4), dpi=100)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(3, 7)

    rng = random.Random(3)
    for r in range(3):
        for c in range(4):
            ax = fig.add_subplot(gs[r, c])
            label = rng.randint(0, 1)
            img, _ = make_patch(label, rng)
            ax.imshow(np.transpose(img, (1, 2, 0))[..., ::-1])  # BGR->RGB
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GREEN if label else RED); s.set_linewidth(2)
    fig.text(0.06, 0.95, "synthetic patches", color=FG, fontsize=10)
    fig.text(0.06, 0.02, "green = target (red balloon)   red = not target",
             color=FG, fontsize=8)

    ax = fig.add_subplot(gs[:, 4:])
    ax.set_facecolor(BG)
    ax.plot(range(1, len(accs) + 1), [a * 100 for a in accs], "-o", color=GREEN, ms=4)
    ax.set_title("from-scratch CNN learns the target", color=FG, fontsize=11)
    ax.set_xlabel("epoch", color=FG); ax.set_ylabel("val accuracy (%)", color=FG)
    ax.set_ylim(50, 101)
    ax.tick_params(colors=FG, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#6e7681")
    ax.grid(alpha=0.25)
    ax.text(0.5, 0.1, f"final {accs[-1]*100:.1f}%", transform=ax.transAxes,
            color=FG, fontsize=10, ha="center", family="monospace")

    fig.tight_layout()
    fig.savefig("docs/media/cnn_discriminator.png", dpi=110,
                facecolor=BG, bbox_inches="tight")
    print(f"wrote docs/media/cnn_discriminator.png (final val acc {accs[-1]*100:.1f}%)")


if __name__ == "__main__":
    main()
