"""Synthetic labelled patches for the target discriminator.

Two classes:
    1 = fireable target  — a RED balloon (the designated target)
    0 = not a target     — a balloon of the wrong colour, a non-balloon shape,
                           or plain background

So the net must learn *colour and shape together* ("only the red balloon"),
not a trivial balloon-vs-noise cue. Patches are 32×32, channel-first, [0,1].
Uses cv2/numpy (no torch) — generating data needs no model.
"""

from __future__ import annotations

import random

PATCH = 32
CLASS_NAMES = ["not_target", "balloon"]


def _balloon(img, cx, cy, rx, ry, colour, rng):
    import cv2
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, colour, -1)
    cv2.ellipse(img, (cx - rx // 3, cy - ry // 3), (max(1, rx // 5), max(1, ry // 5)),
                0, 0, 360, (255, 255, 255), -1)  # highlight
    cv2.line(img, (cx, cy + ry), (cx, min(PATCH - 1, cy + ry + ry // 2)), (40, 40, 40), 1)


def make_patch(label: int, rng: random.Random | None = None):
    """Return ``(image (3,32,32) float32 in [0,1], label)``."""
    import cv2
    import numpy as np

    rng = rng or random.Random()
    img = (np.random.rand(PATCH, PATCH, 3) * 70 + 30).astype("uint8")  # noisy bg

    rx = rng.randint(6, 11)
    ry = int(rx * rng.uniform(1.05, 1.35))
    cx = rng.randint(rx + 1, PATCH - rx - 1)
    cy = rng.randint(ry + 1, PATCH - ry - 1)

    if label == 1:
        # Designated target: a RED balloon (BGR red, with some variation).
        colour = (rng.randint(0, 50), rng.randint(0, 50), rng.randint(180, 255))
        _balloon(img, cx, cy, rx, ry, colour, rng)
    else:
        kind = rng.random()
        if kind < 0.5:
            # Wrong-colour balloon (blue / green / yellow) — same shape, not target.
            colour = rng.choice([
                (rng.randint(180, 255), 0, 0),     # blue
                (0, rng.randint(180, 255), 0),     # green
                (0, rng.randint(180, 255), rng.randint(180, 255)),  # yellow
            ])
            _balloon(img, cx, cy, rx, ry, colour, rng)
        elif kind < 0.8:
            # Non-balloon shape in a random colour (incl. red rectangles).
            colour = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            cv2.rectangle(img, (cx - rx, cy - ry), (cx + rx, cy + ry), colour, -1)
        # else: plain background (no object)

    x = img.astype(np.float32) / 255.0
    return np.transpose(x, (2, 0, 1)), label   # (3,32,32)


def make_dataset(n: int, seed: int = 0):
    """Balanced dataset -> (X (N,3,32,32) float32, y (N,) int64)."""
    import numpy as np
    rng = random.Random(seed)
    xs, ys = [], []
    for i in range(n):
        label = i % 2                  # balanced
        x, y = make_patch(label, rng)
        xs.append(x)
        ys.append(y)
    return np.stack(xs).astype("float32"), np.array(ys, dtype="int64")
