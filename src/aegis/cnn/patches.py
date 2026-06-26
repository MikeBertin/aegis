"""Synthetic labelled patches for the target discriminator.

Two classes:
    1 = fireable target  — a RED balloon (the designated target)
    0 = not a target     — a balloon of the wrong colour, OR a non-balloon shape
                           (often RED), OR plain background

The negatives deliberately include plenty of **red non-balloon shapes** (red
squares/triangles). Without them the net just learns "red == target" and a red
square fools it; with them it must use *colour AND shape* together. Patches are
32×32, channel-first, [0,1]. Uses cv2/numpy (no torch) — data needs no model.
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

    red = (rng.randint(0, 50), rng.randint(0, 50), rng.randint(180, 255))  # BGR red

    if label == 1:
        # Designated target: a RED balloon.
        _balloon(img, cx, cy, rx, ry, red, rng)
    else:
        kind = rng.random()
        if kind < 0.35:
            # Wrong-colour balloon — same shape, wrong colour.
            colour = rng.choice([
                (rng.randint(180, 255), 0, 0),     # blue
                (0, rng.randint(180, 255), 0),     # green
                (0, rng.randint(180, 255), rng.randint(180, 255)),  # yellow
            ])
            _balloon(img, cx, cy, rx, ry, colour, rng)
        elif kind < 0.85:
            # Non-balloon shape — RED half the time, so the net can't rely on
            # colour alone. Square/rectangle or triangle.
            colour = red if rng.random() < 0.5 else (
                rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            if rng.random() < 0.6:
                cv2.rectangle(img, (cx - rx, cy - ry), (cx + rx, cy + ry), colour, -1)
            else:
                pts = np.array([[cx, cy - ry], [cx - rx, cy + ry], [cx + rx, cy + ry]])
                cv2.fillPoly(img, [pts], colour)
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
