"""Synthetic 'balloon' dataset generator.

Two jobs:
  1. Smoke-test the whole M4 pipeline (build -> train -> export) with zero real
     data, so the plumbing is proven before you collect a single photo.
  2. Bootstrap — a fine-tune seeded on synthetic balloons then refined on real
     captures converges faster than real-only from scratch.

A 'balloon' is a coloured ellipse with a little highlight on a noisy
background. Not photoreal, but enough to exercise detect/train/export.
"""

from __future__ import annotations

import random

CLASS_NAMES = ["balloon"]


def make_sample(width: int = 320, height: int = 320, rng: random.Random | None = None):
    """Return ``(image_bgr, [(0, xyxy)])`` with one synthetic balloon."""
    import cv2
    import numpy as np

    rng = rng or random.Random()
    img = (np.random.rand(height, width, 3) * 80 + 40).astype("uint8")  # noisy bg

    rx = rng.randint(width // 12, width // 6)
    ry = int(rx * rng.uniform(1.1, 1.4))  # balloons are taller than wide
    cx = rng.randint(rx + 2, width - rx - 2)
    cy = rng.randint(ry + 2, height - ry - 2)
    colour = (rng.randint(60, 255), rng.randint(60, 255), rng.randint(60, 255))

    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, colour, -1)
    cv2.ellipse(img, (cx - rx // 3, cy - ry // 3), (rx // 5, ry // 5), 0, 0, 360,
                (255, 255, 255), -1)  # highlight
    cv2.line(img, (cx, cy + ry), (cx, cy + ry + ry // 2), (30, 30, 30), 1)  # string

    xyxy = (cx - rx, cy - ry, cx + rx, cy + ry)
    return img, [(0, xyxy)]


def make_samples(n: int, seed: int = 0, **kw):
    rng = random.Random(seed)
    return [make_sample(rng=rng, **kw) for _ in range(n)]
