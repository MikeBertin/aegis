"""Inference wrapper + the pure decision logic for the target discriminator.

The learned CNN scores a candidate patch; :func:`is_valid_target` (pure) turns
that score into a fire/no-fire call. In the pipeline this is an extra *learned*
gate on top of the detector + SafetyGate: even if YOLO says "balloon", the
discriminator confirms it's the *designated* (red) balloon before it's fireable.
"""

from __future__ import annotations

from .model import PATCH, build_net


def is_valid_target(score: float, threshold: float = 0.6) -> bool:
    """Pure decision: is the discriminator confident enough to treat the patch
    as a valid fireable target?"""
    return score >= threshold


def extract_patch(frame, xyxy):
    """Crop ``xyxy`` from a BGR frame and resize to the net's (3, PATCH, PATCH)."""
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((3, PATCH, PATCH), dtype=np.float32)
    crop = cv2.resize(crop, (PATCH, PATCH)).astype(np.float32) / 255.0
    return np.transpose(crop, (2, 0, 1))


class TargetDiscriminator:
    """Loads our trained CNN and scores patches (lazy torch import)."""

    def __init__(self, weights_path: str | None = None) -> None:
        import torch

        self._torch = torch
        self.net = build_net()
        if weights_path:
            self.net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.net.eval()

    def score(self, patch) -> float:
        """Probability the patch is the designated target. ``patch`` is
        (3, PATCH, PATCH) in [0, 1]."""
        torch = self._torch
        x = torch.tensor(patch[None], dtype=torch.float32)
        with torch.no_grad():
            probs = torch.softmax(self.net(x)[0], dim=0)
        return float(probs[1])

    def is_target(self, patch, threshold: float = 0.6) -> bool:
        return is_valid_target(self.score(patch), threshold)
