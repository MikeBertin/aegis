"""Block-matching stereo from scratch — disparity from an image pair.

`stereo.py` has the *geometry* (`Z = focal·baseline / disparity`) but assumes the
disparity is given. This computes it: for each pixel in the (rectified) left
image, slide a small block along the same row of the right image and find the
horizontal shift `d` that matches best (smallest SAD/SSD). That shift is the
disparity; the geometry then turns the disparity map into a depth map.

Vectorised: for each candidate disparity we shift the whole right image, take the
per-pixel cost, and box-sum it over the block window (built here) — a cost volume
whose per-pixel argmin is the disparity. Pure NumPy, tested against a synthetic
pair with known ground-truth disparity.
"""

from __future__ import annotations

import numpy as np

from .stereo import StereoRig


def _box_sum(x, k: int):
    """Sum over each k×k window (k odd), same output size. Built from shifted
    adds — k² vectorised passes, no library filter."""
    pad = k // 2
    xp = np.pad(x, pad, mode="constant")
    h, w = x.shape
    out = np.zeros((h, w), dtype=np.float64)
    for di in range(k):
        for dj in range(k):
            out += xp[di:di + h, dj:dj + w]
    return out


def block_match_disparity(
    left, right, max_disparity: int = 32, block_size: int = 7, method: str = "sad"
):
    """Dense disparity map (pixels). Left/right are 2D grayscale arrays.

    Right-image features sit to the *left* of their left-image position by the
    disparity, so we search shifts ``d = 0..max_disparity``.
    """
    if block_size % 2 == 0:
        raise ValueError("block_size must be odd")
    L = np.asarray(left, dtype=np.float64)
    R = np.asarray(right, dtype=np.float64)
    h, w = L.shape

    cost_volume = np.full((max_disparity + 1, h, w), np.inf)
    for d in range(max_disparity + 1):
        shifted = np.full_like(R, 0.0)
        shifted[:, d:] = R[:, : w - d] if d else R          # right shifted by d
        diff = L - shifted
        pix = np.abs(diff) if method == "sad" else diff * diff
        cost = _box_sum(pix, block_size)
        if d:
            cost[:, :d] = np.inf                            # no valid match in the border
        cost_volume[d] = cost

    return np.argmin(cost_volume, axis=0).astype(np.float64)


def disparity_to_depth(disparity, rig: StereoRig, min_disparity: float = 0.5):
    """Disparity map -> depth map (metres) via the stereo geometry. Pixels with
    near-zero disparity (infinitely far / no match) come back as ``inf``."""
    disparity = np.asarray(disparity, dtype=np.float64)
    depth = np.full_like(disparity, np.inf)
    valid = disparity > min_disparity
    depth[valid] = rig.focal_px * rig.baseline_m / disparity[valid]
    return depth


def make_synthetic_pair(size: int = 128, bg_disp: int = 4, fg_disp: int = 16, seed: int = 0):
    """A rectified stereo pair with known per-region disparity, for testing/viz.

    Returns ``(left, right, true_disparity)``. A foreground square sits closer
    (larger disparity) than the background.
    """
    rng = np.random.RandomState(seed)
    h = w = size
    dmax = max(bg_disp, fg_disp)
    texture = rng.randint(0, 256, (h, w + 2 * dmax + 2)).astype(np.float64)

    disp = np.full((h, w), bg_disp, dtype=np.int64)
    q = size // 4
    disp[q:3 * q, q:3 * q] = fg_disp

    xs = np.arange(w)
    left = texture[:, xs + dmax]
    right = np.empty((h, w))
    for y in range(h):
        # A left feature at column x sits at right column x - disp (shifted left).
        right[y] = texture[y, xs + dmax + disp[y]]
    return left, right, disp.astype(np.float64)
