"""Fire-authorisation logic — the safety spine of AEGIS.

Pure, dependency-free, exhaustively tested. Every fire decision in the system
passes through :meth:`SafetyGate.evaluate`. The policy is deliberately
defence-in-depth so that no single misconfiguration can authorise firing at a
living thing:

    1. Must be ARMED (human-in-the-loop — a physical switch in M3).
    2. Target must be in the configurable *fireable* allowlist (inanimate).
    3. HARD denylist (people, animals) overrides everything — a target on the
       denylist can never be fired at even if mistakenly added to the allowlist.
    4. Must be LOCKED (aim error small) — never fire mid-slew.
    5. Interlock — never fire if a person/animal overlaps or is near the target.

Track-all, fire-inanimate-only is enforced here, not just documented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .tracker import Detection

# Living things — NEVER a valid target, regardless of configuration.
HARD_NO_FIRE: frozenset[str] = frozenset(
    {"person", "cat", "dog", "bird", "horse", "sheep", "cow", "bear", "teddy bear"}
)

# Sensible inanimate defaults the pretrained COCO model can already detect,
# plus "balloon" from M4's custom-trained detector.
DEFAULT_FIREABLE: frozenset[str] = frozenset(
    {"balloon", "sports ball", "bottle", "cup", "frisbee", "apple", "orange", "vase"}
)


@dataclass(frozen=True)
class FireDecision:
    permit: bool
    reason: str

    def __bool__(self) -> bool:  # so `if decision:` reads naturally
        return self.permit


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter)


def _expand(box: tuple[float, float, float, float], margin: float):
    x1, y1, x2, y2 = box
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    return (x1 - mw, y1 - mh, x2 + mw, y2 + mh)


class SafetyGate:
    def __init__(
        self,
        fireable: frozenset[str] = DEFAULT_FIREABLE,
        no_fire: frozenset[str] = HARD_NO_FIRE,
        person_margin: float = 0.25,  # inflate living-thing boxes before overlap test
    ) -> None:
        # The hard denylist always wins, even if a name is in both sets.
        self.fireable = frozenset(fireable) - frozenset(no_fire)
        self.no_fire = frozenset(no_fire)
        self.person_margin = person_margin

    def evaluate(
        self,
        target: Optional[Detection],
        detections: Sequence[Detection],
        *,
        locked: bool,
        armed: bool,
    ) -> FireDecision:
        if not armed:
            return FireDecision(False, "DISARMED")
        if target is None:
            return FireDecision(False, "no target")
        if target.label in self.no_fire:
            return FireDecision(False, f"forbidden target: {target.label}")
        if target.label not in self.fireable:
            return FireDecision(False, f"non-fireable: {target.label}")
        if not locked:
            return FireDecision(False, "not locked")

        # Interlock: a living thing overlapping/near the target blocks the shot.
        for d in detections:
            if d is target or d.label not in self.no_fire:
                continue
            if iou(_expand(d.xyxy, self.person_margin), target.xyxy) > 0.0:
                return FireDecision(False, f"{d.label} near target")

        return FireDecision(True, "CLEAR")
