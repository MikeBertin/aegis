"""YOLO detector wrapper.

Thin adapter around Ultralytics YOLO that turns raw model output into the
dependency-free :class:`~aegis.tracker.Detection` objects the rest of the
pipeline speaks. Ultralytics (and torch) are imported lazily so that importing
this module — or running the tracker tests — does not require the heavy stack.
"""

from __future__ import annotations

from typing import Optional

from .tracker import Detection


class Detector:
    def __init__(
        self,
        model: str = "yolo11n.pt",
        conf: float = 0.35,
        target_classes: Optional[set[str]] = None,
    ) -> None:
        from ultralytics import YOLO  # lazy: pulls in torch

        self._model = YOLO(model)
        self._names: dict[int, str] = self._model.names
        self.conf = conf
        self.target_classes = target_classes

    def detect(self, frame) -> list[Detection]:
        """Run one forward pass on a BGR frame and return detections."""
        results = self._model.predict(
            frame, conf=self.conf, verbose=False
        )[0]

        detections: list[Detection] = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            label = self._names.get(class_id, str(class_id))
            if self.target_classes is not None and label not in self.target_classes:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    class_id=class_id,
                    label=label,
                    confidence=float(box.conf[0]),
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return detections
