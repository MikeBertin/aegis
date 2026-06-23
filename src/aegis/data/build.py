"""Write an Ultralytics-format dataset directory from annotated samples.

A *sample* is ``(image_bgr_ndarray, [(class_id, xyxy_pixels), ...])``. This
materialises the standard layout the trainer expects:

    <out>/images/{train,val}/*.jpg
    <out>/labels/{train,val}/*.txt
    <out>/data.yaml
"""

from __future__ import annotations

import os
from typing import Sequence

from .dataset import data_yaml, label_line, split_dataset, xyxy_to_yolo


def build_dataset(
    out_dir: str,
    samples: Sequence,  # list of (image_bgr, [(class_id, (x1,y1,x2,y2)), ...])
    class_names: Sequence[str],
    val_frac: float = 0.2,
    seed: int = 0,
) -> str:
    """Write images + labels + data.yaml. Returns the data.yaml path."""
    import cv2  # lazy: only needed to write image files

    out_dir = os.path.abspath(out_dir)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    indexed = list(enumerate(samples))
    split = split_dataset(indexed, val_frac=val_frac, seed=seed)

    for fold, rows in (("train", split.train), ("val", split.val)):
        for idx, (image, annotations) in rows:
            h, w = image.shape[:2]
            stem = f"img_{idx:05d}"
            cv2.imwrite(os.path.join(out_dir, "images", fold, stem + ".jpg"), image)
            lines = [
                label_line(cid, xyxy_to_yolo(box, w, h))
                for cid, box in annotations
            ]
            with open(os.path.join(out_dir, "labels", fold, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(data_yaml(out_dir, class_names))
    return yaml_path
