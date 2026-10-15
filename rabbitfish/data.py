"""Dataset access and the preprocessing the ESP32 firmware reproduces exactly."""
from pathlib import Path

import cv2
import numpy as np

from .paths import DATASET

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def images(split, limit=None, root=None):
    """Sorted image paths of one split (<root>/<split>/images), or of a folder that holds images directly."""
    root = Path(root or DATASET)
    folder = root / split / "images" if (root / split / "images").is_dir() else root / split
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return paths[:limit] if limit else paths


def letterbox(img_bgr, width, height):
    """BGR image -> (1,3,H,W) float32 RGB in [0,1], plus (scale, pad_x, pad_y) to map boxes back.

    Keeps the aspect ratio, pads with grey 114, bilinear resize. firmware/.../detector.cpp
    (Detector::preprocess) does the same, quantising each pixel on the way.
    """
    h0, w0 = img_bgr.shape[:2]
    r = min(width / w0, height / h0)
    nw, nh = round(w0 * r), round(h0 * r)
    px, py = (width - nw) // 2, (height - nh) // 2
    canvas = np.full((height, width, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return x, (r, px, py)


def label_path(img_path):
    """YOLO label file of an image: .../images/x.jpg -> .../labels/x.txt."""
    img_path = Path(img_path)
    return img_path.parent.parent / "labels" / (img_path.stem + ".txt")


def load_labels(img_path, img_shape, scale_pad=(1.0, 0, 0)):
    """YOLO label file -> (classes (n,), boxes (n,4) xyxy), in letterboxed pixels when scale_pad is given."""
    lbl = label_path(img_path)
    rows = np.loadtxt(lbl, ndmin=2) if lbl.exists() and lbl.stat().st_size else np.zeros((0, 5))
    h0, w0 = img_shape[:2]
    r, px, py = scale_pad
    cx, cy, w, h = rows[:, 1] * w0, rows[:, 2] * h0, rows[:, 3] * w0, rows[:, 4] * h0
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) * r + [px, py, px, py]
    return rows[:, 0].astype(int), boxes
