"""Per-image detection scores of the float model vs esp-ppq's simulated quantised model.

The quick way to compare quantisation settings (results/esp32/quantisation_study.md was made
with it): mAP hides it when quantisation squashes the class scores, per-image scores do not.
For each image it prints the top scores and the best score on each labelled object, for both
models, and at the end how many labelled objects each finds at the device's 0.25 threshold.

Writes to runs/<run>/esp32/probe-<tag>/ (model.espdl with self-test values, probe.json).

Usage (from the repository root)
    python scripts/probe_scores.py --run yolo26n-b8-i320 --cls-clamp 10 --box16
    python scripts/probe_scores.py --run yolo26n-b8-i320 --cls-clamp 10 --box16 --blockwise --device cuda
    python scripts/probe_scores.py --run yolo26n-b8-i320 --cls-clamp 10 --box16 --int16 'model.23.cv3'
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rabbitfish.paths import DATASET, RUNS  # noqa: E402
from rabbitfish import export  # noqa: E402
from rabbitfish.data import letterbox, load_labels  # noqa: E402
from rabbitfish.metrics import DEVICE_CONF, decode  # noqa: E402

DEFAULT_IMAGES = ["P7020005", "P7020006", "P7020016", "P7020025", "P7020027", "P7020040", "P7020062"]


def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def summarise(outs, strides, img_path, img_shape, scale_pad):
    """Top-4 scores after NMS, and the best score on each labelled object (IoU > 0.5, same class)."""
    boxes, scores, classes = decode(outs, strides, 0.001, nms_iou=0.7)
    order = np.argsort(-scores)
    true_cls, true_boxes = load_labels(img_path, img_shape, scale_pad)
    best = []
    for tc, tb in zip(true_cls, true_boxes):
        ok = [scores[i] for i in order if classes[i] == tc and iou(boxes[i], tb) > 0.5]
        best.append(round(float(max(ok)), 3) if ok else None)
    return {"top4": [round(float(scores[i]), 3) for i in order[:4]], "per_label": best,
            "label_classes": [int(c) for c in true_cls]}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--head", default="o2m", choices=["o2m", "o2o"])
    p.add_argument("--cls-clamp", type=float, default=0.0)
    p.add_argument("--box16", action="store_true")
    p.add_argument("--quant", default="w8a8", choices=["w8a8", "w8a16", "w16a16"])
    p.add_argument("--int16", action="append", default=[], metavar="REGEX",
                   help="ops whose name matches also run in int16 (repeatable), e.g. 'model.23.cv3' = class branch")
    p.add_argument("--blockwise", action="store_true", help="esp-ppq blockwise reconstruction (use a GPU)")
    p.add_argument("--calib-images", type=int, default=256)
    p.add_argument("--split", default="val")
    p.add_argument("--images", nargs="*", default=DEFAULT_IMAGES, help="image stems in the split")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tag", default="", help="suffix for the output folder")
    args = p.parse_args()

    w, h = args.width, args.height
    tag = (f"{w}x{h}-{args.quant}-{args.head}" + (f"-c{args.cls_clamp:g}" if args.cls_clamp else "")
           + ("-box16" if args.box16 else "") + ("-br" if args.blockwise else "") + (f"-{args.tag}" if args.tag else ""))
    out_dir = RUNS / args.run / "esp32" / f"probe-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    t0 = time.time()

    strides, out_names, _ = export.export_onnx(RUNS / args.run / "weights" / "best.pt", onnx_path, w, h,
                                               args.head, args.cls_clamp)
    export.simplify_onnx(onnx_path)
    int16_ops = export.node_names(onnx_path, producing=[n for n in out_names if n.startswith("box")]) if args.box16 else []
    for pattern in args.int16:
        int16_ops += [n for n in export.node_names(onnx_path, pattern) if n not in int16_ops]
    print(f"int16 ops: {len(int16_ops)}")
    graph = export.quantise(onnx_path, out_dir / "model.espdl", out_names, w, h, quant=args.quant,
                            int16_ops=int16_ops, blockwise=args.blockwise, calib_images=args.calib_images,
                            device=args.device, test_split=args.split)
    float_fn, quant_fn = export.runners(onnx_path, graph, out_names, args.device)

    results = {}
    for stem in args.images:
        img_path = DATASET / args.split / "images" / f"{stem}.jpg"
        img = cv2.imread(str(img_path))
        x, sp = letterbox(img, w, h)
        f = summarise(float_fn(x), strides, img_path, img.shape, sp)
        q = summarise(quant_fn(x), strides, img_path, img.shape, sp)
        results[stem] = {"float": f, "quant": q}
        print(f"{stem}  labels {f['label_classes']}")
        print(f"   float  top4 {f['top4']}  per label {f['per_label']}")
        print(f"   quant  top4 {q['top4']}  per label {q['per_label']}")

    pairs = [(fv, qv) for r in results.values() for fv, qv in zip(r["float"]["per_label"], r["quant"]["per_label"])
             if fv is not None]
    kept = sum(qv is not None and qv >= DEVICE_CONF for _, qv in pairs)
    ratio = np.mean([(qv or 0) / fv for fv, qv in pairs])
    print(f"SUMMARY {out_dir.name}: labelled objects found at conf {DEVICE_CONF}: quant {kept}/{len(pairs)}, "
          f"float {sum(fv >= DEVICE_CONF for fv, _ in pairs)}/{len(pairs)}; mean quant/float score {ratio:.2f}")
    (out_dir / "probe.json").write_text(json.dumps(results, indent=1))
    print(f"done in {time.time() - t0:.0f} s -> {out_dir / 'probe.json'}")


if __name__ == "__main__":
    main()
