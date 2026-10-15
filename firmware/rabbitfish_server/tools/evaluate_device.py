"""mAP of the detector running on the ESP32, through its HTTP API, on a dataset split.

Each image is prepared like the web page does it (shrunk to fit the model input, JPEG quality
92) and posted to /detect with conf 0.001, as for a val run. Boxes are scored with ultralytics'
method (greedy IoU matching at 0.50:0.95, 101-point interpolated AP), so the result compares
directly with esp32_export.py's report.json: same model, same split, now including the JPEG
round trip and the device's own decoding.

Needs only numpy and opencv. Usage (from the repo root):
    python firmware/rabbitfish_server/tools/evaluate_device.py                       # dataset/val, rabbitfish.local
    python firmware/rabbitfish_server/tools/evaluate_device.py --split test --host <device-ip>
    python firmware/rabbitfish_server/tools/evaluate_device.py --root data --split samples   # the bundled samples
The dataset root is --root, else $RABBITFISH_DATASET, else <repo>/dataset. Writes
runs/<run>/esp32/<variant>/device_eval_<split>.json when that export folder exists, else --out.
"""
import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
IOUV = np.linspace(0.5, 0.95, 10)


def http_json(url, data=None):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "image/jpeg"} if data else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def box_iou(a, b):
    """(n,4) x (m,4) xyxy -> (n,m)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def match(pred_cls, pred_boxes, true_cls, true_boxes):
    """(n_pred, 10) bool true positives, ultralytics' greedy matching."""
    correct = np.zeros((len(pred_cls), len(IOUV)), bool)
    if not len(pred_cls) or not len(true_cls):
        return correct
    iou = box_iou(true_boxes, pred_boxes) * (true_cls[:, None] == pred_cls[None])
    for i, t in enumerate(IOUV):
        li, pi = np.nonzero(iou >= t)
        if not len(li):
            continue
        m = np.stack([li, pi, iou[li, pi]], 1)
        m = m[m[:, 2].argsort()[::-1]]
        m = m[np.unique(m[:, 1], return_index=True)[1]]
        m = m[np.unique(m[:, 0], return_index=True)[1]]
        correct[m[:, 1].astype(int), i] = True
    return correct


def compute_ap(recall, precision):
    """101-point interpolated AP of one precision-recall curve (ultralytics' compute_ap)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate(([1.0], precision, [0.0])))))
    x = np.linspace(0, 1, 101)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return trapezoid(np.interp(x, mrec, mpre), x)


def ap_per_class(tp, conf, pred_cls, target_cls, nc):
    """(nc, 10) AP per class and IoU threshold."""
    order = np.argsort(-conf)
    tp, pred_cls = tp[order], pred_cls[order]
    ap = np.zeros((nc, len(IOUV)))
    for c in range(nc):
        sel = pred_cls == c
        n_l, n_p = int((target_cls == c).sum()), int(sel.sum())
        if not n_l or not n_p:
            continue
        tpc = tp[sel].cumsum(0)
        fpc = (1 - tp[sel]).cumsum(0)
        recall = tpc / (n_l + 1e-16)
        precision = tpc / (tpc + fpc)
        for j in range(len(IOUV)):
            ap[c, j] = compute_ap(recall[:, j], precision[:, j])
    return ap


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="rabbitfish.local")
    parser.add_argument("--split", default="val")
    parser.add_argument("--root", type=Path, default=None, help="dataset root holding <split>/images and <split>/labels")
    parser.add_argument("--out", type=Path, default=None, help="where to write the JSON result")
    parser.add_argument("--limit", type=int, default=None, help="only the first N images")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality of the upload")
    args = parser.parse_args()

    base = f"http://{args.host}"
    info = http_json(f"{base}/info")
    in_w, in_h = info["input"]["width"], info["input"]["height"]
    names = info["classes"]
    print(f"device: {info['run']} / {info['variant']}, input {in_w}x{in_h}, "
          f"self-test {'passed' if info['device']['self_test_passed'] else 'FAILED'}")

    root = args.root or Path(os.environ.get("RABBITFISH_DATASET", REPO / "dataset"))
    img_dir = root / args.split / "images"
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.limit]
    stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
    timing, round_trip = [], []
    for k, p in enumerate(paths, 1):
        img = cv2.imread(str(p))
        h0, w0 = img.shape[:2]
        s = min(1.0, in_w / w0, in_h / h0)
        w, h = round(w0 * s), round(h0 * s)
        small = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)  # as esp32_export.py's letterbox
        jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, args.quality])[1].tobytes()
        t0 = time.time()
        res = http_json(f"{base}/detect?conf=0.001&max_det=300", jpeg)
        round_trip.append(time.time() - t0)
        timing.append(res["timing_ms"])

        dets = res["detections"]
        pred_boxes = np.array([d["box"] for d in dets], float).reshape(-1, 4)
        pred_cls = np.array([d["class_id"] for d in dets], int)
        lbl = p.parent.parent / "labels" / (p.stem + ".txt")
        rows = np.loadtxt(lbl, ndmin=2) if lbl.exists() and lbl.stat().st_size else np.zeros((0, 5))
        cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
        true_boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        stats["tp"].append(match(pred_cls, pred_boxes, rows[:, 0].astype(int), true_boxes))
        stats["conf"].append(np.array([d["score"] for d in dets], float))
        stats["pred_cls"].append(pred_cls)
        stats["target_cls"].append(rows[:, 0].astype(int))
        if k % 20 == 0 or k == len(paths):
            print(f"  {k}/{len(paths)}  last round trip {round_trip[-1] * 1000:.0f} ms")

    st = {k: np.concatenate(v) for k, v in stats.items()}
    ap = ap_per_class(st["tp"], st["conf"], st["pred_cls"], st["target_cls"], len(names))
    result = {
        "device": {"run": info["run"], "variant": info["variant"]},
        "split": args.split, "images": len(paths), "upload": f"{w}x{h} JPEG q{args.quality}",
        "mAP50": float(ap[:, 0].mean()), "mAP50-95": float(ap.mean()),
        **{n: {"mAP50": float(ap[i, 0]), "mAP50-95": float(ap[i].mean())} for i, n in enumerate(names)},
        "mean_timing_ms": {k: round(float(np.mean([t[k] for t in timing])), 1) for k in timing[0]},
        "mean_round_trip_ms": round(float(np.mean(round_trip)) * 1000, 1),
        "simulated_int8": {"mAP50": info["val_mAP50"]["int8_simulated"],
                           "mAP50-95": info["val_mAP50-95"]["int8_simulated"]},
    }
    print(json.dumps(result, indent=1))
    out = args.out or REPO / "runs" / info["run"] / "esp32" / info["variant"] / f"device_eval_{args.split}.json"
    if (args.out or out.parent.exists()) and not args.limit:
        out.write_text(json.dumps(result, indent=2))
        print(f"written to {out}")


if __name__ == "__main__":
    main()
