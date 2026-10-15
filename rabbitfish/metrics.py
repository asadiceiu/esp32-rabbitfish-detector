"""Decoding of the raw YOLO26 outputs (the firmware does the same) and ultralytics-style mAP."""
import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms
from ultralytics.utils.metrics import ap_per_class, box_iou

from .data import images, letterbox, load_labels

IOUV = np.linspace(0.5, 0.95, 10)  # mAP50-95 thresholds, as in ultralytics
DEVICE_CONF = 0.25  # the web page's default confidence threshold


def decode(outs, strides, conf_thr, nms_iou=None, max_det=300):
    """Raw outputs [box_s8, cls_s8, box_s16, ...] -> (boxes xyxy (k,4), scores (k,), classes (k,)).

    For stride s and grid cell (gx, gy), with box = (l, t, r, b) distances in cells:
        x1 = (gx + 0.5 - l) * s,  y1 = (gy + 0.5 - t) * s,  x2 = (gx + 0.5 + r) * s,  y2 = (gy + 0.5 + b) * s
        score = sigmoid(class logit)
    Without nms_iou: ultralytics' end-to-end path (top max_det (cell, class) pairs). With it:
    ultralytics' val path (every pair above conf_thr, per-class NMS, top max_det). The firmware's
    Detector::postprocess is the second one.
    """
    all_boxes, all_scores = [], []
    for i, s in enumerate(strides):
        box, cls = outs[2 * i][0], outs[2 * i + 1][0]  # (4,h,w), (nc,h,w)
        h, w = box.shape[1:]
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
        l, t, r, b = box
        all_boxes.append((np.stack([gx - l, gy - t, gx + r, gy + b]) * s).reshape(4, -1).T)
        all_scores.append((1 / (1 + np.exp(-np.clip(cls, -50, 50)))).reshape(cls.shape[0], -1).T)
    boxes, scores = np.concatenate(all_boxes), np.concatenate(all_scores)  # (N,4), (N,nc)
    nc = scores.shape[1]
    flat = scores.ravel()
    if nms_iou is not None:
        idx = np.nonzero(flat > conf_thr)[0]
        b, sc, c = boxes[idx // nc], flat[idx], idx % nc
        keep = batched_nms(torch.from_numpy(b).float(), torch.from_numpy(sc).float(),
                           torch.from_numpy(c), nms_iou).numpy()[:max_det]
        return b[keep], sc[keep], c[keep]
    k = min(max_det, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[flat[idx] > conf_thr]
    return boxes[idx // nc], flat[idx], idx % nc


def match(pred_cls, pred_boxes, true_cls, true_boxes):
    """(n_pred, 10) bool: which predictions are true positives at each IoU threshold (ultralytics' matching)."""
    correct = np.zeros((len(pred_cls), len(IOUV)), bool)
    if not len(pred_cls) or not len(true_cls):
        return correct
    iou = box_iou(torch.from_numpy(true_boxes).float(), torch.from_numpy(pred_boxes).float()).numpy()
    iou = iou * (true_cls[:, None] == pred_cls[None])
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


def evaluate(run_fn, strides, width, height, names, split, nms_iou, limit=None, conf_thr=0.001, root=None):
    """mAP of a model given as run_fn: (1,3,H,W) float32 -> list of raw outputs (numpy).

    Besides mAP it reports P and R at DEVICE_CONF: what the web page shows by default. mAP only
    ranks the scores, so it stays high even when quantisation squashes every score to ~0.01,
    while R@0.25 drops (see results/esp32/quantisation_study.md).
    """
    stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
    for p in images(split, limit, root):
        img = cv2.imread(str(p))
        x, sp = letterbox(img, width, height)
        boxes, scores, cls = decode(run_fn(x), strides, conf_thr, nms_iou)
        tcls, tboxes = load_labels(p, img.shape, sp)
        stats["tp"].append(match(cls, boxes, tcls, tboxes))
        stats["conf"].append(scores)
        stats["pred_cls"].append(cls)
        stats["target_cls"].append(tcls)
    s = {k: np.concatenate(v) for k, v in stats.items()}
    res = ap_per_class(s["tp"], s["conf"], s["pred_cls"], s["target_cls"], names=names)
    p, r, ap, ucls = res[2], res[3], res[5], res[6]
    out = {"P": float(p.mean()), "R": float(r.mean()), "mAP50": float(ap[:, 0].mean()), "mAP50-95": float(ap.mean())}
    kept = s["conf"] >= DEVICE_CONF
    out[f"P@{DEVICE_CONF}"] = float(s["tp"][kept, 0].mean()) if kept.any() else 0.0
    out[f"R@{DEVICE_CONF}"] = float(s["tp"][kept, 0].sum() / max(len(s["target_cls"]), 1))
    for j, c in enumerate(ucls):
        out[names[int(c)]] = {"P": float(p[j]), "R": float(r[j]), "mAP50": float(ap[j, 0]), "mAP50-95": float(ap[j].mean())}
    return out


def output_errors(float_fn, quant_fn, out_names, width, height, split, n=32, root=None):
    """Per output tensor: float value range and how far the quantised result is from it (SNR in dB)."""
    acc = {k: {"min": np.inf, "max": -np.inf, "sig": 0.0, "err": 0.0} for k in out_names}
    for p in images(split, n, root):
        x, _ = letterbox(cv2.imread(str(p)), width, height)
        for k, f, q in zip(out_names, float_fn(x), quant_fn(x)):
            a = acc[k]
            a["min"], a["max"] = min(a["min"], float(f.min())), max(a["max"], float(f.max()))
            a["sig"] += float((f ** 2).sum())
            a["err"] += float(((f - q) ** 2).sum())
    return {k: {"min": round(a["min"], 3), "max": round(a["max"], 3),
                "snr_db": round(10 * np.log10(a["sig"] / max(a["err"], 1e-12)), 1)} for k, a in acc.items()}
