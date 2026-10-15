"""Export a trained YOLO26 detector to ESP-DL (.espdl) for the ESP32-S3 and measure what quantisation costs.

The step-by-step version is notebooks/02_quantise_for_esp32.ipynb. Needs requirements-esp32.txt.

    1. ONNX export of one head with the box decoding left out (rabbitfish.export.export_onnx).
    2. esp-ppq quantisation, calibrated on train images (rabbitfish.export.quantise). Recommended:
       --cls-clamp 10 --box16 --blockwise --device cuda. Without --blockwise the int8 model's
       class scores collapse (see results/esp32/quantisation_study.md).
    3. mAP and recall at the device's 0.25 threshold, for the float ONNX model and the int8 model
       in esp-ppq's simulator, through the same decoder the firmware uses.

Preprocessing, identical here and on the device: letterbox the RGB frame to W x H keeping the
aspect ratio, pad with 114, divide by 255. Default input 320 x 256: the frames are 4:3 and both
sides must be multiples of 32.

Output: runs/<run>/esp32/<W>x<H>-<quant>-<head>[-c<clamp>][-box16][-br]/ with model.onnx,
model.espdl and report.json. Copy it into the firmware with
firmware/rabbitfish_server/tools/select_model.py.

Usage (from the repository root)
    python scripts/esp32_export.py --run yolo26n-b8-i320 --cls-clamp 10 --box16 --blockwise --device cuda
    python scripts/esp32_export.py --weights models/yolo26n-i320/best.pt --cls-clamp 10 --box16   # quick, CPU
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rabbitfish.paths import RUNS  # noqa: E402  (sets YOLO_CONFIG_DIR)
from rabbitfish import export, metrics  # noqa: E402
from rabbitfish.data import images  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="training run under runs/, e.g. yolo26n-b8-i320 (uses weights/best.pt)")
    src.add_argument("--weights", type=Path, help="any YOLO26 checkpoint, e.g. models/yolo26n-i320/best.pt")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=None, help="default: width * 3/4 rounded up to a multiple of 32")
    p.add_argument("--target", default="esp32s3")
    p.add_argument("--head", default="o2m", choices=["o2m", "o2o"])
    p.add_argument("--cls-clamp", type=float, default=0.0, help="clamp class logits to +-this (0: off)")
    p.add_argument("--box16", action="store_true", help="int16 for the three final box convolutions")
    p.add_argument("--blockwise", action="store_true", help="esp-ppq blockwise reconstruction (GPU, ~30 min)")
    p.add_argument("--quant", default="w8a8", choices=["w8a8", "w8a16", "w16a16"])
    p.add_argument("--calib-images", type=int, default=256, help="train images used to calibrate")
    p.add_argument("--split", default="val", help="split to measure on")
    p.add_argument("--eval-limit", type=int, default=None, help="only the first N images (quick check)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    width = args.width
    height = args.height or -(-width * 3 // (4 * 32)) * 32
    if width % 32 or height % 32:
        raise SystemExit("width and height must be multiples of 32")
    weights = args.weights or RUNS / args.run / "weights" / "best.pt"
    run = args.run or weights.parent.name
    tag = (f"{width}x{height}-{args.quant}-{args.head}" + (f"-c{args.cls_clamp:g}" if args.cls_clamp else "")
           + ("-box16" if args.box16 else "") + ("-br" if args.blockwise else ""))
    out_dir = RUNS / run / "esp32" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path, espdl_path = out_dir / "model.onnx", out_dir / "model.espdl"
    nms_iou = 0.7 if args.head == "o2m" else None
    t0 = time.time()

    print(f"[1/4] ONNX export of {weights} at {width}x{height}, head {args.head}")
    strides, out_names, names = export.export_onnx(weights, onnx_path, width, height, args.head, args.cls_clamp)
    export.simplify_onnx(onnx_path)

    print(f"[2/4] quantising for {args.target} ({args.quant}{', blockwise' if args.blockwise else ''})")
    int16_ops = export.node_names(onnx_path, producing=[n for n in out_names if n.startswith("box")]) if args.box16 else []
    graph = export.quantise(onnx_path, espdl_path, out_names, width, height, quant=args.quant, int16_ops=int16_ops,
                            blockwise=args.blockwise, calib_images=args.calib_images, device=args.device,
                            target=args.target, test_split=args.split)
    float_fn, quant_fn = export.runners(onnx_path, graph, out_names, args.device,
                                        threads=int(os.environ.get("OMP_NUM_THREADS", 4)))

    print(f"[3/4] {args.split}: float ONNX")
    float_res = metrics.evaluate(float_fn, strides, width, height, names, args.split, nms_iou, args.eval_limit)
    print(json.dumps(float_res, indent=1))
    print(f"[4/4] {args.split}: {args.quant} simulated")
    quant_res = metrics.evaluate(quant_fn, strides, width, height, names, args.split, nms_iou, args.eval_limit)
    print(json.dumps(quant_res, indent=1))
    out_err = metrics.output_errors(float_fn, quant_fn, out_names, width, height, args.split)

    report = {
        "run": run, "weights": weights.name, "head": args.head, "target": args.target, "quant": args.quant,
        "cls_clamp": args.cls_clamp, "blockwise": args.blockwise, "int16_ops": int16_ops,
        "test_image": images(args.split, 1)[0].name,
        "input": {"width": width, "height": height, "layout": "NCHW RGB", "scale": "1/255",
                  "letterbox_pad": 114, "name": "images"},
        "outputs": out_names, "strides": strides, "classes": names,
        "decode": "x1=(gx+0.5-l)*s y1=(gy+0.5-t)*s x2=(gx+0.5+r)*s y2=(gy+0.5+b)*s; score=sigmoid(cls); "
                  + (f"per-class NMS at IoU {nms_iou}" if nms_iou else "no NMS"),
        "calib_images": args.calib_images, "eval_split": args.split, "eval_limit": args.eval_limit,
        "mAP_float": float_res, "mAP_quant": quant_res, "output_errors": out_err,
        "drop_mAP50-95": float_res["mAP50-95"] - quant_res["mAP50-95"],
        "files": {"onnx": onnx_path.name, "espdl": espdl_path.name, "espdl_bytes": espdl_path.stat().st_size},
        "seconds": round(time.time() - t0),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    c = metrics.DEVICE_CONF
    rf, rq = float_res[f"R@{c}"], quant_res[f"R@{c}"]
    print(f"mAP50-95 float {float_res['mAP50-95']:.3f} -> {args.quant} {quant_res['mAP50-95']:.3f}; "
          f"recall at conf {c}: float {rf:.3f} -> {args.quant} {rq:.3f}")
    if rq < 0.9 * rf:
        print(f"WARNING: quantisation squashes the scores: the device would miss {1 - rq / max(rf, 1e-9):.0%} of what "
              f"the float model finds at conf {c}, whatever the mAP says. Use --blockwise.")
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
