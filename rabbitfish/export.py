"""ONNX export of the raw YOLO26 head and esp-ppq quantisation for the ESP32-S3.

Needs the ESP32 environment (requirements-esp32.txt): esp-ppq, onnx, onnxsim, onnxruntime.
"""
import re

import cv2
import torch
from ultralytics import YOLO

from .data import images, letterbox


def export_onnx(weights, onnx_path, width, height, head_kind="o2m", cls_clamp=0.0):
    """ONNX of one YOLO26 head with raw per-stride box/cls outputs; returns (strides, output names, class names).

    The outputs are, per stride 8/16/32, box (1,4,h,w) = left/top/right/bottom distances in grid
    cells and cls (1,nc,h,w) = class logits. Box decoding is left out of the graph so boxes and
    scores get separate quantisation scales and the device needs no TopK/Gather operators.

    head_kind: "o2m" (one-to-many, needs NMS; what ultralytics val uses) or "o2o" (NMS-free;
    quantises badly). cls_clamp: clamp the logits to +-this inside the model (the raw range reaches
    about -170 on certain background, which wastes the int8 range).
    """
    yolo = YOLO(str(weights))
    model = yolo.model.float().eval()
    head = model.model[-1]
    if getattr(head, "one2one_cv2", None) is None:
        raise SystemExit(f"{weights} has no one-to-one head; this expects a YOLO26 (end-to-end) model")
    if head.reg_max != 1:
        raise SystemExit(f"reg_max={head.reg_max}; the device decoder assumes YOLO26's reg_max=1 (no DFL)")
    head.end2end = head_kind == "o2o"  # fuse() drops the other head
    model.fuse(verbose=False)
    box_head, cls_head = (head.one2one_cv2, head.one2one_cv3) if head_kind == "o2o" else (head.cv2, head.cv3)
    strides = [int(s) for s in head.stride]

    def raw_forward(x):
        out = []
        for i in range(head.nl):
            cls = cls_head[i](x[i])
            out += [box_head[i](x[i]), cls.clamp(-cls_clamp, cls_clamp) if cls_clamp else cls]
        return tuple(out)

    head.forward = raw_forward
    out_names = [f"{kind}_s{s}" for s in strides for kind in ("box", "cls")]
    torch.onnx.export(model, torch.zeros(1, 3, height, width), str(onnx_path), opset_version=13,
                      input_names=["images"], output_names=out_names, do_constant_folding=True)
    return strides, out_names, dict(yolo.names)


def simplify_onnx(onnx_path):
    """Simplify in place (esp-ppq would do it anyway), so the node names used below are final.

    Runs onnxsim in a separate process: called inside a Jupyter kernel, onnxsim can hang
    indefinitely (seen on a Linux cluster), while from a plain Python process it takes ~2 s.
    """
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "onnxsim", str(onnx_path), str(onnx_path)], check=True,
                   stdout=subprocess.DEVNULL)


def node_names(onnx_path, pattern=None, producing=None):
    """Names of ONNX nodes: all, those matching a regex, or those producing the given tensors."""
    import onnx
    nodes = onnx.load(str(onnx_path)).graph.node
    if producing is not None:
        return [n.name for n in nodes if set(n.output) & set(producing)]
    return [n.name for n in nodes if pattern is None or re.search(pattern, n.name)]


def quantise(onnx_path, espdl_path, out_names, width, height, *, quant="w8a8", int16_ops=(), blockwise=False,
             blockwise_steps=None, calib_images=256, device="cpu", target="esp32s3", test_split="val"):
    """Quantise with esp-ppq and write the .espdl; returns the quantised graph (for simulation).

    blockwise: esp-ppq's blockwise reconstruction. Without it, plain w8a8 squashes the class
    scores of this model to about 0.01-0.3 (most fish fall below the 0.25 threshold) even though
    mAP barely changes. With it, R@0.25 matches the float model. About 30 min on an H100;
    impractically slow on a CPU.

    int16_ops: node names to run in int16. The three final box convolutions are worth it
    (finer box steps); see results/esp32/quantisation_study.md for why more int16 is not.

    The .espdl embeds the first image of test_split and its expected outputs, so the firmware's
    self-test can check the device computes exactly what the simulator does.
    """
    from esp_ppq.api import espdl_quantize_onnx
    from esp_ppq.api.setting import QuantizationSettingFactory
    from esp_ppq.core import TargetPlatform

    setting = QuantizationSettingFactory.espdl_setting(num_of_bits=16 if quant == "w16a16" else 8)
    for op in int16_ops:
        setting.dispatching_table.append(operation=op, platform=TargetPlatform.ESPDL_S3_INT16)
    setting.blockwise_reconstruction = blockwise
    setting.blockwise_reconstruction_setting.collecting_device = device
    if blockwise_steps:  # esp-ppq's default is 5000 per block
        setting.blockwise_reconstruction_setting.steps = blockwise_steps
    test_img = torch.from_numpy(letterbox(cv2.imread(str(images(test_split, 1)[0])), width, height)[0])
    calib = [torch.from_numpy(letterbox(cv2.imread(str(p)), width, height)[0][0])
             for p in images("train", calib_images)]
    return espdl_quantize_onnx(
        str(onnx_path), str(espdl_path), calib_dataloader=torch.utils.data.DataLoader(calib, batch_size=1),
        calib_steps=len(calib), input_shape=[1, 3, height, width], target=target, quant_type=quant,
        collate_fn=lambda b: b.to(device), device=device, error_report=False,
        setting=setting if quant == "w8a8" or int16_ops or blockwise else None, inputs=[test_img],
        export_test_values=True, test_output_names=out_names)


def runners(onnx_path, graph, out_names, device="cpu", threads=4):
    """(float_fn, quant_fn): both take a (1,3,H,W) float32 array and return the raw outputs as numpy."""
    import onnxruntime as ort
    from esp_ppq.executor import TorchExecutor

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads  # explicit, or onnxruntime pins threads to cores (fails on clusters)
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    executor = TorchExecutor(graph=graph, device=device)

    def float_fn(x):
        return sess.run(out_names, {"images": x})

    def quant_fn(x):
        outs = executor.forward(inputs=torch.from_numpy(x).to(device), output_names=out_names)
        return [o.cpu().numpy() for o in outs]

    return float_fn, quant_fn
