# The detector on the ESP32-S3

Deployed model: YOLO26n trained at 320 px (`models/yolo26n-i320/best.pt`), exported with
`--cls-clamp 10 --box16 --blockwise` to w8a8 (`models/yolo26n-i320-esp32s3/model.espdl`, 2.9 MB),
running on an ESP32-S3-N16R8 at 240 MHz.

## Accuracy

All rows use the same decoder (`rabbitfish/metrics.py:decode`, ported to the firmware), the
o2m head with per-class NMS at IoU 0.7, and ultralytics' matching and 101-point AP. Input 320x256.

| model | where | split | mAP50 | mAP50-95 | recall at conf 0.25 |
|---|---|---|---|---|---|
| float (ONNX) | PC | val | 0.959 | 0.668 | 0.948 |
| int8, esp-ppq simulation | PC | val | 0.955 | 0.640 | 0.948 |
| int8 on the device, via HTTP | ESP32-S3 | val | 0.948 | 0.640 | |
| int8 on the device, via HTTP | ESP32-S3 | **test** | **0.947** | **0.643** | |

Per class on the device (test): fishbody mAP50-95 0.756, fishhead 0.530.

- The device numbers include the JPEG round trip of the web page (images shrunk to 320x240,
  JPEG quality 92): `firmware/rabbitfish_server/tools/evaluate_device.py`.
  Raw results: `models/yolo26n-i320-esp32s3/device_eval_{val,test}.json`.
- The val device run used an earlier export with the same settings (a separate quantisation run,
  identical simulated scores on the probe images); the test run used the committed file.
- The float row is lower than `ultralytics val` in `results/model_comparison.md` (0.687 on val):
  that uses ultralytics' own letterbox and the checkpoint's default head. Compare float with int8 inside this table.

## Speed and memory

| | |
|---|---|
| inference (dual core) | 952 ms (single core: 1480 ms) |
| JPEG decode / letterbox+quantise / decode+NMS | 22 / 17 / 1 ms |
| whole request, browser to boxes, on home Wi-Fi | ~1.07 s |
| model in flash | 2.9 MB (6 MB partition) |
| free PSRAM / internal RAM while serving | ~5 MB / ~86 KB |

## How the quantisation was chosen

See [`quantisation_study.md`](quantisation_study.md): why plain int8 looked fine by mAP but
missed most fish on the device, what did not fix it (more int16 layers, w8a16), and what did
(blockwise reconstruction).
