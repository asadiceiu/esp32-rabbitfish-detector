# Quantisation study: getting YOLO26n onto an ESP32-S3 without losing the fish

## Summary

1. Plain int8 (w8a8) quantisation with esp-ppq kept **mAP50 at 0.92** (float 0.96), so it
   looked fine. On the device it found only **12 of 22** labelled objects in 7 val images at
   the page's 0.25 threshold (the float model: 21): the class scores were squashed (fish at 0.8 in float came out at
   0.01-0.3) while their *ranking* survived, which is all mAP measures.
2. The device was not at fault: it reproduces esp-ppq's simulation bit for bit (self-test), and the
   simulation shows the same low scores.
3. More int16 did not help: int16 class branch, head or neck kept the scores low or made them
   worse; full 16-bit activations (w8a16) are accurate in simulation but ESP-DL does not reproduce
   them on the device.
4. **Blockwise reconstruction** fixed it: simulated recall at 0.25 equals the float model's
   (0.948), mAP50 0.955, and on the device **47 of 52** objects in the first 20 val images are
   found at 0.25 (test-split mAP50 0.947).

The lesson for any int8 detector: check **recall at the threshold you deploy with**, not only mAP.
`scripts/esp32_export.py` now reports `R@0.25` and warns when it drops.

## 1. Full exports (val split, 264 images)

`scripts/esp32_export.py`, mAP through the device's decoder; float is the ONNX model of the same run.

| run | input | settings | float mAP50 / 50-95 | int8 mAP50 / 50-95 | int8 R@0.25 |
|---|---|---|---|---|---|
| yolo26n @640 | 320x256 | w8a8, **o2o** head (NMS-free) | 0.907 / 0.613 | 0.736 / 0.481 | |
| yolo26n @640 | 320x256 | w8a8, o2m head | 0.944 / 0.639 | 0.937 / 0.598 | |
| yolo26n @640 | 320x256 | **w8a16** | 0.944 / 0.639 | 0.944 / 0.620 | (fails on device) |
| yolo26n @256 | 256x192 | w8a8, o2m | 0.950 / 0.660 | 0.911 / 0.591 | |
| yolo26n @320 | 320x256 | w8a8, o2m | 0.959 / 0.668 | 0.914 / 0.614 | |
| yolo26n @320 | 320x256 | + `--cls-clamp 10 --box16` | 0.959 / 0.668 | 0.917 / 0.620 | low (see 2) |
| **yolo26n @320** | 320x256 | **+ `--blockwise`** (deployed) | 0.959 / 0.668 | **0.955 / 0.640** | **0.948** (float 0.948) |

Takeaways: the o2o head quantises badly; training at the deployment resolution (320) beats
exporting a 640-trained model at 320; `--cls-clamp` and `--box16` help a little; blockwise
reconstruction is what matters. (R@0.25 was added to the report after the first exports.)

## 2. Score check on 7 val images (22 labelled objects)

`scripts/probe_scores.py`: re-quantises with the given settings and compares the best score on each
labelled object (IoU > 0.5, same class) between float and the int8 simulation. Base settings:
yolo26n @320, w8a8, o2m, `--cls-clamp 10 --box16`. The float model finds 21/22 at 0.25.

| change | int16 ops | objects found at 0.25 | int8 / float score |
|---|---|---|---|
| none (the export in section 1) | 3 | 11/22 | 0.34 |
| class branch (`model.23.cv3`) in int16 | 42 | 12/22 | 0.37 |
| whole head (`model.23.*`) in int16, no clamp | 60 | 12/22 | 0.37 |
| neck and head (`model.19`-`23`) in int16 | | 8/22 | 0.27 |
| everything from `model.13` in int16 | | 0/22 | 0.00 |
| layerwise equalisation + bias correction | 3 | 14/22 | 0.47 |
| LSQ fine-tuning (GPU, ~15 min) | 3 | 20/22 | 0.87 |
| **blockwise reconstruction** (GPU, ~28 min) | 3 | **20/22** | **0.95** |

The error is spread over the network (esp-ppq's graphwise analysis puts the largest cumulative
errors in the neck, `model.19`-`22`, including the attention block), not concentrated in the
head, which is why making the head int16 does not help. Mixing int8 and int16 layers made it
worse in esp-ppq's simulation, the more boundaries the worse.

## 3. Device vs simulation

| model | self-test (bit-exact vs esp-ppq) | device inference | device result |
|---|---|---|---|
| w8a8 `-c10-box16` | pass | 952 ms | 12/22 objects at 0.25 on the 7 probe images (simulation: 11/22): squashed scores, as simulated |
| w8a16 | **fail** (first output differs: 1358 vs 1163) | 1429 ms | 3/52 objects at 0.25 in 20 val images, although the simulation is near float |
| w8a8 `-c10-box16-br` (deployed) | pass | 952 ms | 47/52 at 0.25 in 20 val images; test mAP50-95 0.643 |

ESP-DL 3.3.13 and esp-ppq 1.3.11 (both the latest at the time) disagree on 16-bit activations for
this model, so w8a16 is not usable on the device despite its simulated accuracy; int16 for
individual ops (the three box convolutions) is reproduced exactly. w8a16 also needs ~75 KB more
internal RAM and leaves too little for the Wi-Fi driver unless Wi-Fi starts first.

## Settings that did not matter much

- **Resize method of the upload** (OpenCV linear/area vs browser/Pillow): moved individual scores of
  the squashed models by up to ~0.15, but did not explain the low scores.
- **Colour order**: verified RGB end to end (swapping R and B on the client made results worse).
