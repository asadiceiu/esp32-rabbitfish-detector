# RabbitFish detector on an ESP32-S3

Detects rabbitfish **bodies and heads** in underwater survey photos with a YOLO26-nano model that
runs **on a microcontroller**: an ESP32-S3 with 8 MB PSRAM, as a small Wi-Fi server you open in
any browser. This repository is the whole pipeline:

```
 dataset ──► 01 train YOLO26n (PyTorch, GPU) ──► best.pt ──► 02 quantise to int8 (esp-ppq) ──► model.espdl
                                                                                                  │
                     03 test from Python  ◄── HTTP /detect ◄── ESP32-S3 firmware (ESP-DL) ◄──────┘
```

| | mAP50 | mAP50-95 | time per photo |
|---|---|---|---|
| YOLO26n @320, float, PC (test split) | 0.982 | 0.706 | |
| **same model, int8, on the ESP32-S3** (test split) | **0.947** | **0.643** | **~1.0 s** |

Measured on the held-out test split (97 photos from a separate survey day), on the device through
its HTTP API. Details: [`results/esp32/README.md`](results/esp32/README.md).

## What is where

| folder | contents |
|---|---|
| [`notebooks/`](notebooks) | the pipeline step by step: [`01_train_yolo26`](notebooks/01_train_yolo26.ipynb), [`02_quantise_for_esp32`](notebooks/02_quantise_for_esp32.ipynb), [`03_test_on_esp32`](notebooks/03_test_on_esp32.ipynb) |
| [`rabbitfish/`](rabbitfish) | shared Python code: letterbox preprocessing and box decoding (mirrored in the firmware), mAP, ONNX export, esp-ppq quantisation |
| [`scripts/`](scripts) | the same steps as command-line tools: `train_yolo26.py`, `esp32_export.py`, `probe_scores.py` |
| [`firmware/rabbitfish_server/`](firmware/rabbitfish_server) | ESP-IDF project: HTTP inference server with a web page, the deployed model included |
| [`models/`](models) | `yolo26n-i320/best.pt` (trained, 5 MB) and `yolo26n-i320-esp32s3/` (`model.espdl`, export report, device results) |
| [`results/`](results) | model comparison, training curves, the quantisation study, YOLO11 baselines |
| [`data/`](data) | dataset format and 12 labelled sample images (the full dataset is not included) |
| [`hpc/`](hpc) | optional SLURM job templates for training and export on a cluster |

## Quick start

**Run the detector on the board** (no Python needed). You need an ESP32-S3 with 8 MB PSRAM
(e.g. ESP32-S3-DevKitC-1 N16R8) and [ESP-IDF 5.5](https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32s3/get-started/):

```bash
cd firmware/rabbitfish_server
cp main/wifi_secrets.h.example main/wifi_secrets.h   # your 2.4 GHz Wi-Fi name and password
idf.py set-target esp32s3 && idf.py -p <port> flash monitor
```
Then open `http://rabbitfish.local/` and drop in photos. See the [firmware README](firmware/rabbitfish_server/README.md).

**Reproduce the model.** Python 3.10, a CUDA GPU, and the dataset in `dataset/` ([`data/README.md`](data/README.md)):

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-esp32.txt      # includes requirements.txt
jupyter lab notebooks/                      # 01 -> 02 -> 03
```
or without notebooks:
```bash
python scripts/train_yolo26.py --size n --imgsz 320                       # ~30 min on an H100
python scripts/esp32_export.py --run yolo26n-b8-i320 --cls-clamp 10 --box16 --blockwise --device cuda   # ~35 min
python firmware/rabbitfish_server/tools/select_model.py runs/yolo26n-b8-i320/esp32/320x256-w8a8-o2m-c10-box16-br
```

Every notebook has a quick mode (`RABBITFISH_QUICK=1`) that runs end to end in minutes with tiny
settings; that is how they are tested.

## How it works

**Model.** YOLO26 nano (2.5 M parameters), COCO-pretrained, fine-tuned at 320 px. On this data
the small and medium models and 640 px input are not more accurate
([`results/model_comparison.md`](results/model_comparison.md)), and nano at 320x256 is what fits
the microcontroller's time budget.

**Export.** The network is exported to ONNX without its box decoding: per stride (8, 16, 32) it
outputs raw box distances and class logits. Boxes and scores then get their own int8 scales, and
the device needs no TopK/Gather operators; the decoding (a few lines, `rabbitfish/metrics.py:decode`)
is ported to C++. Class logits are clamped to ±10 inside the model and the three final box
convolutions run in int16.

**Quantisation.** esp-ppq w8a8 with **blockwise reconstruction**. Plain int8 kept mAP50 at 0.92
but squashed the class scores so much that at the usual 0.25 threshold the device missed about
half the fish; mAP did not show it because it only depends on how predictions are ranked.
Blockwise reconstruction restores the scores (recall at 0.25 equal to float). The full story,
including what did not work: [`results/esp32/quantisation_study.md`](results/esp32/quantisation_study.md).

**Device.** The firmware decodes the uploaded JPEG, letterboxes it into the int8 input exactly
like the Python code, runs ESP-DL on both cores (~950 ms), decodes boxes and applies NMS. At boot
it checks itself: the `.espdl` carries one image and the outputs esp-ppq computed for it, and
the device must reproduce them bit for bit. Photos are only held in RAM while they are processed.

## Results at a glance

- [`results/model_comparison.md`](results/model_comparison.md): five YOLO26 models (n/s/m, 256/320/640 px) on val and test
- [`results/training/`](results/training): loss and mAP curves, PR curves and confusion matrices per run
- [`results/esp32/`](results/esp32): accuracy, speed and memory on the device, and the quantisation study
- [`results/baselines_yolo11/`](results/baselines_yolo11): YOLO11 n/s/m baselines trained earlier with the same recipe

## Acknowledgements

Built on [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics),
Espressif's [ESP-DL](https://github.com/espressif/esp-dl) and [esp-ppq](https://github.com/espressif/esp-ppq), and
[ESP-IDF](https://github.com/espressif/esp-idf).
