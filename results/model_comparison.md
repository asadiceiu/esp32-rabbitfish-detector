# YOLO26 models: float accuracy (PyTorch, `ultralytics val`)

All trained with the same recipe (`scripts/train_yolo26.py`, batch 8, AdamW, cosine LR, up to 500 epochs with patience 100); only size and training/eval image size differ. `val` picked the best checkpoint; `test` (97 images from a separate survey day) was never used for any decision. Evaluated with `ultralytics val` at the training image size (rect, batch 1).

| model | params (M) | split | mAP50 | mAP50-95 | P | R | fishbody mAP50-95 | fishhead mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| YOLO26-n @640 | 2.52 | val | 0.960 | 0.685 | 0.875 | 0.922 | 0.810 | 0.560 |
| YOLO26-n @640 | 2.52 | test | 0.978 | 0.708 | 0.965 | 0.946 | 0.810 | 0.605 |
| YOLO26-s @640 | 9.93 | val | 0.952 | 0.684 | 0.852 | 0.946 | 0.802 | 0.566 |
| YOLO26-s @640 | 9.93 | test | 0.974 | 0.724 | 0.964 | 0.928 | 0.806 | 0.641 |
| YOLO26-m @640 | 21.75 | val | 0.957 | 0.694 | 0.867 | 0.947 | 0.814 | 0.575 |
| YOLO26-m @640 | 21.75 | test | 0.968 | 0.722 | 0.973 | 0.923 | 0.821 | 0.623 |
| YOLO26-n @320 (deployed) | 2.52 | val | 0.962 | 0.687 | 0.895 | 0.917 | 0.813 | 0.561 |
| YOLO26-n @320 (deployed) | 2.52 | test | 0.982 | 0.706 | 0.957 | 0.947 | 0.827 | 0.585 |
| YOLO26-n @256 | 2.52 | val | 0.959 | 0.684 | 0.870 | 0.932 | 0.804 | 0.564 |
| YOLO26-n @256 | 2.52 | test | 0.982 | 0.699 | 0.957 | 0.940 | 0.816 | 0.582 |

The nano model at 320 px is as accurate as the larger and higher-resolution models on this data, so it is the one deployed to the ESP32-S3. On-device results: [`esp32/README.md`](esp32/README.md). YOLO11 baselines trained earlier with the same recipe: [`baselines_yolo11/`](baselines_yolo11/).
