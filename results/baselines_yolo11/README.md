# YOLO11 baselines

YOLO11 n/s/m detectors trained earlier on the same data with the same recipe (batch 4, 8 and 16),
scored with `ultralytics val` at 640 px. They are the reference the YOLO26 models in
[`../model_comparison.md`](../model_comparison.md) were trained to match; YOLO11 checkpoints are
not included.

- `val_comparison.md`, `test_comparison.md`: ranked tables; `*_metrics.csv`: the same with per-class columns and timings
  (`train_log` paths refer to the original training folders, which are not part of this repository)
- `figures/`: val vs test, per-class mAP, accuracy vs speed
