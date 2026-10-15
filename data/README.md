# Data

## The full dataset (not in this repository)

Underwater survey photos of rabbitfish (Olympus TG-6, 1920x1440, 4:3) with two box classes:

| id | class | what is boxed |
|---|---|---|
| 0 | `fishbody` | the whole fish |
| 1 | `fishhead` | the head (for re-identification work downstream) |

| split | images | used for |
|---|---|---|
| `train` | 846 | training and int8 calibration |
| `val` | 264 | choosing the best epoch and the quantisation settings |
| `test` | 97 | final scores only; a separate survey day, never tuned on |

To run the notebooks on the full data, put it in `dataset/` at the repository root (ignored by
git) or anywhere else and set `RABBITFISH_DATASET`:

```
dataset/
  data.yaml              # copy of data/data.yaml
  train/images/*.jpg   train/labels/*.txt
  val/images/*.jpg     val/labels/*.txt
  test/images/*.jpg    test/labels/*.txt
```

Labels are YOLO format: one line per object, `class cx cy w h`, normalised to 0-1.

## Samples (`samples/`)

12 labelled images from the **test** split (22 bodies, 20 heads), so no model here has seen them.
They are enough to try the trained model, the quantised model and the device without the
dataset (notebook 02, step 5; notebook 03; `evaluate_device.py --root data --split samples`).
Their EXIF metadata has been removed; the pixels are unchanged.
