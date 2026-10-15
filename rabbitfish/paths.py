"""Project locations. Everything is relative to the repository root, so the code runs from any checkout."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The full dataset is not in the repository (see data/README.md). Put it in <repo>/dataset or point
# RABBITFISH_DATASET at it. data/samples holds a few labelled test images for quick checks.
DATASET = Path(os.environ.get("RABBITFISH_DATASET", ROOT / "dataset"))
SAMPLES = ROOT / "data" / "samples"
RUNS = ROOT / "runs"      # training runs and ESP32 exports (not committed)
MODELS = ROOT / "models"  # the trained and quantised models that are committed
FIRMWARE = ROOT / "firmware" / "rabbitfish_server"
WEIGHTS = ROOT / "weights"  # pretrained COCO checkpoints (yolo26n.pt, ...), downloaded on first use

# Keep ultralytics' settings file inside the project instead of the user's home directory.
if "YOLO_CONFIG_DIR" not in os.environ:
    (ROOT / ".ultralytics").mkdir(exist_ok=True)  # ultralytics falls back to /tmp when it does not exist
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
