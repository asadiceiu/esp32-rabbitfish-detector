# Running on a SLURM cluster (optional)

Everything also runs from the notebooks on a single GPU machine. These job files are for
unattended runs on a cluster; they take the account and partition on the `sbatch` command line,
so nothing cluster-specific is stored here.

1. Clone the repository on the cluster and copy the dataset to `dataset/` (or set `RABBITFISH_DATASET`).
2. Create two conda environments (Python 3.10), installing PyTorch for your CUDA version first:
   ```bash
   conda create -n yolo python=3.10 -y && conda activate yolo
   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   conda create -n espdl python=3.10 -y && conda activate espdl
   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements-esp32.txt
   ```
3. If compute nodes have no internet, download the pretrained weights once on the login node:
   `python -c "from ultralytics import YOLO; YOLO('weights/yolo26n.pt')"` (from the repository root).
4. Point `env.sh` at your conda install (`export RABBITFISH_CONDA=...`) and submit from the repository root:

| job | what | typical time |
|---|---|---|
| `train.job` | `scripts/train_yolo26.py`, one array task per model | ~30 min (n @320, H100) |
| `export_esp32.job` | `scripts/esp32_export.py` with blockwise reconstruction | ~35 min (H100) |
| `probe_scores.job` | `scripts/probe_scores.py`, per-image score check | ~2 min (CPU) |

Ask for close to these times: short jobs start sooner on a busy cluster.
