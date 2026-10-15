# Sourced by the job files: activates the conda environment and moves to the repository.
# Set these once in your shell profile (or edit the defaults):
#   RABBITFISH_REPO     path of this repository on the cluster
#   RABBITFISH_CONDA    path of conda.sh, e.g. ~/miniforge3/etc/profile.d/conda.sh
#   RABBITFISH_DATASET  dataset location, if not <repo>/dataset
# Environments: "yolo" (requirements.txt) for training, "espdl" (requirements-esp32.txt) for export.
REPO="${RABBITFISH_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONDA_SH="${RABBITFISH_CONDA:-$HOME/miniforge3/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${1:-yolo}"
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
