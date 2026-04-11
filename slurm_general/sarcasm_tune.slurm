#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00

# --- Optional: activate your environment / modules ---
VENV="${HOME}/cs4248-venv"
source "${VENV}/bin/activate"

set -euo pipefail

# Paths (adjust as needed)
SCRIPT_DIR="${HOME}/cs4248"
PY_DRIVER="${SCRIPT_DIR}/scripts/conceptnet_hyperparam_tune.py"

# Variables
INPUT="${SCRIPT_DIR}/data/Sarcasm_Headlines_Dataset_All.json"
# We inject RUN_NAME from the wrapper so each combination gets its own folder
OUTPUT_DIR="${SCRIPT_DIR}/outputs/${RUN_NAME}_${SLURM_JOB_ID}"

TRIALS=30
SEED=42

# Move to the project root
cd "${SCRIPT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Explicitly add the project root to Python's search path
export PYTHONPATH="${SCRIPT_DIR}"

# Run
# Note: ${CNET_FLAG} is unquoted. If it is empty, Bash will ignore it. 
# If it contains "--no-conceptnet", it will be passed as an argument.
srun python "${PY_DRIVER}" \
  --model-type "${MODEL_TYPE}" \
  --input "${INPUT}" \
  --output "${OUTPUT_DIR}" \
  --n-trials "${TRIALS}" \
  --text-format "${FORMAT}" \
  ${CNET_FLAG} \
  ${POS_FLAG}
