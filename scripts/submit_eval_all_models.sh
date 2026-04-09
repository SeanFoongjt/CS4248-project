#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"

ORIG_DIR="runs/eval_all_models/original_test"
SHUF_DIR="runs/eval_all_models/shuffled_test"

mkdir -p "${REPO_ROOT}/${ORIG_DIR}/slurm"
mkdir -p "${REPO_ROOT}/${SHUF_DIR}/slurm"

echo "Submitting original-test evaluation job..."
ORIG_JOB_ID="$(
  sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=eval-all-original
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-47
#SBATCH --output=${ORIG_DIR}/slurm/%x_%j.out
#SBATCH --error=${ORIG_DIR}/slurm/%x_%j.err

set -euo pipefail
cd "${REPO_ROOT}"
source .venv/bin/activate
export PYTHONPATH="${REPO_ROOT}:\${PYTHONPATH:-}"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.cuda.get_device_name(0))"
python scripts/evaluate_original_test_set.py --output ${ORIG_DIR}/${TIMESTAMP}
EOF
)"

echo "Submitting shuffled-test evaluation job..."
SHUF_JOB_ID="$(
  sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=eval-all-shuffled
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-47
#SBATCH --output=${SHUF_DIR}/slurm/%x_%j.out
#SBATCH --error=${SHUF_DIR}/slurm/%x_%j.err

set -euo pipefail
cd "${REPO_ROOT}"
source .venv/bin/activate
export PYTHONPATH="${REPO_ROOT}:\${PYTHONPATH:-}"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.cuda.get_device_name(0))"
python scripts/experiment_shuffle_description.py --output ${SHUF_DIR}/${TIMESTAMP}
EOF
)"

echo "Submitted original-test job: ${ORIG_JOB_ID}"
echo "Submitted shuffled-test job: ${SHUF_JOB_ID}"
