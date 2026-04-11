#!/bin/bash
set -euo pipefail

RUN_ROOT="runs/random_section_replacement"
SLURM_DIR="${RUN_ROOT}/slurm"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_DIR="${RUN_ROOT}/${TIMESTAMP}"
SBATCH_PATH="${SLURM_DIR}/random_section_replacement_${TIMESTAMP}.sbatch"

mkdir -p "${SLURM_DIR}" "${OUTPUT_DIR}"

cat > "${SBATCH_PATH}" <<EOF
#!/bin/bash
#SBATCH --job-name=random-section-repl
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-47
#SBATCH --output=${SLURM_DIR}/%x_%j.out
#SBATCH --error=${SLURM_DIR}/%x_%j.err

set -euo pipefail
cd "\${SLURM_SUBMIT_DIR:-\$PWD}"
source .venv/bin/activate
export PYTHONPATH="\$PWD:\${PYTHONPATH:-}"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.cuda.get_device_name(0))"
python scripts/experiment_random_section_replacement.py --output "${OUTPUT_DIR}"
EOF

echo "Submitting random section replacement experiment:"
echo "  sbatch: ${SBATCH_PATH}"
echo "  output: ${OUTPUT_DIR}"
sbatch "${SBATCH_PATH}"
