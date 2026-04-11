#!/bin/bash
#SBATCH --job-name=random-section-repl
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-47
#SBATCH --output=runs/random_section_replacement/slurm/%x_%j.out
#SBATCH --error=runs/random_section_replacement/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.cuda.get_device_name(0))"
python scripts/experiment_random_section_replacement.py --output "runs/random_section_replacement/2026-04-11_22-07-01"
