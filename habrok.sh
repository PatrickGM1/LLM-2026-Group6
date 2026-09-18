#!/bin/bash
#SBATCH --job-name=xed-lora
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -e
cd "$(dirname "$0")"
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0 2>/dev/null || module load Python CUDA

export HF_HOME=/scratch/$USER/hf_cache
export UV_CACHE_DIR=/scratch/$USER/uv_cache
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venv-llm
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv sync
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"

for c in en ro both; do
  uv run python -m llm_2026_group6.train_lora --train_on $c --dev_limit 500
  uv run python -m llm_2026_group6.prompt --adapter runs/lora_Qwen2.5-1.5B-Instruct_${c}_s42/best --lang en --batch 64
  uv run python -m llm_2026_group6.prompt --adapter runs/lora_Qwen2.5-1.5B-Instruct_${c}_s42/best --lang ro --batch 64
done
