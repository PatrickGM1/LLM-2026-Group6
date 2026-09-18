#!/bin/bash
#SBATCH --job-name=xed-sweep
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -e
cd "$(dirname "$0")"
module purge
module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0 2>/dev/null || module load Python CUDA
export HF_HOME=/scratch/$USER/hf_cache UV_CACHE_DIR=/scratch/$USER/uv_cache UV_PROJECT_ENVIRONMENT=/scratch/$USER/venv-llm
export PATH="$HOME/.local/bin:$PATH"

A=runs/lora_Qwen2.5-1.5B-Instruct_both_s42/best
for t in 0.01 0.02 0.05 0.1 0.2 0.3; do
  uv run python -m llm_2026_group6.prompt --adapter $A --lang ro --batch 64 --threshold $t
done
uv run python -m llm_2026_group6.summary
