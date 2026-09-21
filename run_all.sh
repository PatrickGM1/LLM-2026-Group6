#!/bin/bash
# reproduces every number in the report, in order. ~2h on an A100.
# on habrok: sbatch run_all.sh, or just bash run_all.sh inside a gpu session
#SBATCH --job-name=xed
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -e
cd "$(dirname "$0")"
mkdir -p logs
if [ -n "$SLURM_JOB_ID" ] || [ -d /scratch/$USER ]; then
  module purge
  module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0 2>/dev/null || module load Python CUDA
  export HF_HOME=/scratch/$USER/hf_cache UV_CACHE_DIR=/scratch/$USER/uv_cache UV_PROJECT_ENVIRONMENT=/scratch/$USER/venv-llm
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv sync

P="uv run python -m llm_2026_group6"
M1=Qwen/Qwen2.5-1.5B-Instruct
M2=utter-project/EuroLLM-1.7B-Instruct
A=runs/lora_Qwen2.5-1.5B-Instruct
T=0.15

$P.split

# prompting, both models, dev
for m in $M1 $M2; do for s in 0 5; do for l in en ro; do
  $P.prompt --model $m --shots $s --lang $l --batch 64
done; done; done

# lora, 3 conditions x 3 seeds, checkpoint picked on dev
for s in 42 1 2; do for c in en ro both; do
  $P.train_lora --train_on $c --seed $s --dev_limit 500
done; done

# threshold sweep on dev (bilingual adapter, romanian) -> T above
for t in 0.01 0.02 0.05 0.1 0.15 0.2 0.3; do
  $P.prompt --adapter ${A}_both_s42/best --lang ro --batch 64 --threshold $t
done

# all adapters on dev, greedy and threshold
for c in en ro both; do for l in en ro; do
  $P.prompt --adapter ${A}_${c}_s42/best --lang $l --batch 64
  $P.prompt --adapter ${A}_${c}_s42/best --lang $l --batch 64 --threshold $T
done; done

# test, once
for m in $M1 $M2; do for s in 0 5; do for l in en ro; do
  $P.prompt --model $m --shots $s --lang $l --batch 64 --split test
done; done; done
for s in 42 1 2; do for c in en ro both; do for l in en ro; do
  $P.prompt --adapter ${A}_${c}_s${s}/best --lang $l --batch 64 --split test
  $P.prompt --adapter ${A}_${c}_s${s}/best --lang $l --batch 64 --split test --threshold $T
done; done; done

$P.summary
