#!/bin/bash
# everything on dev with the final decoding, then second model prompting
set -e
cd "$(dirname "$0")"
export HF_HOME=/scratch/$USER/hf_cache UV_CACHE_DIR=/scratch/$USER/uv_cache UV_PROJECT_ENVIRONMENT=/scratch/$USER/venv-llm
export PATH="$HOME/.local/bin:$PATH"

T=${T:-0.15}
M2=utter-project/EuroLLM-1.7B-Instruct

for c in en ro both; do
  for l in en ro; do
    uv run python -m llm_2026_group6.prompt --adapter runs/lora_Qwen2.5-1.5B-Instruct_${c}_s42/best --lang $l --batch 64 --threshold $T
  done
done

for s in 0 5; do
  for l in en ro; do
    uv run python -m llm_2026_group6.prompt --model $M2 --shots $s --lang $l --batch 64
  done
done

uv run python -m llm_2026_group6.summary
