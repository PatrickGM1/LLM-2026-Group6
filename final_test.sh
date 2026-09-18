#!/bin/bash
# extra seeds, then the one and only test evaluation of every condition
set -e
cd "$(dirname "$0")"
export HF_HOME=/scratch/$USER/hf_cache UV_CACHE_DIR=/scratch/$USER/uv_cache UV_PROJECT_ENVIRONMENT=/scratch/$USER/venv-llm
export PATH="$HOME/.local/bin:$PATH"

T=0.15
M1=Qwen/Qwen2.5-1.5B-Instruct
M2=utter-project/EuroLLM-1.7B-Instruct

for s in 1 2; do
  for c in en ro both; do
    uv run python -m llm_2026_group6.train_lora --train_on $c --dev_limit 500 --seed $s
  done
done

for s in 42 1 2; do
  for c in en ro both; do
    for l in en ro; do
      uv run python -m llm_2026_group6.prompt --adapter runs/lora_Qwen2.5-1.5B-Instruct_${c}_s${s}/best --lang $l --batch 64 --threshold $T --split test
      uv run python -m llm_2026_group6.prompt --adapter runs/lora_Qwen2.5-1.5B-Instruct_${c}_s${s}/best --lang $l --batch 64 --split test
    done
  done
done

for m in $M1 $M2; do
  for s in 0 5; do
    for l in en ro; do
      uv run python -m llm_2026_group6.prompt --model $m --shots $s --lang $l --batch 64 --split test
    done
  done
done

uv run python -m llm_2026_group6.summary
