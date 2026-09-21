# Cross-lingual multi-label emotion classification (XED, English + Romanian)

Group 6, Introduction to LLMs 2026. Compares prompting and LoRA fine-tuning of a small
instruct model on the XED emotion dataset, English and Romanian.

## Layout

```
data/raw/        xed files: en-annotated.tsv, ro-projections.tsv, pairs-ro.txt
data/splits/     train/dev/test.jsonl, made by split.py (gitignored, deterministic)
docs/            project brief
src/llm_2026_group6/
  split.py       70/20/10 split by alignment id, multi-label stratified, seed 42
  metrics.py     label list, output parser, metrics from the brief
  prompt.py      zero/five-shot prompting, adapter eval, threshold decoding
  train_lora.py  LoRA/QLoRA fine-tuning, checkpoint selection on dev
  summary.py     table of every run in runs/
  baselines.py   zero / majority / random floors
  errors.py      side by side disagreements of two runs, for the error analysis
run_all.sh       every experiment in order
runs/            outputs, one folder per run (gitignored)
```

## Setup

```
uv sync
```
Needs a GPU. torch is pinned to the cu126 wheels in pyproject.toml. On a 4GB card add `--4bit`.

## Run

```
uv run python -m llm_2026_group6.split
uv run python -m llm_2026_group6.prompt --model Qwen/Qwen2.5-1.5B-Instruct --shots 5 --lang ro
uv run python -m llm_2026_group6.train_lora --train_on both
uv run python -m llm_2026_group6.prompt --adapter runs/lora_Qwen2.5-1.5B-Instruct_both_s42/best --lang ro --threshold 0.15
uv run python -m llm_2026_group6.summary
```
`bash run_all.sh` does the whole thing (all conditions, 3 seeds, dev + test). Each run writes
`preds.jsonl` and `metrics.json` (or `train_config.json`) to `runs/<name>/`.

## Setup we used

Primary model Qwen2.5-1.5B-Instruct, second model EuroLLM-1.7B-Instruct (prompting only).
LoRA r=16, alpha=32, dropout 0.05 on all attention and MLP projections, lr 2e-4 cosine, batch 16,
3 epochs, max length 320, fp16, seeds 42/1/2. Checkpoint = best dev micro-F1.
Decoding: greedy, or threshold decoding (label included if its probability > 0.15, tuned on dev).
Hardware: A100 40GB on Habrok, also runs on an RTX 2070 (8GB).
