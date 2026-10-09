# Cross-lingual multi-label emotion classification

XED emotion dataset, English + Romanian. Group 6, Introduction to LLMs 2026.

We compare two ways of doing the task on a small instruct model: plain prompting
versus LoRA fine-tuning.

Start with `docs/notes.md` for what we did, why, the results, and what's still open.
The assignment itself is in `docs/project_brief.md`.


## Layout

```
data/raw/        xed files as downloaded: en-annotated.tsv, ro-projections.tsv, pairs-ro.txt
data/splits/     train/dev/test.jsonl, produced by split.py, committed so the split is fixed

docs/            project brief, notes
docs/figures/    plots for the report, produced by plots.py

src/llm_2026_group6/
  split.py       70/20/10 split by alignment id, multi-label stratified, seed 42
  metrics.py     label list, output parser, all metrics from the brief
  prompt.py      zero/five-shot prompting, adapter evaluation, threshold decoding
  train_lora.py  LoRA/QLoRA fine-tuning, checkpoint selection on dev
  baselines.py   zero / majority / random floors
  summary.py     one table of every run in runs/, mean +- std over seeds
  errors.py      side by side disagreements of two runs, for the error analysis
  plots.py       label distribution, per-label F1, threshold sweep figures

run_all.sh       every experiment in order (also a slurm script for Habrok)
runs/            outputs, one folder per run (gitignored, full set lives on Habrok)
```


## Setup

```
uv sync
```

Needs a CUDA GPU. torch is pinned to the cu126 wheels in `pyproject.toml`.

Depending on your card:

- 4GB: add `--4bit` to the prompt/train commands
- 8GB: fine as is
- A100: you can bump it up with `--batch 64`


## Commands

Build the splits once, then run whatever you need:

```
uv run python -m llm_2026_group6.split
uv run python -m llm_2026_group6.baselines
uv run python -m llm_2026_group6.prompt --model Qwen/Qwen2.5-1.5B-Instruct --shots 5 --lang ro
uv run python -m llm_2026_group6.train_lora --train_on both
uv run python -m llm_2026_group6.prompt --adapter runs/lora_Qwen2.5-1.5B-Instruct_both_s42/best --lang ro --threshold 0.15
uv run python -m llm_2026_group6.summary
uv run python -m llm_2026_group6.errors <run_a> <run_b>
```

Some flags:

- `--split test` to evaluate on the test set
- `--limit N` to try things on a few rows
- `--prompt short|long` for the wording variants

`bash run_all.sh` reproduces everything (~2h on an A100).

Each run writes `runs/<name>/preds.jsonl` + `metrics.json` (or `train_config.json` + `best/`
adapter). Run names encode the condition, so for example:

```
lora_Qwen2.5-1.5B-Instruct_both_s42_ro_test_t0.15
```

is the bilingual adapter, seed 42, Romanian test set, threshold decoding at 0.15.


## Final configuration

- Primary model: Qwen2.5-1.5B-Instruct
- Second model: EuroLLM-1.7B-Instruct (prompting only)
- LoRA: r=16, alpha=32, dropout 0.05 on all attention and MLP projections
- Training: lr 2e-4 cosine, batch 16, 3 epochs, max length 320, fp16, seeds 42/1/2
- Checkpoint: best dev micro-F1
- Decoding: greedy, or label threshold 0.15 (tuned on dev)
- Hardware: A100 40GB on Habrok
