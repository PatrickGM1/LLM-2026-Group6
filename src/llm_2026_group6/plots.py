# makes the three report figures from the splits and the runs/ metrics
# uv run python -m llm_2026_group6.plots
# writes png files into docs/figures/

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import LABELS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"


def load(run):
    return json.load(open(ROOT / "runs" / run / "metrics.json"))


def label_distribution():
    # count how often each label appears in each split
    counts = {}
    for split in ["train", "dev", "test"]:
        c = np.zeros(len(LABELS))
        for line in open(ROOT / "data" / "splits" / f"{split}.jsonl"):
            c += np.array(json.loads(line)["labels"])
        counts[split] = c

    x = np.arange(len(LABELS))
    w = 0.27
    plt.figure(figsize=(9, 4))
    for i, split in enumerate(["train", "dev", "test"]):
        plt.bar(x + (i - 1) * w, counts[split], w, label=split)
    plt.xticks(x, LABELS, rotation=30, ha="right")
    plt.ylabel("lines with label")
    plt.title("XED Romanian label distribution per split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "label_distribution.png", dpi=150)
    plt.close()


def threshold_sweep():
    # bilingual adapter on Romanian dev, greedy vs the threshold sweep
    ts = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    base = "lora_Qwen2.5-1.5B-Instruct_both_s42_ro_dev"
    metrics = ["micro_f1", "samples_jaccard", "micro_precision", "micro_recall"]
    names = {"micro_f1": "micro-F1", "samples_jaccard": "Jaccard",
             "micro_precision": "precision", "micro_recall": "recall"}

    plt.figure(figsize=(7, 4.5))
    for key in metrics:
        y = [load(f"{base}_t{t}")[key] for t in ts]
        plt.plot(ts, y, marker="o", label=names[key])
    greedy = load(base)
    for key in metrics:
        plt.axhline(greedy[key], ls=":", lw=0.8, color="grey")
    plt.axvline(0.15, ls="--", color="red", lw=0.8, label="chosen t=0.15")
    plt.xlabel("threshold t")
    plt.ylabel("score (dotted = greedy)")
    plt.title("Threshold decoding sweep, bilingual LoRA, Romanian dev")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "threshold_sweep.png", dpi=150)
    plt.close()


def per_label_f1():
    # per-label F1 on Romanian test for the main conditions (seed 42)
    conditions = {
        "Qwen 0-shot": "prompt_Qwen2.5-1.5B-Instruct_0shot_ro_test",
        "Qwen 5-shot": "prompt_Qwen2.5-1.5B-Instruct_5shot_ro_test",
        "LoRA EN": "lora_Qwen2.5-1.5B-Instruct_en_s42_ro_test_t0.15",
        "LoRA RO": "lora_Qwen2.5-1.5B-Instruct_ro_s42_ro_test_t0.15",
        "LoRA both": "lora_Qwen2.5-1.5B-Instruct_both_s42_ro_test_t0.15",
    }
    grid = np.array([[load(run)["per_label"][l]["f1-score"] for l in LABELS]
                     for run in conditions.values()])

    plt.figure(figsize=(8, 4))
    plt.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(label="F1")
    plt.xticks(np.arange(len(LABELS)), LABELS, rotation=30, ha="right")
    plt.yticks(np.arange(len(conditions)), list(conditions))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            plt.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                     color="white" if grid[i, j] < 0.5 else "black", fontsize=8)
    plt.title("Per-label F1, Romanian test")
    plt.tight_layout()
    plt.savefig(OUT / "per_label_f1.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    label_distribution()
    threshold_sweep()
    per_label_f1()
    print(f"wrote 3 figures to {OUT}")
