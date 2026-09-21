# floor numbers: predict nothing, predict the most common label, or sample labels at train frequency
# uv run python -m llm_2026_group6.baselines

import json

import numpy as np

from llm_2026_group6.metrics import LABELS, evaluate
from llm_2026_group6.prompt import ROOT, load

rng = np.random.default_rng(42)
train = np.array([r["labels"] for r in load("train")])
freq = train.mean(0)
majority = [int(i == freq.argmax()) for i in range(8)]

for split in ["dev", "test"]:
    gold = np.array([r["labels"] for r in load(split)])
    n = len(gold)
    for name, pred in [("zero", np.zeros_like(gold)),
                       ("majority", np.tile(majority, (n, 1))),
                       ("random", (rng.random((n, 8)) < freq).astype(int))]:
        print(f"== baseline_{name}_{split} ==")
        m = evaluate(pred, gold, verbose=True)
        out = ROOT / "runs" / f"baseline_{name}_{split}"
        out.mkdir(parents=True, exist_ok=True)
        json.dump(m, open(out / "metrics.json", "w"), indent=2)
