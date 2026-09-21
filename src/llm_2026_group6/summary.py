# prints one table with every run in runs/
# uv run python -m llm_2026_group6.summary

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

rows = []
for f in sorted(glob.glob(str(ROOT / "runs/*/metrics.json"))):
    m = json.load(open(f))
    rows.append((Path(f).parent.name, m["micro_f1"], m["samples_jaccard"], m["macro_f1"], m["micro_precision"],
                 m["micro_recall"], m["hamming_loss"], m.get("malformed_rate", 0), m["config"].get("seconds", 0)))
for f in sorted(glob.glob(str(ROOT / "runs/*/results.json"))):
    r = json.load(open(f))
    for lang, m in r["test"].items():
        rows.append((f"{Path(f).parent.name}_test_{lang}", m["micro_f1"], m["samples_jaccard"], m["macro_f1"],
                     m["micro_precision"], m["micro_recall"], m["hamming_loss"], 0, 0))

print(f"{'run':55s} {'miF1':>6s} {'jacc':>6s} {'maF1':>6s} {'prec':>6s} {'rec':>6s} {'hamm':>6s} {'malf':>6s} {'sec':>5s}")
for r in rows:
    print(f"{r[0]:55s} " + " ".join(f"{x:6.3f}" for x in r[1:8]) + f" {r[8]:5.0f}")

for f in sorted(glob.glob(str(ROOT / "runs/*/train_config.json"))):
    c = json.load(open(f))
    print(f"\n{Path(f).parent.name}: best={c['best']} train {c['train_seconds']}s on {c['hardware']}, dev scores {c['dev_scores']}")

# mean +- std over seeds for the lora runs
groups = defaultdict(list)
for r in rows:
    m = re.match(r"(lora_.*)_s\d+(_.*)", r[0])
    if m:
        groups[m.group(1) + m.group(2)].append(r[1:7])
if groups:
    print(f"\n{'condition (mean +- std over seeds)':50s} {'n':>2s} {'miF1':>13s} {'jacc':>13s} {'maF1':>13s} {'hamm':>13s}")
    for k in sorted(groups):
        a = np.array(groups[k])
        mu, sd = a.mean(0), a.std(0)
        print(f"{k:50s} {len(a):2d} " + " ".join(f"{mu[i]:.3f}+-{sd[i]:.3f}" for i in [0, 1, 2, 5]))
