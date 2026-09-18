# prints one table with every run in runs/
# uv run python -m llm_2026_group6.summary

import glob
import json
from pathlib import Path

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
