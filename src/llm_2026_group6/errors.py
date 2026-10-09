# compare two runs on the same split, print where they disagree with the gold labels
# uv run python -m llm_2026_group6.errors lora_Qwen2.5-1.5B-Instruct_both_s42_ro_test_t0.15 prompt_Qwen2.5-1.5B-Instruct_5shot_ro_test

import json
import sys

from llm_2026_group6.metrics import LABELS
from llm_2026_group6.prompt import ROOT

names = lambda v: ",".join(l for l, x in zip(LABELS, v) if x) or "-"


def read(run):
    return {r["id"]: r for r in map(json.loads, open(ROOT / "runs" / run / "preds.jsonl", encoding="utf-8"))}


a, b = read(sys.argv[1]), read(sys.argv[2])
k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
a_only, b_only, both_wrong = [], [], []
for i, ra in a.items():
    rb = b[i]
    ok_a, ok_b = ra["pred"] == ra["gold"], rb["pred"] == rb["gold"]
    if ok_a and not ok_b:
        a_only.append((ra, rb))
    elif ok_b and not ok_a:
        b_only.append((ra, rb))
    elif not ok_a and not ok_b:
        both_wrong.append((ra, rb))

print(f"{len(a)} rows. only A exact: {len(a_only)}, only B exact: {len(b_only)}, both wrong: {len(both_wrong)}\n")
for title, rows in [(f"A right, B wrong ({sys.argv[1]})", a_only), (f"B right, A wrong ({sys.argv[2]})", b_only), ("both wrong", both_wrong)]:
    print(f"--- {title} ---")
    for ra, rb in rows[:k]:
        print(f"{ra['text']}\n   gold={names(ra['gold'])}  A={names(ra['pred'])}  B={names(rb['pred'])}  raw_B={rb['output']!r}")
    print()
