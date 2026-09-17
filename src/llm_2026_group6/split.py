import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

SEED = 42
DATA = Path(__file__).resolve().parents[2] / "data"
RAW = DATA / "raw"
LABELS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]


def norm(s):
    return re.sub(r"[^\w]+", "", s.lower())


def read_tsv(path, ncols):
    with open(path, encoding="utf-8", newline="") as f:
        rows = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        # ro-projections has some broken lines in it, skip them
        return [[c.strip() for c in r] for r in rows if len(r) == ncols]


def to_vec(s):
    ids = [int(x) for x in s.split(",")]
    return [1 if i + 1 in ids else 0 for i in range(8)]


ro_labels = {norm(t): lab for t, lab in read_tsv(RAW / "ro-projections.tsv", 2)}

rows = []
seen = set()
for en_file, ro_file, en_idx, ro_idx, en, ro in read_tsv(RAW / "pairs-ro.txt", 6):
    k = norm(ro)
    # about half the pairs are danish->romanian, we only want the english ones
    if not en_file.startswith("en/") or k not in ro_labels or k in seen:
        continue
    seen.add(k)
    rows.append(
        {
            "id": f"{en_file}:{en_idx}|{ro_file}:{ro_idx}",
            "en": en,
            "ro": ro,
            "labels": to_vec(ro_labels[k]),
        }
    )

# group by english text so stuff like "Yes." doesn't end up in both train and test
groups = defaultdict(list)
for r in rows:
    groups[norm(r["en"])].append(r)
keys = list(groups)
y = np.array([np.max([r["labels"] for r in groups[k]], axis=0) for k in keys])
X = np.zeros((len(keys), 1))


def strat_split(idx, frac):
    s = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=SEED)
    a, b = next(s.split(X[idx], y[idx]))
    return idx[a], idx[b]


rest, test = strat_split(np.arange(len(keys)), 0.1)
train, dev = strat_split(rest, 2 / 9)  # 2/9 of 90% = 20%

out = DATA / "splits"
out.mkdir(exist_ok=True)
for name, idx in [("train", train), ("dev", dev), ("test", test)]:
    part = [r for i in idx for r in groups[keys[i]]]
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = np.sum([r["labels"] for r in part], axis=0)
    print(name, len(part), dict(zip(LABELS, counts.tolist())))

print("total", len(rows))
