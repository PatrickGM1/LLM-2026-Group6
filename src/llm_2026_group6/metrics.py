# shared stuff: label list, output parser, metrics from section 7 of the brief

import json
import re

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, f1_score, hamming_loss,
                             jaccard_score, precision_score, recall_score)

LABELS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]


def parse_output(text):
    m = re.search(r"\[.*?\]", text, re.S)
    if not m:
        return [0] * 8, True
    try:
        items = json.loads(m.group())
    except json.JSONDecodeError:
        return [0] * 8, True
    if not isinstance(items, list):
        return [0] * 8, True
    found = {x.strip().lower() for x in items if isinstance(x, str)} & set(LABELS)
    if not found:
        return [0] * 8, True
    return [int(l in found) for l in LABELS], False


def evaluate(pred, gold, malformed=None, verbose=False):
    pred = np.asarray(pred).astype(int)
    gold = np.asarray(gold).astype(int)
    m = {
        "micro_f1": f1_score(gold, pred, average="micro", zero_division=0),
        "samples_jaccard": jaccard_score(gold, pred, average="samples", zero_division=0),
        "macro_f1": f1_score(gold, pred, average="macro", zero_division=0),
        "micro_precision": precision_score(gold, pred, average="micro", zero_division=0),
        "micro_recall": recall_score(gold, pred, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(gold, pred),
        "exact_match": accuracy_score(gold, pred),
        "empty_pred_rate": float((pred.sum(1) == 0).mean()),
    }
    if malformed is not None:
        m["malformed_rate"] = float(np.mean(malformed))
    m["per_label"] = classification_report(gold, pred, target_names=LABELS, zero_division=0, output_dict=True)
    if verbose:
        for k, v in m.items():
            if k != "per_label":
                print(f"{k:16s} {v:.4f}")
        print(classification_report(gold, pred, target_names=LABELS, zero_division=0, digits=3))
    return m
