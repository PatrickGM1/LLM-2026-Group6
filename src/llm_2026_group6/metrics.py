# shared stuff: label list, output parser, metrics from section 7 of the brief

import json
import re

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, f1_score, hamming_loss,
                             jaccard_score, precision_score, recall_score)

LABELS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]


def parse_output(text):
    try:
        items = json.loads(re.search(r"\[.*?\]", text, re.S).group())
        found = {x.strip().lower() for x in items if isinstance(x, str)} & set(LABELS)
    except (AttributeError, json.JSONDecodeError):
        found = set()
    return [int(l in found) for l in LABELS], not found


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
