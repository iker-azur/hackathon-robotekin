"""Dataset validation helpers."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List


def validate_example(ex: Dict[str, Any]) -> List[str]:
    errs = []
    if not ex.get("text"):
        errs.append("empty_text")
    actions = ex.get("actions")
    if not isinstance(actions, list) or not actions:
        errs.append("missing_actions")
    for a in actions or []:
        if "intent" not in a:
            errs.append("missing_intent")
        if "parameters" not in a:
            errs.append("missing_parameters")
        if "negated" not in a:
            errs.append("missing_negated")
    return errs


def validate_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors = []
    texts = [r.get("text", "") for r in records]
    # duplicates
    seen = set()
    dup = 0
    for t in texts:
        key = " ".join(t.lower().split())
        if key in seen:
            dup += 1
        seen.add(key)
    # intent coverage
    intents = Counter()
    for r in records:
        for a in r.get("actions", []):
            intents[a.get("intent")] += 1
    # near-duplicate (simple): count pairs with very high similarity? skip heavy; report unique ratio
    report = {
        "total": len(records),
        "unique_texts": len(seen),
        "duplicate_texts": dup,
        "intent_counts": dict(intents),
        "validation_errors": errors,
    }
    return report


def check_split_leakage(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    # detect exact text overlap between train and test/val/challenge
    train_texts = set(" ".join(r["text"].lower().split()) for r in splits.get("train", []))
    out = {}
    for name, recs in splits.items():
        if name == "train":
            continue
        overlap = sum(1 for r in recs if " ".join(r["text"].lower().split()) in train_texts)
        out[name] = {"overlap_with_train": overlap, "size": len(recs)}
    return out
