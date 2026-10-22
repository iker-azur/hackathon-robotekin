import random
from collections import defaultdict

from roboqc.dataset import templates as T
from roboqc.dataset.generator import generate_dataset
from roboqc.commands.taxonomy import ALL_INTENTS


def test_literal_pattern_values_are_kept():
    rng = random.Random(0)
    for i, spec in enumerate(T.PATTERN_SPECS):
        _, actions, _ = T._run_pattern(rng, spec, i)
        for param, key in spec.get("params", {}).items():
            if key not in ("obj", "color", "dest", "dcolor", "quality", "q"):
                assert actions[0]["parameters"][param] == key, spec["text"]


def test_quantity_label_matches_text():
    rng = random.Random(1)
    for i, spec in enumerate(T.PATTERN_SPECS):
        if "quantity" in spec.get("params", {}) or "{obj_phrase}" in spec["text"]:
            continue
        _, actions, _ = T._run_pattern(rng, spec, i)
        assert actions[0]["parameters"]["quantity"] == "UNSPECIFIED", spec["text"]


def test_splits_do_not_share_families():
    splits, _ = generate_dataset(n_samples=12000, seed=7)
    fams = {name: {r["family"] for r in recs} for name, recs in splits.items()}
    assert not fams["train"] & fams["validation"]
    assert not fams["train"] & fams["test"]
    assert not fams["validation"] & fams["test"]


def test_every_intent_is_validated_and_tested():
    splits, _ = generate_dataset(n_samples=12000, seed=7)
    seen = {name: defaultdict(set) for name in ("train", "validation", "test")}
    for name in seen:
        for r in splits[name]:
            seen[name][r["actions"][0]["intent"]].add(r["family"])
    for intent in ALL_INTENTS:
        if intent == "UNKNOWN":
            continue
        assert seen["train"][intent], intent
        assert seen["validation"][intent], intent
        assert seen["test"][intent], intent
