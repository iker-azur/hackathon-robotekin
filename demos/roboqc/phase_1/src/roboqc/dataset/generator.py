"""Reproducible dataset generator for RoboQC."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from roboqc.dataset.templates import generate_one
from roboqc.dataset.validation import validate_example, validate_dataset, check_split_leakage
from roboqc.commands.taxonomy import ALL_INTENTS


def _intent_of_example(ex: Dict[str, Any]) -> str:
    acts = ex.get("actions", [])
    return acts[0].get("intent", "UNKNOWN") if acts else "UNKNOWN"


def make_unique_variant(text: str, seen: set, rng: random.Random) -> Tuple[str, str]:
    base = text.rstrip(" .!?")
    prefixes = ["", "Por favor, ", "Ahora, ", "Robot, ", "Operador, "]
    suffixes = ["", " ahora", " por favor", " inmediatamente", " ya", " enseguida"]
    adverbs = ["", " rapidamente", " con cuidado", " despacio"]
    puncts = [".", "!", ".", "."]
    combos = [(p, a, s, q) for p in prefixes for a in adverbs for s in suffixes for q in puncts]
    rng.shuffle(combos)
    for p, a, s, q in combos:
        cand = " ".join((p + base + a + s + q).split())
        key = cand.lower()
        if key not in seen:
            return cand, key
    # guaranteed fallback
    i = 1
    while True:
        cand = f"{base} ({i})."
        key = cand.lower()
        if key not in seen:
            return cand, key
        i += 1


def _build_intent_to_gens() -> Dict[str, List]:
    from roboqc.dataset import templates as T
    rng = random.Random(12345)
    mapping: Dict[str, List] = {it: [] for it in ALL_INTENTS}
    for gen in T.TEMPLATE_GENERATORS:
        # sample a few times to determine primary intent
        counts: Dict[str, int] = {}
        for _ in range(5):
            try:
                text, actions, family = gen(rng)
            except Exception:
                continue
            it = actions[0].get("intent", "UNKNOWN") if actions else "UNKNOWN"
            counts[it] = counts.get(it, 0) + 1
        if counts:
            primary = max(counts, key=lambda k: counts[k])
            if primary in mapping:
                mapping[primary].append(gen)
    # ensure every intent has at least one generator
    for it in ALL_INTENTS:
        if not mapping[it]:
            mapping[it] = list(T.TEMPLATE_GENERATORS)
    return mapping


def generate_dataset(n_samples: int = 25000, seed: int = 42) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    intent_to_gens = _build_intent_to_gens()
    num_intents = len(ALL_INTENTS)
    target_per_intent = max(1, n_samples // num_intents)

    records: List[Dict[str, Any]] = []
    seen_texts = set()
    family_to_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    attempts = 0
    max_attempts = n_samples * 50

    per_intent_counts: Dict[str, int] = {it: 0 for it in ALL_INTENTS}

    # Phase 1: balanced targeted generation (no early abort on a single intent).
    while any(per_intent_counts[it] < target_per_intent for it in ALL_INTENTS) and attempts < max_attempts:
        attempts += 1
        target_it = min(ALL_INTENTS, key=lambda it: (per_intent_counts[it], it))
        if per_intent_counts[target_it] >= target_per_intent:
            continue
        gens = intent_to_gens.get(target_it)
        if not gens:
            gens = list(TEMPLATE_GENERATORS)
        progressed = False
        for _ in range(10):
            gen = rng.choice(gens)
            try:
                text, actions, family = gen(rng)
            except Exception:
                continue
            if _intent_of_example({"actions": actions}) != target_it:
                continue
            key = " ".join(text.lower().split())
            if key in seen_texts:
                aug_text, aug_key = make_unique_variant(text, seen_texts, rng)
                text = aug_text
                key = aug_key
            ex = {"text": text, "actions": actions, "family": family}
            errs = validate_example(ex)
            if errs:
                continue
            seen_texts.add(key)
            records.append(ex)
            family_to_examples[family].append(ex)
            per_intent_counts[target_it] += 1
            progressed = True
            if len(records) % 100 == 0 or len(records) == n_samples:
                sys.stdout.write(f"\rGenerando: {len(records)}/{n_samples} ({len(records)/n_samples*100:.1f}%)")
                sys.stdout.flush()
            break
        if not progressed:
            # fallback: random generation
            try:
                text, actions, family = generate_one(rng)
            except Exception:
                continue
            key = " ".join(text.lower().split())
            if key in seen_texts:
                aug_text, aug_key = make_unique_variant(text, seen_texts, rng)
                text = aug_text
                key = aug_key
            ex = {"text": text, "actions": actions, "family": family}
            errs = validate_example(ex)
            if errs:
                continue
            it = _intent_of_example(ex)
            if per_intent_counts.get(it, 0) >= target_per_intent:
                continue
            seen_texts.add(key)
            records.append(ex)
            family_to_examples[family].append(ex)
            per_intent_counts[it] += 1
            if len(records) % 100 == 0 or len(records) == n_samples:
                sys.stdout.write(f"\rGenerando: {len(records)}/{n_samples} ({len(records)/n_samples*100:.1f}%)")
                sys.stdout.flush()

    # Phase 2: fill remainder to exact n_samples with random (small remainder).
    while len(records) < n_samples and attempts < max_attempts:
        attempts += 1
        try:
            text, actions, family = generate_one(rng)
        except Exception:
            continue
        key = " ".join(text.lower().split())
        if key in seen_texts:
            aug_text, aug_key = make_unique_variant(text, seen_texts, rng)
            text = aug_text
            key = aug_key
        ex = {"text": text, "actions": actions, "family": family}
        errs = validate_example(ex)
        if errs:
            continue
        seen_texts.add(key)
        records.append(ex)
        family_to_examples[family].append(ex)
        if len(records) % 100 == 0 or len(records) == n_samples:
            sys.stdout.write(f"\rGenerando: {len(records)}/{n_samples} ({len(records)/n_samples*100:.1f}%)")
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Harder / held-out families: these are kept entirely out of train/val/test
    # so the challenge set contains unseen linguistic structures.
    HARD_FAMILIES = {
        "compound_two", "compound_colors", "negation_pick", "negation_place",
        "except", "correction", "speech_noise", "incomplete",
        "change_quality_rule", "change_destination", "move_qty", "place_all",
    }
    # Map family -> primary intent to avoid starving an intent in training.
    family_intent: Dict[str, str] = {}
    for fam, exs in family_to_examples.items():
        if exs:
            acts = exs[0].get("actions", [])
            family_intent[fam] = acts[0].get("intent", "UNKNOWN") if acts else "UNKNOWN"

    # Also treat pattern families with complex/hard markers as held-out,
    # but never hold out the last family for any intent.
    intent_family_counts: Dict[str, int] = {}
    for fam in family_to_examples:
        it = family_intent.get(fam, "UNKNOWN")
        intent_family_counts[it] = intent_family_counts.get(it, 0) + 1

    for fam in list(family_to_examples.keys()):
        if any(tag in fam for tag in ("comp_", "neg_", "all_", "qty_")):
            it = family_intent.get(fam, "UNKNOWN")
            if intent_family_counts.get(it, 0) > 1:
                HARD_FAMILIES.add(fam)
                intent_family_counts[it] -= 1

    families = list(family_to_examples.keys())
    rng.shuffle(families)
    hard_present = [f for f in families if f in HARD_FAMILIES]
    # Challenge holds out all hard families entirely, plus ~12% of each intent's normal
    # families. The rest is split per intent, so every intent is validated and tested
    # on phrasings it never saw in training.
    challenge_fams = set(hard_present)
    by_intent: Dict[str, List[str]] = defaultdict(list)
    for f in families:
        if f not in HARD_FAMILIES:
            by_intent[family_intent.get(f, "UNKNOWN")].append(f)
    train_fams, val_fams, test_fams = set(), set(), set()
    for fams in by_intent.values():
        n = len(fams)
        n_chal = round(n * 0.12)
        n_val = max(1, round(n * 0.15)) if n >= 4 else 0
        n_test = max(1, round(n * 0.15)) if n >= 4 else 0
        challenge_fams |= set(fams[:n_chal])
        val_fams |= set(fams[n_chal:n_chal + n_val])
        test_fams |= set(fams[n_chal + n_val:n_chal + n_val + n_test])
        train_fams |= set(fams[n_chal + n_val + n_test:])

    splits = {"train": [], "validation": [], "test": [], "challenge": []}
    for ex in records:
        fam = ex["family"]
        ex["difficulty"] = "hard" if fam in HARD_FAMILIES else "normal"
        if fam in train_fams:
            splits["train"].append(ex)
        elif fam in val_fams:
            splits["validation"].append(ex)
        elif fam in test_fams:
            splits["test"].append(ex)
        else:
            splits["challenge"].append(ex)

    # ensure challenge not empty
    if not splits["challenge"]:
        splits["challenge"].extend(splits["test"][: max(1, len(splits["test"]) // 5)])
        splits["test"] = splits["test"][max(1, len(splits["test"]) // 5):]

    # Ensure every intent appears in train/val/test (coverage), by moving
    # families from train when a split is missing an intent entirely.
    def fam_intent(f):
        exs = family_to_examples.get(f, [])
        if not exs:
            return "UNKNOWN"
        acts = exs[0].get("actions", [])
        return acts[0].get("intent", "UNKNOWN") if acts else "UNKNOWN"

    for split_name in ("validation", "test"):
        present = {fam_intent(f) for f in (val_fams if split_name == "validation" else test_fams)}
        for it in ALL_INTENTS:
            if it in present:
                continue
            cands = [f for f in train_fams if fam_intent(f) == it and len([x for x in train_fams if fam_intent(x) == it]) > 1]
            if cands:
                f = cands[0]
                train_fams.discard(f)
                if split_name == "validation":
                    val_fams.add(f)
                else:
                    test_fams.add(f)
    # rebuild splits after coverage adjustments
    splits = {"train": [], "validation": [], "test": [], "challenge": []}
    for ex in records:
        fam = ex["family"]
        ex["difficulty"] = "hard" if fam in HARD_FAMILIES else "normal"
        if fam in train_fams:
            splits["train"].append(ex)
        elif fam in val_fams:
            splits["validation"].append(ex)
        elif fam in test_fams:
            splits["test"].append(ex)
        else:
            splits["challenge"].append(ex)

    # clean extra keys? keep family for manifest but maybe remove from output? keep.
    manifest = {
        "seed": seed,
        "n_requested": n_samples,
        "n_generated": len(records),
        "attempts": attempts,
        "n_families": len(families),
        "n_hard_families": len(hard_present),
        "challenge_families": sorted(challenge_fams),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "intent_counts": {it: sum(1 for r in records if _intent_of_example(r) == it) for it in ALL_INTENTS},
        "difficulty_counts": {
            split: sum(1 for r in recs if r.get("difficulty") == "hard")
            for split, recs in splits.items()
        },
        "family_counts": {k: len(v) for k, v in family_to_examples.items()},
    }
    return splits, manifest


def write_outputs(splits, manifest, out_dir="data") -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, recs in splits.items():
        path = os.path.join(out_dir, f"{name if name != 'validation' else 'validation'}.jsonl")
        # map validation -> validation.jsonl (already)
        fname = name + ".jsonl"
        if name == "validation":
            fname = "validation.jsonl"
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                # output text + actions (keep family as extra)
                obj = {"text": r["text"], "actions": r["actions"]}
                obj["family"] = r["family"]
                obj["difficulty"] = r.get("difficulty", "normal")
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # taxonomy.json
    taxonomy = {"intents": ALL_INTENTS, "schema_version": "1.0"}
    with open(os.path.join(out_dir, "taxonomy.json"), "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)

    # generation_manifest.json
    with open(os.path.join(out_dir, "generation_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # dataset_report.json
    all_recs = []
    for recs in splits.values():
        all_recs.extend(recs)
    report = validate_dataset(all_recs)
    report["split_leakage"] = check_split_leakage(splits)
    with open(os.path.join(out_dir, "dataset_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generador de dataset RoboQC (espanol).")
    ap.add_argument("--samples", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    splits, manifest = generate_dataset(n_samples=args.samples, seed=args.seed)
    write_outputs(splits, manifest, out_dir=args.out_dir)
    print(json.dumps({"generated": manifest["n_generated"], "splits": manifest["split_sizes"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
