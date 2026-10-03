"""Show what the v2 dataset templates produce, per language. No model, no training.

    python command_model_v2/show_templates.py
    python command_model_v2/show_templates.py --seed 3 --count 3
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import generate_dataset as G  # noqa: E402


def spans_of(phrase) -> dict:
    return {k: v for k, v in phrase.spans.items() if k.startswith("destination") and v}


def show(title, make, langs, count, seed):
    print(f"--- {title} ---")
    for lang in langs:
        for _ in range(count):
            sample = make(G.GRAMMARS[lang], random.Random(seed), words, lang)
            seed += 1
            print(f"  [{lang}] {sample.text}")
            print(f"         intent={sample.intent} template={sample.template} "
                  f"sort_by={sample.sort_by} command_place="
                  f"{sample.place.kind if sample.place else None}")
            for it, ph in sample.items:
                dest = f"{it.dest.kind}/{it.dest.color}" if it.dest else None
                print(f"         item {it.kind}/{it.color} qty={it.quantity} -> dest={dest} "
                      f"spans={spans_of(ph)}")
            print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--count", type=int, default=2)
    args = parser.parse_args()

    global words
    words = G.vocab_split("train")
    langs = ("es", "en", "eu")

    show("sort/into_matching", G._sort_into_matching, langs, args.count, args.seed)
    show("sequence/each_own_place", G._each_own_place, langs, args.count, args.seed)

    print("--- give, clear and sort: the templates each intent actually produces ---")
    for intent in ("give", "clear", "sort"):
        print(f"  {intent}:")
        seen: dict[str, str] = {}
        for lang in langs:
            for seed in range(args.count * 12):
                sample = G.build(intent, lang, random.Random(seed), words)
                seen.setdefault(f"{sample.template}", f"[{lang}] {sample.text}")
        for template, example in seen.items():
            print(f"    {template:28s} {example}")
    return 0


words = None

if __name__ == "__main__":
    raise SystemExit(main())