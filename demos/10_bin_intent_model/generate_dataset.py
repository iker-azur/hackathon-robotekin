"""Step 1: generate the intent dataset (Spanish, English, Basque) for the two-box bin-sorting cell.

    python generate_dataset.py                 # 20,000 training sentences per language
    python generate_dataset.py --show 40       # print a sample of each language and stop

Writes, to ``data/`` (all JSON Lines, one sample per line, five classification answers each):

    train.jsonl         20,000 sentences per language, train words only
    eval.jsonl          sentences held out of train (same words): the trainer's validation loss
    calib.jsonl         sentences with CALIB words that never occur in train: fits the probability calibration
    test_unseen.jsonl   sentences with TEST words that occur in neither: measures generalisation
    stats.md            counts per language and intent

No sentence (compared without case, accents or punctuation) appears in two files. The vocabulary lives in
``grammar_es.py`` / ``grammar_en.py`` / ``grammar_eu.py`` as ``train`` / ``calib`` / ``test`` word tiers.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from grammar_common import ACTION_INTENTS, INTENTS, LANGUAGES, TASKS, Ctx, lower_first, plain, upper_first
from grammar_en import English
from grammar_es import Spanish
from grammar_eu import Basque

HERE = Path(__file__).resolve().parent
GRAMMARS = {"es": Spanish(), "en": English(), "eu": Basque()}

# share of each intent in a language's sentences (sums to 1)
QUOTAS = {"move": 0.26, "sort": 0.22, "pick": 0.065, "both": 0.075, "reply": 0.075, "ask": 0.055, "stop": 0.035,
          "cancel": 0.0225, "repeat": 0.0225, "yes": 0.025, "no": 0.025, "hello": 0.025, "chat": 0.055, "refuse": 0.04}
NEEDS_NOVEL = {"move", "sort", "pick", "both", "ask", "refuse", "reply"}  # grammar-built: an unseen word must appear
CONFUSIONS = {  # what a speech recogniser plausibly writes instead (the label is still recoverable)
    "es": {"caja": "casa", "perno": "berno", "pernos": "bernos", "blanca": "planca", "grande": "grane", "piezas": "pieza"},
    "en": {"bolt": "bold", "bolts": "bolds", "white": "wide", "pieces": "peaces", "bad": "bat"},
    "eu": {"kutxa": "kuxa", "pernoa": "bernoa", "handia": "handi", "txarrak": "txarra", "piezak": "pieza"},
}


def labels(intent="refuse", feature="any", quality="any", box="none", count="none") -> dict:
    return {"intent": intent, "feature": feature, "quality": quality, "box": box, "count": count}


def pieces(ctx: Ctx):
    feature = ctx.weighted({"any": 0.40, "bolt": 0.32, "no bolt": 0.28})
    quality = ctx.weighted({"any": 0.40, "bad": 0.32, "good": 0.28})
    count = ctx.weighted({"all": 0.60, "one": 0.17, "two": 0.12, "three": 0.11})
    return feature, quality, count


def other_box(box: str, rng: random.Random) -> str:
    return rng.choice([b for b in ("big", "white", "other") if b != box])


def build(lang, ctx: Ctx, intent: str):
    """One ``(text, labels)`` pair for an intent."""
    rng = ctx.rng
    if intent in ACTION_INTENTS:
        feature, quality, count = pieces(ctx)
        if intent == "pick":
            box = "none"
        elif intent == "move":
            box = ctx.weighted({"big": 0.37, "white": 0.37, "other": 0.14, "none": 0.12})
        else:
            box = ctx.weighted({"big": 0.30, "white": 0.30, "other": 0.08, "none": 0.32})
        truth = labels(intent, feature, quality, box, count)
        if intent != "pick" and box != "none" and ctx.chance(0.07):
            return correction(lang, ctx, intent, feature, quality, box, count), truth
        return lang.action(ctx, intent, feature, quality, box, count), truth
    if intent == "both":
        return lang.both(ctx), labels("both")
    if intent == "reply":
        kind = ctx.weighted({"box": 0.40, "np": 0.45, "count": 0.15})
        if kind == "box":
            box = ctx.weighted({"big": 0.4, "white": 0.4, "other": 0.2})
            return lang.reply(ctx, "any", "any", box, "none"), labels("reply", box=box)
        if kind == "count":
            count = ctx.weighted({"all": 0.4, "one": 0.3, "two": 0.15, "three": 0.15})
            return lang.reply(ctx, "any", "any", "none", count), labels("reply", count=count)
        feature, quality, count = pieces(ctx)
        if feature == "any" and quality == "any":
            quality = rng.choice(["bad", "good"])
        return lang.reply(ctx, feature, quality, "none", count), labels("reply", feature, quality, "none", count)
    if intent in ("stop", "cancel", "repeat", "yes", "no", "hello", "chat"):
        return lang.control(ctx, intent), labels(intent)
    if intent == "ask":
        return lang.ask(ctx), labels("ask")
    return lang.refuse(ctx), labels("refuse")


def correction(lang, ctx: Ctx, intent, feature, quality, box, count) -> str:
    """"Move the good ones to the big box, no, the white one": the labels are the last thing said."""
    rng = ctx.rng
    marker = rng.choice(lang.markers)
    kind = ctx.weighted({"full": 0.5, "box": 0.35, "piece": 0.15})
    if kind == "box":
        wrong = other_box(box, rng)
        first = lang.action(ctx, intent, feature, quality, wrong, count)
        return f"{first}, {marker} {lang.box_ph(ctx, box, intent)}"
    if kind == "piece":
        if quality != "any" and ctx.chance(0.5):
            wf, wq = feature, ("bad" if quality != "bad" else "good")
        elif feature != "any":
            wf, wq = ("bolt" if feature != "bolt" else "no bolt"), quality
        else:
            wf, wq = feature, ("bad" if quality != "bad" else "good")
        first = lang.action(ctx, intent, wf, wq, box, count)
        return f"{first}, {marker} {lang.np(ctx, feature, quality, count, pron=False)[0]}"
    wf, wq, wbox = rng.choice(["bolt", "no bolt", "any"]), rng.choice(["bad", "good", "any"]), other_box(box, rng)
    first = lang.action(ctx, intent, wf, wq, wbox, count)
    return f"{first}, {marker} {lower_first(lang.action(ctx, intent, feature, quality, box, count))}"


# --- what a speech recogniser does to text -------------------------------------------------------------------

def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def typo(word: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(word) - 1)
    return rng.choice([word[:i] + word[i + 1:], word[:i] + word[i] + word[i:], word[:i] + word[i + 1] + word[i] + word[i + 2:],
                       word[:i] + rng.choice("aeiounrst") + word[i + 1:]])


def finish(text: str, code: str, rng: random.Random) -> str:
    text = " ".join(text.split())
    roll = rng.random()
    if roll < 0.35:
        text = upper_first(text)
    elif roll < 0.55:
        text = text.lower()
    if rng.random() < 0.30 and text[-1] not in "?!.":
        text += rng.choice([".", "!", "."])
    if rng.random() < 0.25:  # speech recognition often drops punctuation
        text = text.replace("¿", "").replace("¡", "").rstrip("?!.")
    if code != "en" and rng.random() < 0.12:
        text = strip_accents(text)
    words = text.split()
    if rng.random() < 0.02:
        table = CONFUSIONS[code]
        for i, w in enumerate(words):
            if w.lower().strip(".,?!¿¡") in table and rng.random() < 0.7:
                words[i] = table[w.lower().strip(".,?!¿¡")]
                break
    elif rng.random() < 0.05:
        long = [i for i, w in enumerate(words) if len(w) >= 6 and w.isalpha()]
        if long:
            i = rng.choice(long)
            words[i] = typo(words[i], rng)
    return " ".join(words)


def record(text: str, truth: dict, code: str, tier: str) -> dict:
    return {"input": text,
            "output": {"classifications": [{"task": task, "labels": options, "true_label": [truth[task]]}
                                           for task, options in TASKS.items()]},
            "meta": {"lang": code, "tier": tier, "intent": truth["intent"]}}


def generate(code: str, tier: str, total: int, seen: set, rng: random.Random) -> list:
    lang, rows = GRAMMARS[code], []
    for intent, share in QUOTAS.items():
        want, got, attempts = round(total * share), 0, 0
        while got < want and attempts < want * 60:
            attempts += 1
            ctx = Ctx(rng, tier)
            text, truth = build(lang, ctx, intent)
            if tier != "train" and intent in NEEDS_NOVEL and not ctx.novel:
                continue
            text = finish(lang.wrap(ctx, text, intent), code, rng)
            key = plain(text)
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(record(text, truth, code, tier))
            got += 1
        if got < want:
            print(f"  note: {code}/{tier}/{intent}: {got} of {want} distinct sentences (vocabulary exhausted)")
    return rows


def write(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-language", type=int, default=20000, help="training sentences per language")
    parser.add_argument("--eval", type=int, default=1000, help="held-out validation sentences per language")
    parser.add_argument("--calib", type=int, default=800, help="calibration sentences per language (CALIB words)")
    parser.add_argument("--test", type=int, default=1000, help="unseen-word test sentences per language (TEST words)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=HERE / "data")
    parser.add_argument("--show", type=int, default=0, help="print this many samples per language and stop")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")

    splits = {"train": [], "eval": [], "calib": [], "test_unseen": []}
    for code in LANGUAGES:
        rng = random.Random(f"{args.seed}-{code}")
        seen: set = set()
        if args.show:
            rows = generate(code, "train", args.show * 8, seen, rng)
            random.Random(1).shuffle(rows)
            for row in rows[:args.show]:
                got = {c["task"]: c["true_label"][0] for c in row["output"]["classifications"]}
                print(f"[{code}] {row['input']!r:90} {got['intent']:6} {got['feature']:8} {got['quality']:5} "
                      f"{got['box']:6} {got['count']}")
            continue
        # unseen-word sets first, so the training sentences can never coincide with them
        splits["test_unseen"] += generate(code, "test", args.test, seen, rng)
        splits["calib"] += generate(code, "calib", args.calib, seen, rng)
        splits["eval"] += generate(code, "train", args.eval, seen, rng)
        splits["train"] += generate(code, "train", args.per_language, seen, rng)
    if args.show:
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    lines = ["# Dataset statistics", ""]
    for name, rows in splits.items():
        rng.shuffle(rows)
        write(args.output / f"{name}.jsonl", rows)
        lines += [f"## {name}: {len(rows)} sentences", "", "| intent | " + " | ".join(LANGUAGES) + " |",
                  "| --- | " + " | ".join("---" for _ in LANGUAGES) + " |"]
        counts = Counter((r["meta"]["lang"], r["meta"]["intent"]) for r in rows)
        for intent in INTENTS:
            lines.append(f"| {intent} | " + " | ".join(str(counts[(code, intent)]) for code in LANGUAGES) + " |")
        lines.append("")
        print(f"{name:12} {len(rows):6} sentences  " + "  ".join(
            f"{code}={sum(1 for r in rows if r['meta']['lang'] == code)}" for code in LANGUAGES))
    (args.output / "stats.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
