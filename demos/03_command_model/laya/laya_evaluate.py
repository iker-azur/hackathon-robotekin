"""Measure a Laya checkpoint on the command test sets (sentence-level decisions).

Asks the four questions of laya_questions.py about every sentence of a test
file and compares the chosen options with the right ones:

    intent, place_kind, place_color, sort_by   accuracy of each
    sentence                                   all four right
    stop caught / false action rate            the same safety numbers as ../evaluate.py

Run:

    python .\\demos\\03_command_model\\laya\\laya_evaluate.py                                  # zero-shot laya-multilingual
    python .\\demos\\03_command_model\\laya\\laya_evaluate.py --model demos\\03_command_model\\models\\laya-commands
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
from laya_questions import QUESTIONS, answers_of  # noqa: E402

ACTIONS = {"pick_and_place", "pick", "give", "sort", "clear", "sequence"}
BASE = "convaiinnovations/laya-multilingual"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", default=["test_real.jsonl", "test_unseen.jsonl"])
    parser.add_argument("--model", default=BASE, help="Hub id or local folder of a Laya checkpoint")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    try:
        import truststore  # Norton re-signs HTTPS; see demos/02_voice

        truststore.inject_into_ssl()
    except ImportError:
        pass
    import laya

    agent = laya.load(args.model)
    agent.predict("coge la bola roja", QUESTIONS)  # warm up
    name = Path(args.model).name
    for file in args.files:
        lines = [json.loads(line) for line in (PROJECT / "data" / file).open(encoding="utf-8")]
        lines = lines[:args.limit] if args.limit else lines
        right = collections.defaultdict(list)
        per_lang = collections.defaultdict(list)
        stops, false_actions, non_actions, errors = [], 0, 0, []
        start = time.perf_counter()
        for line in lines:
            want = answers_of(line["meta"]["command"])
            answers = agent.predict(line["input"], QUESTIONS)["answers"]
            got = {q: answers[q]["choice"] for q in QUESTIONS}
            for q in QUESTIONS:
                right[q].append(got[q] == want[q])
            whole = got == want
            right["sentence"].append(whole)
            per_lang[line["meta"]["lang"]].append(whole)
            if want["intent"] == "stop":
                stops.append(got["intent"] == "stop")
            if want["intent"] not in ACTIONS:
                non_actions += 1
                false_actions += got["intent"] in ACTIONS
            if not whole:
                errors.append({"input": line["input"], "want": want, "got": got,
                               "confidence": {q: answers[q].get("confidence") for q in QUESTIONS}})
        ms = 1000 * (time.perf_counter() - start) / len(lines)

        def pct(values):
            return f"{100 * sum(values) / len(values):.1f}%" if values else "-"

        report = [f"# Laya `{args.model}` on {file}", "", f"{len(lines)} sentences, {ms:.0f} ms per sentence", "",
                  "| intent | place_kind | place_color | sort_by | **all four** |", "| --- | --- | --- | --- | --- |",
                  "| " + " | ".join(pct(right[q]) for q in QUESTIONS) + f" | **{pct(right['sentence'])}** |", "",
                  "all four right per language: " + ", ".join(f"{k} {pct(v)}" for k, v in sorted(per_lang.items())),
                  f"- stop caught: {pct(stops)} ({sum(stops)}/{len(stops)})",
                  f"- false action rate: {100 * false_actions / max(non_actions, 1):.1f}% ({false_actions}/{non_actions})",
                  "", f"## Errors ({len(errors)})", ""]
        report += [f"- {e['input']!r}: want `{e['want']}` got `{e['got']}`" for e in errors[:50]]
        out = PROJECT / "results" / f"laya_{name}_{Path(file).stem}.md"
        out.parent.mkdir(exist_ok=True)
        out.write_text("\n".join(report) + "\n", encoding="utf-8")
        print("\n".join(report[:12]), f"\n  -> {out.relative_to(PROJECT)}\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
