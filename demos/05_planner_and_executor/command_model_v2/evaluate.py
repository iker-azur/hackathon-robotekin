"""Step 4: measure the trained model.

Runs CommandModel on a test file and compares every predicted command with the
expected one (the "command" in each line's "meta"):

    intent     exact match
    items      same number of items, and for each (in order) the same kind,
               colour, quantity and position; the name is compared only for
               kind "other", where it is what tells the robot what to take
    place      same kind and colour (or both missing)
    sort_by    exact match
    command    all of the above: what the robot would actually do right

and two safety numbers:

    stop caught         share of "stop" sentences recognised as stop
    false action rate   share of sentences that must not move the robot
                        (stop, cancel, questions, greetings, off-topic,
                        not_doable...) that were read as an action

Test files (in ./data):

    test_real.jsonl    98 hand-written sentences, never generated: the one that matters
    test_unseen.jsonl  generated with words that are not in the training data
    eval_commands.jsonl generated like the training data (how well it learnt)

Writes results/<file>_report.md and results/<file>_errors.jsonl.

Run:

    python .\\demos\\03_command_model\\evaluate.py                      # test_real + test_unseen
    python .\\demos\\03_command_model\\evaluate.py eval_commands.jsonl --limit 600
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))

from command_model import ITEMLESS, CommandModel, plain  # noqa: E402

ACTIONS = {"pick_and_place", "pick", "give", "sort", "clear", "sequence"}


def same_item(expected: dict, got: dict) -> bool:
    if any(expected[k] != got[k] for k in ("kind", "color", "quantity", "position")):
        return False
    # ADDED IN THIS COPY: the per-item destination is part of being right. Without this a
    # command that sent the red ball to the blue box and the blue ball to the red box scored
    # as correct as one that sent each to its own box, which is the whole point of the
    # multi-destination templates.
    if same_place(expected.get("place"), got.get("place")) is False:
        return False
    if expected["kind"] == "other":
        return plain(expected["name"] or "") == plain(got["name"] or "")
    return True


def same_place(expected, got) -> bool:
    """Two destinations agree. Either may be None, meaning "use the command's place"."""
    if expected is None and got is None:
        return True
    if expected is None or got is None:
        return False
    return expected.get("kind") == got.get("kind") and expected.get("color") == got.get("color")


def compare(expected: dict, got: dict) -> dict:
    # For intents that do not use items (a greeting, "stop"...) the robot ignores them: so does the score.
    wanted = [] if expected["intent"] in ITEMLESS else expected["items"]
    items = len(wanted) == len(got["items"]) and all(same_item(e, g) for e, g in zip(wanted, got["items"]))
    place = expected["place"] == (got["place"] and {"kind": got["place"]["kind"], "color": got["place"]["color"]})
    result = {"intent": expected["intent"] == got["intent"], "items": items, "place": place,
              "sort_by": expected["sort_by"] == got["sort_by"]}
    result["command"] = all(result.values())
    return result


def evaluate(model: CommandModel, name: str, limit: int | None,
             data_dir: Path = DATA, results_dir: Path = RESULTS) -> dict:
    lines = [json.loads(line) for line in (data_dir / name).open(encoding="utf-8")]
    lines = lines[:limit] if limit else lines
    scores = collections.defaultdict(list)
    per_lang = collections.defaultdict(lambda: collections.defaultdict(list))
    per_intent = collections.defaultdict(list)
    stops, false_actions, non_actions = [], 0, 0
    errors = []
    start = time.perf_counter()
    for line in lines:
        expected, lang = line["meta"]["command"], line["meta"]["lang"]
        got = model.parse(line["input"])
        result = compare(expected, got)
        for key, ok in result.items():
            scores[key].append(ok)
            per_lang[lang][key].append(ok)
        per_intent[expected["intent"]].append(result["intent"])
        if expected["intent"] == "stop":
            stops.append(got["intent"] == "stop")
        if expected["intent"] not in ACTIONS:
            non_actions += 1
            false_actions += got["intent"] in ACTIONS
        if not result["command"]:
            errors.append({"input": line["input"], "lang": lang, "expected": expected, "got": got,
                           "wrong": [k for k, ok in result.items() if not ok and k != "command"]})
    seconds = time.perf_counter() - start

    def pct(values):
        return f"{100 * sum(values) / len(values):.1f}%" if values else "-"

    report = [f"# {name}", "", f"{len(lines)} sentences, {1000 * seconds / len(lines):.0f} ms per sentence", "",
              "| | intent | items | place | sort_by | **command** |", "| --- | --- | --- | --- | --- | --- |",
              "| all | " + " | ".join(pct(scores[k]) for k in ("intent", "items", "place", "sort_by")) +
              f" | **{pct(scores['command'])}** |"]
    for lang, lang_scores in sorted(per_lang.items()):
        report.append(f"| {lang} | " + " | ".join(pct(lang_scores[k]) for k in ("intent", "items", "place", "sort_by"))
                      + f" | **{pct(lang_scores['command'])}** |")
    report += ["", f"- stop caught: {pct(stops)} ({sum(stops)}/{len(stops)})",
               f"- false action rate: {100 * false_actions / max(non_actions, 1):.1f}% "
               f"({false_actions}/{non_actions} sentences that must not move the robot)", "",
               "| intent | recognised |", "| --- | --- |"]
    report += [f"| {intent} | {pct(ok)} ({len(ok)}) |" for intent, ok in sorted(per_intent.items())]
    report += ["", f"## Errors ({len(errors)})", ""]
    for error in errors[:60]:
        report.append(f"- `{error['lang']}` {error['input']!r}: wrong {', '.join(error['wrong'])}  \n"
                      f"  expected `{json.dumps(error['expected'], ensure_ascii=False)}`  \n"
                      f"  got `{json.dumps({k: v for k, v in error['got'].items() if k != 'confidence'}, ensure_ascii=False)}`")

    results_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    (results_dir / f"{stem}_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    with (results_dir / f"{stem}_errors.jsonl").open("w", encoding="utf-8") as sink:
        for error in errors:
            sink.write(json.dumps(error, ensure_ascii=False) + "\n")
    print("\n".join(report[:12 + len(per_lang)]))
    return {k: sum(v) / len(v) for k, v in scores.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", default=["test_real.jsonl", "test_unseen.jsonl"])
    parser.add_argument("--limit", type=int, help="Only the first N sentences of each file")
    parser.add_argument("--model", help="Model folder (default: models/command-model/best)")
    parser.add_argument("--data-dir", type=Path, default=DATA, help="directory with evaluation JSONL files")
    parser.add_argument("--results-dir", type=Path, default=RESULTS, help="where evaluation reports are written")
    args = parser.parse_args()

    model = CommandModel(args.model) if args.model else CommandModel()
    for name in args.files:
        evaluate(model, name, args.limit, args.data_dir, args.results_dir)
        print(f"  -> {args.results_dir / (Path(name).stem + '_report.md')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
