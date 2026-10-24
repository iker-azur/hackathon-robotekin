"""Step 3: measure the model, calibrate its probabilities, and time it.

    python evaluate.py --calibrate                 # fit one temperature per question on data/calib.jsonl, then report
    python evaluate.py                             # report with the saved calibration.json
    python evaluate.py --model models/bin-intent/best --device cpu --latency

Four sets, from easiest to fairest:

    eval          held-out sentences built from the training words (same distribution as training)
    test_unseen   sentences built with words that never occur in training (verbs, nouns, adjectives, box names)
    calib         a second set of unseen words; only used here, to fit the temperatures
    real          ``test_real.txt``: sentences written by hand, not by the grammars

Per set it reports: accuracy per question, whole-sentence exact match, how often the probability is right
(expected calibration error and a reliability table), how much exact-match accuracy rises when low-confidence
answers are refused, the big/white box confusion, and the safety numbers (a sentence that must not move the robot
being read as an order; "stop" being missed).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from grammar_common import ACTION_INTENTS, TASKS

HERE = Path(__file__).resolve().parent
ORDERS = set(ACTION_INTENTS) | {"both"}
KEYS = list(TASKS)


def load_jsonl(path: Path) -> list:
    rows = []
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        truth = {c["task"]: c["true_label"][0] for c in record["output"]["classifications"]}
        rows.append({"text": record["input"], "lang": record["meta"]["lang"], "truth": truth})
    return rows


def load_real(path: Path) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        lang, text, *values = [part.strip() for part in line.split(" | ")]
        rows.append({"text": text, "lang": lang, "truth": dict(zip(KEYS, values))})
    return rows


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def score(classifier, schema, config, rows: list) -> list:
    """Raw logits per question for each row: ``[{task: array in TASKS order}]``."""
    scores = classifier.batch_score([r["text"] for r in rows], schema, config=config)
    return [{task: np.array([s.tasks[task][label] for label in TASKS[task]]) for task in TASKS} for s in scores]


def fit_temperatures(rows: list, logits: list) -> dict:
    """Per question, the temperature that minimises the negative log-likelihood of the true option."""
    out = {}
    for task, options in TASKS.items():
        index = [options.index(r["truth"][task]) for r in rows]
        best = (math.inf, 1.0)
        for temperature in np.exp(np.linspace(math.log(0.3), math.log(30), 240)):
            nll = -np.mean([math.log(max(softmax(l[task], temperature)[i], 1e-12)) for l, i in zip(logits, index)])
            best = min(best, (nll, float(temperature)))
        out[task] = round(best[1], 3)
    return out


def predictions(rows: list, logits: list, temperature: dict) -> list:
    out = []
    for l in logits:
        probs = {task: softmax(l[task], temperature.get(task, 1.0)) for task in TASKS}
        out.append({task: (TASKS[task][int(p.argmax())], float(p.max())) for task, p in probs.items()})
    return out


def sentence_confidence(pred: dict) -> float:
    """The weakest answer among the questions that matter for what the sentence was read as."""
    tasks = ["intent"] + (["feature", "quality", "box", "count"] if pred["intent"][0] in ("move", "sort", "pick") else [])
    return min(pred[t][1] for t in tasks)


def ece(confidences: list, correct: list, bins: int = 10) -> float:
    confidences, correct = np.array(confidences), np.array(correct, dtype=float)
    total = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        mask = (confidences > lo) & (confidences <= lo + 1 / bins)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(total)


def report(name: str, rows: list, preds: list) -> str:
    n = len(rows)
    out = [f"# {name}", "", f"{n} sentences", ""]
    hit = {task: [p[task][0] == r["truth"][task] for r, p in zip(rows, preds)] for task in TASKS}
    exact = [all(hit[t][i] for t in TASKS) for i in range(n)]
    out += ["| | " + " | ".join(KEYS) + " | **all five** |", "| --- | " + " | ".join("---" for _ in KEYS) + " | --- |"]
    for lang in ["all"] + sorted({r["lang"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if lang == "all" or r["lang"] == lang]
        out.append(f"| {lang} ({len(idx)}) | " + " | ".join(f"{np.mean([hit[t][i] for i in idx]):.1%}" for t in KEYS)
                   + f" | **{np.mean([exact[i] for i in idx]):.1%}** |")
    out += ["", "## Is the percentage right?", "",
            "| question | accuracy | mean confidence | calibration error (ECE) |", "| --- | --- | --- | --- |"]
    for task in KEYS:
        conf = [p[task][1] for p in preds]
        out.append(f"| {task} | {np.mean(hit[task]):.1%} | {np.mean(conf):.1%} | {ece(conf, hit[task]):.3f} |")
    conf = [sentence_confidence(p) for p in preds]
    out += ["", "Refusing the answers the model is least sure of (confidence = weakest question that matters):", "",
            "| confidence at least | sentences kept | all five right among them |", "| --- | --- | --- |"]
    for threshold in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        keep = [i for i in range(n) if conf[i] >= threshold]
        if keep:
            out.append(f"| {threshold:.0%} | {len(keep) / n:.1%} | {np.mean([exact[i] for i in keep]):.1%} |")

    big_white = [(r["truth"]["box"], p["box"][0]) for r, p in zip(rows, preds)
                 if r["truth"]["intent"] in ("move", "sort") and r["truth"]["box"] in ("big", "white")]
    if big_white:
        wrong = sum(1 for t, g in big_white if g in ("big", "white") and g != t)
        miss = sum(1 for t, g in big_white if g not in ("big", "white"))
        out += ["", "## The two boxes", "",
                f"{len(big_white)} orders named the big or the white box. Sent to the wrong one of the two: "
                f"**{wrong}** ({wrong / len(big_white):.2%}). Read as another box or none: {miss} ({miss / len(big_white):.2%}).", ""]
    intents = defaultdict(list)
    for r, p in zip(rows, preds):
        intents[r["truth"]["intent"]].append(p["intent"][0])
    out += ["## Safety", ""]
    quiet = [(r, p) for r, p in zip(rows, preds) if r["truth"]["intent"] not in ORDERS | {"reply"}]
    false_action = [(r, p) for r, p in quiet if p["intent"][0] in ORDERS]
    out.append(f"- Sentences that must not move the robot read as an order: **{len(false_action)} of {len(quiet)}** "
               f"({len(false_action) / max(len(quiet), 1):.2%})")
    stops = intents.get("stop", [])
    if stops:
        out.append(f"- stop recognised: {sum(1 for g in stops if g == 'stop')} of {len(stops)}")
    out += ["", "## Per intent (recognised)", "", "| intent | recognised |", "| --- | --- |"]
    for intent, guesses in sorted(intents.items()):
        out.append(f"| {intent} | {sum(1 for g in guesses if g == intent) / len(guesses):.1%} ({len(guesses)}) |")
    errors = [(r, p) for r, p, ok in zip(rows, preds, exact) if not ok]
    out += ["", f"## Errors ({len(errors)}), first 40", ""]
    for r, p in errors[:40]:
        diff = ", ".join(f"{t}: {r['truth'][t]} -> {p[t][0]} ({p[t][1]:.0%})" for t in KEYS if p[t][0] != r["truth"][t])
        out.append(f"- `{r['lang']}` {r['text']!r}: {diff}")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=HERE / "models" / "bin-intent" / "best")
    parser.add_argument("--sets", nargs="+", default=["eval", "test_unseen", "real"])
    parser.add_argument("--calibrate", action="store_true", help="fit the temperatures on data/calib.jsonl and save them")
    parser.add_argument("--device", default=None)
    parser.add_argument("--half", action="store_true", help="fp16 on the GPU")
    parser.add_argument("--latency", action="store_true", help="time single sentences and a batch")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")

    import torch
    from gliner2.classification import ClassificationConfig, ClassificationSchema, Classifier

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    classifier = Classifier.from_pretrained(str(args.model)).to(device=device).eval()
    if args.half and device == "cuda":
        classifier.to(dtype=torch.float16)
    schema = ClassificationSchema()
    for task, options in TASKS.items():
        schema.single(task, options)
    config = ClassificationConfig(decoder="independent", batch_size=64)

    calibration_file = args.model.parent / "calibration.json"
    if args.calibrate:
        calib = load_jsonl(HERE / "data" / "calib.jsonl")
        fitted = fit_temperatures(calib, score(classifier, schema, config, calib))
        calibration_file.write_text(json.dumps({"temperature": fitted, "fitted_on": "calib.jsonl", "n": len(calib)},
                                               indent=2), encoding="utf-8")
        print("temperatures", fitted)
    temperature = json.loads(calibration_file.read_text())["temperature"] if calibration_file.exists() else {}

    args.results.mkdir(exist_ok=True)
    for name in args.sets:
        rows = load_real(HERE / "test_real.txt") if name == "real" else load_jsonl(HERE / "data" / f"{name}.jsonl")
        logits = score(classifier, schema, config, rows)
        before, after = predictions(rows, logits, {}), predictions(rows, logits, temperature)
        text = report(name + (" (calibrated)" if temperature else ""), rows, after)
        if temperature:
            text += "\n## Without calibration (temperature 1)\n\n| question | ECE before | ECE after |\n| --- | --- | --- |\n"
            for task in KEYS:
                hit = [p[task][0] == r["truth"][task] for r, p in zip(rows, before)]
                hit2 = [p[task][0] == r["truth"][task] for r, p in zip(rows, after)]
                text += (f"| {task} | {ece([p[task][1] for p in before], hit):.3f} | "
                         f"{ece([p[task][1] for p in after], hit2):.3f} |\n")
        (args.results / f"{name}_report.md").write_text(text, encoding="utf-8")
        exact = np.mean([all(p[t][0] == r["truth"][t] for t in KEYS) for r, p in zip(rows, after)])
        print(f"{name:12} {len(rows):5} sentences   all five right: {exact:.1%}")

    if args.latency:
        texts = [r["text"] for r in load_real(HERE / "test_real.txt")]
        classifier.score(texts[0], schema, config=config)
        sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
        sync()
        start = time.perf_counter()
        for text in texts[:60]:
            classifier.classify(text, schema, config=config)
        sync()
        single = (time.perf_counter() - start) / 60 * 1000
        start = time.perf_counter()
        classifier.batch_classify(texts[:128], schema, config=config)
        sync()
        batch = (time.perf_counter() - start) / min(128, len(texts)) * 1000
        memory = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else float("nan")
        print(f"latency on {device}{' fp16' if args.half else ''}: {single:.1f} ms per sentence alone, "
              f"{batch:.1f} ms per sentence in a batch, peak GPU memory {memory:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
