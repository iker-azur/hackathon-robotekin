"""Use the trained model: a sentence in, a probability for every option out, and the pieces linked to a box.

    from intent_model import BinIntentModel
    model = BinIntentModel()                                   # loads models/bin-intent/best (+ calibration.json)
    model = BinIntentModel("IkerAzurmendi/bin-intent-gliner2")  # or downloads it from the Hugging Face hub
    plan = model.plan("Mueve las piezas con perno a la caja grande")
    plan.show()

    intent   move 98%  sort 1%  ask 0%  ...
    feature  bolt 99%  any 1%  no bolt 0%
    box      big 97%  white 2%  other 1%  none 0%
    rule     pieces [with bolt, all] -> big box   (confidence 97%, box said)

Jev style: the sentence is read once and every question (``intent``, ``feature``, ``quality``, ``box``,
``count``) is answered as a probability for each of its options, from one forward pass of the encoder
(``gliner2.classification.Classifier``). Nothing is generated, so it cannot answer with an option that is not on
the list. The probabilities are calibrated with one temperature per question (``calibration.json``, fitted by
``evaluate.py --calibrate``), so "90%" means about nine times out of ten.

``plan`` links the pieces a sentence names to the box it names. When the sentence names no box, ``routes`` (the
cell's own piece -> box convention, empty by default) can supply it:

    BinIntentModel(routes={("bolt", "any"): "big", ("any", "bad"): "white"})

A sentence with two instructions ("the ones with a bolt to the big box and the rest to the white box") is read as
``both``; ``plan`` then cuts it at the conjunction, scores every possible cut in one batch and keeps the cut where
both halves are orders. It must be loaded before Isaac Sim starts (torch after Kit crashes the process).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from grammar_common import TASKS

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "bin-intent" / "best"
ORDERS = ("move", "sort", "pick")
CONJUNCTION = re.compile(r"\s*(?:,\s*(?:y|and|eta)\b|;|\b(?:y luego|y después|y despues|and then|and after that)\b|"
                         r"\b(?:y|and|eta)\b)\s*", re.IGNORECASE)
SURE_ENOUGH = 0.85  # below this, a plan asks instead of acting


@dataclass
class Rule:
    """Which pieces go to which box, with the probability of each part."""

    feature: str
    quality: str
    count: str
    box: str
    p_feature: float
    p_quality: float
    p_count: float
    p_box: float
    box_options: dict            # probability of every box option, e.g. {"big": 0.97, "white": 0.02, ...}
    box_source: str = "said"     # "said" | "route" (taken from the cell's routes) | "missing"
    conflicts_with_route: bool = False

    @property
    def confidence(self) -> float:
        """The weakest part: how sure the sentence was about the part it was least sure of."""
        return min(self.p_feature, self.p_quality, self.p_count, self.p_box if self.box_source == "said" else 1.0)

    def describe(self) -> str:
        feature = {"bolt": "with bolt", "no bolt": "without bolt", "any": ""}[self.feature]
        quality = {"good": "good", "bad": "bad", "any": ""}[self.quality]
        who = ", ".join(w for w in (quality, feature, self.count) if w)
        box = {"big": "big box", "white": "white box", "other": "a box that is not in this cell", "none": "no box"}[self.box]
        return f"pieces [{who}] -> {box}"


@dataclass
class Reading:
    text: str
    probabilities: dict          # task -> {option: probability}

    def top(self, task: str):
        options = self.probabilities[task]
        label = max(options, key=options.get)
        return label, options[label]

    @property
    def intent(self) -> str:
        return self.top("intent")[0]

    @property
    def confidence(self) -> float:
        """The weakest decision among the questions that matter for this intent."""
        tasks = ["intent"] + (["feature", "quality", "box", "count"] if self.intent in ORDERS else [])
        return min(self.top(task)[1] for task in tasks)

    def show(self, width: int = 4) -> None:
        print(f"{self.text!r}")
        for task, options in self.probabilities.items():
            ranked = sorted(options.items(), key=lambda kv: -kv[1])[:width]
            print(f"  {task:8}" + "  ".join(f"{label} {p:.0%}" for label, p in ranked))


@dataclass
class Plan:
    status: str                  # ok | ask_box | unsure | no_action | stop | cancel | repeat | answer | not_a_cell_box
    reading: Reading
    rules: list = field(default_factory=list)
    note: str = ""

    def show(self) -> None:
        self.reading.show()
        for rule in self.rules:
            print(f"  rule    {rule.describe()}   (confidence {rule.confidence:.0%}, box {rule.box_source})")
        print(f"  status  {self.status}" + (f" - {self.note}" if self.note else ""))


class BinIntentModel:
    def __init__(self, path: str | Path = DEFAULT_MODEL, device: str | None = None, routes: dict | None = None,
                 half: bool = True) -> None:
        import torch
        from gliner2.classification import ClassificationConfig, ClassificationSchema, Classifier

        if re.fullmatch(r"[\w.-]+/[\w.-]+", str(path)) and not Path(path).exists():  # a Hugging Face repo id
            try:
                import truststore  # the Windows certificate store, for downloads behind a proxy

                truststore.inject_into_ssl()
            except ImportError:
                pass
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(str(path))) / "best"
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no model at {path}: run train.py first")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = Classifier.from_pretrained(str(path)).to(device=device).eval()
        if half and device == "cuda":
            self.classifier.to(dtype=torch.float16)
        calibration = path.parent / "calibration.json"
        self.temperature = (json.loads(calibration.read_text(encoding="utf-8"))["temperature"]
                            if calibration.exists() else {})
        self.schema = ClassificationSchema()
        for task, options in TASKS.items():
            self.schema.single(task, options, temperature=float(self.temperature.get(task, 1.0)))
        self.config = ClassificationConfig(decoder="independent", batch_size=64)
        self.routes = dict(routes or {})
        self.read("move the bad pieces to the white box")  # the first GPU call is slow: warm up now

    # -- reading ------------------------------------------------------------------------------------------

    def read_many(self, texts: list[str]) -> list[Reading]:
        results = self.classifier.batch_classify(list(texts), self.schema, config=self.config)
        return [Reading(text, {task: dict(result.probabilities(task)) for task in TASKS})
                for text, result in zip(texts, results)]

    def read(self, text: str) -> Reading:
        return self.read_many([text])[0]

    # -- linking pieces to boxes --------------------------------------------------------------------------

    def _route(self, feature: str, quality: str):
        return self.routes.get((feature, quality)) or self.routes.get((feature, "any")) \
            or self.routes.get(("any", quality))

    def _rule(self, reading: Reading) -> Rule:
        feature, p_feature = reading.top("feature")
        quality, p_quality = reading.top("quality")
        count, p_count = reading.top("count")
        box, p_box = reading.top("box")
        routed = self._route(feature, quality) if (feature, quality) != ("any", "any") else None
        source, conflict = "said", False
        if box == "none":
            if routed:
                box, p_box, source = routed, 1.0, "route"
            else:
                source = "missing"
        elif box in ("big", "white"):
            conflict = bool(routed and routed != box)
        return Rule(feature, quality, count, box, p_feature, p_quality, p_count, p_box,
                    dict(reading.probabilities["box"]), source, conflict)

    def plan(self, text: str) -> Plan:
        return self._plan(self.read(text))

    def _plan(self, reading: Reading) -> Plan:
        intent, p_intent = reading.top("intent")
        if intent in ("stop", "cancel", "repeat"):
            return Plan(intent, reading)
        if intent == "reply":
            return Plan("answer", reading)
        if intent == "both":
            return self._plan_both(reading)
        if intent not in ORDERS:
            return Plan("no_action", reading, note=intent)
        rule = self._rule(reading)
        if p_intent < SURE_ENOUGH or rule.confidence < SURE_ENOUGH:
            return Plan("unsure", reading, [rule], "ask the person to say it again")
        if intent == "pick" and rule.box != "none":
            intent = "move"  # "pick" never carries a destination: a named box means the person wants it put there
        if intent == "pick":
            return Plan("ok", reading, [rule])
        if rule.box == "other":
            return Plan("not_a_cell_box", reading, [rule], "that box is not one of the two in this cell")
        if rule.box == "none":
            return Plan("ask_box", reading, [rule], "which box?")
        return Plan("ok", reading, [rule], "conflicts with the cell's routes" if rule.conflicts_with_route else "")

    def _plan_both(self, reading: Reading) -> Plan:
        text = reading.text
        cuts = [m for m in CONJUNCTION.finditer(text) if text[:m.start()].strip() and text[m.end():].strip()]
        if not cuts:
            return Plan("unsure", reading, note="two instructions, but no place to cut the sentence")
        halves = [(text[:m.start()].strip(" ,;"), text[m.end():].strip(" ,;")) for m in cuts]
        readings = self.read_many([part for pair in halves for part in pair])
        best = None
        for left, right in zip(readings[0::2], readings[1::2]):
            if left.intent in ORDERS and right.intent in ORDERS + ("reply",):
                score = min(left.confidence, right.confidence)
                if best is None or score > best[0]:
                    best = (score, left, right)
        if best is None:
            return Plan("unsure", reading, note="could not split the two instructions")
        rules = [self._rule(best[1]), self._rule(best[2])]
        sure = min(rule.confidence for rule in rules) >= SURE_ENOUGH and all(r.box in ("big", "white") for r in rules)
        return Plan("ok" if sure else "unsure", reading, rules, "two rules")


if __name__ == "__main__":
    import sys

    model = BinIntentModel()
    for sentence in sys.argv[1:] or ["Mueve las piezas con perno a la caja grande",
                                     "Clasifica las piezas malas en la caja blanca"]:
        model.plan(sentence).show()
