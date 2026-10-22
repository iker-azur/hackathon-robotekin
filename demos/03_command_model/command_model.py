"""Use the trained command model: sentence in, robot command out.

    from command_model import CommandModel
    model = CommandModel()                      # loads models/command-model/best
    model.parse("coge dos bolas rojas y mételas en la caja azul")

returns

    {"intent": "pick_and_place",
     "items": [{"kind": "ball", "name": "bolas", "color": "red", "quantity": 2, "position": null}],
     "place": {"kind": "box", "color": "blue"},
     "sort_by": "none",
     "confidence": 0.97}

Two passes of the same model, like Jev's "System One" questions answered in
parallel:
  1. on the sentence: intent, place, place colour, sort criterion, and one
     record per item with the words for its name, colour, quantity, position;
  2. on the item phrases, all at once: what kind of thing each is and whether
     it is one or several.
The colour/number/position words are mapped to values with vocabulary.yaml.

Run it on a sentence:

    python .\\demos\\03_command_model\\command_model.py "hartu bi bola gorri eta sartu kutxan"
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "command-model" / "best"
VOCAB = yaml.safe_load((HERE / "vocabulary.yaml").read_text(encoding="utf-8"))

sys.path.insert(0, str(HERE))
from generate_dataset import COLORS, INTENTS, ITEM_COUNTS, ITEM_KINDS, PLACE_KINDS, SORT_BY  # noqa: E402


def plain(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", word.lower().strip(" .,!?¿¡"))
                   if unicodedata.category(c) != "Mn")


def _word_table(section: str) -> dict[str, object]:
    """Every word of a vocabulary section (all languages and forms) -> its value."""
    table = {}
    for value, languages in VOCAB[section].items():
        for forms in languages.values():
            for form in (forms.values() if isinstance(forms, dict) else forms):
                table[plain(str(form))] = value
    return table


COLOR_WORDS = _word_table("colors")
NUMBER_WORDS = _word_table("numbers")
POSITION_WORDS = _word_table("positions")


# Intents whose command does not use items: the robot ignores them, so they are dropped
# ("cuéntame un chiste" would otherwise come with an item "chiste").
ITEMLESS = {"sort", "clear", "stop", "repeat", "cancel", "answer_yes", "answer_no", "greeting", "off_topic",
            "not_doable"}


def lookup(word: str | None, table: dict):
    """Exact match first, then the longest known word the given one starts with ("gorriarekin" -> red)."""
    if not word:
        return None
    word = plain(word)
    if word in table:
        return table[word]
    starts = [known for known in table if len(known) >= 3 and word.startswith(known)]
    return table[max(starts, key=len)] if starts else None


def in_phrase(phrase: str, table: dict, skip: str | None = None):
    """The first word of an item's own phrase that the table knows ("dos bolas rojas" -> 2 / red).

    Reading the values from each item's phrase keeps them with the right item. `skip` is
    the item's name, so "la naranja" (the fruit) is not read as the colour orange.
    """
    for word in re.findall(r"[^\W\d_]+", phrase):
        if skip and plain(word) == plain(skip):
            continue
        if plain(word) in table:
            return table[plain(word)]
    return None


class CommandModel:
    def __init__(self, path: str | Path = MODEL, device: str | None = None):
        import torch
        from gliner2 import AutoExtractor

        if not Path(path).exists():
            raise FileNotFoundError(f"no model at {path}: run train.py first")
        self.model = AutoExtractor.from_pretrained(str(path))
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.schema = (
            self.model.create_schema()
            .classification("intent", INTENTS)
            .classification("place_kind", PLACE_KINDS)
            .classification("place_color", COLORS + ["none"])
            .classification("sort_by", SORT_BY)
            .structure("item", mode="natural", anchor="mention")
            .field("mention", dtype="str").field("name", dtype="str").field("color_word", dtype="str")
            .field("quantity_word", dtype="str").field("position_word", dtype="str"))
        self.item_tasks = {"item_kind": ITEM_KINDS, "item_count": ITEM_COUNTS}
        self.parse("coge la bola roja")  # the first GPU call is slow: warm up now

    def parse(self, text: str, decisions: dict | None = None) -> dict:
        """Sentence -> command. `decisions` ({task: (label, confidence)} for intent, place_kind,
        place_color, sort_by) replaces this model's own answers for those tasks, e.g. with a
        Laya model's; the items are still extracted here."""
        found = self.model.extract(text, self.schema, include_confidence=True)
        label = {task: found[task]["label"] for task in ("intent", "place_kind", "place_color", "sort_by")}
        sure = {task: found[task]["confidence"] for task in label}
        for task, (value, confidence) in (decisions or {}).items():
            label[task], sure[task] = value, confidence

        def text_of(field):
            return field["text"] if isinstance(field, dict) and field.get("text") else None

        records = [r for r in found.get("item") or [] if text_of(r.get("mention"))]
        if label["intent"] in ITEMLESS:
            records = []
        records.sort(key=lambda r: text.lower().find(text_of(r["mention"]).lower()))  # in the order they were said
        confidences = [sure["intent"]]
        items = []
        if records:
            mentions = [text_of(r["mention"]) for r in records]
            kinds = self.model.batch_classify_text(mentions, self.item_tasks, include_confidence=True)
            for record, kind, mention in zip(records, kinds, mentions):
                name = text_of(record.get("name"))
                # values from the item's own phrase first, the model's extracted words second
                color = in_phrase(mention, COLOR_WORDS, skip=name) or lookup(text_of(record.get("color_word")), COLOR_WORDS)
                count = in_phrase(mention, NUMBER_WORDS, skip=name)
                if count is None:
                    count = lookup(text_of(record.get("quantity_word")), NUMBER_WORDS)
                several = kind["item_count"]["label"] == "several"
                position = (in_phrase(mention, POSITION_WORDS)
                            or lookup(text_of(record.get("position_word")), POSITION_WORDS))
                items.append({
                    "kind": kind["item_kind"]["label"],
                    "name": name,
                    "color": color,
                    "quantity": count if count is not None else ("all" if several else 1),
                    "position": position,
                })
                confidences += [record["mention"]["confidence"], kind["item_kind"]["confidence"]]
        place = None
        if label["place_kind"] != "none":
            place = {"kind": label["place_kind"],
                     "color": None if label["place_color"] == "none" else label["place_color"]}
            confidences.append(sure["place_kind"])
        return {"intent": label["intent"], "items": items, "place": place, "sort_by": label["sort_by"],
                "confidence": round(min(confidences), 3)}


if __name__ == "__main__":
    model = CommandModel()
    for sentence in sys.argv[1:] or ["coge dos bolas rojas y mételas en la caja azul"]:
        start = time.perf_counter()
        command = model.parse(sentence)
        print(f"{sentence!r} ({(time.perf_counter() - start) * 1000:.0f} ms)\n  {json.dumps(command, ensure_ascii=False)}")
