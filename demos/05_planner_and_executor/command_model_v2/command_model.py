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


def _place_word_table() -> dict[str, str]:
    """Every receptacle word, in every language and form, to its PLACE_KINDS value.

    ADDED IN THIS COPY, for the per-item destination. ``_word_table`` cannot build this one:
    it flattens a ``{value: {lang: forms}}`` mapping, whereas ``vocabulary.yaml`` puts places
    behind two more levels::

        places:
          train:
            box: {es: [[caja, f]], en: [[box]], eu: [[kutxa, kutxan, kutxara]]}
          test_only:
            box: {es: [[contenedor, m]], ...}

    so there is a ``split`` level and then a noun-level list whose first element is the bare
    stem. Every element of every entry is registered, because the same stem appears in
    several inflections and only the longest match is worth keeping - which is what
    ``lookup`` already does.
    """
    table: dict[str, str] = {}
    for split in ("train", "test_only"):
        for kind, per_lang in (VOCAB["places"].get(split) or {}).items():
            for forms in per_lang.values():
                for entry in forms:
                    for form in (entry if isinstance(entry, list) else [entry]):
                        table[plain(str(form))] = kind
    return table


PLACE_WORDS = _place_word_table()


# Intents whose command does not use items: the robot ignores them, so they are dropped
# ("cuéntame un chiste" would otherwise come with an item "chiste").
# ADDED IN THIS COPY, and corrected against ground.py's NO_ACTION, which is the list that
# actually decides what gets executed:
#
# * "sort" and "clear" REMOVED. They used to be here because neither ever carried items.
#   They both do now: sort/into_matching and sequence/each_own_place attach a destination to
#   each item, and clear/filter names the things to remove. Keeping them here would throw
#   that work away before the schema ever saw it, and `sort` in particular was 100% correct
#   on intent while being structurally incapable of sorting.
# * "question" ADDED. It was missing here but present in ground.NO_ACTION, so "¿Qué ves?" -
#   which carries an item for every object it names - had its items grounded against the
#   scene and could come back asking the person "¿cuál?" about a question that was never
#   ambiguous. Demo 03's own error dump shows exactly that happening on two test sentences.
ITEMLESS = {"stop", "repeat", "cancel", "answer_yes", "answer_no", "question",
            "greeting", "off_topic", "not_doable"}


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


def _destination_in(text: str, segment: str, kind_required: bool = True) -> dict | None:
    """The receptacle a destination segment names, or None.

    ADDED IN THIS COPY. Reads the first receptacle word and the first colour word out of the
    stretch of sentence between one item's mention and the next.

    ``kind_required=False`` returns a colour even when no receptacle word is recognised, for
    the case where the classification head supplied the kind from open vocabulary but not the
    colour. Measured: "pon el cubo rojo en el recuadro rojo" came back as other/None, because
    "recuadro" is not one of the words PLACE_WORDS knows, so there was nothing to anchor the
    colour read to and the whole lookup gave up - dropping "rojo" even though it was right
    there in the sentence.
    """
    if not segment.strip():
        return None
    plain_segment = plain(segment)
    kind = None
    for word in sorted(PLACE_WORDS, key=len, reverse=True):
        if word and word in plain_segment:
            kind = PLACE_WORDS[word]
            break
    if kind is None and kind_required:
        return None
    color = None
    for word in sorted(COLOR_WORDS, key=len, reverse=True):
        if word and word in plain_segment:
            color = COLOR_WORDS[word]
            break
    if kind is None and color is None:
        return None
    return {"kind": kind, "color": color}


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
            .field("quantity_word", dtype="str").field("position_word", dtype="str")
            # ADDED IN THIS COPY. Where THIS item goes, for a command that sends different
            # things to different places. Flat fields on the existing record, never a nested
            # structure: calling .structure() on a record builder emits a sibling record
            # rather than a child, verified against the installed gliner2 2.0.0. And never
            # choices= on a repeated record's field: demo 03's README records that such a
            # field does not depend on which record it belongs to - the second item copied
            # the first one's colour - so the words are extracted and looked up instead.
            .field("destination_kind_word", dtype="str")
            .field("destination_color_word", dtype="str"))
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
                raw_name = text_of(record.get("name"))
                # A repeated record's fields can come from the wrong record. Measured: in
                # "put the red cube in the blue box and the green ball in the red tray" BOTH
                # items came back with name "cube", and in "mete las bolas rojas ... y las
                # azules" both came back "bolas". So an extracted name is only believed when
                # it literally occurs in that item's own mention.
                name = raw_name if (raw_name and plain(raw_name) in plain(mention)) else None
                # values from the item's own phrase first, the model's extracted words second
                color = in_phrase(mention, COLOR_WORDS, skip=name) or lookup(text_of(record.get("color_word")), COLOR_WORDS)
                count = in_phrase(mention, NUMBER_WORDS, skip=name)
                if count is None:
                    count = lookup(text_of(record.get("quantity_word")), NUMBER_WORDS)
                several = kind["item_count"]["label"] == "several"
                position = (in_phrase(mention, POSITION_WORDS)
                            or lookup(text_of(record.get("position_word")), POSITION_WORDS))
                # ADDED IN THIS COPY: this item's own destination, when the sentence gave it
                # one. Both words are read out of the record rather than out of the whole
                # sentence, because a multi-destination sentence carries two colours and two
                # receptacle words and the command-level answer can only hold one of each.
                dest_kind = lookup(text_of(record.get("destination_kind_word")), PLACE_WORDS)
                dest_color = lookup(text_of(record.get("destination_color_word")), COLOR_WORDS)
                dest = {"kind": dest_kind, "color": dest_color} if dest_kind else None
                items.append({
                    "kind": kind["item_kind"]["label"],
                    "name": name,
                    "color": color,
                    "quantity": count if count is not None else ("all" if several else 1),
                    "position": position,
                    "place": dest,
                })
                confidences += [record["mention"]["confidence"], kind["item_kind"]["confidence"]]
        if label["intent"] == "clear":
            # ADDED IN THIS COPY. `clear` had to leave ITEMLESS so that "quita las bolas rojas
            # de la mesa" could carry a filter, but that also lets the model hang an item off
            # a bare "recoge la mesa": measured over the 98 hand-written sentences it invented
            # one for several. A bare "clear the table" means everything, and "everything" names
            # nothing - no noun and no colour. So the test is that, not the kind: "todo" and
            # "everything" classify as kind "any", which no amount of filtering on kind can
            # tell from a real "las bolas". Naming nothing is what distinguishes them.
            items = [i for i in items if not (i["name"] is None and i["color"] is None)]

        # ADDED IN THIS COPY: fill in per-item destinations the model's new fields missed.
        #
        # Measured: after 2 epochs from demo 03's checkpoint, all 12 hand-written
        # multi-destination sentences came back with `place: None` on every item, and the
        # receptacle had been absorbed into the command-level `place` instead - sometimes the
        # wrong one of the two. That is not random: `place_kind` and `place_color` are
        # classification heads that demo 03 trained for hours, while `destination_kind_word` is
        # a new string field inside a repeated record, and the old head simply wins.
        #
        # The model's item mentions ARE correct and correctly ordered, so the destination is
        # read out of the sentence between one mention and the next with the same lookup the
        # rest of this file uses. Restricted to multi-item commands: on a single-item command
        # the item's destination is the command's place anyway, and filling it in would change
        # the shape of every existing command.
        #
        # The model's own fields are consulted first, so if the head is trained further this
        # stops firing.
        if len(items) >= 2:
            # A coordinated phrase shares one noun: "las bolas rojas y las azules" is two
            # groups of balls, not one group of balls and one group of somethings. The second
            # mention has no noun in it at all, so it arrives with no name and its kind falls
            # back to "any". Carry the noun forward - but only when there is no kind to carry,
            # so that a second group that does name itself ("la bola roja ... y dos azules") is
            # left with whatever its own mention said.
            for index in range(1, len(items)):
                if items[index]["kind"] != "any" or items[index]["color"] is None:
                    continue
                if items[index]["name"] is not None:
                    continue
                previous = items[index - 1]
                if previous["name"] is not None and previous["kind"] in ("ball", "cube"):
                    items[index]["kind"] = previous["kind"]
                    items[index]["name"] = previous["name"]

            spans = []
            for record in records:
                mention = text_of(record["mention"])
                start = text.lower().find(mention.lower()) if mention else -1
                spans.append((start, start + len(mention) if mention else -1))
            for index, record in enumerate(records):
                if items[index].get("place") is not None:
                    continue
                begin = spans[index][1]
                end = spans[index + 1][0] if index + 1 < len(spans) and spans[index + 1][0] >= 0 else len(text)
                found = _destination_in(text, text[begin:end] if begin >= 0 else "")
                if found is not None:
                    items[index]["place"] = found

            # A command whose items each have their own destination has no single
            # destination, so the command-level place is None rather than a copy of the first
            # one. The classification heads have only one place_kind and one place_color to
            # write into and will fill them with whichever receptacle they are most confident
            # about, which for "the reds in the red box and the blues in the blue tray" was
            # measured to be sometimes the wrong one of the two.
            own = [i["place"] for i in items if i.get("place")]
            if len(own) == len(items):
                label["place_kind"] = "none"
                label["place_color"] = "none"

            # "Each colour in a receptacle of that colour" is what sorting by colour means,
            # so recognise it from the answer rather than from the verb. Measured: the model
            # returns `sequence` for these, which operationally it is - the executor does the
            # same thing either way - but the difference is exactly what `sort_by` exists to
            # record, and the surface forms of the two templates are nearly identical
            # ("mete las bolas rojas en la caja roja y las azules..." against "mete la bola
            # roja en la caja azul y el cubo verde..."), so the verb carries no signal to learn.
            if (label["intent"] == "sequence"
                    and all(i.get("color") and i["place"] and i["place"].get("color") == i["color"]
                            for i in items)):
                label["intent"] = "sort"
                label["sort_by"] = "color"
        # ADDED IN THIS COPY: fill in the command-level destination the heads left blank.
        #
        # The per-item fallback above is restricted to multi-item commands, because on a
        # single-item command the item's destination IS the command's place. But that left the
        # single-item case resting entirely on `place_kind` and `place_color`, and the head
        # drops the colour when the item and the destination are the same colour: measured,
        # "pon el cubo rojo en la caja roja" parsed as place box/None - the kind was right and
        # "roja" had vanished, so grounding asked which of the two boxes was meant about a box
        # that had been named unambiguously. "pon la bola verde en la bandeja azul" was fine,
        # because there the two colours differ and the head has one to hold.
        #
        # The text is the authority here, not the head: the colour is a word sitting right
        # next to the receptacle word, and reading it out of the sentence is the same lookup
        # the rest of this file already uses. Only fills a blank, so a head that does answer
        # keeps its answer.
        if label["place_kind"] != "none" and label["place_color"] == "none":
            tail = ""
            if records:
                mention = text_of(records[-1]["mention"]) or ""
                begin = text.lower().rfind(mention.lower())
                tail = text[begin + len(mention):] if mention and begin >= 0 else ""
            named = _destination_in(text, tail, kind_required=False)
            if named is not None and named["color"]:
                label["place_color"] = named["color"]

        place = None
        if label["place_kind"] != "none":
            place = {"kind": label["place_kind"],
                     "color": None if label["place_color"] == "none" else label["place_color"]}
            confidences.append(sure["place_kind"])

        if label["intent"] == "sort" and not any(i.get("place") for i in items):
            # A sort with no destination named per group carries no ITEMS for the planner to
            # pair with receptacles: "separa las piezas por colores" is an instruction about
            # the whole table, and the dataset trains it with an empty item list; "separa las
            # bolas rojas en la caja roja y las azules en la bandeja azul" is two groups each
            # with somewhere to go, and those are kept.
            #
            # Without this the model volunteers items for the first form - measured, it
            # returned a ball group and a cube group for "separa las bolas de los cubos",
            # where the training data has none.
            #
            # CHANGED HERE: what the sentence DID say is kept, but not in `items`. "separa los
            # cubos por colores" names no destinations and does say what to sort, and
            # throwing that away made the planner sort the red BALL as well, putting a cube
            # and a ball into the same 5.6 cm cavity. It is carried as `sort_filter` instead of
            # as items, because putting it in items costs 3 points of exact-command accuracy
            # (measured: 84.6% to 81.6% overall, 85.7% to 82.7% on the legacy set) against a
            # ground truth that genuinely has no items there. Only a filter that names neither
            # kind nor colour is dropped, which is the same test "clear" uses to tell a real
            # group from "everything".
            sort_filter = [i for i in items
                           if i.get("kind") not in (None, "", "any") or i.get("color")]
            items = []
        else:
            sort_filter = []
        # `confidence` above is the MINIMUM over everything, which is the honest one number for
        # "how sure was it, overall" - it is the weakest link, not an average. It is also useless
        # for showing a person *where* the doubt is: a sort at 0.99 and a pick at 0.99 read the
        # same, and so does a sort at 0.99 whose destination was a guess.
        #
        # So the per-task confidences go out too, under `confidences`. They are the model's own
        # scores for each field it decided, unmodified - the minimum is derived from them rather
        # than the other way round, so this adds no new judgement and cannot disagree with the
        # number already published. The dashboard shows them beside each option it selected.
        return {"intent": label["intent"], "items": items, "place": place, "sort_by": label["sort_by"],
                "sort_filter": sort_filter,
                "confidence": round(min(confidences), 3),
                "confidences": {task: round(float(value), 3) for task, value in sure.items()}}


if __name__ == "__main__":
    model = CommandModel()
    for sentence in sys.argv[1:] or ["coge dos bolas rojas y mételas en la caja azul"]:
        start = time.perf_counter()
        command = model.parse(sentence)
        print(f"{sentence!r} ({(time.perf_counter() - start) * 1000:.0f} ms)\n  {json.dumps(command, ensure_ascii=False)}")
