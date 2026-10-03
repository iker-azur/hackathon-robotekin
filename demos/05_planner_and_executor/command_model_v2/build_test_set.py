"""Build command_model_v2/data/test_real.jsonl: demo 03's test set plus the v2 cases.

Demo 03's 98 hand-written sentences are copied verbatim so the v2 model is scored on
exactly the same test set, and the numbers are comparable with
``demos/03_command_model/results/test_real_report.md`` (85.7% exact). They are NOT
retrained on, by design: this file is the only evaluation data that was written by a
person rather than generated.

On top of them this adds the sentences demo 03's schema could not represent at all - a sort
into two matching coloured receptacles, two items to two different places, a give that ends
in a tray, and a clear with a filter - in all three languages, plus a handful of older forms
to catch a regression in them.

The expected commands are written by hand from what the sentence means, not generated, which
is the point: a bug in the generator's labelling shows up here as a failing case.

    python command_model_v2/build_test_set.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DEMO_03 = HERE.parent.parent / "03_command_model" / "data" / "test_real.jsonl"


def case(text, lang, intent, items=None, place=None, sort_by="none", source="hand-written"):
    return {
        "input": text,
        "meta": {
            "lang": lang,
            "source": source,
            "command": {
                "intent": intent,
                "items": items or [],
                "place": place,
                "sort_by": sort_by,
            },
        },
    }


def item(kind, color=None, quantity=1, position=None, place=None, name=None):
    return {"kind": kind, "name": name, "color": color, "quantity": quantity,
            "position": position, "place": place}


def dest(kind, color=None):
    return {"kind": kind, "color": color}


NEW = [
    # --- sort into two matching coloured receptacles -------------------------------
    case("mete las bolas rojas en la caja roja y las azules en la bandeja azul", "es", "sort",
         [item("ball", "red", "all", place=dest("box", "red")),
          item("ball", "blue", "all", place=dest("tray", "blue"))], sort_by="color"),
    case("pon los cubos verdes en la bandeja verde y los rojos en el cesto rojo", "es", "sort",
         [item("cube", "green", "all", place=dest("tray", "green")),
          item("cube", "red", "all", place=dest("basket", "red"))], sort_by="color"),
    case("separa por colores: las piezas azules dentro de la caja azul y las rojas en la bandeja roja", "es", "sort",
         [item("any", "blue", "all", place=dest("box", "blue")),
          item("any", "red", "all", place=dest("tray", "red"))], sort_by="color"),
    case("put the red balls in the red box and the blue ones in the blue tray", "en", "sort",
         [item("ball", "red", "all", place=dest("box", "red")),
          item("ball", "blue", "all", place=dest("tray", "blue"))], sort_by="color"),
    case("sort the green cubes in the green basket and the red cubes into the red tray", "en", "sort",
         [item("cube", "green", "all", place=dest("basket", "green")),
          item("cube", "red", "all", place=dest("tray", "red"))], sort_by="color"),
    case("sartu pilota gorriak kutxa gorrian eta pilota urdinak erretilu urdinean", "eu", "sort",
         [item("ball", "red", "all", place=dest("box", "red")),
          item("ball", "blue", "all", place=dest("tray", "blue"))], sort_by="color"),
    case("jarri pilota berdeak saski berdean eta pilota gorriak kutxa gorrian", "eu", "sort",
         [item("ball", "green", "all", place=dest("basket", "green")),
          item("ball", "red", "all", place=dest("box", "red"))], sort_by="color"),

    # --- two items, two destinations ----------------------------------------------
    case("pon el cubo rojo en la caja azul y la bola verde en la bandeja roja", "es", "sequence",
         [item("cube", "red", 1, place=dest("box", "blue")),
          item("ball", "green", 1, place=dest("tray", "red"))]),
    case("mete la bola roja en la caja azul y el cubo verde en el cesto verde", "es", "sequence",
         [item("ball", "red", 1, place=dest("box", "blue")),
          item("cube", "green", 1, place=dest("basket", "green"))]),
    case("put the red cube in the blue box and the green ball in the red tray", "en", "sequence",
         [item("cube", "red", 1, place=dest("box", "blue")),
          item("ball", "green", 1, place=dest("tray", "red"))]),
    case("place the blue ball in the red box and the red cube in the green tray", "en", "sequence",
         [item("ball", "blue", 1, place=dest("box", "red")),
          item("cube", "red", 1, place=dest("tray", "green"))]),
    case("jarri kubo gorria kutxa urdinean eta pilota berdea erretilu gorrian", "eu", "sequence",
         [item("cube", "red", 1, place=dest("box", "blue")),
          item("ball", "green", 1, place=dest("tray", "red"))]),

    # --- give, ending in a tray ----------------------------------------------------
    case("pásame la bola roja y déjalo en la bandeja", "es", "give",
         [item("ball", "red")], place=dest("tray")),
    case("dame el cubo azul y déjalo en la bandeja", "es", "give",
         [item("cube", "blue")], place=dest("tray")),
    case("hand me the red ball and leave it in the tray", "en", "give",
         [item("ball", "red")], place=dest("tray")),
    case("bring me the blue cube and leave it in the tray", "en", "give",
         [item("cube", "blue")], place=dest("tray")),
    case("pasa iezadazu bola gorria eta utzi erretilura", "eu", "give",
         [item("ball", "red")], place=dest("tray")),

    # --- clear with a filter -------------------------------------------------------
    case("quita las bolas rojas de la mesa y ponlas en la caja azul", "es", "clear",
         [item("ball", "red", "all")], place=dest("box", "blue")),
    case("take the red balls off the table and put them in the blue box", "en", "clear",
         [item("ball", "red", "all")], place=dest("box", "blue")),
    case("kendu pilota gorriak mahaitik eta jarri kutxa urdinean", "eu", "clear",
         [item("ball", "red", "all")], place=dest("box", "blue")),

    # --- regressions on the forms that already worked ------------------------------
    case("coge la bola roja", "es", "pick", [item("ball", "red")]),
    case("coge la bola roja y ponla en la caja azul", "es", "pick_and_place",
         [item("ball", "red")], place=dest("box", "blue")),
    case("coge las dos bolas rojas y ponlas en la caja azul", "es", "sequence",
         [item("ball", "red", 2)], place=dest("box", "blue")),
    case("recoge la mesa", "es", "clear"),
    case("mete todo en la caja azul", "es", "clear", place=dest("box", "blue")),
    case("separa las bolas de los cubos", "es", "sort", sort_by="kind"),
    case("clasifica las piezas por color", "es", "sort", sort_by="color"),
    case("pick up the red ball", "en", "pick", [item("ball", "red")]),
    case("hartu bola gorria", "eu", "pick", [item("ball", "red")]),
    case("para", "es", "stop"),
    case("stop", "en", "stop"),
    case("¿qué ves en la mesa?", "es", "question"),
    case("la de la izquierda", "es", "answer_value", [item("any", None)]),
    case("sí", "es", "answer_yes"),
    case("no", "es", "answer_no"),
    case("otra vez", "es", "repeat"),
    case("cancela", "es", "cancel"),
    case("hola", "es", "greeting"),
]


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    lines: list[dict] = []
    copied = 0
    if DEMO_03.exists():
        for raw in DEMO_03.open(encoding="utf-8"):
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            # Verbatim, with no adjustment. An earlier version rewrote the `sort` rows to
            # carry a group item, on the grounds that this demo gave `sort` items; it turned
            # out to be `command_model.parse` that needed adjusting instead, and rewriting the
            # expectations only moved the goalposts. A hand-written test set that is edited to
            # agree with the model is not a measurement.
            record["meta"]["source"] = "demo03"
            lines.append(record)
            copied += 1
    else:
        print(f"WARNING: {DEMO_03} not found, writing the v2 cases only")

    lines.extend(NEW)
    with (DATA / "test_real.jsonl").open("w", encoding="utf-8") as sink:
        for record in lines:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    # The two populations, written out separately so each can be scored on its own. Mixing
    # them into one headline number hides both: the 98 legacy sentences measure whether this
    # demo broke what demo 03 already did, and the 38 new ones measure whether the new
    # capability works at all. One number over both would move for either reason.
    subsets = {
        "test_legacy_only.jsonl": [r for r in lines if r["meta"]["source"].startswith("demo03")],
        "test_v2_only.jsonl": [r for r in lines if r["meta"]["source"] == "hand-written"],
    }
    for name, subset in subsets.items():
        with (DATA / name).open("w", encoding="utf-8") as sink:
            for record in subset:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    intents: dict[str, int] = {}
    for record in lines:
        name = record["meta"]["command"]["intent"]
        intents[name] = intents.get(name, 0) + 1
    print(f"wrote {len(lines)} sentences to {DATA / 'test_real.jsonl'}")
    print(f"  {copied} copied verbatim from demo 03, {len(NEW)} new for v2")
    for name, subset in subsets.items():
        print(f"  {name}: {len(subset)} sentences")
    print(f"  intents: {dict(sorted(intents.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())