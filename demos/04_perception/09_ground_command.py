"""Say a command, look at the table, and either name the objects or ask which one.

Demo 08 ends a sentence with a command and a spoken confirmation, but it never checks
the command against anything real, because there was nothing real to check against.
This script closes that loop using the scene JSON from ``09_camera_to_scene.py`` and the
command model from demo 03, which are the two halves of the project's loop:

    what was said  ->  a command  ->  what is on the table  ->  an object, or a question

Three situations are shown, and the third is the one worth watching:

    "coge la bola verde"            one green ball, so the robot can act
    "coge la bola roja"             two red balls, so it asks instead of guessing
    "coge la bola roja de la izquierda"
                                     two red balls and a position word, so it acts

The answer to the robot's own question goes back through the same command model, which
is trained on exactly this: ``answer_value`` for *"La de la izquierda"*
(``laya_questions.py``). So the whole dialogue is two turns of a model that already
exists, with no new training.

The microphone belongs to demos/02_voice/06_voice_command.py and the voice to
demos/03_command_model/08_voice_hybrid.py, so this script takes text. It is the piece
that has to work when there is no microphone, and it is the piece demo 12 will call.

Run:

    python .\\demos\\04_perception\\09_ground_command.py --scene out\\scenes\\trial000.json
    python .\\demos\\04_perception\\09_ground_command.py --scene out\\scenes\\trial000.json --text "coge la bola roja" --lang es
    python .\\demos\\04_perception\\09_ground_command.py --scene out\\scenes\\trial000.json --text "coge la bola roja" --answer "la de la izquierda" --lang es
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import describe  # noqa: E402
import ground as grounding  # noqa: E402
from scene import Scene  # noqa: E402

HERE = Path(__file__).resolve().parent
COMMAND_MODEL = HERE.parent / "03_command_model"

# What to try when neither --text nor --answer is given: the three situations above, plus
# the same ambiguous command in Basque and in English, because the ambiguity is the
# interesting part and it should not look different in the three languages.
DEFAULTS = [
    ("es", "coge la bola verde y ponla en la caja azul"),
    ("es", "coge la bola roja"),
    ("es", "coge la bola roja de la izquierda"),
    ("eu", "hartu bola gorria"),
    ("en", "pick up the red ball"),
]

EMPTY_COMMAND = {"intent": None, "items": [], "place": None, "sort_by": "none"}


def load_command_model():
    """Demo 03's CommandModel, imported by path because there are no __init__.py files.

    Its module import pulls in ``yaml`` and ``gliner2``, and constructing it loads a
    1.1 GB checkpoint, so it is only called when a sentence really has to be understood.
    """
    path = COMMAND_MODEL / "command_model.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run demos/03_command_model/train.py first")
    spec = importlib.util.spec_from_file_location("command_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(COMMAND_MODEL))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(COMMAND_MODEL))
    return module.CommandModel()


def understand(model, text: str) -> dict:
    """A command for a sentence, or a clearly marked empty one when there is no model."""
    if model is None:
        return dict(EMPTY_COMMAND)
    return model.parse(text)


def slim(command: dict) -> dict:
    """The command without the fields grounding does not use, to keep the log readable."""
    return {k: v for k, v in command.items() if k in ("intent", "items", "place", "confidence")}


def reply_for(command: dict, scene: Scene, result: dict, lang: str) -> str:
    """The sentence the robot would say.

    A question it had to ask wins over everything else, because that is what the person
    has to act on. Otherwise a question about the scene is answered from the scene, which
    is what replaces demo 08's "todavía no puedo ver la mesa".
    """
    for decided in result["items"]:
        if decided.question:
            return decided.question.get(lang, "")
    if result["place"].question:
        return result["place"].question.get(lang, "")
    if command.get("intent") == "question":
        return describe.answer(scene, command, lang)
    targets = ", ".join(f"#{i}" for group in result["executable"] for i in group)
    return {
        "es": f"Vale, cojo {targets or 'nada'}.",
        "en": f"OK, picking up {targets or 'nothing'}.",
        "eu": f"Ados, {targets or 'ezer'} hartuko dut.",
    }[lang]


def show(command: dict, scene: Scene, result: dict, lang: str, heard: str) -> None:
    """Print one turn of the dialogue, the way demo 12's dashboard will show it."""
    print(f"  heard    : {heard}")
    print(f"  command  : {json.dumps(slim(command), ensure_ascii=False)}")
    for item, decided in zip(command.get("items", []), result["items"]):
        named = ", ".join(f"#{t.id}" for t in decided.candidates) or "none"
        print(
            f"  grounded : {grounding.describe_item(item, lang)} -> {decided.status} "
            f"[{decided.target_ids or named}]"
        )
    if result["place"].status not in ("no_place",):
        where = f"#{result['place'].place.id}" if result["place"].place else result["place"].candidates
        print(f"  place    : {result['place'].status} {where}")
    print(f"  reply    : {reply_for(command, scene, result, lang)}")
    print(f"  targets  : {result['executable']} of {len(command.get('items', []))} item(s)")
    print("")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", type=Path, required=True,
                        help="A scene JSON written by 09_camera_to_scene.py")
    parser.add_argument("--text", default=None, help="One sentence instead of the built-in examples")
    parser.add_argument("--answer", default=None,
                        help="The person's reply to a question the robot asked, as a second turn")
    parser.add_argument("--lang", choices=["es", "en", "eu"], default="es",
                        help="Language of the reply (default: es)")
    parser.add_argument("--command", default=None,
                        help="A command JSON, to bypass the model entirely")
    parser.add_argument("--command-file", type=Path, default=None,
                        help="The same JSON in a file. Easier than --command on Windows, "
                             "where a JSON blob has to survive the shell's quoting")
    parser.add_argument("--all-langs", action="store_true",
                        help="Print the robot's question in all three languages")
    args = parser.parse_args()

    if not args.scene.exists():
        print(f"ERROR: no scene at {args.scene}. Run 09_camera_to_scene.py first.")
        return 1
    scene = Scene.read(args.scene)

    turns: list[tuple[str, str]] = []
    if args.answer:
        turns.append((args.lang, args.answer))
    elif args.text:
        turns.append((args.lang, args.text))
    else:
        turns.extend(DEFAULTS)

    if args.command_file:
        if not args.command_file.exists():
            print(f"ERROR: no command file at {args.command_file}")
            return 1
        try:
            command = json.loads(args.command_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            print(f"ERROR: {args.command_file} is not valid JSON: {error}")
            return 1
        result = grounding.ground_command(scene, command)
        show(command, scene, result, args.lang, f"(given with {args.command_file.name})")
        return 0

    if args.command:
        try:
            command = json.loads(args.command)
        except json.JSONDecodeError as error:
            print(f"ERROR: --command is not valid JSON: {error}")
            return 1
        result = grounding.ground_command(scene, command)
        show(command, scene, result, args.lang, "(given with --command)")
        return 0

    model = None
    try:
        model = load_command_model()
    except Exception as error:
        print(f"ERROR: the command model could not be loaded: {error}")
        print("       run demos/03_command_model/train.py, or pass --command with a command")
        print("       JSON to try the grounding on its own")
        return 1

    print("")
    print(scene.summary())
    print("")

    for heard, text in turns:
        command = understand(model, text)
        result = grounding.ground_command(scene, command)
        show(command, scene, result, args.lang, text)
        if args.all_langs:
            for decided in result["items"]:
                if decided.question:
                    for lang, said in decided.question.items():
                        print(f"    [{lang}] {said}")
            print("")
        ambiguous = [d for d in result["items"] if d.status == "ambiguous"]
        if ambiguous:
            word = grounding.options_of(scene, ambiguous[0].candidates)
            hint = f"la de la {word['position']}" if word else f"object #{ambiguous[0].candidates[0].id}"
            print(f"  next turn: answer the question, e.g. --answer \"{hint}\"")
            print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
