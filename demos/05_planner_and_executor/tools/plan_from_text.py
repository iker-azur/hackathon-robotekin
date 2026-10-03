"""Type a sentence, see what the robot would do about it. No simulator.

Text goes through the v2 command model, then grounding, then the planner, against a recorded
scene JSON. Everything between the sentence and the list of steps is the same code the real
demo runs; only the final motion needs Isaac.

    python tools/plan_from_text.py
    python tools/plan_from_text.py --scene trial013 --text "mete las bolas rojas en la caja roja y las azules en la bandeja azul"
    python tools/plan_from_text.py --lang eu
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import planner  # noqa: E402
from scene import Scene  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
COMMAND_MODEL = ROOT / "command_model_v2" / "command_model.py"

DEFAULTS = [
    ("es", "coge la bola roja"),
    ("es", "coge la bola roja y ponla en la caja azul"),
    ("es", "pon el cubo rojo en la caja azul y la bola verde en la bandeja roja"),
    ("es", "mete las bolas rojas en la caja roja y las azules en la bandeja azul"),
    ("es", "separa las piezas por colores"),
    ("es", "separa las bolas de los cubos"),
    ("es", "pásame la bola verde y déjalo en la bandeja"),
    ("es", "quita las bolas rojas de la mesa y ponlas en la caja azul"),
    ("es", "recoge la mesa y ponlo todo en la caja azul"),
    ("es", "para"),
    ("es", "¿qué ves en la mesa?"),
    ("en", "put the red balls in the red box and the blue ones in the blue tray"),
    ("eu", "jarri pilota gorria kutxa gorrian eta pilota berdea erretilu gorrian"),
]


def load_command_model():
    spec = importlib.util.spec_from_file_location("command_model_v2", COMMAND_MODEL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.CommandModel()


def show(model, scene: Scene, lang: str, text: str) -> None:
    command = model.parse(text)
    plan = planner.plan_command(command, scene)
    print(f"\n  {text!r}  [{lang}]")
    print(f"    intent   : {command.get('intent')}  sort_by={command.get('sort_by')}  "
          f"confidence={command.get('confidence')}")
    for it in command.get("items", []):
        own = (it.get("place") or {}).get("kind")
        print(f"    item     : {it.get('kind')} {it.get('color')} qty={it.get('quantity')}"
              + (f" -> {own}" if own else ""))
    if command.get("place"):
        print(f"    place    : {command['place']}")
    print(f"    plan     : {plan.summary}")
    for i, step in enumerate(plan.steps):
        pose = "  ".join(f"{v:+.4f}" for v in step.pose) if step.pose else "-"
        grip = "open" if step.grip > 0 else "close"
        print(f"      {i:2d} {step.action:9s} {grip:5s} {pose}   {step.note}")
    for line in plan.unresolved:
        print(f"    UNRESOLVED: {line}")
    if plan.question:
        print(f"    ASKS     : {plan.question.get('es') or plan.question}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="trial000")
    parser.add_argument("--text", action="append", help="Repeatable; overrides the defaults")
    parser.add_argument("--lang", default="es", choices=("es", "en", "eu"))
    args = parser.parse_args()

    path = FIXTURES / f"{args.scene}.json"
    if not path.exists():
        print(f"ERROR: no fixture at {path}; have {[p.stem for p in FIXTURES.glob('*.json')]}")
        return 1
    scene = Scene.read(path)

    print(f"scene {args.scene}: {scene.summary()}")
    model = load_command_model()
    for thing in scene.containers:
        print(f"    container #{thing.id} {thing.color} at "
              f"({thing.pos_m[0]:+.3f}, {thing.pos_m[1]:+.3f})")

    if args.text:
        pairs = [(args.lang, t) for t in args.text]
    else:
        pairs = DEFAULTS
    for lang, text in pairs:
        show(model, scene, lang, text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())