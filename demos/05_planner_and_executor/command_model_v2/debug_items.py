"""What parse() produces per item, including the extracted noun, for one sentence.

    python command_model_v2/debug_items.py "mete las bolas rojas en la caja roja y las azules en la bandeja azul"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from command_model import CommandModel  # noqa: E402


def main() -> int:
    model = CommandModel()
    for sentence in sys.argv[1:]:
        print(f"\n{sentence!r}")
        print(f"  {json.dumps(model.parse(sentence), ensure_ascii=False, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())