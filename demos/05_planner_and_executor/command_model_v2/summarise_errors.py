"""Summarise an evaluate.py error dump: which intents fail, which fields, and why.

    python command_model_v2/summarise_errors.py test_legacy_only
    python command_model_v2/summarise_errors.py test_real
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def brief(item) -> tuple:
    return (item.get("kind"), item.get("color"), item.get("quantity"), item.get("position"),
            (item.get("place") or {}).get("kind"), (item.get("place") or {}).get("color"))


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "test_real"
    path = RESULTS / f"{name}_errors.jsonl"
    if not path.exists():
        print(f"no error dump at {path}")
        return 1
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    print(f"{len(rows)} errors in {path.name}")
    print()

    print("errors per expected intent:")
    for intent, count in collections.Counter(r["expected"]["intent"] for r in rows).most_common():
        print(f"  {intent:16s} {count}")
    print()

    fields = collections.Counter()
    for row in rows:
        for name_ in row["wrong"]:
            fields[name_] += 1
    print("wrong fields:", dict(fields.most_common()))
    print()

    only_items = [r for r in rows if r["wrong"] == ["items"]]
    print(f"--- the {len(only_items)} items-only errors (kind, colour, qty, position, place) ---")
    for row in only_items[:14]:
        print(f"  {row['input'][:66]}")
        print(f"     want {[brief(i) for i in row['expected']['items']]}")
        print(f"     got  {[brief(i) for i in row['got']['items']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())