"""Compare a v2 error dump against demo 03's, separating the intentional contract change.

Demo 03 treated `sort` and `clear` as intents that carry no items, and encoded that in its
hand-written test set. This demo deliberately gave both intents items - `sort` needs one per
colour group with its own destination, `clear` needs a filter - so those hand-written rows
now disagree with the model for a reason that is not an error at all.

This separates the two: errors that follow from the changed contract, and errors that would
be errors under either contract.

    python command_model_v2/compare_errors.py test_legacy_only
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Intents whose item list this demo deliberately changed.
CHANGED = ("sort", "clear")


def brief(item) -> tuple:
    return (item.get("kind"), item.get("color"), item.get("quantity"), item.get("position"))


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "test_real"
    path = RESULTS / f"{name}_errors.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    print(f"{len(rows)} errors in {path.name}")

    contract, real = [], []
    for row in rows:
        is_contract = (row["expected"]["intent"] in CHANGED
                       and row["wrong"] == ["items"]
                       and row["expected"]["items"] == []
                       and bool(row["got"]["items"]))
        (contract if is_contract else real).append(row)

    print(f"  from the changed sort/clear item contract: {len(contract)}")
    print(f"  errors under either contract              : {len(real)}")
    print()
    print("--- errors under either contract ---")
    by_intent = collections.Counter(r["expected"]["intent"] for r in real)
    print("  per intent:", dict(by_intent.most_common()))
    fields = collections.Counter(w for r in real for w in r["wrong"])
    print("  per field :", dict(fields.most_common()))
    print()
    for row in real[:16]:
        want = [brief(i) for i in row["expected"]["items"]]
        got = [brief(i) for i in row["got"]["items"]]
        print(f"  {row['input'][:60]}")
        print(f"     wrong={row['wrong']}  want {want}  got {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())