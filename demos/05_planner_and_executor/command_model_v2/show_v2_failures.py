"""What the v2 model actually returns for the multi-destination sentences, side by side.

The accuracy number says 44.7% on the new sentences; this says why, one sentence at a time.
Only the sentences that carry a per-item destination are shown, because those are the ones
the new schema exists for.

    python command_model_v2/show_v2_failures.py
    python command_model_v2/show_v2_failures.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from command_model import CommandModel  # noqa: E402

DATA = HERE / "data"


def brief(item):
    place = item.get("place") or {}
    return (item.get("kind"), item.get("color"), place.get("kind"), place.get("color"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="Include the sentences that pass")
    parser.add_argument("--file", default="test_v2_only.jsonl")
    args = parser.parse_args()

    rows = [json.loads(line) for line in (DATA / args.file).open(encoding="utf-8")]
    rows = [r for r in rows if any(i.get("place") for i in r["meta"]["command"]["items"])]
    print(f"{len(rows)} sentence(s) with a per-item destination\n")

    model = CommandModel()
    right = 0
    for row in rows:
        got = model.parse(row["input"])
        want = row["meta"]["command"]
        ok = (got["intent"] == want["intent"]
              and [brief(i) for i in got["items"]] == [brief(i) for i in want["items"]]
              and (got["place"] or {}) == (want["place"] or {}))
        right += ok
        if ok and not args.all:
            continue
        mark = "OK  " if ok else "FAIL"
        print(f"{mark} [{row['meta']['lang']}] {row['input']}")
        print(f"       want intent={want['intent']} items={[brief(i) for i in want['items']]} "
              f"place={want['place']}")
        print(f"       got  intent={got['intent']} items={[brief(i) for i in got['items']]} "
              f"place={got['place']}")
    print(f"\ncorrect on the per-item destination: {right}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())