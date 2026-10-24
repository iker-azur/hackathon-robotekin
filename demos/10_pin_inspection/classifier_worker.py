"""JSON stdin/line-event worker used by the Studio; heavy imports stay outside Qt."""

import argparse
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

from paired_data import CAMERAS, load_pairs, split_pairs


def emit(payload):
    print("@@PIN@@" + json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("scan", "prepare", "train", "predict"))
    args = parser.parse_args()
    try:
        settings = json.load(sys.stdin)
        if args.action in ("scan", "prepare"):
            report = load_pairs(settings["data_root"], settings.get("selected_repos"))
            sessions = {}
            for pair in report["pairs"]:
                for repo in {source["repo_id"] for source in pair["sources"]}:
                    sessions.setdefault(repo, Counter())[pair["label"]] += 1
            result = {"event": "data", **report, "sessions": sessions}
            if args.action == "prepare":
                split = split_pairs(report["pairs"], settings.get("seed", 42))
                result["split_counts"] = {key: dict(Counter(pair["label"] for pair in split[key]))
                                          for key in ("train", "validation")}
            emit(result)
        elif args.action == "train":
            from training import train
            result = train(settings, emit)
            emit(result)
            return 0 if result["status"] == "completed" else 130
        else:
            from model import predict
            result = predict(Path(settings["checkpoint"]), {camera: settings.get(camera) for camera in CAMERAS})
            emit({"event": "prediction", **result})
    except Exception as error:  # noqa: BLE001 -- serialize worker errors and retain the complete traceback
        traceback.print_exc()
        emit({"event": "error", "message": str(error)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
