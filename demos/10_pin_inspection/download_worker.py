"""Internal download process: receives the window's settings through stdin."""

import json
from pathlib import Path
import sys

from huggingface_hub import HfApi, snapshot_download

from download_datasets import download, label_map


def main():
    settings = json.load(sys.stdin)
    aliases = label_map(settings["with_pin_labels"], settings["without_pin_labels"])
    return download(settings["owner"], Path(settings["output"]), aliases, HfApi(), snapshot_download)


if __name__ == "__main__":
    raise SystemExit(main())
