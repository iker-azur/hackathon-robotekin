"""Turn the command dataset into Laya training items.

Laya is trained on "items": one per (sentence, question), each the tokens of
[sentence + question + options] with the position of every option's marker,
and a target probability for each option. This script builds them from the
JSON Lines files of ../generate_dataset.py (and ../add_asr_noise.py), asking
the four questions of laya_questions.py about every sentence.

Laya's own training data has *soft* targets (a teacher model's probabilities);
ours has one right answer, so the target puts 0.9 on it and spreads 0.1 over the
other options (label smoothing). The calibration step of laya_train.py then
fits how confident the model should be.

Writes ../data/laya_train_items.pt (a list of dicts: ids, markers, qtype, target, label).

Run:

    python .\\demos\\03_command_model\\laya\\laya_prepare.py                     # 4000 sentences per language
    python .\\demos\\03_command_model\\laya\\laya_prepare.py --per-language 1000 # quicker
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
sys.path.insert(0, str(HERE))
from laya_questions import QUESTIONS, answers_of  # noqa: E402

BASE = "convaiinnovations/laya-multilingual"
SMOOTHING = 0.1


def base_model_dir(model: str = BASE) -> str:
    """Local folder of a Laya checkpoint (downloaded once from the Hub)."""
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config

    folder = model if Path(model).exists() else snapshot_download(model)
    _fix_tokenizer_config(folder)
    return folder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-language", type=int, default=4000, help="Sentences per language")
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    try:
        import truststore  # Norton re-signs HTTPS; see demos/02_voice

        truststore.inject_into_ssl()
    except ImportError:
        pass
    import torch
    from laya.common import QTYPES, build_sequence, render_options
    from transformers import AutoTokenizer

    folder = base_model_dir()
    tok = AutoTokenizer.from_pretrained(os.path.join(folder, "tokenizer"))
    cfg = json.loads(Path(folder, "rl_agent_config.json").read_text(encoding="utf-8"))

    lines = [json.loads(line) for line in (DATA / "train_commands.jsonl").open(encoding="utf-8")]
    rng = random.Random(args.seed)
    chosen = []
    for lang in ("es", "en", "eu"):
        of_lang = [line for line in lines if line["meta"]["lang"] == lang]
        chosen += rng.sample(of_lang, min(args.per_language, len(of_lang)))
    if (DATA / "train_asr.jsonl").exists():  # speech-recognition variants: all of them
        chosen += [json.loads(line) for line in (DATA / "train_asr.jsonl").open(encoding="utf-8")]

    items, skipped, counts = [], 0, collections.Counter()
    for line in chosen:
        right = answers_of(line["meta"]["command"])
        for name, question in QUESTIONS.items():
            options = list(question["criteria"])
            k = len(options)
            target = [SMOOTHING / (k - 1)] * k
            target[options.index(right[name])] = 1.0 - SMOOTHING
            spec = {"t": question["type"], "ins": question["instructions"], "crit": question["criteria"]}
            seq, markers = build_sequence(tok, line["input"], spec, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options({"t": spec["t"], "crit": spec["crit"]})):
                skipped += 1  # an option was cut off by the length budget
                continue
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[question["type"]],
                          "target": target, "label": options.index(right[name])})
            counts[name] += 1
    rng.shuffle(items)
    out = DATA / "laya_train_items.pt"
    torch.save(items, out)
    print(f"{len(chosen)} sentences -> {len(items)} items ({dict(counts)}), {skipped} skipped -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
