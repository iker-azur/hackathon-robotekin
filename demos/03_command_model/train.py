"""Step 3: fine-tune GLiNER2.5-multi on the generated data.

Reads the JSON Lines files written by generate_dataset.py (and, if present,
add_asr_noise.py) and trains one model on all of them at once:

    data/train_commands.jsonl    sentences -> intent, place, sort, item records
    data/train_item_kinds.jsonl  item phrases -> kind, one/several
    data/train_asr.jsonl         (optional) the same kind of sentences after speech recognition

Training settings worth knowing:
  * full fine-tuning of all 287M parameters, in fp32: the trainer's default for
    this architecture (bf16) produced non-finite losses on this task;
  * encoder learning rate 2e-5, task heads 5e-4, 10% warm-up;
  * the checkpoint with the lowest loss on data/eval_*.jsonl is kept as
    models/command-model/best; the other checkpoints (1.1 GB each) are deleted.

About 3 epochs over 34k samples takes roughly an hour on an RTX 5060 Ti.

Run:

    python .\\demos\\03_command_model\\train.py
    python .\\demos\\03_command_model\\train.py --epochs 5
    python .\\demos\\03_command_model\\train.py --max-steps 50     # quick check that it runs
    python .\\demos\\03_command_model\\train.py --base demos\\03_command_model\\models\\command-model\\best ^
        --epochs 1 --output demos\\03_command_model\\models\\command-model-more    # continue training
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT = HERE / "models" / "command-model"
BASE_MODEL = "fastino/gliner2.5-multi-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop early (for a quick check)")
    parser.add_argument("--base", default=BASE_MODEL, help="Model to start from (e.g. a previous best, to continue)")
    parser.add_argument("--output", default=str(OUTPUT), help="Folder for the checkpoints")
    args = parser.parse_args()
    output = Path(args.output)

    try:
        import truststore  # Norton re-signs HTTPS; see demos/02_voice

        truststore.inject_into_ssl()
    except ImportError:
        pass

    train_files = [DATA / "train_commands.jsonl", DATA / "train_item_kinds.jsonl"]
    if (DATA / "train_asr.jsonl").exists():
        train_files.append(DATA / "train_asr.jsonl")
    eval_files = [DATA / "eval_commands.jsonl", DATA / "eval_item_kinds.jsonl"]
    missing = [f.name for f in train_files + eval_files if not f.exists()]
    if missing:
        raise SystemExit(f"missing {missing}: run generate_dataset.py first")
    for f in train_files:
        with f.open(encoding="utf-8") as lines:
            print(f"  {f.name}: {sum(1 for _ in lines)} samples")

    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    model = AutoExtractor.from_pretrained(args.base)
    config = TrainingConfig(
        output_dir=str(output), experiment_name="command-model",
        num_epochs=args.epochs, max_steps=args.max_steps,
        batch_size=args.batch_size, eval_batch_size=32,
        encoder_lr=2e-5, task_lr=5e-4, warmup_ratio=0.1,
        eval_strategy="epoch", save_best=True, logging_steps=100,
        num_workers=0,           # Windows: no data-loader subprocesses
        fp16=False, bf16=False,  # full precision: bf16 gave non-finite losses on this task
        seed=1,
    )
    summary = ExtractorTrainer(model, config).train(
        train_data=[str(f) for f in train_files], eval_data=[str(f) for f in eval_files])

    # Keep only the best checkpoint (the one command_model.py loads).
    for checkpoint in output.iterdir():
        if checkpoint.is_dir() and (checkpoint.name.startswith("checkpoint-") or checkpoint.name == "final"):
            shutil.rmtree(checkpoint)
    (output / "training_summary.json").write_text(
        json.dumps({"base": args.base, "epochs": args.epochs, "train_files": [f.name for f in train_files],
                    "summary": summary}, indent=2, default=str), encoding="utf-8")
    print(f"saved {output / 'best'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
