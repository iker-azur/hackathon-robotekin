"""Step 2: fine-tune GLiNER2.5-multi on the bin-sorting intent data.

Reads ``data/train.jsonl`` (60,000 sentences) and validates on ``data/eval.jsonl``. Each sample asks five
classification questions (intent, feature, quality, box, count) about one sentence, so the model answers all
five in a single forward pass and gives a probability for every option of every question.

Same recipe as ``05_planner_and_executor/command_model_v2/train.py``, which is the one that was measured on this
machine: full fine-tuning in fp32 (bf16 gave non-finite losses on this architecture), encoder learning rate
2e-5, heads 5e-4, 10% warm-up, best checkpoint by validation loss. TF32 matmuls are switched on: they are
faster than fp32 and keep its exponent range.

    python train.py --max-steps 60 --output models/speed-check     # quick check of speed and memory
    python train.py                                                # 2 epochs, writes models/bin-intent/best
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from grammar_common import TASKS

HERE = Path(__file__).resolve().parent
BASE_MODEL = "fastino/gliner2.5-multi-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=-1, help="stop early (for a quick check)")
    parser.add_argument("--base", default=BASE_MODEL, help="model to start from")
    parser.add_argument("--output", default=str(HERE / "models" / "bin-intent"))
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    output = Path(args.output)
    if (output / "best").exists():
        raise SystemExit(f"{output / 'best'} already exists: pass another --output so a trained model is not overwritten")

    try:
        import truststore  # the Windows certificate store, for Hugging Face downloads behind a proxy

        truststore.inject_into_ssl()
    except ImportError:
        pass
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    train_file, eval_file = args.data_dir / "train.jsonl", args.data_dir / "eval.jsonl"
    for path in (train_file, eval_file):
        if not path.exists():
            raise SystemExit(f"missing {path}: run generate_dataset.py first")
    with train_file.open(encoding="utf-8") as lines:
        print(f"{train_file.name}: {sum(1 for _ in lines)} samples", flush=True)

    model = AutoExtractor.from_pretrained(args.base)
    config = TrainingConfig(
        output_dir=str(output), experiment_name="bin-intent",
        num_epochs=args.epochs, max_steps=args.max_steps,
        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        encoder_lr=2e-5, task_lr=5e-4, warmup_ratio=0.1,
        eval_strategy="epoch", save_best=True, logging_steps=100,
        num_workers=0,           # Windows: no data-loader subprocesses
        fp16=False, bf16=False,  # bf16 gave non-finite losses on this task
        seed=args.seed,
    )
    summary = ExtractorTrainer(model, config).train(train_data=[str(train_file)], eval_data=[str(eval_file)])

    for checkpoint in output.iterdir():  # keep only the best checkpoint
        if checkpoint.is_dir() and (checkpoint.name.startswith("checkpoint-") or checkpoint.name == "final"):
            shutil.rmtree(checkpoint)
    (output / "labels.json").write_text(json.dumps(TASKS, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "training_summary.json").write_text(
        json.dumps({"base": args.base, "epochs": args.epochs, "batch_size": args.batch_size,
                    "train_file": train_file.name, "summary": summary}, indent=2, default=str), encoding="utf-8")
    print(f"saved {output / 'best'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
