"""Step 3: fine-tune GLiNER2.5-multi on the generated data.

Reads the JSON Lines files written by generate_dataset.py (and, if present,
add_asr_noise.py) and trains one model on all of them at once:

    data/train_commands.jsonl    sentences -> intent, place, sort, item records
    data/train_item_kinds.jsonl  item phrases -> kind, one/several
    data/train_asr.jsonl         (optional) the same kind of sentences after speech recognition

Training settings worth knowing:
  * full fine-tuning is available by default; ``--use-lora`` trains low-rank
    adapters and explicitly merges the best adapter into ``best_merged``;
  * fp32 is used because bf16 produced non-finite losses on this task;
  * encoder learning rate 2e-5, task heads 5e-4, 10% warm-up;
  * the checkpoint with the lowest loss on data/eval_*.jsonl is kept as
    ``best``; LoRA runs also produce a standard full checkpoint at ``best_merged``.

Training time depends on GPU contention. Sharing an RTX 5060 Ti with Isaac Sim
can make a multi-epoch run several times slower.

Run:

    python train.py
    python train.py --epochs 5
    python train.py --max-steps 50     # quick check that it runs
    python train.py --base models\\command-model\\best --epochs 1 --output models\\command-model-more
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
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="micro-batches per optimizer update; use a small batch to limit VRAM")
    parser.add_argument("--eval-batch-size", type=int, default=32,
                        help="evaluation micro-batch size")
    parser.add_argument("--use-lora", action="store_true",
                        help="train low-rank adapters and save a merged, loadable full checkpoint")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop early (for a quick check)")
    parser.add_argument("--base", default=BASE_MODEL, help="Model to start from (e.g. a previous best, to continue)")
    parser.add_argument("--output", default=str(OUTPUT), help="Folder for the checkpoints")
    parser.add_argument("--data-dir", type=Path, default=DATA,
                        help="directory containing generated train/eval JSONL files")
    args = parser.parse_args()
    output = Path(args.output)
    data_dir = args.data_dir

    try:
        import truststore  # Norton re-signs HTTPS; see demos/02_voice

        truststore.inject_into_ssl()
    except ImportError:
        pass

    train_files = [data_dir / "train_commands.jsonl", data_dir / "train_item_kinds.jsonl"]
    if (data_dir / "train_asr.jsonl").exists():
        train_files.append(data_dir / "train_asr.jsonl")
    eval_files = [data_dir / "eval_commands.jsonl", data_dir / "eval_item_kinds.jsonl"]
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
        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_lora=args.use_lora, lora_r=args.lora_r,
        # Save a PEFT-native adapter, then merge it explicitly below into a
        # normal GLiNER2 checkpoint for Demo 11 inference.
        save_adapter_only=True,
        encoder_lr=2e-5, task_lr=5e-4, warmup_ratio=0.1,
        eval_strategy="epoch", save_best=True, logging_steps=100,
        num_workers=0,           # Windows: no data-loader subprocesses
        fp16=False, bf16=False,  # full precision: bf16 gave non-finite losses on this task
        seed=1,
    )
    summary = ExtractorTrainer(model, config).train(
        train_data=[str(f) for f in train_files], eval_data=[str(f) for f in eval_files])

    if args.use_lora:
        from merge_lora_checkpoint import merge_checkpoint

        merge_checkpoint(args.base, output / "best", output / "best_merged")

    # Keep only the best checkpoint (the one command_model.py loads).
    for checkpoint in output.iterdir():
        if checkpoint.is_dir() and (checkpoint.name.startswith("checkpoint-") or checkpoint.name == "final"):
            shutil.rmtree(checkpoint)
    (output / "training_summary.json").write_text(
        json.dumps({"base": args.base, "epochs": args.epochs, "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "use_lora": args.use_lora, "lora_r": args.lora_r,
                    "train_files": [f.name for f in train_files],
                    "summary": summary}, indent=2, default=str), encoding="utf-8")
    print(f"saved {output / ('best_merged' if args.use_lora else 'best')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
