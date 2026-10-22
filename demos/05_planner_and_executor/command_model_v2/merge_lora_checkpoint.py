"""Merge a GLiNER2 PEFT LoRA adapter into a standard loadable checkpoint.

Example (PowerShell):
    python merge_lora_checkpoint.py `
        --base models/command-model/best `
        --adapter models/command-model-shapes-lora-candidate/best `
        --output models/command-model-shapes-lora-candidate/best_merged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_checkpoint(base_path: str | Path, adapter_path: str | Path, output_path: str | Path) -> int:
    """Merge one saved PEFT adapter into a standard GLiNER2 checkpoint."""
    from gliner2 import AutoExtractor
    from peft import PeftModel

    base_ref = str(base_path)
    if Path(base_ref).exists():
        base_ref = str(Path(base_ref).resolve())
    adapter_path, output_path = Path(adapter_path), Path(output_path)
    base = AutoExtractor.from_pretrained(base_ref)
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    merged = model.merge_and_unload(safe_merge=True)
    output_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_path), safe_serialization=True)

    # GLiNER2 stores its tokenizer on the processor rather than the model config.
    tokenizer = getattr(getattr(base, "processor", None), "tokenizer", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_path))

    metadata = {
        "base": str(base_path),
        "adapter": str(adapter_path),
        "merged": True,
        "method": "PEFT PeftModel.merge_and_unload(safe_merge=True)",
    }
    (output_path / "merge_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    size = sum(p.stat().st_size for p in output_path.rglob("*") if p.is_file())
    print(f"saved merged checkpoint: {output_path} ({size / (1024 * 1024):.1f} MiB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="base GLiNER2 checkpoint")
    parser.add_argument("--adapter", type=Path, required=True, help="PEFT adapter directory")
    parser.add_argument("--output", type=Path, required=True, help="merged checkpoint directory")
    args = parser.parse_args()

    return merge_checkpoint(args.base, args.adapter, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
