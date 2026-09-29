"""Fine-tune Laya (laya-multilingual) on the command decisions, on one GPU.

A single-process port of Laya's own trainer (`train_ddp.py` in
notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb, Apache-2.0,
https://github.com/NandhaKishorM/laya). The original runs under torchrun on two
Kaggle T4s with NCCL, which Windows does not have; the training itself is kept:

  * RLCD: for each item, GROUP_SIZE noisy versions of the option logits are
    sampled and rewarded with strictly proper scoring rules (spherical + ranked
    probability score) against the target; a policy-gradient term pushes
    towards the better-rewarded ones (GRPO-style), plus a soft cross-entropy term;
  * exploration noise annealed from 0.4 to 0.1 over the epochs;
  * AdamW with separate learning rates for the encoder (2.5e-5) and the head
    (1e-4), cosine schedule, fp16 autocast, gradient checkpointing, clip 1.0;
  * a calibration slice (10%, at most 400 items) is held out before training,
    and after the last epoch one temperature per question type is fitted on
    it, so the confidences mean what they say.

Reads ../data/laya_train_items.pt (from laya_prepare.py).
Writes ../models/laya-commands/ (model.safetensors, encoder/, tokenizer/,
rl_agent_config.json), loadable with laya.load(folder).

Run:

    python .\\demos\\03_command_model\\laya\\laya_train.py
    python .\\demos\\03_command_model\\laya\\laya_train.py --epochs 1 --limit 2000   # quick check
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
from laya_prepare import base_model_dir  # noqa: E402

OUTPUT = PROJECT / "models" / "laya-commands"


def collate(items, pad_id):
    import torch

    n, length = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    attention = torch.zeros((n, length), dtype=torch.long)
    marker_pos = torch.zeros((n, kmax), dtype=torch.long)
    marker_mask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        attention[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        marker_pos[i, :k] = torch.tensor(it["markers"])
        marker_mask[i, :k] = True
        target[i, :len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": attention, "marker_pos": marker_pos, "marker_mask": marker_mask,
            "target": target, "qtype": torch.tensor([it["qtype"] for it in items])}


def fit_temperature(pairs):
    """One temperature for a question type, by LBFGS on held-out (logits, target) pairs."""
    import torch

    if len(pairs) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in pairs)
    logits = torch.full((len(pairs), kmax), -1e4)
    targets = torch.zeros((len(pairs), kmax))
    for i, (z, t) in enumerate(pairs):
        logits[i, :len(z)] = torch.tensor(z)
        targets[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = -(targets * torch.log_softmax(logits / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def forward(model, batch, device):
    import torch

    with torch.autocast("cuda", dtype=torch.float16):
        logits, activity = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                 batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                                 batch["qtype"].to(device))
    return logits.float(), activity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4, help="Effective batch = micro-batch x this")
    parser.add_argument("--limit", type=int, help="Use only the first N items (quick check)")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    try:
        import truststore  # Norton re-signs HTTPS; see demos/02_voice

        truststore.inject_into_ssl()
    except ImportError:
        pass
    import torch
    from laya.common import build_model, proper_reward
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    group_size, lr_encoder, lr_head, sigma_start, sigma_end = 4, 2.5e-5, 1.0e-4, 0.4, 0.1
    device = torch.device("cuda")
    folder = base_model_dir()
    cfg = json.loads(Path(folder, "rl_agent_config.json").read_text(encoding="utf-8"))
    cfg["gradient_checkpointing"] = True
    tok = AutoTokenizer.from_pretrained(os.path.join(folder, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(folder, "encoder"))
    model.load_state_dict(load_file(os.path.join(folder, "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device).train()

    items = torch.load(PROJECT / "data" / "laya_train_items.pt", weights_only=False)
    items = items[:args.limit] if args.limit else items
    # The calibration slice is held out before training: fitting temperatures on items the
    # model trained on would measure the fit, not the calibration (Laya's docs explain why).
    order = list(range(len(items)))
    random.Random(20260922).shuffle(order)
    n_calib = min(400, len(items) // 10)
    calib = [items[i] for i in sorted(order[:n_calib])]
    train = [items[i] for i in sorted(order[n_calib:])]

    encoder_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW([{"params": encoder_params, "lr": lr_encoder},
                                   {"params": head_params, "lr": lr_head}], weight_decay=0.01)
    updates = math.ceil(len(train) / (args.micro_batch * args.grad_accum)) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    print(f"{len(train)} training items ({len(calib)} held out for calibration), {args.epochs} epochs, "
          f"{updates} updates", flush=True)

    start = time.time()
    for epoch in range(args.epochs):
        random.Random(42 + epoch).shuffle(train)
        sigma = sigma_start + (sigma_end - sigma_start) * (epoch / max(1, args.epochs - 1))
        total, batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, first in enumerate(range(0, len(train), args.micro_batch), start=1):
            batch = collate(train[first:first + args.micro_batch], tok.pad_token_id)
            logits, activity = forward(model, batch, device)
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            # 1. sample noisy logit vectors (zero-mean over the options)
            eps = torch.randn((group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            # 2. reward each sample with proper scoring rules; advantage = reward above the group mean
            with torch.no_grad():
                reward = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                advantage = reward - reward.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)
            # 3. policy-gradient term + soft cross-entropy term
            log_prob = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(advantage * log_prob).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / args.grad_accum + 0.0 * activity.sum()
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0 or first + args.micro_batch >= len(train):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total += loss.item() * args.grad_accum
            batches += 1
            if batches % 200 == 0:
                print(f"  epoch {epoch + 1}/{args.epochs} step {batches} loss {total / batches:.4f} "
                      f"reward {reward.mean().item():.3f} ({time.time() - start:.0f} s)", flush=True)
        print(f"=== epoch {epoch + 1}/{args.epochs} done in {time.time() - start:.0f} s, "
              f"mean loss {total / max(1, batches):.4f}", flush=True)

    # Fit one temperature per question type on the held-out slice.
    model.eval()
    held_out = []
    with torch.no_grad():
        for first in range(0, len(calib), 32):
            chunk = calib[first:first + 32]
            logits, _ = forward(model, collate(chunk, tok.pad_token_id), device)
            for row, it in zip(logits.cpu().numpy(), chunk):
                held_out.append((it["qtype"], row[:len(it["markers"])], it["target"]))
    temperatures = [fit_temperature([(z, t) for qt, z, t in held_out if qt == qtype]) for qtype in range(3)]
    print("calibration temperatures (choice, score, noul):", [round(t, 3) for t in temperatures])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()},
              str(output / "model.safetensors"))
    model.encoder.config.save_pretrained(str(output / "encoder"))
    tok.save_pretrained(str(output / "tokenizer"))
    cfg.update(fine_tuned=True, model_name="laya-commands", temperature=temperatures)
    cfg.pop("temperature_by_options", None)  # would override the fitted temperatures at inference
    cfg.pop("gradient_checkpointing", None)
    (output / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
