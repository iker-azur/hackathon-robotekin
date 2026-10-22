"""Compact JEPA-inspired text representation model for Spanish intent classification.

A small character-level encoder predicts a target (full sentence) latent
representation from a context (partially masked sentence) representation.
An EMA target encoder and a variance regulariser are used as anti-collapse
measures. A linear classification head is trained jointly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False
    torch = None
    nn = None
    F = None


def is_available() -> bool:
    return _TORCH


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
MASK_TOKEN = "<mask>"


class CharVocab:
    def __init__(self):
        self.stoi: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1, MASK_TOKEN: 2}
        self.itos: List[str] = [PAD_TOKEN, UNK_TOKEN, MASK_TOKEN]

    def build(self, texts: List[str]) -> None:
        chars = set()
        for t in texts:
            for ch in t.lower():
                chars.add(ch)
        for ch in sorted(chars):
            if ch not in self.stoi:
                self.stoi[ch] = len(self.itos)
                self.itos.append(ch)

    def encode(self, text: str, max_len: int = 128) -> List[int]:
        ids = [self.stoi.get(ch.lower(), self.stoi[UNK_TOKEN]) for ch in text.lower()]
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids = ids + [self.stoi[PAD_TOKEN]] * (max_len - len(ids))
        return ids

    def mask_sequence(self, ids: List[int], mask_prob: float = 0.15) -> List[int]:
        out = list(ids)
        for i in range(len(out)):
            if out[i] in (0,):  # skip pad
                continue
            if torch.rand(1).item() < mask_prob:  # type: ignore
                out[i] = self.stoi[MASK_TOKEN]
        return out

    def to_dict(self) -> Dict:
        return {"stoi": self.stoi, "itos": self.itos}

    @classmethod
    def from_dict(cls, d: Dict) -> "CharVocab":
        v = cls()
        v.stoi = dict(d["stoi"])
        v.itos = list(d["itos"])
        return v


if _TORCH:
    class TextEncoder(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int = 32, hidden: int = 64, latent: int = 64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.conv1 = nn.Conv1d(embed_dim, hidden, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
            self.proj = nn.Linear(hidden, latent)

        def forward(self, x):
            # x: (B, L) long
            e = self.embedding(x).transpose(1, 2)  # (B, E, L)
            h = torch.relu(self.conv1(e))
            h = torch.relu(self.conv2(h))
            h = h.max(dim=2).values  # (B, hidden)
            z = self.proj(h)
            return z

    class JEPAIntentModel(nn.Module):
        def __init__(self, vocab_size: int, num_classes: int, embed_dim: int = 32,
                     hidden: int = 64, latent: int = 64):
            super().__init__()
            self.embed_dim = embed_dim
            self.hidden = hidden
            self.latent = latent
            self.context_encoder = TextEncoder(vocab_size, embed_dim, hidden, latent)
            self.target_encoder = TextEncoder(vocab_size, embed_dim, hidden, latent)
            self.predictor = nn.Sequential(
                nn.Linear(latent, latent),
                nn.ReLU(),
                nn.Linear(latent, latent),
            )
            self.classifier = nn.Linear(latent, num_classes)
            # init target encoder = context encoder
            self.target_encoder.load_state_dict(self.context_encoder.state_dict())
            for p in self.target_encoder.parameters():
                p.requires_grad = False

        def encode(self, x):
            return self.context_encoder(x)

        def forward_train(self, context_ids, target_ids):
            z_c = self.context_encoder(context_ids)
            with torch.no_grad():
                z_t = self.target_encoder(target_ids)
            z_pred = self.predictor(z_c)
            logits = self.classifier(z_c)
            return z_pred, z_t, logits

        @torch.no_grad()
        def update_target_encoder(self, tau: float = 0.996):
            for ps, pt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                pt.data.mul_(tau).add_(ps.data, alpha=1 - tau)

    def variance_loss(z, gamma: float = 1.0):
        # VICReg-style variance: keep std above gamma
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.mean(torch.relu(gamma - std))

    def jepa_loss(z_pred, z_t, logits, labels, lambda_var: float = 0.1):
        pred_loss = F.mse_loss(z_pred, z_t)
        cls_loss = F.cross_entropy(logits, labels)
        var = variance_loss(z_pred) + variance_loss(z_t.detach())
        total = pred_loss + cls_loss + lambda_var * var
        return total, {"pred": pred_loss.item(), "cls": cls_loss.item(), "var": var.item()}
else:
    TextEncoder = None
    JEPAIntentModel = None
    def variance_loss(*a, **k):
        raise RuntimeError("torch no disponible")
    def jepa_loss(*a, **k):
        raise RuntimeError("torch no disponible")
