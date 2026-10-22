"""Training and evaluation for RoboQC intent models (CPU only)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Make src importable when run as script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from roboqc.commands.entities import extract_entities, entities_to_parameters
from roboqc.commands.parser import classify_intent_rule
from roboqc.commands.schema import StructuredCommand
from roboqc.commands.taxonomy import ALL_INTENTS
from roboqc.intent.baseline import BaselineIntentClassifier, is_available as baseline_available
from roboqc.intent.minilm import MiniLMIntentClassifier, is_available as minilm_available
from roboqc.intent.jepa_model import (
    CharVocab, JEPAIntentModel, jepa_loss, is_available as torch_available,
)

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
    _SK = True
except Exception:
    _SK = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def records_to_xy(records: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    texts = []
    labels = []
    for r in records:
        text = r.get("text", "")
        actions = r.get("actions") or []
        if not actions:
            labels.append("UNKNOWN")
            texts.append(text)
            continue
        # Use the primary (first) action intent only. Flattening compound
        # actions under the same text creates conflicting labels and hurts
        # single-intent classification.
        intent = actions[0].get("intent", "UNKNOWN")
        texts.append(text)
        labels.append(intent)
    return texts, labels


def train_baseline(data_dir: str, model_dir: str, seed: int = 42, **kwargs) -> Dict[str, Any]:
    if not baseline_available():
        raise RuntimeError("scikit-learn no disponible.")
    set_seed(seed)
    train = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    val = load_jsonl(os.path.join(data_dir, "validation.jsonl"))
    tr_x, tr_y = records_to_xy(train)
    va_x, va_y = records_to_xy(val)

    clf = BaselineIntentClassifier()
    t0 = time.perf_counter()
    clf.fit(tr_x, tr_y, seed=seed)
    train_time = time.perf_counter() - t0

    # eval
    preds = [clf.predict(t) for t in va_x]
    metrics = compute_metrics(va_y, preds)
    metrics["train_time_s"] = train_time
    metrics["train_size"] = len(tr_x)
    metrics["val_size"] = len(va_x)

    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "baseline.joblib")
    clf.save(path)
    # save labels
    with open(os.path.join(model_dir, "baseline_labels.json"), "w", encoding="utf-8") as f:
        json.dump(clf.labels, f, ensure_ascii=False, indent=2)

    with open(os.path.join(model_dir, "baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def train_minilm(data_dir: str, model_dir: str, seed: int = 42, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs) -> Dict[str, Any]:
    if not minilm_available():
        raise RuntimeError("sentence-transformers o scikit-learn no disponibles.")
    set_seed(seed)
    train = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    val = load_jsonl(os.path.join(data_dir, "validation.jsonl"))
    tr_x, tr_y = records_to_xy(train)
    va_x, va_y = records_to_xy(val)

    clf = MiniLMIntentClassifier(model_name=model_name)
    t0 = time.perf_counter()
    clf.fit(tr_x, tr_y, seed=seed)
    train_time = time.perf_counter() - t0

    preds = clf.predict_batch(va_x)
    metrics = compute_metrics(va_y, preds)
    metrics["train_time_s"] = train_time
    metrics["train_size"] = len(tr_x)
    metrics["val_size"] = len(va_x)

    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "minilm.joblib")
    clf.save(path)
    with open(os.path.join(model_dir, "minilm_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    if not _SK:
        # fallback simple accuracy
        acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / max(1, len(y_true))
        return {"accuracy": acc, "macro_f1": None}
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0,
                                                         labels=sorted(set(y_true) | set(y_pred)))
    per_class = {}
    labels = sorted(set(y_true) | set(y_pred))
    for i, lab in enumerate(labels):
        per_class[lab] = {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])}
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": labels,
    }


def train_jepa(data_dir: str, model_dir: str, seed: int = 42, epochs: int = 10,
               batch_size: int = 64, lr: float = 1e-3, patience: int = 3,
               cpu_threads: int = 2, **kwargs) -> Dict[str, Any]:
    if not torch_available():
        raise RuntimeError("PyTorch no disponible.")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.set_num_threads(cpu_threads)
    set_seed(seed)

    train = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    val = load_jsonl(os.path.join(data_dir, "validation.jsonl"))
    tr_x, tr_y = records_to_xy(train)
    va_x, va_y = records_to_xy(val)

    # label mapping
    labels = sorted(set(tr_y) | set(va_y) | set(ALL_INTENTS))
    label2idx = {l: i for i, l in enumerate(labels)}

    vocab = CharVocab()
    vocab.build(tr_x + va_x)

    max_len = 128
    def encode_texts(xs):
        return torch.tensor([vocab.encode(t, max_len) for t in xs], dtype=torch.long)
    tr_ids = encode_texts(tr_x)
    va_ids = encode_texts(va_x)
    tr_labels = torch.tensor([label2idx[y] for y in tr_y], dtype=torch.long)
    va_labels = torch.tensor([label2idx[y] for y in va_y], dtype=torch.long)

    model = JEPAIntentModel(vocab_size=len(vocab.stoi), num_classes=len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    history = []
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        # shuffle
        perm = torch.randperm(len(tr_ids))
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(tr_ids), batch_size):
            idx = perm[start:start + batch_size]
            batch_ids = tr_ids[idx]
            batch_labels = tr_labels[idx]
            # context = masked
            context_list = [vocab.mask_sequence(ids.tolist()) for ids in batch_ids]
            context_ids = torch.tensor(context_list, dtype=torch.long)
            target_ids = batch_ids
            optimizer.zero_grad()
            z_pred, z_t, logits = model.forward_train(context_ids, target_ids)
            loss, parts = jepa_loss(z_pred, z_t, logits, batch_labels)
            loss.backward()
            optimizer.step()
            model.update_target_encoder()
            total_loss += loss.item()
            n_batches += 1
        # val
        model.eval()
        with torch.no_grad():
            z = model.encode(va_ids)
            logits = model.classifier(z)
            preds_idx = logits.argmax(dim=1).tolist()
        preds = [labels[i] for i in preds_idx]
        metrics = compute_metrics(va_y, preds)
        val_f1 = metrics.get("macro_f1") or 0.0
        history.append({"epoch": epoch + 1, "train_loss": total_loss / max(1, n_batches), "val_macro_f1": val_f1, "val_acc": metrics.get("accuracy")})
        print(f"epoch {epoch+1}/{epochs} loss={total_loss/max(1,n_batches):.4f} val_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_time = time.perf_counter() - t0
    # final eval
    model.eval()
    with torch.no_grad():
        z = model.encode(va_ids)
        logits = model.classifier(z)
        preds_idx = logits.argmax(dim=1).tolist()
    preds = [labels[i] for i in preds_idx]
    metrics = compute_metrics(va_y, preds)
    metrics["train_time_s"] = train_time
    metrics["train_size"] = len(tr_x)
    metrics["val_size"] = len(va_x)
    metrics["best_val_macro_f1"] = best_val_f1
    metrics["history"] = history

    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, "jepa.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab.to_dict(),
            "labels": labels,
            "config": {"embed_dim": 32, "hidden": 64, "latent": 64, "max_len": max_len},
        },
        ckpt_path,
    )
    with open(os.path.join(model_dir, "jepa_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def evaluate_entities(data_path: str) -> Dict[str, Any]:
    recs = load_jsonl(data_path)
    # Compare deterministic extraction to ground truth first action
    total = 0
    field_correct = {"object_type": 0, "object_color": 0, "quantity": 0,
                    "object_quality": 0, "destination_type": 0, "destination_color": 0}
    field_total = {k: 0 for k in field_correct}
    for r in recs:
        text = r.get("text", "")
        actions = r.get("actions") or []
        if not actions:
            continue
        gt = actions[0].get("parameters", {}) or {}
        ents = extract_entities(text)
        params = entities_to_parameters(ents)
        total += 1
        for field in field_correct:
            if field in gt:
                field_total[field] += 1
                gv = gt.get(field)
                pv = params.get(field)
                # normalize quantity comparison
                if field == "quantity":
                    if gv == "UNSPECIFIED":
                        gv = "UNSPECIFIED"
                    if pv == "UNSPECIFIED":
                        pv = "UNSPECIFIED"
                if gv == pv:
                    field_correct[field] += 1
    out = {"examples": total}
    for field in field_correct:
        denom = field_total[field] if field_total[field] > 0 else total
        out[field + "_accuracy"] = field_correct[field] / denom if denom else 0.0
        out[field + "_count"] = field_total[field]
    return out


def evaluate_command_exact(data_path: str, classifier=None, confidence_fn=None) -> Dict[str, Any]:
    from roboqc.commands.pipeline import understand_text
    recs = load_jsonl(data_path)
    total = 0
    exact = 0
    intent_correct = 0
    for r in recs:
        text = r.get("text", "")
        gt_actions = r.get("actions") or []
        cmd = understand_text(text, classifier=classifier, confidence_fn=confidence_fn)
        total += 1
        # intent compare (first action)
        if cmd.actions and gt_actions:
            got_intent = cmd.actions[0].intent if hasattr(cmd.actions[0], "intent") else cmd.actions[0]["intent"]
            if got_intent == gt_actions[0].get("intent"):
                intent_correct += 1
        # exact match of serialized actions
        got = [a.to_dict() if hasattr(a, "to_dict") else a for a in cmd.actions]
        # normalize: drop confidence
        for a in got:
            if isinstance(a, dict):
                a.pop("confidence", None)
                a.pop("requires_confirmation", None)
        gt_norm = []
        for a in gt_actions:
            d = dict(a)
            d.pop("confidence", None)
            d.pop("requires_confirmation", None)
            gt_norm.append(d)
        if got == gt_norm:
            exact += 1
    return {
        "examples": total,
        "intent_accuracy": intent_correct / total if total else 0.0,
        "exact_match": exact / total if total else 0.0,
    }


def evaluate_splits(model_dir: str, data_dir: str) -> Dict[str, Any]:
    """Score every saved model on validation, test and challenge (accuracy, macro-F1, confusions)."""
    from roboqc.intent.inference import load_engine, MODEL_FILES
    report: Dict[str, Any] = {}
    for name, fname in MODEL_FILES.items():
        if not os.path.exists(os.path.join(model_dir, fname)):
            continue
        engine = load_engine(model_dir=model_dir, preferred=name)
        engine.load()
        report[name] = {}
        for split in ("validation", "test", "challenge"):
            xs, ys = records_to_xy(load_jsonl(os.path.join(data_dir, f"{split}.jsonl")))
            preds = [engine.predict_topk(t, 1)[0][0] for t in xs]
            report[name][split] = compute_metrics(ys, preds)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de intencion RoboQC (CPU).")
    parser.add_argument("--model", choices=["baseline", "jepa", "minilm", "auto"], default="baseline")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--evaluate-entities", action="store_true")
    parser.add_argument("--evaluate-splits", action="store_true",
                        help="Puntua cada modelo guardado en validation, test y challenge.")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    if args.model in ("baseline", "auto"):
        print("Entrenando baseline...")
        m = train_baseline(args.data_dir, args.model_dir, seed=args.seed)
        print(json.dumps({k: v for k, v in m.items() if k not in ("per_class", "confusion_matrix")}, ensure_ascii=False, indent=2))

    if args.model in ("minilm", "auto"):
        print("Entrenando MiniLM...")
        m = train_minilm(args.data_dir, args.model_dir, seed=args.seed, model_name=args.model_name)
        print(json.dumps({k: v for k, v in m.items() if k not in ("per_class", "confusion_matrix")}, ensure_ascii=False, indent=2))

    if args.model in ("jepa", "auto"):
        print("Entrenando JEPA...")
        m = train_jepa(args.data_dir, args.model_dir, seed=args.seed, epochs=args.epochs,
                       batch_size=args.batch_size, lr=args.lr, patience=args.patience,
                       cpu_threads=args.cpu_threads)
        print(json.dumps({k: v for k, v in m.items() if k not in ("per_class", "confusion_matrix", "history")}, ensure_ascii=False, indent=2))

    if args.model == "auto":
        candidates = {}
        for name, fname in (("baseline", "baseline_metrics.json"), ("minilm", "minilm_metrics.json"), ("jepa", "jepa_metrics.json")):
            p = os.path.join(args.model_dir, fname)
            if os.path.exists(p):
                m = json.load(open(p, encoding="utf-8"))
                candidates[name] = m.get("macro_f1") or 0.0
        if not candidates:
            raise RuntimeError("No hay metricas disponibles.")
        selected = max(candidates, key=lambda k: candidates[k])
        with open(os.path.join(args.model_dir, "selected_model.txt"), "w", encoding="utf-8") as f:
            f.write(selected)
        print(f"Modelo seleccionado: {selected} ({', '.join(f'{k} F1={v:.4f}' for k,v in candidates.items())})")

    if args.evaluate_splits:
        report = evaluate_splits(args.model_dir, args.data_dir)
        with open(os.path.join(args.model_dir, "split_eval.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        for name, splits in report.items():
            row = "  ".join(f"{s} acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f}" for s, m in splits.items())
            print(f"{name:8} {row}")

    if args.evaluate_entities:
        for split in ("test.jsonl", "challenge.jsonl"):
            p = os.path.join(args.data_dir, split)
            if os.path.exists(p):
                print(split, json.dumps(evaluate_entities(p), ensure_ascii=False))


if __name__ == "__main__":
    main()
