"""Colab training and evaluation helpers. No training runs at import time."""

import copy
import json
import random
import time
from pathlib import Path

import numpy as np

from .models import IntentBundle
from .voice import canonical, understand


def load_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def target(record):
    return record["intent"] + "|" + record["mode"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def export_minilm_onnx(output, sample):
    """Export the full sentence encoder (including pooling), then validate INT8."""
    import torch
    import joblib
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from sentence_transformers import SentenceTransformer
    from .models import ONNXSentenceEncoder
    out = Path(output)
    original = SentenceTransformer(str(out / "encoder"),device="cpu",model_kwargs={"attn_implementation":"eager"})
    original.eval()
    destination = out / "encoder_int8"
    destination.mkdir(exist_ok=True)
    original.tokenizer.save_pretrained(destination)
    class Encoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model({"input_ids":input_ids,"attention_mask":attention_mask})["sentence_embedding"]
    tokens = original.tokenize(sample[:2])
    fp32_path = destination / "model.onnx"
    torch.onnx.export(Encoder(original).eval(),(tokens["input_ids"],tokens["attention_mask"]),str(fp32_path),
        input_names=["input_ids","attention_mask"],output_names=["sentence_embedding"],opset_version=17,dynamo=False,
        dynamic_axes={"input_ids":{0:"batch",1:"sequence"},"attention_mask":{0:"batch",1:"sequence"},"sentence_embedding":{0:"batch"}})
    quantize_dynamic(str(fp32_path),str(destination / "model_int8.onnx"),weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"],per_channel=True)
    config = {"model_file":"model_int8.onnx","max_seq_length":original.max_seq_length,
              "embedding_dim":original.get_sentence_embedding_dimension()}
    (destination / "onnx_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    a = original.encode(sample,show_progress_bar=False)
    fp32 = ort.InferenceSession(str(fp32_path),providers=["CPUExecutionProvider"])
    reference_tokens = original.tokenizer(sample,padding=True,truncation=True,max_length=original.max_seq_length,return_tensors="np")
    b = fp32.run(None,{item.name:reference_tokens[item.name].astype(np.int64) for item in fp32.get_inputs()})[0]
    np.testing.assert_allclose(a,b,atol=1e-4,rtol=1e-3)
    quantized = ONNXSentenceEncoder(destination)
    c = quantized.encode(sample)
    cosine = np.sum(a*c,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(c,axis=1))
    head = joblib.load(out / "minilm.joblib")
    agreement = float(np.mean(head.predict(a) == head.predict(c)))
    return {"minimum_embedding_cosine":float(cosine.min()),"classification_agreement":agreement,
            "passed":bool(cosine.min() >= .99 and agreement == 1.0),"backend":"onnxruntime"}


def train_linear(train, validation, output, encoder_name=None, seed=42, epochs=40, patience=5):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import f1_score
    out = Path(output)
    name = "minilm" if encoder_name else "tfidf"
    if encoder_name:
        import torch
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(encoder_name, device="cuda" if torch.cuda.is_available() else "cpu")
        encoder.save(str(out / "encoder"))
        x = encoder.encode([r["text"] for r in train], batch_size=64, show_progress_bar=True)
        vx = encoder.encode([r["text"] for r in validation], batch_size=64, show_progress_bar=True)
        np.save(out / "train_embeddings.npy", x)
        np.save(out / "validation_embeddings.npy", vx)
        vectorizer = None
    else:
        vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=30000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000)),
        ])
        x = vectorizer.fit_transform([r["text"] for r in train])
        vx = vectorizer.transform([r["text"] for r in validation])
    y, vy = np.array([target(r) for r in train]), [target(r) for r in validation]
    model = SGDClassifier(loss="log_loss", random_state=seed, alpha=1e-5)
    best, stale, history = -1, 0, []
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(len(y))
        model.partial_fit(x[order], y[order], classes=np.unique(y))
        score = f1_score(vy, model.predict(vx), average="macro", zero_division=0)
        history.append({"epoch": epoch + 1, "macro_f1": float(score)})
        if score > best:
            best, stale = score, 0
            best_model = copy.deepcopy(model)
            artifact = best_model if vectorizer is None else Pipeline([("features", vectorizer), ("classifier", best_model)])
            joblib.dump(artifact, out / f"{name}.joblib")
        else:
            stale += 1
        (out / f"{name}_history.json").write_text(json.dumps(history), encoding="utf-8")
        if stale >= patience:
            break
    return {"validation_macro_f1": float(best), "threshold": 0.0}


def train_jepa(train, validation, output, seed=42, epochs=30, patience=5):
    import torch
    from roboqc.intent.jepa_model import CharVocab, JEPAIntentModel, jepa_loss
    from sklearn.metrics import f1_score
    seed_all(seed)
    out = Path(output)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = sorted({target(r) for r in train})
    vocab = CharVocab()
    vocab.build([r["text"] for r in train])
    max_len = 512
    if any(len(r["text"]) > max_len for r in train + validation):
        raise ValueError("JEPA max_len would truncate a labelled instruction")
    x = torch.tensor([vocab.encode(r["text"], max_len) for r in train])
    y = torch.tensor([labels.index(target(r)) for r in train])
    vx = torch.tensor([vocab.encode(r["text"], max_len) for r in validation])
    vy = [target(r) for r in validation]
    model = JEPAIntentModel(len(vocab.stoi), len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = out / "jepa_checkpoint.pt"
    first, best, stale, history = 0, -1.0, 0, []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first, best, stale, history = state["epoch"], state["best"], state["stale"], state["history"]
    for epoch in range(first, epochs):
        model.train()
        for ids in torch.randperm(len(x)).split(64):
            full = x[ids].to(device)
            masked = full.clone()
            masked[(torch.rand_like(full.float()) < .15) & (full != 0)] = 2
            optimizer.zero_grad()
            pred, target_z, logits = model.forward_train(masked, full)
            loss, _ = jepa_loss(pred, target_z, logits, y[ids].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target_encoder()
        model.eval()
        predictions = []
        with torch.inference_mode():
            for batch in vx.split(128):
                predictions += [labels[i] for i in model.classifier(model.encode(batch.to(device))).argmax(-1).cpu().tolist()]
        score = float(f1_score(vy, predictions, average="macro", zero_division=0))
        history.append({"epoch": epoch + 1, "macro_f1": score})
        if score > best:
            best, stale = score, 0
            torch.save({"state_dict": model.cpu().state_dict(), "vocab": vocab.to_dict(), "labels": labels, "max_len": max_len}, out / "jepa.pt")
            model.to(device)
        else:
            stale += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                    "best": best, "stale": stale, "history": history}, checkpoint)
        (out / "jepa_history.json").write_text(json.dumps(history), encoding="utf-8")
        if stale >= patience:
            break
    return {"validation_macro_f1": best, "threshold": 0.0}


def train_openjev(train, validation, output, epochs=3, seed=42):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              EarlyStoppingCallback, Trainer, TrainingArguments)
    from sklearn.metrics import f1_score
    from .dataset import pairs
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Colab GPU for OpenJev LoRA training.")
    name, sub = "AlexWortega/openjev", "qwen3.5-0.8b-nli-v5"
    tokenizer = AutoTokenizer.from_pretrained(name, subfolder=sub)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(name, subfolder=sub, torch_dtype=torch.float32)
    mapping = {k.lower(): int(v) for k, v in model.config.label2id.items()}
    if set(mapping) != {"entailment", "contradiction", "neutral"}:
        raise ValueError(f"Inspect upstream label mapping before training: {mapping}")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    template = getattr(model.config, "nli_template", None)
    if not template:
        raise ValueError("Inspect the pretrained OpenJev nli_template before training")
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
                                           target_modules=["q_proj", "v_proj"], lora_dropout=.05))
    def prepare(records):
        rows = [{"premise": p, "hypothesis": h, "labels": mapping[label]} for r in records for p, h, label in pairs(r)]
        data = Dataset.from_list(rows)
        return data.map(lambda b: tokenizer([template.format(premise=p, hypothesis=h) for p,h in zip(b["premise"], b["hypothesis"])], truncation=True, max_length=512), batched=True,
                        remove_columns=["premise", "hypothesis"])
    out = Path(output)
    args = TrainingArguments(output_dir=str(out / "openjev_checkpoints"), num_train_epochs=epochs,
        per_device_train_batch_size=2, per_device_eval_batch_size=4, gradient_accumulation_steps=8,
        learning_rate=2e-4, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", save_total_limit=2, seed=seed, report_to="none",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = Trainer(model=model, args=args, train_dataset=prepare(train), eval_dataset=prepare(validation),
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        compute_metrics=lambda p: {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro", zero_division=0)})
    checkpoints = list((out / "openjev_checkpoints").glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if checkpoints else None)
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(out / "openjev", safe_serialization=True)
    tokenizer.save_pretrained(out / "openjev")
    return {"threshold": 0.0, "upstream": name, "subfolder": sub}


def evaluate(records, model=None, clear_cache=True):
    from sklearn.metrics import classification_report, confusion_matrix
    predictions, timings = [], []
    if model is not None and clear_cache:
        model._cache.clear()
    for record in records:
        start = time.perf_counter()
        response = understand(record["text"], model)
        timings.append((time.perf_counter() - start) * 1000)
        predictions.append(response)
    true_status = [r["expected"]["status"] for r in records]
    pred_status = [p["status"] for p in predictions]
    true_intents = [(r["expected"]["command"] or {}).get("intent", "unsupported") for r in records]
    pred_intents = [(p["command"] or {}).get("intent", "unsupported") for p in predictions]
    def fields(command):
        return {(r["category"], r["destination"]) for r in (command or {}).get("rules", [])}
    tp = fp = fn = exact = false_accept = valid_count = 0
    field_correct = {"category": 0, "destination": 0, "mode": 0}
    entity_counts = {"category":[0,0,0],"destination":[0,0,0]}
    for record, prediction in zip(records, predictions):
        gold = record["expected"]
        equal = prediction["status"] == gold["status"]
        if gold["status"] in ("valid", "needs_clarification"):
            equal &= canonical(prediction["command"]) == canonical(gold["command"])
        exact += equal
        false_accept += gold["status"] != "valid" and prediction["status"] == "valid"
        a, b = fields(gold["command"]), fields(prediction["command"])
        tp += len(a & b); fp += len(b - a); fn += len(a - b)
        for field, index in [("category", 0), ("destination", 1)]:
            gold_values,pred_values = {v[index] for v in a},{v[index] for v in b}
            field_correct[field] += gold_values == pred_values
            totals = entity_counts[field]
            totals[0] += len(gold_values & pred_values)
            totals[1] += len(pred_values-gold_values)
            totals[2] += len(gold_values-pred_values)
        field_correct["mode"] += (gold["command"] or {}).get("mode") == (prediction["command"] or {}).get("mode")
        valid_count += gold["status"] == "valid"
    labels = sorted(set(true_status + pred_status))
    report = {"n": len(records), "complete_command_accuracy": exact / max(1, len(records)),
        "false_acceptance_rate": false_accept / max(1, len(records) - valid_count),
        "relation_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "entity_accuracy": {k: v / max(1, len(records)) for k, v in field_correct.items()},
        "entity_f1": {k: 2*v[0]/max(1,2*v[0]+v[1]+v[2]) for k,v in entity_counts.items()},
        "intent": classification_report(true_intents, pred_intents, output_dict=True, zero_division=0),
        "status": classification_report(true_status, pred_status, output_dict=True, zero_division=0),
        "confusion_labels": labels, "confusion_matrix": confusion_matrix(true_status, pred_status, labels=labels).tolist(),
        "latency_ms": {"p50": float(np.percentile(timings, 50)), "p95": float(np.percentile(timings, 95))}}
    if model is not None:
        raw = [model.predict(r["text"]) for r in records]
        report["raw_model_intent"] = classification_report([r["intent"] for r in records], [r["intent"] for r in raw], output_dict=True, zero_division=0)
        report["raw_model_mode_accuracy"] = float(np.mean([a["mode"] == b["mode"] for a,b in zip(records, raw)]))
    return report, predictions


def select_and_calibrate(output, validation):
    """Only validation chooses thresholds and model; test is never consulted."""
    out = Path(output)
    manifest = json.loads((out / "manifest.json").read_text())
    ranks = []
    for kind in manifest["models"]:
        model = IntentBundle(out, kind)
        model.predict(validation[0]["text"])
        start = time.perf_counter()
        for record in validation[1:17]:
            model.predict(record["text"])
        uncached_latency = (time.perf_counter()-start)*1000 / max(1,len(validation[1:17]))
        best = None
        for threshold in [0.0, .5, .7, .85, .95]:
            model.threshold = threshold
            report, _ = evaluate(validation, model, clear_cache=False)
            rank = (report["false_acceptance_rate"], -report["complete_command_accuracy"], -report["raw_model_intent"]["macro avg"]["f1-score"],uncached_latency)
            if best is None or rank < best[0]:
                best = rank, threshold, report
        manifest["models"][kind].update(threshold=best[1], validation=best[2],uncached_cpu_ms=uncached_latency)
        ranks.append((best[0], kind))
        del model
    manifest["selected"] = min(ranks)[1]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
