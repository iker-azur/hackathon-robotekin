"""Portable intent bundles. Heavy backends are imported only when selected."""

import json
from pathlib import Path

LABELS = ["sort|quality", "sort|color", "stop|none", "resume|none", "cancel|none", "unsupported|none"]


def decode(label, confidence=1.0):
    intent, mode = label.split("|")
    return {"intent": intent, "mode": mode, "confidence": float(confidence)}


class ONNXSentenceEncoder:
    """Sentence embeddings through ONNX Runtime, independent of Optimum."""

    def __init__(self, path):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        path = Path(path)
        self.config = json.loads((path / "onnx_config.json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.session = ort.InferenceSession(str(path / self.config["model_file"]), providers=["CPUExecutionProvider"])

    def encode(self, texts, show_progress_bar=False, batch_size=32):
        import numpy as np
        outputs = []
        for start in range(0,len(texts),batch_size):
            tokens = self.tokenizer(texts[start:start+batch_size],padding=True,truncation=True,
                                    max_length=self.config["max_seq_length"],return_tensors="np")
            inputs = {item.name: tokens[item.name].astype(np.int64) for item in self.session.get_inputs()}
            outputs.append(self.session.run(["sentence_embedding"],inputs)[0])
        return np.concatenate(outputs) if outputs else np.empty((0,self.config["embedding_dim"]),dtype=np.float32)


class IntentBundle:
    def __init__(self, path, kind=None):
        self.root = Path(path)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != "industrial-1.0":
            raise ValueError("Unsupported industrial artifact version")
        self.kind = kind or self.manifest["selected"]
        if self.kind not in self.manifest["models"]:
            raise ValueError(f"Model {self.kind} was not exported")
        self.threshold = self.manifest["models"][self.kind].get("threshold", 0.0)
        self._cache = {}
        if self.kind == "tfidf":
            import joblib
            self.model = joblib.load(self.root / "tfidf.joblib")
        elif self.kind == "minilm":
            import joblib
            self.model = joblib.load(self.root / "minilm.joblib")
            info = self.manifest["models"][self.kind]
            path = self.root / info.get("encoder_path", "encoder")
            if info.get("encoder_backend") == "onnx_native":
                self.encoder = ONNXSentenceEncoder(path)
            else:
                from sentence_transformers import SentenceTransformer
                kwargs = {"backend": info.get("encoder_backend", "torch")}
                if info.get("encoder_file"):
                    kwargs["model_kwargs"] = {"file_name": info["encoder_file"]}
                self.encoder = SentenceTransformer(str(path), device="cpu", **kwargs)
        elif self.kind == "jepa":
            import torch
            from roboqc.intent.jepa_model import CharVocab, JEPAIntentModel
            ckpt = torch.load(self.root / "jepa.pt", map_location="cpu", weights_only=True)
            self.vocab = CharVocab.from_dict(ckpt["vocab"])
            self.labels = ckpt["labels"]
            self.max_len = ckpt["max_len"]
            self.model = JEPAIntentModel(len(self.vocab.stoi), len(self.labels))
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
        elif self.kind == "openjev":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.root / "openjev", local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.root / "openjev", local_files_only=True).eval()
        else:
            raise ValueError(f"Unknown model: {self.kind}")

    def relation_scores(self, text, hypotheses):
        import torch
        template = getattr(self.model.config, "nli_template", None)
        if not template:
            raise ValueError("OpenJev artifact must define its pretrained nli_template")
        encoded = self.tokenizer([template.format(premise=text, hypothesis=h) for h in hypotheses], return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.inference_mode():
            probs = self.model(**encoded).logits.softmax(-1)
        mapping = {str(k).lower(): int(v) for k, v in self.model.config.label2id.items()}
        ent = mapping.get("entailment")
        if ent is None:
            raise ValueError("OpenJev artifact is missing the entailment label mapping")
        return probs[:, ent].cpu().numpy()

    def predict(self, text):
        if text in self._cache:
            value = self._cache[text]
            return decode("unsupported|none",value["confidence"]) if value["confidence"] < self.threshold else value
        if self.kind in ("tfidf", "minilm"):
            x = [text] if self.kind == "tfidf" else self.encoder.encode([text], show_progress_bar=False)
            p = self.model.predict_proba(x)[0]
            index = p.argmax()
            value = decode(str(self.model.classes_[index]), p[index])
        elif self.kind == "jepa":
            import torch
            if len(text) > self.max_len:
                return decode("unsupported|none", 0.0)
            with torch.inference_mode():
                p = self.model.classifier(self.model.encode(torch.tensor([self.vocab.encode(text, self.max_len)]))).softmax(-1)[0]
            value = decode(self.labels[int(p.argmax())], float(p.max()))
        else:
            hs = ["La orden pide clasificar por calidad.", "La orden pide clasificar por color.",
                  "La orden pide detener la clasificacion.", "La orden pide continuar clasificando.",
                  "La orden pide cancelar la orden.", "La orden solicita una accion no admitida."]
            p = self.relation_scores(text, hs)
            index = p.argmax()
            value = decode(LABELS[index], p[index])
        if len(self._cache) >= 4096:
            self._cache.clear()
        self._cache[text] = value
        return decode("unsupported|none",value["confidence"]) if value["confidence"] < self.threshold else value
