"""Model loading and inference helpers for RoboQC."""

from __future__ import annotations

import json
import importlib.util
import os
import sys
from typing import Callable, Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def minilm_available():
    return all(importlib.util.find_spec(module) is not None for module in ("sentence_transformers","torch","sklearn","joblib"))


def torch_available():
    return importlib.util.find_spec("torch") is not None


class IntentInferenceEngine:
    def __init__(self, model_dir: str = "models", preferred: Optional[str] = None):
        self.model_dir = model_dir
        self.preferred = preferred  # baseline | jepa | auto | None
        self.kind: Optional[str] = None
        self.baseline = None
        self.minilm = None
        self.jepa = None
        self.vocab = None
        self.labels: list = []
        self.config: dict = {}
        self.max_len = 128
        self._loaded = False

    def _resolve_kind(self) -> str:
        manifest_path = os.path.join(self.model_dir, "manifest.json")
        if self.preferred == "industrial" or (not self.preferred and os.path.isfile(manifest_path)):
            with open(manifest_path, encoding="utf-8") as f:
                if json.load(f).get("schema_version") == "industrial-1.0":
                    return "industrial"
        if self.preferred and self.preferred in ("baseline", "minilm", "jepa"):
            return self.preferred
        sel_path = os.path.join(self.model_dir, "selected_model.txt")
        if os.path.exists(sel_path):
            try:
                k = open(sel_path, encoding="utf-8").read().strip()
                if k in ("baseline", "minilm", "jepa"):
                    return k
            except Exception:
                pass
        # default: baseline if exists else jepa
        if os.path.exists(os.path.join(self.model_dir, "minilm.joblib")) and minilm_available():
            return "minilm"
        if os.path.exists(os.path.join(self.model_dir, "baseline.joblib")):
            return "baseline"
        if os.path.exists(os.path.join(self.model_dir, "jepa.pt")):
            return "jepa"
        raise FileNotFoundError("No hay modelo entrenado. Ejecuta 03_train_model.py.")

    def load(self) -> None:
        self.kind = self._resolve_kind()
        if self.kind == "industrial":
            from roboqc.industrial.models import IntentBundle
            self.bundle = IntentBundle(self.model_dir)
        elif self.kind == "baseline":
            from roboqc.intent.baseline import BaselineIntentClassifier
            path = os.path.join(self.model_dir, "baseline.joblib")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Modelo baseline no encontrado: {path}")
            clf = BaselineIntentClassifier(model_path=path)
            clf.load(path)
            self.baseline = clf
            labels_path = os.path.join(self.model_dir, "baseline_labels.json")
            if os.path.exists(labels_path):
                self.labels = json.load(open(labels_path, encoding="utf-8"))
        elif self.kind == "minilm":
            from roboqc.intent.minilm import MiniLMIntentClassifier
            path = os.path.join(self.model_dir, "minilm.joblib")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Modelo MiniLM no encontrado: {path}")
            clf = MiniLMIntentClassifier(model_path=path)
            clf.load(path)
            self.minilm = clf
            self.labels = clf.labels
        else:
            import torch
            from roboqc.intent.jepa_model import CharVocab, JEPAIntentModel
            if not torch_available():
                raise RuntimeError("PyTorch no disponible.")
            path = os.path.join(self.model_dir, "jepa.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Modelo JEPA no encontrado: {path}")
            ckpt = torch.load(path, map_location="cpu")
            self.vocab = CharVocab.from_dict(ckpt["vocab"])
            self.labels = ckpt["labels"]
            self.config = ckpt.get("config", {})
            self.max_len = self.config.get("max_len", 128)
            model = JEPAIntentModel(
                vocab_size=len(self.vocab.stoi),
                num_classes=len(self.labels),
                embed_dim=self.config.get("embed_dim", 32),
                hidden=self.config.get("hidden", 64),
                latent=self.config.get("latent", 64),
            )
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self.jepa = model
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def predict_intent(self, text: str) -> str:
        self._ensure_loaded()
        if self.kind == "industrial":
            prediction = self.bundle.predict(text)
            return {"sort": "SORT_OBJECTS", "stop": "STOP_PROCESS", "resume": "RESUME_PROCESS", "cancel": "CANCEL_ACTION"}.get(prediction["intent"], "UNKNOWN")
        if self.kind == "baseline":
            return self.baseline.predict(text)
        if self.kind == "minilm":
            return self.minilm.predict(text)
        import torch
        ids = self.vocab.encode(text, self.max_len)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            z = self.jepa.encode(x)
            logits = self.jepa.classifier(z)
            idx = int(logits.argmax(dim=1).item())
        return self.labels[idx]

    def predict_confidence(self, text: str) -> float:
        self._ensure_loaded()
        if self.kind == "industrial":
            return self.bundle.predict(text)["confidence"]
        if self.kind == "baseline":
            return self.baseline.predict_proba(text)
        if self.kind == "minilm":
            return self.minilm.predict_proba(text)
        import torch
        ids = self.vocab.encode(text, self.max_len)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            z = self.jepa.encode(x)
            logits = self.jepa.classifier(z)
            proba = torch.softmax(logits, dim=1)
            return float(proba.max().item())

    def predict_topk(self, text: str, k: int = 5):
        self._ensure_loaded()
        if self.kind == "industrial":
            return [(self.predict_intent(text), self.predict_confidence(text))]
        if self.kind == "baseline":
            X = self.baseline.vectorizer.transform([text])
            proba = self.baseline.model.predict_proba(X)[0]
            classes = list(self.baseline.model.classes_)
            pairs = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
            return pairs[:k]
        if self.kind == "minilm":
            X = self.minilm._encode([text])
            proba = self.minilm.model.predict_proba(X)[0]
            classes = list(self.minilm.model.classes_)
            pairs = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
            return pairs[:k]
        import torch
        ids = self.vocab.encode(text, self.max_len)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            z = self.jepa.encode(x)
            logits = self.jepa.classifier(z)
            proba = torch.softmax(logits, dim=1)[0]
        pairs = sorted(zip(self.labels, proba.tolist()), key=lambda x: x[1], reverse=True)
        return pairs[:k]

    def make_classifier_callable(self) -> Callable[[str], str]:
        def _cls(t: str) -> str:
            return self.predict_intent(t)
        return _cls

    def make_confidence_callable(self) -> Callable[[str], float]:
        def _conf(t: str) -> float:
            return self.predict_confidence(t)
        return _conf

    @property
    def loaded(self) -> bool:
        return self._loaded


def load_engine(model_dir: str = "models", preferred: Optional[str] = None) -> IntentInferenceEngine:
    eng = IntentInferenceEngine(model_dir=model_dir, preferred=preferred)
    return eng


MODEL_FILES = {"baseline": "baseline.joblib", "minilm": "minilm.joblib", "jepa": "jepa.pt"}


def available_models(model_dir: str = "models") -> Dict[str, bool]:
    """Map each intent model to whether its weights exist and its runtime is installed."""
    runtime = {"baseline": True, "minilm": minilm_available(), "jepa": torch_available()}
    models = {
        name: os.path.exists(os.path.join(model_dir, fname)) and runtime[name]
        for name, fname in MODEL_FILES.items()
    }
    models["industrial"] = os.path.isfile(os.path.join(model_dir, "manifest.json"))
    return models
