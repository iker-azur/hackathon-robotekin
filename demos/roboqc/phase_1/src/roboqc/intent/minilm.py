"""Sentence-Transformers (MiniLM) based intent classifier (CPU, transfer learning)."""

from __future__ import annotations

import os
from typing import Any, List, Optional

try:
    from sentence_transformers import SentenceTransformer
    _ST = True
except Exception:
    _ST = False
    SentenceTransformer = None

try:
    import joblib
except Exception:
    joblib = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    _SK = True
except Exception:
    _SK = False


def is_available() -> bool:
    return _ST and _SK and (joblib is not None)


class MiniLMIntentClassifier:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 model_path: Optional[str] = None):
        self.model_name = model_name
        self.model_path = model_path
        self.encoder = None
        self.model = None
        self.labels: List[str] = []
        self._loaded = False

    def _load_encoder(self):
        if self.encoder is None:
            if not _ST:
                raise RuntimeError("sentence-transformers no disponible.")
            self.encoder = SentenceTransformer(self.model_name)

    def _encode(self, texts: List[str]):
        self._load_encoder()
        return self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=32)

    def fit(self, texts: List[str], labels: List[str], seed: int = 42) -> None:
        if not is_available():
            raise RuntimeError("sentence-transformers o scikit-learn no disponibles.")
        import numpy as np
        from collections import Counter
        print("Codificando textos con MiniLM...")
        X = self._encode(texts)
        print("Entrenando clasificador lineal...")
        counts = Counter(labels)
        min_count = min(counts.values()) if counts else 0
        base = LogisticRegression(
            max_iter=2000,
            C=2.0,
            class_weight="balanced",
            random_state=seed,
        )
        if min_count >= 3:
            self.model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        elif min_count >= 2:
            self.model = CalibratedClassifierCV(base, method="sigmoid", cv=2)
        else:
            self.model = base
        self.model.fit(X, labels)
        self.labels = sorted(set(labels))

    def predict(self, text: str) -> str:
        self._ensure_loaded()
        X = self._encode([text])
        return str(self.model.predict(X)[0])

    def predict_batch(self, texts: List[str]) -> List[str]:
        self._ensure_loaded()
        if not texts:
            return []
        X = self._encode(texts)
        return [str(p) for p in self.model.predict(X)]

    def predict_proba(self, text: str) -> float:
        self._ensure_loaded()
        X = self._encode([text])
        proba = self.model.predict_proba(X)[0]
        return float(proba.max())

    def predict_proba_batch(self, texts: List[str]) -> List[float]:
        self._ensure_loaded()
        if not texts:
            return []
        X = self._encode(texts)
        proba = self.model.predict_proba(X)
        return [float(row.max()) for row in proba]

    def _ensure_loaded(self):
        if self._loaded:
            return
        if self.model is not None:
            self._loaded = True
            return
        if self.model_path and os.path.exists(self.model_path):
            self.load(self.model_path)
        else:
            raise FileNotFoundError("Modelo MiniLM no encontrado. Entrena primero.")

    def save(self, path: str) -> None:
        if joblib is None:
            raise RuntimeError("joblib no disponible.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {"model_name": self.model_name, "model": self.model, "labels": self.labels},
            path,
        )

    def load(self, path: str) -> None:
        if joblib is None:
            raise RuntimeError("joblib no disponible.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo MiniLM no encontrado: {path}")
        obj = joblib.load(path)
        self.model_name = obj.get("model_name", self.model_name)
        self.model = obj["model"]
        self.labels = obj.get("labels", [])
        self._loaded = True
        # encoder loaded lazily
