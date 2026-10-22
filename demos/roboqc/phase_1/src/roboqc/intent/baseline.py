"""Lightweight TF-IDF + linear classifier baseline for Spanish intent recognition."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    _SKLEARN = True
except Exception:  # pragma: no cover
    _SKLEARN = False


def is_available() -> bool:
    return _SKLEARN


class BaselineIntentClassifier:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.vectorizer = None
        self.model = None
        self.labels: List[str] = []
        self._loaded = False

    def fit(self, texts: List[str], labels: List[str], seed: int = 42) -> None:
        if not _SKLEARN:
            raise RuntimeError("scikit-learn no esta instalado.")
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
            lowercase=True,
        )
        X = self.vectorizer.fit_transform(texts)
        from collections import Counter
        counts = Counter(labels)
        min_count = min(counts.values()) if counts else 0
        base = LogisticRegression(
            max_iter=1000,
            C=2.0,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
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
        if not self._loaded and self.model is None:
            if self.model_path and os.path.exists(self.model_path):
                self.load(self.model_path)
            else:
                raise FileNotFoundError("Modelo baseline no encontrado. Entrena primero con 03_train_model.py.")
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("Modelo baseline no cargado.")
        X = self.vectorizer.transform([text])
        pred = self.model.predict(X)[0]
        return str(pred)

    def predict_proba(self, text: str) -> float:
        if not self._loaded and self.model is None:
            if self.model_path and os.path.exists(self.model_path):
                self.load(self.model_path)
            else:
                raise FileNotFoundError("Modelo baseline no encontrado.")
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        return float(proba.max())

    def save(self, path: str) -> None:
        if joblib is None:
            raise RuntimeError("joblib no esta instalado.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {"vectorizer": self.vectorizer, "model": self.model, "labels": self.labels},
            path,
        )

    def load(self, path: str) -> None:
        if joblib is None:
            raise RuntimeError("joblib no esta instalado.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo baseline no encontrado: {path}")
        obj = joblib.load(path)
        self.vectorizer = obj["vectorizer"]
        self.model = obj["model"]
        self.labels = obj.get("labels", [])
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded or self.model is not None
