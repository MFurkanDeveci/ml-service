"""Kaydedilmiş modeli yükler ve tahmin yapar. API katmanından bağımsızdır, ayrı test edilir."""

from pathlib import Path

import joblib
import numpy as np

from ml_service.schemas import IrisFeatures, Prediction


class ModelNotFoundError(RuntimeError):
    pass


class ModelService:
    def __init__(self, artifact_path: Path):
        if not artifact_path.exists():
            raise ModelNotFoundError(
                f"Model dosyası bulunamadı: {artifact_path}. Önce 'uv run ml-train' çalıştır."
            )
        bundle = joblib.load(artifact_path)
        self.pipeline = bundle["pipeline"]
        self.feature_names: list[str] = bundle["feature_names"]
        self.class_names: list[str] = bundle["class_names"]
        self.metrics: dict[str, float] = bundle["metrics"]
        self.version: str = bundle["version"]

    def predict(self, items: list[IrisFeatures]) -> list[Prediction]:
        # Sütun sırasını isimden kuruyoruz: JSON'daki alan sırası ne olursa olsun doğru eşleşir
        X = np.array([[getattr(item, name) for name in self.feature_names] for item in items])
        probas = self.pipeline.predict_proba(X)
        return [
            Prediction(
                label=self.class_names[int(row.argmax())],
                probabilities={
                    name: round(float(p), 4) for name, p in zip(self.class_names, row, strict=True)
                },
            )
            for row in probas
        ]
