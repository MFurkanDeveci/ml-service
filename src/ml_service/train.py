"""Modeli eğitir, değerlendirir ve tek bir dosya (artifact) olarak kaydeder."""

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_service.config import Settings

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SEED = 42


def train(artifact_path: Path) -> dict:
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Ölçekleme modelin İÇİNDE: API'de aynı ön işleme otomatik uygulanır (training-serving skew yok)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    pipeline.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, pipeline.predict(X_test))

    metrics = {
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "test_accuracy": round(float(test_acc), 4),
    }
    bundle = {
        "pipeline": pipeline,
        "feature_names": FEATURE_NAMES,
        "class_names": [str(c) for c in data.target_names],
        "metrics": metrics,
        "version": datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    return bundle


def main() -> None:
    settings = Settings()
    bundle = train(settings.artifact_path)
    print(f"Model kaydedildi: {settings.artifact_path}")
    print(f"Sürüm: {bundle['version']}")
    print(json.dumps(bundle["metrics"], indent=2))


if __name__ == "__main__":
    main()
