"""Model katmanının testleri: HTTP olmadan, doğrudan Python'dan."""

import pytest

from ml_service.model import ModelNotFoundError, ModelService
from ml_service.schemas import IrisFeatures


def test_model_quality_gate(artifact_path):
    # Kalite kapısı: model bu eşiğin altına düşerse deploy edilmemeli
    model = ModelService(artifact_path)
    assert model.metrics["test_accuracy"] >= 0.9


def test_probabilities_sum_to_one(artifact_path, setosa):
    model = ModelService(artifact_path)
    prediction = model.predict([IrisFeatures(**setosa)])[0]
    assert prediction.label == "setosa"
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-3)


def test_missing_artifact_gives_clear_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="ml-train"):
        ModelService(tmp_path / "yok.joblib")
