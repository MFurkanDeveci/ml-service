"""Testler için ortak hazırlık: geçici klasörde model eğitir, test istemcisi kurar."""

import pytest
from fastapi.testclient import TestClient

from ml_service.api import create_app
from ml_service.config import Settings
from ml_service.train import train


@pytest.fixture(scope="session")
def artifact_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("artifacts") / "model.joblib"
    train(path)
    return path


@pytest.fixture
def client(artifact_path):
    settings = Settings(artifact_path=artifact_path, max_batch_size=5)
    # "with" bloğu lifespan'i çalıştırır, yani model gerçekten yüklenir
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def setosa():
    return {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
