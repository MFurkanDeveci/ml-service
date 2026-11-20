"""FastAPI uygulaması: HTTP katmanı. İş mantığı model.py'de, burada sadece yönlendirme var."""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request

from ml_service.config import Settings
from ml_service.model import ModelService
from ml_service.schemas import (
    BatchRequest,
    BatchResponse,
    IrisFeatures,
    ModelInfo,
    PredictionResponse,
)

logger = logging.getLogger("ml_service")


def get_model(request: Request) -> ModelService:
    """Dependency: lifespan'de yüklenen modeli endpoint'lere verir."""
    return request.app.state.model


ModelDep = Annotated[ModelService, Depends(get_model)]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Model her istekte değil, uygulama açılırken BİR KEZ yüklenir
        app.state.model = ModelService(settings.artifact_path)
        logger.info("Model yüklendi, sürüm=%s", app.state.model.version)
        yield

    app = FastAPI(title="Iris Tahmin Servisi", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        logger.info(
            "%s %s -> %s (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/model/info", response_model=ModelInfo)
    def model_info(model: ModelDep) -> ModelInfo:
        return ModelInfo(
            version=model.version,
            feature_names=model.feature_names,
            class_names=model.class_names,
            metrics=model.metrics,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(features: IrisFeatures, model: ModelDep):
        prediction = model.predict([features])[0]
        return PredictionResponse(**prediction.model_dump(), model_version=model.version)

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(request: BatchRequest, model: ModelDep):
        if len(request.items) > settings.max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=f"En fazla {settings.max_batch_size} kayıt gönderilebilir.",
            )
        return BatchResponse(predictions=model.predict(request.items), model_version=model.version)

    return app


app = create_app()
