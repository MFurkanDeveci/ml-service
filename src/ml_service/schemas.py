"""API'nin girdi ve çıktı sözleşmesi. Pydantic geçersiz veriyi modele ulaşmadan reddeder."""

from pydantic import BaseModel, ConfigDict, Field


class IrisFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid",  # Bilinmeyen alan gelirse 422 döner; yazım hataları sessizce yutulmaz
        json_schema_extra={
            "examples": [
                {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
            ]
        },
    )

    sepal_length: float = Field(gt=0, le=15, description="Çanak yaprak uzunluğu (cm)")
    sepal_width: float = Field(gt=0, le=15, description="Çanak yaprak genişliği (cm)")
    petal_length: float = Field(gt=0, le=15, description="Taç yaprak uzunluğu (cm)")
    petal_width: float = Field(gt=0, le=15, description="Taç yaprak genişliği (cm)")


class Prediction(BaseModel):
    label: str
    probabilities: dict[str, float]


class PredictionResponse(Prediction):
    model_version: str


class BatchRequest(BaseModel):
    items: list[IrisFeatures] = Field(min_length=1)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str


class ModelInfo(BaseModel):
    version: str
    feature_names: list[str]
    class_names: list[str]
    metrics: dict[str, float]
