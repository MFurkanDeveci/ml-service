"""Uygulama ayarları: hepsi ortam değişkeninden veya .env dosyasından okunur."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    artifact_path: Path = Path("artifacts/model.joblib")
    log_level: str = "INFO"
    max_batch_size: int = 100
