# ---- 1. aşama: bağımlılıkları kur ve modeli eğit ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# Önce sadece bağımlılık dosyaları: kod değişince bu katman cache'ten gelir, build hızlanır
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

# Ders amaçlı model build sırasında eğitiliyor. Gerçek projede model registry'den indirilir.
RUN .venv/bin/ml-train

# ---- 2. aşama: sadece çalışması gerekenler, root olmayan kullanıcıyla ----
FROM python:3.12-slim
RUN useradd --create-home appuser
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app /app

ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "ml_service.api:app", "--host", "0.0.0.0", "--port", "8000"]