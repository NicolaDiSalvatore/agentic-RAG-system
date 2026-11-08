FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY scripts ./scripts
RUN mkdir -p /app/.cache && chown -R appuser:appuser /app/.cache

USER appuser

EXPOSE 8000

# Ready means the persisted collection exists, has points, and matches its
# ingestion manifest. The index is built by the `ingest` compose service.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready')"

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
