# Slim rather than alpine: the ONNX Runtime wheels are built against glibc, and
# on musl they fall back to compiling from source -- a long build for no gain.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep model weights outside the image so a rebuild does not re-download
    # them and the layer cache stays small.
    FASTEMBED_CACHE_PATH=/models

WORKDIR /app

# Dependencies first, in their own layer: they change far less often than the
# source, so editing a Python file does not reinstall the whole stack.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY tools/ ./tools/
COPY pyproject.toml ./

ENV PYTHONPATH=/app/src:/app

# A non-root user, because the container only ever reads mounted data.
RUN useradd --create-home --uid 1000 auditor \
    && mkdir -p /models \
    && chown -R auditor:auditor /app /models
USER auditor

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

# One worker on purpose. Each worker loads its own copy of the embedding and
# reranker models; on a small host, four workers is four times the RAM for a
# service that is bound by the LLM call, not by Python.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
