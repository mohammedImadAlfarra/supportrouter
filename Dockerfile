# SupportRouter API image (Phase 9). Serves only; it never trains.
# No model, data or secret inside: models/ and mlruns/ are mounted by Docker Compose at run
# time, the key comes from `.env` through `env_file`, the checksum is verified on
# load.
FROM python:3.11.17-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Non-root user with the usual first-user uid, so files it writes to the mounted logs/ stay
# owned by the developer on the host.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Dependencies first: this layer is rebuilt only when requirements.txt changes, not on every
# code edit. requirements.txt is exported from uv.lock with hashes, so pip runs in
# hash-checking mode; the extra index serves the CPU-only torch wheel (torch==...+cpu).
COPY requirements.txt .
RUN pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Source last (changes most often). Only what the service imports: package, API, config.
COPY --chown=app:app src/ src/
COPY --chown=app:app app/ app/
COPY --chown=app:app configs/ configs/
RUN mkdir -p logs models mlruns && chown app:app logs models mlruns

# requirements.txt is exported with --no-emit-project, so the package is found via PYTHONPATH.
# Offline flags: the T1 tokenizer and weights come from the mounted models/ folder only.
ENV PYTHONPATH=/app/src \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

USER app
EXPOSE 8000

# Slim images have no curl; the standard library is enough for the health probe.
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
