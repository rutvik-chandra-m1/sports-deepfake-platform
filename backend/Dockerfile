# Backend image: FastAPI + PyTorch inference service.
#
# Two things about this file are deliberate and easy to get wrong:
#
# 1. TORCH IS INSTALLED FROM THE CPU INDEX, FIRST.
#    The default PyPI wheel bundles CUDA libraries (~2.5GB unpacked) that are
#    dead weight here -- there is no GPU in the target environment and the
#    pipeline runs on CPU by design. Installing torch from
#    download.pytorch.org/whl/cpu first means the later `-r requirements.txt`
#    sees the pin already satisfied and leaves it alone.
#
# 2. libGL AND libglib ARE REQUIRED EVEN THOUGH WE USE opencv-*-headless.
#    requirements.txt pins opencv-python-headless precisely to avoid GUI
#    bindings -- but MediaPipe declares a hard dependency on
#    opencv-contrib-python (the non-headless build), so pip installs it
#    anyway and `import cv2` resolves to a binary linked against libGL.so.1.
#    Without these packages the image builds fine and then dies at the first
#    import with "libGL.so.1: cannot open shared object file".
#
# Build from the PROJECT ROOT (it needs scripts/ and models/ too):
#   docker build -f backend/Dockerfile -t sdp-backend .

FROM python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- system libraries -------------------------------------------------------
# libgl1 + libglib2.0-0: see note 2 above (OpenCV/MediaPipe native deps).
# curl: used by the compose healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- python dependencies ----------------------------------------------------
# Copied before the source so edits to application code don't invalidate the
# (very slow) dependency layer.
COPY backend/requirements.txt backend/requirements.txt

RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.13.0 torchvision==0.28.0 \
    && pip install --no-cache-dir -r backend/requirements.txt

# --- application ------------------------------------------------------------
COPY backend/ backend/
COPY scripts/ scripts/
# Trained artefacts (probe head + fusion calibration). Small, and the service
# silently degrades to untrained heuristics without them, so they are baked in
# rather than mounted.
COPY models/configs/ models/configs/

# --- model cache ------------------------------------------------------------
# Pull the ~330MB transformer, Haar cascade and MediaPipe bundle at BUILD
# time. A container that has to reach the public internet to answer its first
# request is not deployable behind a firewall, and every replica would
# re-download independently.
ENV MODELS_DIR=/app/models/pretrained
RUN python scripts/warm_model_cache.py

# --- runtime ----------------------------------------------------------------
ENV APP_ENV=production \
    DATABASE_URL=sqlite:////data/db/app.db \
    UPLOAD_DIR=/data/uploads \
    REPORTS_DIR=/data/reports \
    BACKEND_HOST=0.0.0.0 \
    BACKEND_PORT=8000

# Runs as a non-root user. /data is the only writable path the app needs.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/uploads /data/reports /data/db \
    && chown -R appuser:appuser /data /app
USER appuser

EXPOSE 8000

COPY --chown=appuser:appuser backend/docker-entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
