# ── Base image — override with --build-arg for GPU support ──
# CPU (default):
#   docker build -t supertonic .
# GPU (CUDA 12):
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.1-runtime-ubuntu22.04 -t supertonic:gpu .
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# ── System dependencies ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python setup ──
WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard] pydantic python-multipart

# Copy package source
COPY supertonic/ supertonic/
COPY scripts/ scripts/
COPY pyproject.toml .

# Install the package in editable mode
RUN pip install -e ".[serve]"

# ── Environment defaults (overridable at runtime) ──
ENV SUPERTONIC_SERVE_MODEL=supertonic-3
ENV SUPERTONIC_SERVE_USE_ONNX=1
ENV SUPERTONIC_SERVE_HOST=0.0.0.0
ENV SUPERTONIC_SERVE_PORT=7788
ENV SUPERTONIC_SERVE_WORKERS=1
ENV SUPERTONIC_SERVE_DEVICE=""
ENV SUPERTONIC_SERVE_CORS=""
ENV SUPERTONIC_SERVE_CUSTOM_STYLES_DIR=""
ENV SUPERTONIC_LOG_LEVEL=INFO
ENV SUPERTONIC_CACHE_DIR=/cache/supertonic
ENV SUPERTONIC_INTRA_OP_THREADS=""
ENV SUPERTONIC_INTER_OP_THREADS=""

# ── Expose port ──
EXPOSE 7788

# ── Entrypoint ──
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
