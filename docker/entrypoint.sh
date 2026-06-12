#!/bin/bash
set -e

MODEL="${SUPERTONIC_SERVE_MODEL:-supertonic-3}"
CACHE_DIR="${SUPERTONIC_CACHE_DIR:-/cache/supertonic}"

# ── Ensure model files are present ──
MODEL_TTS_JSON="${CACHE_DIR}/tts.json"
MODEL_SAFETENSORS="${CACHE_DIR}/model.safetensors"

if [ ! -f "${MODEL_TTS_JSON}" ] || [ ! -f "${MODEL_SAFETENSORS}" ]; then
    echo "Model files not found in ${CACHE_DIR}."
    LOCKFILE="${CACHE_DIR}/.download.lock"
    mkdir -p "${CACHE_DIR}"
    exec 9>"${LOCKFILE}"
    flock 9
    # Double-check after acquiring lock (another worker may have done it)
    if [ ! -f "${MODEL_TTS_JSON}" ] || [ ! -f "${MODEL_SAFETENSORS}" ]; then
        # ── Try host cache first ──
        if [ -f "/host_cache/supertonic3/tts.json" ] && [ -f "/host_cache/supertonic3/model.safetensors" ]; then
            echo "  Copying model from host cache (/host_cache/supertonic3)..."
            cp -r /host_cache/supertonic3/* "${CACHE_DIR}/"
        elif [ -f "/host_cache/supertonic/model.safetensors" ]; then
            echo "  Copying model from host cache (/host_cache/supertonic)..."
            cp -r /host_cache/supertonic/* "${CACHE_DIR}/"
        else
            echo "  Downloading ONNX model from HuggingFace Hub..."
            supertonic download --model "${MODEL}"

            echo "  Converting ONNX weights to PyTorch safetensors..."
            python scripts/load_onnx_weights.py --output-dir "${CACHE_DIR}"
        fi
        echo "Model ready."
    else
        echo "  Model already available."
    fi
    exec 9>&-
fi

# ── Build serve.py arguments from environment variables ──
ARGS=(
    --host "${SUPERTONIC_SERVE_HOST:-0.0.0.0}"
    --port "${SUPERTONIC_SERVE_PORT:-7788}"
    --workers "${SUPERTONIC_SERVE_WORKERS:-1}"
    --model "${MODEL}"
)

# ONNX vs PyTorch
if [ "${SUPERTONIC_SERVE_USE_ONNX:-1}" = "0" ]; then
    ARGS+=(--no-use-onnx)
    if [ -n "${SUPERTONIC_SERVE_DEVICE}" ]; then
        ARGS+=(--device "${SUPERTONIC_SERVE_DEVICE}")
    fi
fi

# Optional flags
if [ -n "${SUPERTONIC_SERVE_CORS}" ]; then
    ARGS+=(--cors "${SUPERTONIC_SERVE_CORS}")
fi
if [ -n "${SUPERTONIC_SERVE_CUSTOM_STYLES_DIR}" ]; then
    ARGS+=(--custom-styles-dir "${SUPERTONIC_SERVE_CUSTOM_STYLES_DIR}")
fi
if [ -n "${SUPERTONIC_SERVE_RELOAD}" ] && [ "${SUPERTONIC_SERVE_RELOAD}" = "1" ]; then
    ARGS+=(--reload)
fi
if [ -n "${SUPERTONIC_SERVE_LOG_LEVEL}" ]; then
    ARGS+=(--log-level "${SUPERTONIC_SERVE_LOG_LEVEL}")
fi

echo "Starting supertonic serve with: ${ARGS[*]}"
exec python scripts/serve.py "${ARGS[@]}"
