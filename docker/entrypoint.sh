#!/bin/bash
set -e

# ── Build serve.py arguments from environment variables ──
ARGS=(
    --host "${SUPERTONIC_SERVE_HOST:-0.0.0.0}"
    --port "${SUPERTONIC_SERVE_PORT:-7788}"
    --workers "${SUPERTONIC_SERVE_WORKERS:-1}"
    --model "${SUPERTONIC_SERVE_MODEL:-supertonic-3}"
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
