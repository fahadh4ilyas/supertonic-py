#!/usr/bin/env python3
"""Standalone uvicorn runner for Supertonic serve with full CLI control.

Runs the same :func:`supertonic.server.create_app` but exposes every flag
as a CLI argument so you can use ``--reload``, ``--workers``, and all
uvicorn options directly.  Unlike ``supertonic serve``, this script
supports uvicorn's reload mode (because uvicorn needs an *import string*,
which cannot carry constructor arguments).

Usage::

    python scripts/serve.py --host 0.0.0.0 --port 7788 --reload
    python scripts/serve.py --model supertonic-3 --no-use-onnx --workers 2

All arguments after ``--`` are forwarded to uvicorn as-is.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Insert the repo root so we can always ``import supertonic`` when running
# this script from anywhere in the tree.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ──────────────────────────────────────────────────────────────────
# Factory – called by uvicorn (with ``factory=True``).
# Reads configuration from environment variables set by main().
# ──────────────────────────────────────────────────────────────────


def create_app():
    """uvicorn-compatible FastAPI factory (reads env vars)."""
    from supertonic.server.app import create_app as _create_app  # noqa: PLC0415

    cors_raw = os.environ.get("SUPERTONIC_SERVE_CORS", "")
    cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()] if cors_raw else None
    custom_styles_raw = os.environ.get("SUPERTONIC_SERVE_CUSTOM_STYLES_DIR", "")
    custom_styles_dir = Path(custom_styles_raw) if custom_styles_raw else None
    filter_chars = os.environ.get("SUPERTONIC_FILTER_CHARS", "0") == "1"

    return _create_app(
        model=os.environ.get("SUPERTONIC_SERVE_MODEL", "supertonic-3"),
        use_onnx=os.environ.get("SUPERTONIC_SERVE_USE_ONNX", "1") == "1",
        device=os.environ.get("SUPERTONIC_SERVE_DEVICE") or None,
        cors_origins=cors_origins,
        custom_styles_dir=custom_styles_dir,
        filter_chars=filter_chars,
    )


# ──────────────────────────────────────────────────────────────────
# Main – parse flags, set env vars, exec uvicorn
# ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supertonic serve with full uvicorn control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/serve.py --host 0.0.0.0 --port 7788 --reload
  python scripts/serve.py --model supertonic-3 --no-use-onnx --workers 2
  python scripts/serve.py --cors "http://localhost:*,chrome-extension://*"
        """,
    )

    # ---- supertonic serve flags ----
    parser.add_argument(
        "--model",
        type=str,
        default="supertonic-3",
        help="Model to load (default: supertonic-3)",
    )
    parser.add_argument(
        "--use-onnx",
        action="store_true",
        default=True,
        help="Use ONNX runtime (default).",
    )
    parser.add_argument(
        "--no-use-onnx",
        action="store_false",
        dest="use_onnx",
        help="Use PyTorch SupertonicModel instead of ONNX.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for PyTorch backend (e.g. 'cuda', 'cpu'). Only used with --no-use-onnx.",
    )
    parser.add_argument(
        "--cors",
        type=str,
        default=None,
        help="Comma-separated CORS origins (default: none)",
    )
    parser.add_argument(
        "--custom-styles-dir",
        type=str,
        default=None,
        help="Directory for user-imported voice style JSONs",
    )
    parser.add_argument(
        "--filter-chars",
        action="store_true",
        default=False,
        help="Silently drop unsupported characters during tokenization instead of raising an error.",
    )

    # ---- uvicorn flags ----
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7788,
        help="Port to listen on (default: 7788)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload on code changes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (multiprocessing; 1 = single worker)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="uvicorn log level (default: info)",
    )

    args = parser.parse_args()

    # ---- Set env vars for the factory ----
    os.environ["SUPERTONIC_SERVE_MODEL"] = args.model
    os.environ["SUPERTONIC_SERVE_USE_ONNX"] = "1" if args.use_onnx else "0"
    if args.device:
        os.environ["SUPERTONIC_SERVE_DEVICE"] = args.device
    if args.cors:
        os.environ["SUPERTONIC_SERVE_CORS"] = args.cors
    if args.custom_styles_dir:
        os.environ["SUPERTONIC_SERVE_CUSTOM_STYLES_DIR"] = args.custom_styles_dir
    if args.filter_chars:
        os.environ["SUPERTONIC_FILTER_CHARS"] = "1"

    # ---- Print startup info ----
    print(f"supertonic serve listening on http://{args.host}:{args.port}")
    print(f"  backend: {'ONNX' if args.use_onnx else 'PyTorch'}")
    print(f"  model:   {args.model}")
    print(f"  docs:    http://{args.host}:{args.port}/docs")

    # ---- Run uvicorn ----
    import uvicorn

    # Build the import string for this module's create_app factory.
    # When running as a script, __name__ is "__main__", but uvicorn needs
    # the real module path for reload to work.  We use the full dotted path.
    factory_path = "scripts.serve:create_app"

    uvicorn.run(
        factory_path,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
        factory=True,
    )


if __name__ == "__main__":
    main()
