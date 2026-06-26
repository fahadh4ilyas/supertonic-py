"""Uploaded voice management for Supertonic serve.

Stores voice styles as safetensors files with {style_ttl, style_dp} tensors
and metadata in the safetensors header __metadata__ (matching vLLM-Omni format).

API follows vLLM-Omni's voice management spec (speech_api.md).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import safetensors.torch
import torch

logger = logging.getLogger(__name__)


def save_voice(
    dir_path: Path,
    name: str,
    style_ttl: torch.Tensor,
    style_dp: torch.Tensor,
    consent: str,
    ref_text: Optional[str] = None,
    speaker_description: Optional[str] = None,
    mime_type: str = "audio/wav",
    file_size: int = 0,
) -> dict:
    """Save a voice to a safetensors file. Overwrites if name exists."""
    dir_path.mkdir(parents=True, exist_ok=True)

    created_at = str(int(time.time()))

    tensors = {
        "style_ttl": style_ttl.cpu().detach().contiguous(),
        "style_dp": style_dp.cpu().detach().contiguous(),
    }

    metadata: dict[str, str] = {
        "name": name,
        "consent": consent,
        "created_at": created_at,
        "mime_type": mime_type,
        "file_size": str(file_size),
    }
    if ref_text:
        metadata["ref_text"] = ref_text
    if speaker_description:
        metadata["speaker_description"] = speaker_description

    safetensors.torch.save_file(tensors, str(dir_path / f"{name}.safetensors"), metadata=metadata)

    meta = {
        "name": name,
        "consent": consent,
        "created_at": int(created_at),
        "mime_type": mime_type,
        "file_size": file_size,
    }
    if ref_text:
        meta["ref_text"] = ref_text
    if speaker_description:
        meta["speaker_description"] = speaker_description
    return meta


def delete_voice(dir_path: Path, name: str) -> bool:
    """Delete a voice. Returns True if deleted, False if not found."""
    path = dir_path / f"{name}.safetensors"
    if path.exists():
        path.unlink()
        return True
    return False


def load_voice(dir_path: Path, name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load style_ttl, style_dp from a saved voice. Returns None if not found."""
    path = dir_path / f"{name}.safetensors"
    if not path.exists():
        return None
    state = safetensors.torch.load_file(str(path))
    return state["style_ttl"], state["style_dp"]


def list_voices(dir_path: Path) -> list[dict]:
    """List all uploaded voices with metadata from the safetensors header."""
    if not dir_path.exists():
        return []
    voices = []
    for path in sorted(dir_path.glob("*.safetensors")):
        try:
            with open(path, "rb") as f:
                data = f.read()
            header_len = int.from_bytes(data[:8], "little")
            header = json.loads(data[8:8 + header_len])
            meta = header.get("__metadata__", {})
            if "name" in meta:
                # Convert string fields back to proper types
                if "created_at" in meta:
                    meta["created_at"] = int(meta["created_at"])
                if "file_size" in meta:
                    meta["file_size"] = int(meta["file_size"])
                voices.append(meta)
        except Exception:
            logger.warning("Failed to read voice file: %s", path)
    return voices


def voice_exists(dir_path: Path, name: str) -> bool:
    """Check if a voice exists."""
    return (dir_path / f"{name}.safetensors").exists()
