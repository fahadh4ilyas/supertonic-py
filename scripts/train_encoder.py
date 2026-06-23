#!/usr/bin/env python3
"""Train the AudioEncoder to extract voice style vectors from reference audio.

Two training modes, driven by the dataset:

    Text-only (synthetic):  text → frozen TTS (with known style) → audio
                            → encoder → predicted style → MSE vs true style
                            + optional TextEncoder/DurationPredictor aux losses

    Real-audio (cloned):    voice_encoder audio → encoder → predicted style
                            → frozen TTS → synthesized audio
                            → Mel loss vs voice_tts target audio

Usage:
    python scripts/train_encoder.py --dataset data.jsonl --output_dir ./checkpoints

Dataset format (JSONL):
    {"text_encoder": "Long text for encoder input...",
     "text_tts": "Shorter text for TTS-through loss",
     "lang": "en",
     "voice_encoder": "path/to/ref_audio.wav",   # optional: real audio mode
     "voice_tts": "path/to/target_audio.wav"}     # optional: Mel reconstruction target
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supertonic.model import SupertonicModel, AudioEncoder, MelSpectrogram
from supertonic.core import UnicodeProcessor
from supertonic.loader import (
    load_voice_style_from_json_file,
    get_cache_dir,
    download_model,
    has_all_onnx_modules,
)

# Import ONNX→PyTorch conversion utilities from sibling script
from scripts.load_onnx_weights import (
    load_all_onnx_weights,
    load_weights_to_model,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Style split is configurable via CLI args (--train_styles, --val_styles).
# Defaults below are used when args are not provided.
_DEFAULT_TRAIN_STYLES = ["F1", "F2", "F3", "F4", "M1", "M2", "M3", "M4"]
_DEFAULT_VAL_STYLES = ["F5", "M5"]

# From tts.json: typical speech rate ≈ 15 chars/sec at speed=1.0
# 15-30s target ⇒ 225-450 chars at speed=1.0, scaled by speed
TARGET_MIN_CHARS = 200
TARGET_MAX_CHARS = 500


# ──────────────────────────────────────────────────────────────────────────────
# Model bootstrapping (download + ONNX→PyTorch conversion)
# ──────────────────────────────────────────────────────────────────────────────


def ensure_model_ready(model_dir: Path) -> Path:
    """Ensure the PyTorch model exists, downloading and converting if needed.

    Checks in order:
        1. model.safetensors exists → ready, return model_dir
        2. ONNX files exist in model_dir/onnx/ → convert to PyTorch
        3. Neither exists → download from HuggingFace, then convert

    Returns the model_dir (which contains model.safetensors + tts.json).
    """
    model_dir = Path(model_dir)
    safetensors_path = model_dir / "model.safetensors"

    if safetensors_path.exists():
        print(f"PyTorch model found: {safetensors_path}")
        return model_dir

    onnx_dir = model_dir / "onnx"
    if not has_all_onnx_modules(model_dir):
        print(f"Model not found in {model_dir}. Downloading supertonic-3 from HuggingFace...")
        download_model(model_dir, "supertonic-3")

    if not safetensors_path.exists():
        print("Converting ONNX weights to PyTorch safetensors...")
        _convert_onnx_to_pytorch(onnx_dir, model_dir)

    return model_dir


def _convert_onnx_to_pytorch(onnx_dir: Path, output_dir: Path) -> None:
    """Convert ONNX models to PyTorch safetensors format."""
    import shutil

    onnx_weights = load_all_onnx_weights(onnx_dir)
    print(f"  Extracted {len(onnx_weights)} ONNX parameter arrays")

    model = SupertonicModel(
        config=str(onnx_dir / "tts.json"),
        unicode_indexer=str(onnx_dir / "unicode_indexer.json"),
    )
    model.eval()

    loaded = load_weights_to_model(onnx_weights, model, verbose=False)
    state_dict = model.state_dict()
    state_dict.update(loaded)

    # Distribute shared k_context to all 4 StyleCrossAttention blocks
    kctx_key = "vector_field.main_blocks.5.k_context"
    if kctx_key in state_dict:
        kctx_val = state_dict[kctx_key]
        for idx in (12, 19, 26):
            other = f"vector_field.main_blocks.{idx}.k_context"
            if other in state_dict:
                state_dict[other] = kctx_val.clone()

    # Distribute shared increments and theta to all 4 RoPE blocks
    for buf_name in ("increments", "theta"):
        src_key = f"vector_field.main_blocks.2.{buf_name}"
        if src_key in state_dict:
            val = state_dict[src_key]
            for idx in (9, 16, 23):
                dst_key = f"vector_field.main_blocks.{idx}.{buf_name}"
                if dst_key in state_dict:
                    state_dict[dst_key] = val.clone()

    model.load_state_dict(state_dict, strict=False)

    # Copy config files
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_dir / "tts.json", output_dir / "tts.json")
    shutil.copy(onnx_dir / "unicode_indexer.json", output_dir / "unicode_indexer.json")

    # Save safetensors
    import safetensors.torch
    safe_state = {k: v.contiguous() if isinstance(v, torch.Tensor) else torch.tensor(v)
                  for k, v in state_dict.items()}
    safetensors.torch.save_file(safe_state, str(output_dir / "model.safetensors"))

    loaded_n = sum(v.numel() for v in loaded.values())
    total_n = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {loaded_n:,}/{total_n:,} params ({100*loaded_n/total_n:.1f}%) → {output_dir / 'model.safetensors'}")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────


class EncoderDataset(Dataset):
    """JSONL dataset with text_encoder, text_tts, and lang keys.

    Pre-tokenizes all texts at init time to avoid CPU bottleneck during training.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        text_processor: UnicodeProcessor | None = None,
        min_chars: int = TARGET_MIN_CHARS,
        max_chars: int = TARGET_MAX_CHARS,
        concat_short: bool = True,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.concat_short = concat_short

        self.samples: list[dict] = []
        self._load()

        # Pre-tokenize all texts (major CPU bottleneck otherwise)
        if text_processor is not None:
            self._tokenize(text_processor)

    def _load(self) -> None:
        with open(self.jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                # text_encoder is required; text_tts defaults to text_encoder
                if "text_encoder" not in sample:
                    continue
                if "text_tts" not in sample:
                    sample["text_tts"] = sample["text_encoder"]
                # Resolve voice paths relative to dataset location
                if "voice_encoder" in sample:
                    sample["_voice_enc_path"] = str(
                        self.jsonl_path.parent / sample["voice_encoder"]
                    )
                else:
                    sample["_voice_enc_path"] = None
                if "voice_tts" in sample:
                    sample["_voice_tts_path"] = str(
                        self.jsonl_path.parent / sample["voice_tts"]
                    )
                else:
                    sample["_voice_tts_path"] = None
                self.samples.append(sample)

        if not self.samples:
            raise ValueError(f"No valid samples found in {self.jsonl_path}")

        print(f"Loaded {len(self.samples)} raw samples from {self.jsonl_path}")

        self._prepare_encoder_texts()
        print(f"After length filtering/concat: {len(self.samples)} samples")

    def _tokenize(self, text_processor: UnicodeProcessor) -> None:
        """Pre-tokenize all texts to numpy arrays."""
        print("Pre-tokenizing texts...", flush=True)
        for sample in tqdm(self.samples, desc="Tokenizing", unit="sample", dynamic_ncols=True):
            sample["_enc_ids"], sample["_enc_mask"] = text_processor(
                [sample["text_encoder"]], sample.get("lang", "na")
            )
            sample["_tts_ids"], sample["_tts_mask"] = text_processor(
                [sample["text_tts"] or sample["text_encoder"]],
                sample.get("lang", "na"),
            )
        print(f"Tokenization complete ({len(self.samples)} samples)")

    def _prepare_encoder_texts(self) -> None:
        """Ensure text_encoder texts are in the 15-30s range.

        If concat_short is True, short texts are concatenated to reach
        the target length. Texts that are too long are truncated.
        """
        prepared = []

        short_texts: list[dict] = []

        for sample in self.samples:
            # Real audio samples don't need text length filtering
            if sample.get("_voice_enc_path") is not None:
                prepared.append(sample)
                continue
            text_len = len(sample["text_encoder"])
            if text_len >= self.min_chars:
                # Good length — use as-is, truncate if too long
                if text_len > self.max_chars:
                    # Truncate at last sentence boundary within range
                    trunc_point = self.max_chars
                    for sep in (". ", "? ", "! ", "\n"):
                        idx = sample["text_encoder"].rfind(sep, 0, self.max_chars)
                        if idx > self.min_chars:
                            trunc_point = idx + len(sep)
                            break
                    sample = {**sample, "text_encoder": sample["text_encoder"][:trunc_point].strip()}
                prepared.append(sample)
            elif self.concat_short:
                short_texts.append(sample)
            # else: skip — too short and not concatenating

        # Concatenate short texts to build longer encoder inputs
        if self.concat_short and short_texts:
            random.shuffle(short_texts)
            buffer_text = ""
            buffer_lang = short_texts[0].get("lang", "na")
            buffer_tts = short_texts[0].get("text_tts", "")

            for sample in short_texts:
                candidate = f"{buffer_text} {sample['text_encoder']}".strip()
                if len(candidate) >= self.min_chars:
                    prepared.append({
                        "text_encoder": candidate[:self.max_chars],
                        "text_tts": buffer_tts or sample.get("text_tts", ""),
                        "lang": buffer_lang,
                        "_voice_enc_path": None,
                        "_voice_tts_path": None,
                    })
                    buffer_text = ""
                else:
                    buffer_text = candidate
                    buffer_lang = sample.get("lang", buffer_lang)
                    if not buffer_tts:
                        buffer_tts = sample.get("text_tts", "")

            # Don't waste the last partial buffer — pad with a repeated segment
            if buffer_text and len(buffer_text) >= 50:
                while len(buffer_text) < self.min_chars:
                    buffer_text = f"{buffer_text}. {buffer_text}"
                prepared.append({
                    "text_encoder": buffer_text[:self.max_chars],
                    "text_tts": buffer_tts or "",
                    "lang": buffer_lang,
                    "_voice_enc_path": None,
                    "_voice_tts_path": None,
                })

        self.samples = prepared

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Collate function
# ──────────────────────────────────────────────────────────────────────────────


def collate_fn(batch: list[dict]) -> dict:
    """Collate a batch — includes pre-tokenized arrays and voice_ref paths."""
    return {
        "_enc_ids": np.stack([s["_enc_ids"] for s in batch]),
        "_enc_mask": np.stack([s["_enc_mask"] for s in batch]),
        "_tts_ids": np.stack([s["_tts_ids"] for s in batch]),
        "_tts_mask": np.stack([s["_tts_mask"] for s in batch]),
        "_voice_enc_path": [s["_voice_enc_path"] for s in batch],
        "_voice_tts_path": [s["_voice_tts_path"] for s in batch],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────


def load_style_vectors(
    styles_dir: Path,
    style_names: list[str],
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Load style_ttl and style_dp from JSON files as torch tensors.

    Returns:
        Dict mapping style name → (style_ttl, style_dp) tensors on device.
    """
    styles = {}
    for name in style_names:
        style_path = styles_dir / f"{name}.json"
        if not style_path.exists():
            print(f"Warning: style file not found: {style_path}")
            continue
        style_obj = load_voice_style_from_json_file(style_path)
        style_ttl = torch.from_numpy(style_obj.ttl).to(device)
        style_dp = torch.from_numpy(style_obj.dp).to(device)
        styles[name] = (style_ttl, style_dp)
    return styles


def generate_audio(
    tts_model: SupertonicModel,
    text_ids_np: np.ndarray,
    text_mask_np: np.ndarray,
    style_ttl: torch.Tensor,
    style_dp: torch.Tensor,
    total_steps: int = 4,
    speed: float = 1.0,
) -> torch.Tensor | None:
    """Generate audio from pre-tokenized text + style using the frozen TTS model.

    Returns:
        Waveform tensor of shape (1, num_samples), or None if generation fails.
    """
    with torch.no_grad():
        seq_len = text_ids_np.shape[1]
        if seq_len < 3:
            return None

        text_ids = torch.from_numpy(text_ids_np).to(style_ttl.device)
        text_mask = torch.from_numpy(text_mask_np).to(style_ttl.device)

        try:
            wav, _dur = tts_model.forward(
                text_ids=text_ids,
                style_ttl=style_ttl,
                style_dp=style_dp,
                text_mask=text_mask,
                total_steps=total_steps,
                speed=speed,
            )
        except RuntimeError as e:
            print(f"Warning: TTS forward failed: {e}")
            return None

        wav_np = wav.cpu().numpy()[0]
        active = np.where(np.abs(wav_np) > 0.002)[0]
        if len(active) > 0:
            margin = int(tts_model.sample_rate * 0.02)
            start = max(0, active[0] - margin)
            end = min(len(wav_np), active[-1] + margin)
            wav = wav[:, start:end]

    return wav


def _load_real_audio(path: str, target_sr: int, device: torch.device) -> torch.Tensor | None:
    """Load a real audio file, resample if needed, return as (1, T) tensor on device."""
    try:
        waveform, sr = torchaudio.load(path)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    except Exception:
        return None
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Trim silence
    wav_np = waveform.numpy()[0]
    active = np.where(np.abs(wav_np) > 0.002)[0]
    if len(active) > 0:
        margin = int(target_sr * 0.02)
        start = max(0, active[0] - margin)
        end = min(len(wav_np), active[-1] + margin)
        waveform = waveform[:, start:end]
    return waveform.to(device)


# Shared mel-spectrogram extractor for TTS-through loss (created once)
_mel_extractor: MelSpectrogram | None = None


def _get_mel(tts_model: SupertonicModel, device: torch.device) -> MelSpectrogram:
    """Get or create a shared MelSpectrogram matching the TTS model's audio config."""
    global _mel_extractor
    if _mel_extractor is None:
        spec = tts_model.config.get("ae", {}).get("encoder", {}).get("spec_processor", {})
        _mel_extractor = MelSpectrogram(
            sample_rate=spec.get("sample_rate", tts_model.sample_rate),
            n_fft=spec.get("n_fft", 2048),
            win_length=spec.get("win_length", 2048),
            hop_length=spec.get("hop_length", 512),
            n_mels=spec.get("n_mels", 228),
        ).to(device)
    return _mel_extractor


def tts_through_loss(
    tts_model: SupertonicModel,
    text_ids_np: np.ndarray,
    text_mask_np: np.ndarray,
    style_ttl_pred: torch.Tensor,
    style_dp_pred: torch.Tensor,
    target_wav: torch.Tensor,
    total_steps: int = 4,
    speed: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TTS-through reconstruction loss: generate audio from predicted style,
    compare mel-spectrograms with the target audio."""
    device = style_ttl_pred.device
    text_ids = np.expand_dims(text_ids_np, axis=0) if text_ids_np.ndim == 1 else text_ids_np
    text_mask = np.expand_dims(text_mask_np, axis=0) if text_mask_np.ndim == 2 else text_mask_np
    text_ids_t = torch.from_numpy(text_ids).to(device)
    text_mask_t = torch.from_numpy(text_mask).to(device)

    dur_pred = tts_model.duration_predictor(text_ids_t, style_dp_pred, text_mask_t)
    dur_actual = target_wav.shape[-1] / tts_model.sample_rate
    dur_loss = F.smooth_l1_loss(
        dur_pred, torch.tensor([dur_actual], device=device),
    )

    wav_pred, _ = tts_model.forward(
        text_ids=text_ids_t, style_ttl=style_ttl_pred, style_dp=style_dp_pred,
        text_mask=text_mask_t, total_steps=total_steps, speed=speed,
    )

    mel = _get_mel(tts_model, device)
    mel_pred = mel(wav_pred)
    mel_target = mel(target_wav.unsqueeze(0))
    min_len = min(mel_pred.shape[-1], mel_target.shape[-1])
    mel_loss = F.mse_loss(mel_pred[..., :min_len], mel_target[..., :min_len])

    return mel_loss, dur_loss


def text_encoder_loss(
    tts_model: SupertonicModel,
    text_ids_np: np.ndarray,
    text_mask_np: np.ndarray,
    style_ttl_pred: torch.Tensor,
    style_ttl_true: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE between TextEncoder outputs for predicted vs true styles."""
    text_ids = np.expand_dims(text_ids_np, axis=0) if text_ids_np.ndim == 1 else text_ids_np
    text_mask = np.expand_dims(text_mask_np, axis=0) if text_mask_np.ndim == 2 else text_mask_np

    with torch.no_grad():
        enc_true = tts_model.text_encoder(
            torch.from_numpy(text_ids).to(style_ttl_pred.device),
            style_ttl_true,
            torch.from_numpy(text_mask).to(style_ttl_pred.device),
        )

    enc_pred = tts_model.text_encoder(
        torch.from_numpy(text_ids).to(style_ttl_pred.device),
        style_ttl_pred,
        torch.from_numpy(text_mask).to(style_ttl_pred.device),
    )

    return F.mse_loss(enc_pred, enc_true)


def duration_predictor_loss(
    tts_model: SupertonicModel,
    text_ids_np: np.ndarray,
    text_mask_np: np.ndarray,
    style_dp_pred: torch.Tensor,
    style_dp_true: torch.Tensor,
) -> torch.Tensor:
    """Compute log-space Huber loss between DurationPredictor outputs."""
    text_ids = np.expand_dims(text_ids_np, axis=0) if text_ids_np.ndim == 1 else text_ids_np
    text_mask = np.expand_dims(text_mask_np, axis=0) if text_mask_np.ndim == 2 else text_mask_np

    with torch.no_grad():
        dur_true = tts_model.duration_predictor(
            torch.from_numpy(text_ids).to(style_dp_pred.device),
            style_dp_true,
            torch.from_numpy(text_mask).to(style_dp_pred.device),
        )

    dur_pred = tts_model.duration_predictor(
        torch.from_numpy(text_ids).to(style_dp_pred.device),
        style_dp_pred,
        torch.from_numpy(text_mask).to(style_dp_pred.device),
    )

    return F.smooth_l1_loss(torch.log(dur_pred.clamp(min=1e-6)), torch.log(dur_true.clamp(min=1e-6)))


# ──────────────────────────────────────────────────────────────────────────────
# Main training
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train AudioEncoder to extract voice style vectors from audio"
    )
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Path to JSONL dataset file")
    parser.add_argument("--model_dir", type=Path,
                        default=get_cache_dir("supertonic-3"),
                        help="Directory with TTS model (tts.json, model.safetensors, voice_styles/). "
                             "Defaults to ~/.cache/supertonic3 (or $SUPERTONIC_CACHE_DIR if set).")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory to save encoder checkpoints")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device (cuda, cpu)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Samples per batch (keep small — TTS generation is memory-heavy)")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--total_steps", type=int, default=8,
                        help="Diffusion steps for audio generation during training (default: 8)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed for generated audio")
    parser.add_argument("--use_tts_encoder", action="store_true",
                        help="Train the encoder inside the loaded TTS model instead of a fresh one")
    parser.add_argument("--lambda_text", type=float, default=0.1,
                        help="Weight for text encoder output consistency loss (0 = disabled)")
    parser.add_argument("--lambda_dur", type=float, default=0.1,
                        help="Weight for duration predictor consistency loss (0 = disabled)")
    parser.add_argument("--lambda_style_ttl", type=float, default=1.0,
                        help="Weight for style_ttl MSE loss (voice character)")
    parser.add_argument("--lambda_style_dp", type=float, default=1.0,
                        help="Weight for style_dp MSE loss (speaking rate)")
    parser.add_argument("--lambda_mel", type=float, default=1.0,
                        help="Weight for mel-spectrogram loss in real-audio path")
    parser.add_argument("--lambda_real_dur", type=float, default=1.0,
                        help="Weight for duration loss in real-audio path")
    parser.add_argument("--num_styles_per_train_step", type=str, default="1",
                        help="Number of training styles per sample: int (e.g. '3') or 'all'. "
                             "Default: 1 (random single style, original behavior)")
    parser.add_argument("--num_styles_per_val_step", type=str, default="all",
                        help="Number of validation styles per sample: int (e.g. '2') or 'all'. "
                             "Default: all")
    parser.add_argument("--train_styles", type=str, nargs="+",
                        default=_DEFAULT_TRAIN_STYLES,
                        help="Training voice style names (default: F1 F2 F3 F4 M1 M2 M3 M4)")
    parser.add_argument("--val_styles", type=str, nargs="+",
                        default=_DEFAULT_VAL_STYLES,
                        help="Validation voice style names (default: F5 M5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (used as fallback for train/val seeds)")
    parser.add_argument("--train_seed", type=int, default=None,
                        help="Seed for training randomness (dataset shuffle, style sampling). "
                             "Falls back to --seed if not set.")
    parser.add_argument("--val_seed", type=int, default=None,
                        help="Seed for validation randomness (sample selection, style sampling). "
                             "Falls back to --seed if not set.")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--val_every", type=int, default=5,
                        help="Run validation every N epochs")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of dataset to use for validation (default: 0.1 = 10%%)")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log detailed metrics every N epochs")
    parser.add_argument("--bar_refresh_every", type=int, default=4,
                        help="Update tqdm batch progress bar every N gradient steps (default: 4)")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Stop after N validation checks without improvement. "
                             "Set to 0 to disable. Default: 10")
    parser.add_argument("--early_stopping_metric", type=str, default="val_style",
                        choices=["val_style", "val_ttl", "val_dp"],
                        help="Which validation metric to monitor for early stopping.")
    parser.add_argument("--amp", type=str, default="bf16", choices=["bf16", "fp16", "none"],
                        help="Mixed precision mode: bf16 (best for Ampere+, no scaler), "
                             "fp16 (needs scaler), none (fp32)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from a checkpoint file")
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    train_seed = args.train_seed if args.train_seed is not None else args.seed
    val_seed = args.val_seed if args.val_seed is not None else args.seed
    train_rng = random.Random(train_seed)
    val_rng = random.Random(val_seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save training config for reproducibility
    config_path = args.output_dir / "train_config.json"
    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Training config saved to {config_path}")

    # ── Ensure model exists (download + convert ONNX→PyTorch if needed) ──────
    args.model_dir = ensure_model_ready(args.model_dir)

    # ── Load frozen TTS model ────────────────────────────────────────────────
    print(f"\nLoading frozen TTS model from {args.model_dir}...")
    tts_model = SupertonicModel.from_pretrained(str(args.model_dir), device=device)
    tts_model.eval()
    for param in tts_model.parameters():
        param.requires_grad = False
    print(f"TTS model loaded. Sample rate: {tts_model.sample_rate}")

    # Load text processor
    indexer_path = args.model_dir / "unicode_indexer.json"
    if not indexer_path.exists():
        raise FileNotFoundError(f"unicode_indexer.json not found in {args.model_dir}")
    text_processor = UnicodeProcessor(str(indexer_path))

    # ── Load voice styles ────────────────────────────────────────────────────
    styles_dir = args.model_dir / "voice_styles"
    train_styles = load_style_vectors(styles_dir, args.train_styles, device)
    val_styles = load_style_vectors(styles_dir, args.val_styles, device)

    if not train_styles:
        raise RuntimeError(f"No training styles loaded from {styles_dir}")
    print(f"Training styles: {list(train_styles.keys())}")
    print(f"Validation styles: {list(val_styles.keys())}")

    # ── Create / select encoder model ────────────────────────────────────────
    if args.use_tts_encoder:
        encoder = tts_model.audio_encoder
        for param in encoder.parameters():
            param.requires_grad = True
        print("Using encoder from loaded TTS model")
    else:
        with open(args.model_dir / "tts.json") as f:
            tts_config = json.load(f)
        encoder = AudioEncoder(config=tts_config)
        encoder = encoder.to(device)
    encoder.train()

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"AudioEncoder params: {total_params:,}")

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2,
    )

    use_amp = args.amp != "none"
    use_scaler = args.amp == "fp16"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    scaler = GradScaler("cuda", enabled=use_scaler) if (use_scaler and torch.cuda.is_available()) else None

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_val_metric = {m: float("inf") for m in ["val_style", "val_ttl", "val_dp"]}
    early_stop_counter = 0

    # ── Resume checkpoint ────────────────────────────────────────────────────
    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed at epoch {start_epoch}, step {global_step}")

    # ── Dataset (moved after text_processor init for pre-tokenization) ──────
    dataset = EncoderDataset(args.dataset, text_processor=text_processor)

    # Validation uses only text-only samples (real-audio samples have no
    # ground-truth style vectors to compare against).
    val_dataset = [s for s in dataset.samples if s.get("_voice_enc_path") is None]
    if val_dataset:
        print(f"Validation text-only samples: {len(val_dataset)}")
    else:
        print("WARNING: no text-only samples available for validation")

    # ── Training loop ────────────────────────────────────────────────────────
    train_style_names = sorted(train_styles.keys())

    epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Training", unit="epoch",
                       dynamic_ncols=True)

    for epoch in epoch_pbar:
        epoch_start = time.time()
        epoch_metrics = defaultdict(float)
        epoch_steps = 0
        epoch_synth_steps = 0  # only synthetic samples (for text/dur metrics)
        epoch_samples_skipped = 0

        # Shuffle and create dataloader
        indices = list(range(len(dataset)))
        train_rng.shuffle(indices)

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=torch.utils.data.SequentialSampler(indices),
            collate_fn=collate_fn,
            drop_last=False,
        )

        optimizer.zero_grad()
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}", unit="batch",
                          leave=False, dynamic_ncols=True)

        for batch_idx, batch in enumerate(batch_pbar):
            for i in range(len(batch["_enc_ids"])):
                enc_ids = batch["_enc_ids"][i]
                enc_mask = batch["_enc_mask"][i]
                tts_ids = batch["_tts_ids"][i]
                tts_mask = batch["_tts_mask"][i]
                voice_enc_path = batch["_voice_enc_path"][i]
                voice_tts_path = batch["_voice_tts_path"][i]

                # Step 1: Get reference audio
                if voice_enc_path is not None:
                    # Real audio: encoder input — process once (no style iteration)
                    audio = _load_real_audio(voice_enc_path, tts_model.sample_rate, device)
                    if audio is None:
                        epoch_samples_skipped += 1
                        continue
                    audio_dur = audio.shape[-1] / tts_model.sample_rate
                    if audio_dur < 10:
                        epoch_samples_skipped += 1
                        continue

                    # Step 2: Encode audio → predicted style
                    with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        style_ttl_pred, style_dp_pred = encoder(audio.unsqueeze(0))

                    # Step 3: Compute losses
                    if voice_tts_path is not None:
                        # Real audio: TTS-through reconstruction loss
                        audio_tts = _load_real_audio(voice_tts_path, tts_model.sample_rate, device)
                        if audio_tts is None:
                            epoch_samples_skipped += 1
                            continue
                        loss_mel, loss_dur = tts_through_loss(
                            tts_model, tts_ids, tts_mask,
                            style_ttl_pred, style_dp_pred, audio_tts,
                            total_steps=args.total_steps, speed=args.speed,
                        )
                        loss = args.lambda_mel * loss_mel + args.lambda_real_dur * loss_dur
                        epoch_metrics["loss_style"] += loss.item()
                        epoch_metrics["loss_style_ttl"] += loss_mel.item()
                        epoch_metrics["loss_style_dp"] += loss_dur.item()
                    else:
                        epoch_samples_skipped += 1
                        continue
                else:
                    # Synthetic: iterate over training styles for this sample
                    sample_total_loss = 0.0
                    sample_num_styles = 0

                    if args.num_styles_per_train_step == "all":
                        step_style_names = train_style_names
                    else:
                        n = max(1, int(args.num_styles_per_train_step))
                        step_style_names = train_rng.sample(
                            train_style_names, min(n, len(train_style_names))
                        )

                    for style_name in step_style_names:
                        style_ttl_true, style_dp_true = train_styles[style_name]

                        audio = generate_audio(
                            tts_model, enc_ids, enc_mask,
                            style_ttl_true, style_dp_true,
                            total_steps=args.total_steps, speed=args.speed,
                        )
                        if audio is None:
                            epoch_samples_skipped += 1
                            continue

                        audio_dur = audio.shape[-1] / tts_model.sample_rate
                        if audio_dur < 10:
                            epoch_samples_skipped += 1
                            continue

                        # Encode audio → predicted style
                        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                            style_ttl_pred, style_dp_pred = encoder(audio.unsqueeze(0))

                        # MSE against known style vectors
                        loss_ttl = F.mse_loss(style_ttl_pred, style_ttl_true)
                        loss_dp = F.mse_loss(style_dp_pred, style_dp_true)
                        loss_style = args.lambda_style_ttl * loss_ttl + args.lambda_style_dp * loss_dp
                        sample_total_loss = sample_total_loss + loss_style
                        sample_num_styles += 1

                        epoch_metrics["loss_style"] += loss_style.item()
                        epoch_metrics["loss_style_ttl"] += loss_ttl.item()
                        epoch_metrics["loss_style_dp"] += loss_dp.item()

                        if args.lambda_text > 0:
                            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                                loss_text = text_encoder_loss(
                                    tts_model, tts_ids, tts_mask,
                                    style_ttl_pred, style_ttl_true,
                                )
                            sample_total_loss = sample_total_loss + args.lambda_text * loss_text
                            epoch_metrics["loss_text"] += loss_text.item()

                        if args.lambda_dur > 0:
                            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                                loss_dur_loss = duration_predictor_loss(
                                    tts_model, tts_ids, tts_mask,
                                    style_dp_pred, style_dp_true,
                                )
                            sample_total_loss = sample_total_loss + args.lambda_dur * loss_dur_loss
                            epoch_metrics["loss_dur"] += loss_dur_loss.item()

                    if sample_num_styles == 0:
                        epoch_samples_skipped += 1
                        continue
                    loss = sample_total_loss / sample_num_styles
                    epoch_synth_steps += 1

                loss = loss / args.grad_accum_steps

                if use_amp and use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_steps += 1

            # Gradient accumulation step
            if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx == len(dataloader) - 1):
                if use_scaler and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                    optimizer.step()

                optimizer.zero_grad()
                global_step += 1

                # Update batch progress bar (throttled)
                if global_step % args.bar_refresh_every == 0:
                    denom = max(epoch_steps, 1)
                    synth_denom = max(epoch_synth_steps, 1)
                    batch_pbar.set_postfix({
                        "loss": f"{epoch_metrics['loss_style'] / denom:.4f}",
                        "ttl": f"{epoch_metrics['loss_style_ttl'] / denom:.4f}",
                        "dp": f"{epoch_metrics['loss_style_dp'] / denom:.4f}",
                        **({"text": f"{epoch_metrics['loss_text'] / synth_denom:.4f}"} if args.lambda_text > 0 else {}),
                        **({"dur": f"{epoch_metrics['loss_dur'] / synth_denom:.4f}"} if args.lambda_dur > 0 else {}),
                        "skip": epoch_samples_skipped,
                    }, refresh=False)

        batch_pbar.close()

        # ── Epoch summary ────────────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        denom = max(epoch_steps, 1)
        synth_denom = max(epoch_synth_steps, 1)
        avg_style = epoch_metrics["loss_style"] / denom

        epoch_pbar.set_postfix({
            "style": f"{avg_style:.4f}",
            "ttl": f"{epoch_metrics['loss_style_ttl'] / denom:.4f}",
            "dp": f"{epoch_metrics['loss_style_dp'] / denom:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            "skip": epoch_samples_skipped,
        })

        # Detailed log every log_every epochs (using tqdm.write to not break bars)
        if (epoch + 1) % args.log_every == 0:
            tqdm.write(
                f"── Epoch {epoch:3d} ── "
                f"Steps: {epoch_steps} | Time: {epoch_time:.1f}s | "
                f"Style: {avg_style:.6f} | "
                f"TTL: {epoch_metrics['loss_style_ttl'] / denom:.6f} | "
                f"DP: {epoch_metrics['loss_style_dp'] / denom:.6f}"
                + (f" | Text: {epoch_metrics['loss_text'] / synth_denom:.6f}"
                   if args.lambda_text > 0 else "")
                + (f" | Dur: {epoch_metrics['loss_dur'] / synth_denom:.6f}"
                   if args.lambda_dur > 0 else "")
                + f" | Skipped: {epoch_samples_skipped}"
            )

        scheduler.step()

        # ── Validation ───────────────────────────────────────────────────────
        if args.val_every > 0 and (epoch + 1) % args.val_every == 0 and val_styles and val_dataset:
            val_metrics = run_validation(
                tts_model, encoder, val_styles, val_dataset,
                device, args, val_rng,
            )
            val_loss = val_metrics["val_style"]
            tqdm.write(
                f"  Val ── style: {val_metrics['val_style']:.6f} | "
                f"ttl: {val_metrics['val_ttl']:.6f} | "
                f"dp: {val_metrics['val_dp']:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(args.output_dir / "best_model.pt",
                                encoder, optimizer, scheduler, epoch, global_step, best_val_loss)
                tqdm.write(f"  → Saved best model (val_style={best_val_loss:.6f})")

            # ── Early stopping ───────────────────────────────────────────────
            monitor = val_metrics[args.early_stopping_metric]
            if monitor < best_val_metric[args.early_stopping_metric]:
                best_val_metric[args.early_stopping_metric] = monitor
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if args.early_stopping_patience > 0 and early_stop_counter >= args.early_stopping_patience:
                tqdm.write(
                    f"  Early stopping triggered after {early_stop_counter} "
                    f"checks without improvement in {args.early_stopping_metric}"
                )
                break

        # ── Checkpoint ───────────────────────────────────────────────────────
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                args.output_dir / f"encoder_epoch{epoch:04d}.pt",
                encoder, optimizer, scheduler, epoch, global_step, best_val_loss,
            )

    # ── Final save ───────────────────────────────────────────────────────────
    save_checkpoint(
        args.output_dir / "final_model.pt",
        encoder, optimizer, scheduler, args.epochs - 1, global_step, best_val_loss,
    )
    tqdm.write(f"\nTraining complete. Model saved to {args.output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_validation(
    tts_model: SupertonicModel,
    encoder: AudioEncoder,
    val_styles: dict[str, tuple[torch.Tensor, torch.Tensor]],
    val_samples: list[dict],
    device: torch.device,
    args: argparse.Namespace,
    val_rng: random.Random,
) -> dict[str, float]:
    """Evaluate encoder on held-out styles using text-only samples.

    *val_samples* must be pre-filtered to text-only entries (no voice_encoder).

    Returns dict with keys: val_style, val_ttl, val_dp.
    """
    encoder.eval()
    total_style = 0.0
    total_ttl = 0.0
    total_dp = 0.0
    count = 0

    num_val_samples = max(1, int(len(val_samples) * args.val_ratio))

    val_style_names = sorted(val_styles.keys())
    indices = val_rng.sample(range(len(val_samples)), min(num_val_samples, len(val_samples)))

    for idx in indices:
        sample = val_samples[idx]

        if args.num_styles_per_val_step == "all":
            step_style_names = val_style_names
        else:
            n = max(1, int(args.num_styles_per_val_step))
            step_style_names = val_rng.sample(
                val_style_names, min(n, len(val_style_names))
            )

        for style_name in step_style_names:
            style_ttl_true, style_dp_true = val_styles[style_name]

            audio = generate_audio(
                tts_model, sample["_enc_ids"], sample["_enc_mask"],
                style_ttl_true, style_dp_true,
                total_steps=args.total_steps, speed=args.speed,
            )
            if audio is None:
                continue

            audio_dur = audio.shape[-1] / tts_model.sample_rate
            if audio_dur < 10:
                continue

            style_ttl_pred, style_dp_pred = encoder(audio.unsqueeze(0).to(device))

            loss_style_ttl = F.mse_loss(style_ttl_pred, style_ttl_true)
            loss_style_dp = F.mse_loss(style_dp_pred, style_dp_true)
            total_ttl += loss_style_ttl.item()
            total_dp += loss_style_dp.item()
            total_style += (args.lambda_style_ttl * loss_style_ttl + args.lambda_style_dp * loss_style_dp).item()
            count += 1

    encoder.train()
    denom = max(count, 1)
    return {
        "val_style": total_style / denom,
        "val_ttl": total_ttl / denom,
        "val_dp": total_dp / denom,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────────────────────


def save_checkpoint(
    path: Path,
    encoder: AudioEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
) -> None:
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }, path)
    tqdm.write(f"  Checkpoint saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
