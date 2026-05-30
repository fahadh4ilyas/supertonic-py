#!/usr/bin/env python3
"""Train the AudioEncoder to extract voice style vectors from reference audio.

Training approach:
    For each text pair (text_encoder, text_tts) and a randomly selected
    training voice style, audio is generated via the frozen TTS model and
    fed into the AudioEncoder. The encoder is trained to recover the
    original style vectors, optionally with an auxiliary loss that ensures
    the predicted styles produce the same TextEncoder conditioning.

Usage:
    python scripts/train_encoder.py --dataset data.jsonl --output_dir ./checkpoints

Dataset format (JSONL):
    {"text_encoder": "Long text for encoder input...", "text_tts": "Shorter text for TTS-through loss", "lang": "en"}
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supertonic.model import SupertonicModel, AudioEncoder
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

TRAIN_STYLES = ["F1", "F2", "F3", "F4", "M1", "M2", "M3", "M4"]
VAL_STYLES = ["F5", "M5"]

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
                if "text_encoder" not in sample or "text_tts" not in sample:
                    continue
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
                [sample["text_encoder"]], sample.get("lang", "en")
            )
            sample["_tts_ids"], sample["_tts_mask"] = text_processor(
                [sample["text_tts"] or sample["text_encoder"]],
                sample.get("lang", "en"),
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
            buffer_lang = short_texts[0].get("lang", "en")
            buffer_tts = short_texts[0].get("text_tts", "")

            for sample in short_texts:
                candidate = f"{buffer_text} {sample['text_encoder']}".strip()
                if len(candidate) >= self.min_chars:
                    prepared.append({
                        "text_encoder": candidate[:self.max_chars],
                        "text_tts": buffer_tts or sample.get("text_tts", ""),
                        "lang": buffer_lang,
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
                    "text_tts": buffer_tts,
                    "lang": buffer_lang,
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
    """Collate a batch — includes pre-tokenized arrays via np.stack."""
    return {
        "text_encoder": [s["text_encoder"] for s in batch],
        "text_tts": [s["text_tts"] for s in batch],
        "lang": [s.get("lang", "en") for s in batch],
        "_enc_ids": np.stack([s["_enc_ids"] for s in batch]),
        "_enc_mask": np.stack([s["_enc_mask"] for s in batch]),
        "_tts_ids": np.stack([s["_tts_ids"] for s in batch]),
        "_tts_mask": np.stack([s["_tts_mask"] for s in batch]),
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


def text_encoder_loss(
    tts_model: SupertonicModel,
    text_ids_np: np.ndarray,
    text_mask_np: np.ndarray,
    style_ttl_pred: torch.Tensor,
    style_ttl_true: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE between TextEncoder outputs for predicted vs true styles."""
    with torch.no_grad():
        text_ids = torch.from_numpy(text_ids_np).to(style_ttl_pred.device)
        text_mask = torch.from_numpy(text_mask_np).to(style_ttl_pred.device)
        enc_true = tts_model.text_encoder(text_ids, style_ttl_true, text_mask)

    text_ids = torch.from_numpy(text_ids_np).to(style_ttl_pred.device)
    text_mask = torch.from_numpy(text_mask_np).to(style_ttl_pred.device)
    enc_pred = tts_model.text_encoder(text_ids, style_ttl_pred, text_mask)

    return F.mse_loss(enc_pred, enc_true)


def duration_predictor_loss(
    tts_model: SupertonicModel,
    text_ids_np: np.ndarray,
    text_mask_np: np.ndarray,
    style_dp_pred: torch.Tensor,
    style_dp_true: torch.Tensor,
) -> torch.Tensor:
    """Compute log-space Huber loss between DurationPredictor outputs."""
    with torch.no_grad():
        text_ids = torch.from_numpy(text_ids_np).to(style_dp_pred.device)
        text_mask = torch.from_numpy(text_mask_np).to(style_dp_pred.device)
        dur_true = tts_model.duration_predictor(text_ids, style_dp_true, text_mask)

    text_ids = torch.from_numpy(text_ids_np).to(style_dp_pred.device)
    text_mask = torch.from_numpy(text_mask_np).to(style_dp_pred.device)
    dur_pred = tts_model.duration_predictor(text_ids, style_dp_pred, text_mask)

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
    parser.add_argument("--total_steps", type=int, default=4,
                        help="Diffusion steps for audio generation during training (fewer = faster)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed for generated audio")
    parser.add_argument("--lambda_text", type=float, default=0.1,
                        help="Weight for text encoder output consistency loss (0 = disabled)")
    parser.add_argument("--lambda_dur", type=float, default=0.1,
                        help="Weight for duration predictor consistency loss (0 = disabled)")
    parser.add_argument("--max_samples_per_style", type=int, default=None,
                        help="Max samples per style per epoch (limits dataset passes)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--val_every", type=int, default=5,
                        help="Run validation every N epochs")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log metrics every N training steps")
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="Stop after N validation checks without improvement. "
                             "Requires --val_every > 0. Default: no early stopping.")
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    train_styles = load_style_vectors(styles_dir, TRAIN_STYLES, device)
    val_styles = load_style_vectors(styles_dir, VAL_STYLES, device)

    if not train_styles:
        raise RuntimeError(f"No training styles loaded from {styles_dir}")
    print(f"Training styles: {list(train_styles.keys())}")
    print(f"Validation styles: {list(val_styles.keys())}")

    # ── Create encoder model ─────────────────────────────────────────────────
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

    # ── Training loop ────────────────────────────────────────────────────────
    train_style_names = sorted(train_styles.keys())

    epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Training", unit="epoch",
                       dynamic_ncols=True)

    for epoch in epoch_pbar:
        epoch_start = time.time()
        epoch_metrics = defaultdict(float)
        epoch_steps = 0
        epoch_samples_skipped = 0

        # Shuffle and create dataloader
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        if args.max_samples_per_style is not None:
            indices = indices[:args.max_samples_per_style]

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(indices),
            collate_fn=collate_fn,
            drop_last=False,
        )

        optimizer.zero_grad()
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}", unit="batch",
                          leave=False, dynamic_ncols=True)

        for batch_idx, batch in enumerate(batch_pbar):
            # Select a random training style for this batch
            style_name = random.choice(train_style_names)
            style_ttl_true, style_dp_true = train_styles[style_name]

            accumulated_loss = 0.0

            for i in range(len(batch["text_encoder"])):
                enc_ids = batch["_enc_ids"][i]
                enc_mask = batch["_enc_mask"][i]
                tts_ids = batch["_tts_ids"][i]
                tts_mask = batch["_tts_mask"][i]

                # Step 1: Generate reference audio from pre-tokenized text + true style
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

                # Step 2: Encode audio → predicted style
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    style_ttl_pred, style_dp_pred = encoder(audio.unsqueeze(0))

                # Step 3: Compute losses
                loss_style_ttl = F.mse_loss(style_ttl_pred, style_ttl_true)
                loss_style_dp = F.mse_loss(style_dp_pred, style_dp_true)
                loss_style = loss_style_ttl + loss_style_dp

                loss = loss_style
                epoch_metrics["loss_style"] += loss_style.item()
                epoch_metrics["loss_style_ttl"] += loss_style_ttl.item()
                epoch_metrics["loss_style_dp"] += loss_style_dp.item()

                if args.lambda_text > 0 and tts_ids is not None:
                    with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        loss_text = text_encoder_loss(
                            tts_model, tts_ids, tts_mask,
                            style_ttl_pred, style_ttl_true,
                        )
                    loss = loss + args.lambda_text * loss_text
                    epoch_metrics["loss_text"] += loss_text.item()

                if args.lambda_dur > 0 and tts_ids is not None:
                    with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        loss_dur = duration_predictor_loss(
                            tts_model, tts_ids, tts_mask,
                            style_dp_pred, style_dp_true,
                        )
                    loss = loss + args.lambda_dur * loss_dur
                    epoch_metrics["loss_dur"] += loss_dur.item()

                loss = loss / args.grad_accum_steps

                if use_amp and use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulated_loss += loss.item() * args.grad_accum_steps
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

                # Update batch progress bar (throttled: every 4 gradient steps)
                if global_step % 4 == 0:
                    denom = max(epoch_steps, 1)
                    batch_pbar.set_postfix({
                        "loss": f"{epoch_metrics['loss_style'] / denom:.4f}",
                        "ttl": f"{epoch_metrics['loss_style_ttl'] / denom:.4f}",
                        "dp": f"{epoch_metrics['loss_style_dp'] / denom:.4f}",
                        **({"text": f"{epoch_metrics['loss_text'] / denom:.4f}"} if args.lambda_text > 0 else {}),
                        **({"dur": f"{epoch_metrics['loss_dur'] / denom:.4f}"} if args.lambda_dur > 0 else {}),
                        "skip": epoch_samples_skipped,
                    }, refresh=False)

                accumulated_loss = 0.0

        batch_pbar.close()

        # ── Epoch summary ────────────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        denom = max(epoch_steps, 1)
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
                + (f" | Text: {epoch_metrics['loss_text'] / denom:.6f}"
                   if args.lambda_text > 0 else "")
                + (f" | Dur: {epoch_metrics['loss_dur'] / denom:.6f}"
                   if args.lambda_dur > 0 else "")
                + f" | Skipped: {epoch_samples_skipped}"
            )

        scheduler.step()

        # ── Validation ───────────────────────────────────────────────────────
        if args.val_every > 0 and (epoch + 1) % args.val_every == 0 and val_styles:
            val_metrics = run_validation(
                tts_model, encoder, val_styles, dataset,
                text_processor, device, args,
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

            if args.early_stopping_patience is not None and early_stop_counter >= args.early_stopping_patience:
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
    dataset: EncoderDataset,
    text_processor: UnicodeProcessor,
    device: torch.device,
    args: argparse.Namespace,
    num_val_samples: int = 20,
) -> dict[str, float]:
    """Evaluate encoder on held-out styles (F5, M5).

    Returns dict with keys: val_style, val_ttl, val_dp.
    """
    encoder.eval()
    total_style = 0.0
    total_ttl = 0.0
    total_dp = 0.0
    count = 0

    val_style_names = sorted(val_styles.keys())
    indices = random.sample(range(len(dataset)), min(num_val_samples, len(dataset)))

    for idx in indices:
        sample = dataset[idx]
        style_name = random.choice(val_style_names)
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
        total_style += (loss_style_ttl + loss_style_dp).item()
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
