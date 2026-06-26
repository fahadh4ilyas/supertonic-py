#!/usr/bin/env python3
"""Train the full Supertonic TTS model end-to-end with integrated AudioEncoder.

Training approach:
    For each sample, the reference audio (voice_encoder) is fed into the
    AudioEncoder (inside tts_model) to predict style vectors.  The TTS model
    then generates audio from text_tts + predicted style, and losses are
    computed against available ground truth:

    - Style MSE (if true style available & encoder not frozen)
    - Mel reconstruction (if voice_tts target audio available)
    - Duration loss (if true style available)

Usage:
    python scripts/train_tts.py --dataset data.jsonl --output_dir ./checkpoints

Dataset format (JSONL):
    {"voice_encoder": "path/to/ref.wav", "text_tts": "Text to synthesize",
     "lang": "en",                                    # optional, default "na"
     "voice_tts": "path/to/target.wav",               # optional
     "style_name": "M1",                              # optional
     "style_ttl": {"dims": [...], "data": [...]},     # optional
     "style_dp":  {"dims": [...], "data": [...]}}     # optional

    At least one of (voice_tts, style_name, style_ttl+style_dp) must be present.
"""

from __future__ import annotations

import argparse
import json
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

from supertonic.model import SupertonicModel, MelSpectrogram
from supertonic.core import UnicodeProcessor
from supertonic.loader import (
    load_voice_style_from_json_file,
    get_cache_dir,
    download_model,
    has_all_onnx_modules,
)

from scripts.load_onnx_weights import (
    load_all_onnx_weights,
    load_weights_to_model,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_STYLES = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]


# ──────────────────────────────────────────────────────────────────────────────
# Model bootstrapping (download + ONNX→PyTorch conversion)
# ──────────────────────────────────────────────────────────────────────────────


def ensure_model_ready(model_dir: Path) -> Path:
    """Ensure the PyTorch model exists, downloading and converting if needed."""
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

    # Distribute shared k_context
    kctx_key = "vector_field.main_blocks.5.k_context"
    if kctx_key in state_dict:
        kctx_val = state_dict[kctx_key]
        for idx in (12, 19, 26):
            other = f"vector_field.main_blocks.{idx}.k_context"
            if other in state_dict:
                state_dict[other] = kctx_val.clone()

    # Distribute shared increments and theta
    for buf_name in ("increments", "theta"):
        src_key = f"vector_field.main_blocks.2.{buf_name}"
        if src_key in state_dict:
            val = state_dict[src_key]
            for idx in (9, 16, 23):
                dst_key = f"vector_field.main_blocks.{idx}.{buf_name}"
                if dst_key in state_dict:
                    state_dict[dst_key] = val.clone()

    model.load_state_dict(state_dict, strict=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_dir / "tts.json", output_dir / "tts.json")
    shutil.copy(onnx_dir / "unicode_indexer.json", output_dir / "unicode_indexer.json")

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


class TTSDataset(Dataset):
    """JSONL dataset for TTS training.

    Required fields per sample:
        voice_encoder   — path to reference audio for encoder input
        text_tts        — text to synthesize

    Optional fields:
        lang            — language code (default: "na")
        voice_tts       — path to target audio for mel reconstruction loss
        style_name      — built-in voice style name (e.g. "M1")
        style_ttl       — explicit style_ttl dict with "dims" and "data"
        style_dp        — explicit style_dp dict with "dims" and "data"

    At least one of (voice_tts, style_name, style_ttl+style_dp) must be present.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        text_processor: UnicodeProcessor,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.samples: list[dict] = []
        self._load(text_processor)

    def _load(self, text_processor: UnicodeProcessor) -> None:
        with open(self.jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)

                if "voice_encoder" not in sample or "text_tts" not in sample:
                    continue

                has_voice_tts = "voice_tts" in sample
                has_style_name = "style_name" in sample
                has_style_vecs = "style_ttl" in sample and "style_dp" in sample

                if not (has_voice_tts or has_style_name or has_style_vecs):
                    continue

                sample.setdefault("lang", "na")

                # Resolve paths relative to dataset location
                base = self.jsonl_path.parent
                sample["_voice_enc_path"] = str(base / sample["voice_encoder"])
                if has_voice_tts:
                    sample["_voice_tts_path"] = str(base / sample["voice_tts"])
                else:
                    sample["_voice_tts_path"] = None

                self.samples.append(sample)

        if not self.samples:
            raise ValueError(f"No valid samples found in {self.jsonl_path}")

        print(f"Loaded {len(self.samples)} samples from {self.jsonl_path}")

        # Pre-tokenize text_tts
        print("Pre-tokenizing texts...", flush=True)
        for sample in tqdm(self.samples, desc="Tokenizing", unit="sample", dynamic_ncols=True):
            sample["_tts_ids"], sample["_tts_mask"] = text_processor(
                [sample["text_tts"]], sample["lang"]
            )
        print("Tokenization complete")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    """Collate a batch."""
    return {
        "text_tts": [s["text_tts"] for s in batch],
        "lang": [s["lang"] for s in batch],
        "_tts_ids": np.stack([s["_tts_ids"] for s in batch]),
        "_tts_mask": np.stack([s["_tts_mask"] for s in batch]),
        "_voice_enc_path": [s["_voice_enc_path"] for s in batch],
        "_voice_tts_path": [s["_voice_tts_path"] for s in batch],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────


def _load_real_audio(path: str, target_sr: int, device: torch.device) -> torch.Tensor | None:
    """Load a real audio file, resample if needed, return as (1, T) tensor on device."""
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    except ImportError:
        import soundfile as sf
        import scipy.signal
        wav_np, sr = sf.read(path)
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)
        if sr != target_sr:
            num_samples = int(len(wav_np) * target_sr / sr)
            wav_np = scipy.signal.resample(wav_np, num_samples)
        waveform = torch.from_numpy(wav_np.astype(np.float32)).unsqueeze(0)
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


def load_style_vectors(
    styles_dir: Path,
    style_names: list[str],
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Load style_ttl and style_dp from JSON files as torch tensors."""
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


def parse_style_dict(d: dict | None) -> np.ndarray | None:
    """Parse a style dict with 'dims' and 'data' into a numpy array."""
    if d is None:
        return None
    return np.array(d["data"], dtype=np.float32).reshape(*d["dims"])


def resolve_true_style(
    sample: dict,
    builtin_styles: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve true style_ttl and style_dp from sample fields.

    Priority: style_ttl+style_dp > style_name.
    Returns (style_ttl, style_dp) tensors or (None, None).
    """
    style_ttl_dict = sample.get("style_ttl")
    style_dp_dict = sample.get("style_dp")

    if style_ttl_dict is not None and style_dp_dict is not None:
        ttl = torch.from_numpy(parse_style_dict(style_ttl_dict)).to(device)
        dp = torch.from_numpy(parse_style_dict(style_dp_dict)).to(device)
        return ttl, dp

    style_name = sample.get("style_name")
    if style_name is not None and style_name in builtin_styles:
        return builtin_styles[style_name]

    return None, None


# Shared mel-spectrogram extractor (created once)
_mel_extractor: MelSpectrogram | None = None


def _get_mel(tts_model: SupertonicModel, device: torch.device) -> MelSpectrogram:
    """Get or create a shared MelSpectrogram matching the TTS model's config."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Main training
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train Supertonic TTS model end-to-end with integrated AudioEncoder"
    )
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Path to JSONL dataset file")
    parser.add_argument("--model_dir", type=Path,
                        default=get_cache_dir("supertonic-3"),
                        help="Directory with TTS model files")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device (cuda, cpu)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Samples per batch")
    parser.add_argument("--grad_accum_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--total_steps", type=int, default=8,
                        help="Diffusion steps for TTS generation during training (fewer = faster)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed for generated audio")
    parser.add_argument("--freeze_encoder_until", type=int, default=0,
                        help="Freeze AudioEncoder for the first N epochs, then unfreeze. "
                             "0 = never freeze. (default: 0)")
    parser.add_argument("--lambda_style", type=float, default=1.0,
                        help="Weight for style MSE loss (0 = disabled)")
    parser.add_argument("--lambda_mel", type=float, default=1.0,
                        help="Weight for mel reconstruction loss (0 = disabled)")
    parser.add_argument("--lambda_dur", type=float, default=0.1,
                        help="Weight for duration predictor loss (0 = disabled)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (fallback for train/val seeds)")
    parser.add_argument("--train_seed", type=int, default=None,
                        help="Seed for training randomness")
    parser.add_argument("--val_seed", type=int, default=None,
                        help="Seed for validation randomness")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of dataset held out for validation (default: 0.1)")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--val_every", type=int, default=1,
                        help="Run validation every N epochs")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log detailed metrics every N epochs")
    parser.add_argument("--bar_refresh_every", type=int, default=4,
                        help="Update tqdm batch progress bar every N gradient steps")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Stop after N validation checks without improvement. Set 0 to disable.")
    parser.add_argument("--early_stopping_metric", type=str, default="val_loss",
                        choices=["val_loss", "val_style", "val_mel", "val_dur"],
                        help="Which validation metric to monitor for early stopping.")
    parser.add_argument("--amp", type=str, default="bf16", choices=["bf16", "fp16", "none"],
                        help="Mixed precision mode")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from a checkpoint file")
    parser.add_argument("--styles", type=str, nargs="+",
                        default=_DEFAULT_STYLES,
                        help="Built-in voice style names available for style_name resolution "
                             "(default: all 10)")
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

    # ── Ensure model exists ──────────────────────────────────────────────────
    args.model_dir = ensure_model_ready(args.model_dir)

    # ── Load TTS model ───────────────────────────────────────────────────────
    print(f"\nLoading TTS model from {args.model_dir}...")
    tts_model = SupertonicModel.from_pretrained(str(args.model_dir), device=device)
    print(f"TTS model loaded. Sample rate: {tts_model.sample_rate}")

    # ── Load text processor ──────────────────────────────────────────────────
    indexer_path = args.model_dir / "unicode_indexer.json"
    if not indexer_path.exists():
        raise FileNotFoundError(f"unicode_indexer.json not found in {args.model_dir}")
    text_processor = UnicodeProcessor(str(indexer_path))

    # ── Load built-in voice styles ───────────────────────────────────────────
    styles_dir = args.model_dir / "voice_styles"
    all_style_names = list(args.styles)
    builtin_styles = load_style_vectors(styles_dir, all_style_names, device)
    print(f"Loaded {len(builtin_styles)} built-in voice styles")

    # ── Encoder freeze schedule ──────────────────────────────────────────────
    if args.freeze_encoder_until > 0:
        for param in tts_model.audio_encoder.parameters():
            param.requires_grad = False
        print(f"AudioEncoder frozen for first {args.freeze_encoder_until} epochs")

    # ── Optimizer ────────────────────────────────────────────────────────────
    trainable = [p for p in tts_model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters! Check --freeze_encoder_until and model state.")
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(
        trainable,
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
    best_val_metric = {m: float("inf") for m in ["val_loss", "val_style", "val_mel", "val_dur"]}
    early_stop_counter = 0

    # ── Resume ───────────────────────────────────────────────────────────────
    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        tts_model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed at epoch {start_epoch}, step {global_step}")

    # ── Dataset (train/val split) ────────────────────────────────────────────
    full_dataset = TTSDataset(args.dataset, text_processor=text_processor)
    n_val = max(1, int(len(full_dataset) * args.val_ratio))
    n_train = len(full_dataset) - n_val
    indices = list(range(len(full_dataset)))
    val_rng.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    # ── Training loop ────────────────────────────────────────────────────────
    epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Training", unit="epoch",
                       dynamic_ncols=True)

    for epoch in epoch_pbar:
        epoch_start = time.time()
        epoch_metrics = defaultdict(float)
        epoch_steps = 0
        epoch_samples_skipped = 0

        # Unfreeze encoder when reaching the threshold
        if args.freeze_encoder_until > 0 and epoch == args.freeze_encoder_until:
            for param in tts_model.audio_encoder.parameters():
                param.requires_grad = True
            # Add encoder params to existing optimizer (preserves momentum for other params)
            current_lr = optimizer.param_groups[0]["lr"]
            encoder_params = list(tts_model.audio_encoder.parameters())
            optimizer.add_param_group({
                "params": encoder_params,
                "lr": current_lr,
                "weight_decay": args.weight_decay,
                "betas": (0.9, 0.999),
            })
            trainable = [p for p in tts_model.parameters() if p.requires_grad]
            print(f"  → AudioEncoder unfrozen at epoch {epoch} "
                  f"({sum(p.numel() for p in encoder_params):,} params added to optimizer)")

        encoder_frozen = epoch < args.freeze_encoder_until

        tts_model.train()
        if encoder_frozen:
            tts_model.audio_encoder.eval()

        # Shuffle and create dataloader
        train_indices_shuf = list(range(len(train_dataset)))
        train_rng.shuffle(train_indices_shuf)

        dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=torch.utils.data.SequentialSampler(train_indices_shuf),
            collate_fn=collate_fn,
            drop_last=False,
        )

        optimizer.zero_grad()
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}", unit="batch",
                          leave=False, dynamic_ncols=True)

        for batch_idx, batch in enumerate(batch_pbar):
            accumulated_loss = 0.0

            for i in range(len(batch["text_tts"])):
                tts_ids = batch["_tts_ids"][i]
                tts_mask = batch["_tts_mask"][i]
                voice_enc_path = batch["_voice_enc_path"][i]
                voice_tts_path = batch["_voice_tts_path"][i]
                lang = batch["lang"][i]

                # ── Step 1: Encoder forward ──────────────────────────────────
                enc_audio = _load_real_audio(voice_enc_path, tts_model.sample_rate, device)
                if enc_audio is None:
                    epoch_samples_skipped += 1
                    continue

                enc_audio_dur = enc_audio.shape[-1] / tts_model.sample_rate
                if enc_audio_dur < 1.0:
                    epoch_samples_skipped += 1
                    continue

                with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    style_ttl_pred, style_dp_pred = tts_model.audio_encoder(enc_audio.unsqueeze(0))

                # ── Step 2: Resolve true style ───────────────────────────────
                # Look up the original sample via the Subset
                dataset_idx = train_indices[train_indices_shuf[batch_idx * args.batch_size + i]]
                sample = full_dataset[dataset_idx]
                style_ttl_true, style_dp_true = resolve_true_style(sample, builtin_styles, device)
                has_true_style = style_ttl_true is not None

                # ── Step 3: Style MSE loss ───────────────────────────────────
                loss_style = torch.tensor(0.0, device=device)
                if has_true_style and not encoder_frozen and args.lambda_style > 0:
                    loss_style_ttl = F.mse_loss(style_ttl_pred, style_ttl_true)
                    loss_style_dp = F.mse_loss(style_dp_pred, style_dp_true)
                    loss_style = loss_style_ttl + loss_style_dp
                    epoch_metrics["loss_style"] += loss_style.item()
                    epoch_metrics["loss_style_ttl"] += loss_style_ttl.item()
                    epoch_metrics["loss_style_dp"] += loss_style_dp.item()

                # ── Step 4: TTS forward with predicted style ─────────────────
                text_ids_t = torch.from_numpy(tts_ids).to(device)
                text_mask_t = torch.from_numpy(tts_mask).to(device)

                with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    wav_pred, dur_pred = tts_model.forward(
                        text_ids=text_ids_t,
                        style_ttl=style_ttl_pred,
                        style_dp=style_dp_pred,
                        text_mask=text_mask_t,
                        total_steps=args.total_steps,
                        speed=args.speed,
                    )

                # ── Step 5: Mel reconstruction loss ──────────────────────────
                loss_mel = torch.tensor(0.0, device=device)
                loss_dur = torch.tensor(0.0, device=device)

                if voice_tts_path is not None and args.lambda_mel > 0:
                    target_audio = _load_real_audio(voice_tts_path, tts_model.sample_rate, device)
                    if target_audio is not None:
                        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                            mel = _get_mel(tts_model, device)
                            mel_pred = mel(wav_pred)
                            mel_target = mel(target_audio.unsqueeze(0))
                            min_len = min(mel_pred.shape[-1], mel_target.shape[-1])
                            loss_mel = F.mse_loss(mel_pred[..., :min_len], mel_target[..., :min_len])
                            epoch_metrics["loss_mel"] += loss_mel.item()

                        # Duration loss from target waveform length
                        dur_actual = target_audio.shape[-1] / tts_model.sample_rate
                        loss_dur_wav = F.smooth_l1_loss(
                            torch.log(dur_pred.clamp(min=1e-6)),
                            torch.log(torch.tensor(dur_actual, device=device).clamp(min=1e-6)),
                        )
                        loss_dur = loss_dur + loss_dur_wav
                        epoch_metrics["loss_dur"] += loss_dur_wav.item()

                elif has_true_style and args.lambda_mel > 0:
                    # No target audio — generate reference from true style,
                    # then compute both mel and duration losses.
                    with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        wav_true, dur_true = tts_model.forward(
                            text_ids=text_ids_t,
                            style_ttl=style_ttl_true,
                            style_dp=style_dp_true,
                            text_mask=text_mask_t,
                            total_steps=args.total_steps,
                            speed=args.speed,
                        )
                        mel = _get_mel(tts_model, device)
                        mel_pred = mel(wav_pred)
                        mel_true = mel(wav_true)
                        min_len = min(mel_pred.shape[-1], mel_true.shape[-1])
                        loss_mel = F.mse_loss(mel_pred[..., :min_len], mel_true[..., :min_len])
                        epoch_metrics["loss_mel"] += loss_mel.item()

                        # Duration loss: dur_pred (from pred style) vs dur_true (from true style)
                        loss_dur_style = F.smooth_l1_loss(
                            torch.log(dur_pred.clamp(min=1e-6)),
                            torch.log(dur_true.clamp(min=1e-6)),
                        )
                        loss_dur = loss_dur + loss_dur_style
                        epoch_metrics["loss_dur"] += loss_dur_style.item()

                # ── Step 6: Total loss ───────────────────────────────────────
                loss = (
                    args.lambda_style * loss_style
                    + args.lambda_mel * loss_mel
                    + args.lambda_dur * loss_dur
                )
                epoch_metrics["loss_total"] += loss.item()
                epoch_steps += 1

                if loss.item() == 0.0:
                    continue

                loss = loss / args.grad_accum_steps

                if use_amp and use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulated_loss += loss.item() * args.grad_accum_steps

            # Gradient accumulation step
            if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx == len(dataloader) - 1):
                if use_scaler and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    optimizer.step()

                optimizer.zero_grad()
                global_step += 1

                # Update batch progress bar
                if global_step % args.bar_refresh_every == 0:
                    denom = max(epoch_steps, 1)
                    postfix = {
                        "loss": f"{epoch_metrics['loss_total'] / denom:.4f}",
                    }
                    if epoch_metrics["loss_style"] > 0:
                        postfix["style"] = f"{epoch_metrics['loss_style'] / denom:.4f}"
                    if epoch_metrics["loss_mel"] > 0:
                        postfix["mel"] = f"{epoch_metrics['loss_mel'] / denom:.4f}"
                    if epoch_metrics["loss_dur"] > 0:
                        postfix["dur"] = f"{epoch_metrics['loss_dur'] / denom:.4f}"
                    postfix["skip"] = epoch_samples_skipped
                    batch_pbar.set_postfix(postfix, refresh=False)

                accumulated_loss = 0.0

        batch_pbar.close()

        # ── Epoch summary ────────────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        denom = max(epoch_steps, 1)

        epoch_pbar.set_postfix({
            "loss": f"{epoch_metrics['loss_total'] / denom:.4f}",
            "style": f"{epoch_metrics['loss_style'] / denom:.4f}",
            "mel": f"{epoch_metrics['loss_mel'] / denom:.4f}",
            "dur": f"{epoch_metrics['loss_dur'] / denom:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            "skip": epoch_samples_skipped,
        })

        if (epoch + 1) % args.log_every == 0:
            tqdm.write(
                f"── Epoch {epoch:3d} ── "
                f"Steps: {epoch_steps} | Time: {epoch_time:.1f}s | "
                f"Total: {epoch_metrics['loss_total'] / denom:.6f} | "
                f"Style: {epoch_metrics['loss_style'] / denom:.6f} | "
                f"Mel: {epoch_metrics['loss_mel'] / denom:.6f} | "
                f"Dur: {epoch_metrics['loss_dur'] / denom:.6f} | "
                f"Skipped: {epoch_samples_skipped}"
            )

        scheduler.step()

        # ── Validation ───────────────────────────────────────────────────────
        if args.val_every > 0 and (epoch + 1) % args.val_every == 0:
            val_metrics = run_validation(
                tts_model, val_dataset, full_dataset, val_indices,
                device, args, builtin_styles,
            )
            val_loss = val_metrics["val_loss"]
            tqdm.write(
                f"  Val ── loss: {val_metrics['val_loss']:.6f} | "
                f"style: {val_metrics['val_style']:.6f} | "
                f"mel: {val_metrics['val_mel']:.6f} | "
                f"dur: {val_metrics['val_dur']:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(args.output_dir / "best_model.pt",
                                tts_model, optimizer, scheduler, epoch, global_step, best_val_loss)
                tqdm.write(f"  → Saved best model (val_loss={best_val_loss:.6f})")

            # Early stopping
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
                args.output_dir / f"tts_epoch{epoch:04d}.pt",
                tts_model, optimizer, scheduler, epoch, global_step, best_val_loss,
            )

    # ── Final save ───────────────────────────────────────────────────────────
    save_checkpoint(
        args.output_dir / "final_model.pt",
        tts_model, optimizer, scheduler, args.epochs - 1, global_step, best_val_loss,
    )
    tqdm.write(f"\nTraining complete. Model saved to {args.output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_validation(
    tts_model: SupertonicModel,
    val_dataset: torch.utils.data.Subset,
    full_dataset: TTSDataset,
    val_indices: list[int],
    device: torch.device,
    args: argparse.Namespace,
    builtin_styles: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Evaluate on held-out validation set.

    Returns dict with keys: val_loss, val_style, val_mel, val_dur.
    """
    tts_model.eval()
    total_style = 0.0
    total_mel = 0.0
    total_dur = 0.0
    total_loss = 0.0
    count = 0

    for idx in range(len(val_dataset)):
        dataset_idx = val_indices[idx]
        sample = full_dataset[dataset_idx]
        tts_ids = sample["_tts_ids"]
        tts_mask = sample["_tts_mask"]
        voice_enc_path = sample["_voice_enc_path"]
        voice_tts_path = sample.get("_voice_tts_path")

        # Encoder forward
        enc_audio = _load_real_audio(voice_enc_path, tts_model.sample_rate, device)
        if enc_audio is None:
            continue

        enc_audio_dur = enc_audio.shape[-1] / tts_model.sample_rate
        if enc_audio_dur < 1.0:
            continue

        style_ttl_pred, style_dp_pred = tts_model.audio_encoder(enc_audio.unsqueeze(0))

        # Resolve true style
        style_ttl_true, style_dp_true = resolve_true_style(sample, builtin_styles, device)
        has_true_style = style_ttl_true is not None

        # Style MSE
        loss_style = 0.0
        if has_true_style:
            loss_ttl = F.mse_loss(style_ttl_pred, style_ttl_true).item()
            loss_dp = F.mse_loss(style_dp_pred, style_dp_true).item()
            loss_style = loss_ttl + loss_dp
            total_style += loss_style

        # TTS forward
        text_ids_t = torch.from_numpy(tts_ids).to(device)
        text_mask_t = torch.from_numpy(tts_mask).to(device)

        wav_pred, dur_pred = tts_model.forward(
            text_ids=text_ids_t,
            style_ttl=style_ttl_pred,
            style_dp=style_dp_pred,
            text_mask=text_mask_t,
            total_steps=args.total_steps,
            speed=args.speed,
        )

        # Mel loss (and duration-from-waveform)
        loss_mel = 0.0
        loss_dur = 0.0

        if voice_tts_path is not None:
            target_audio = _load_real_audio(voice_tts_path, tts_model.sample_rate, device)
            if target_audio is not None:
                mel = _get_mel(tts_model, device)
                mel_pred = mel(wav_pred)
                mel_target = mel(target_audio.unsqueeze(0))
                min_len = min(mel_pred.shape[-1], mel_target.shape[-1])
                loss_mel = F.mse_loss(mel_pred[..., :min_len], mel_target[..., :min_len]).item()
                total_mel += loss_mel

                # Duration loss from target waveform length
                dur_actual = target_audio.shape[-1] / tts_model.sample_rate
                loss_dur += F.smooth_l1_loss(
                    torch.log(dur_pred.clamp(min=1e-6)),
                    torch.log(torch.tensor(dur_actual, device=device).clamp(min=1e-6)),
                ).item()

        elif has_true_style:
            # No target audio — generate reference from true style,
            # then compute both mel and duration losses.
            wav_true, dur_true = tts_model.forward(
                text_ids=text_ids_t,
                style_ttl=style_ttl_true,
                style_dp=style_dp_true,
                text_mask=text_mask_t,
                total_steps=args.total_steps,
                speed=args.speed,
            )
            mel = _get_mel(tts_model, device)
            mel_pred = mel(wav_pred)
            mel_true = mel(wav_true)
            min_len = min(mel_pred.shape[-1], mel_true.shape[-1])
            loss_mel = F.mse_loss(mel_pred[..., :min_len], mel_true[..., :min_len]).item()
            total_mel += loss_mel

            # Duration loss: dur_pred (from pred style) vs dur_true (from true style)
            loss_dur += F.smooth_l1_loss(
                torch.log(dur_pred.clamp(min=1e-6)),
                torch.log(dur_true.clamp(min=1e-6)),
            ).item()

        total_dur += loss_dur
        total_loss += args.lambda_style * loss_style + args.lambda_mel * loss_mel + args.lambda_dur * loss_dur
        count += 1

    tts_model.train()
    if args.freeze_encoder_until > 0:
        tts_model.audio_encoder.eval()

    denom = max(count, 1)
    return {
        "val_loss": total_loss / denom,
        "val_style": total_style / denom,
        "val_mel": total_mel / denom,
        "val_dur": total_dur / denom,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────────────────────


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
) -> None:
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
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
