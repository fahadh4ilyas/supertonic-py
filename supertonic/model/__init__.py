"""Supertonic TTS model package.

Trainable PyTorch equivalent of the ONNX-based Supertonic TTS pipeline,
consisting of four sub-models:

1. DurationPredictor  – predicts total utterance duration from text + style
2. TextEncoder       – encodes text to style-conditioned embeddings
3. VectorField       – flow-matching diffusion denoiser
4. Vocoder           – neural audio decoder (latent → waveform)

Architecture adapted from ONNX graph analysis of supertonic-3.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import safetensors.torch
import torch
import torchaudio
import torch.nn as nn

from .common import (
    SymmetricPad1d,
    CausalPad1d,
    LayerNorm,
    ConvNeXtBlock,
    ConvNeXtStack,
)
from .attention import (
    RelPosAttention,
    FFN,
    RoPECrossAttention,
    StyleCrossAttention,
    SpeechPromptedAttention,
    StyleTokenLayer,
)
from .duration_predictor import DurationPredictor
from .text_encoder import TextEncoder
from .vector_field import VectorField
from .vocoder import Vocoder
from .encoder import AudioEncoder, MelSpectrogram
from supertonic.core import Style, UnicodeProcessor
from supertonic.loader import load_voice_style_from_json_file, load_voice_style_from_name, list_available_voice_style_names
from supertonic.utils import chunk_text as _chunk_text_util


__all__ = [
    # Common
    "SymmetricPad1d",
    "CausalPad1d",
    "LayerNorm",
    "ConvNeXtBlock",
    "ConvNeXtStack",
    # Attention
    "RelPosAttention",
    "FFN",
    "RoPECrossAttention",
    "StyleCrossAttention",
    "SpeechPromptedAttention",
    "StyleTokenLayer",
    # Sub-models
    "DurationPredictor",
    "TextEncoder",
    "VectorField",
    "Vocoder",
    "AudioEncoder",
    "MelSpectrogram",
    # Full model
    "SupertonicModel",
    # Utilities
    "length_to_mask",
]


class SupertonicModel(nn.Module):
    """Full Supertonic TTS model in PyTorch.

    Args:
        vocab_size: Vocabulary size (default 8322 for BPE tokenizer).
        config: Optional dict matching tts.json structure, or path to a tts.json file.
                When provided, sub-model hyperparameters are derived from the config.
        sample_rate, base_chunk_size, ldim, chunk_compress_factor:
                Overrides when config is not provided.
    """

    def __init__(
        self,
        vocab_size: int = 8322,
        config: dict | str | Path | None = None,
        sample_rate: int = 44100,
        base_chunk_size: int = 512,
        ldim: int = 24,
        chunk_compress_factor: int = 6,
        unicode_indexer: str | Path | None = None,
        model_dir: str | Path | None = None,
        device: str | torch.device | None = None,
    ):
        super().__init__()

        self._device = torch.device(device) if device is not None else torch.device("cpu")

        self.unicode_indexer: Path | None = Path(unicode_indexer) if unicode_indexer is not None else None
        self.model_dir: Path | None = Path(model_dir) if model_dir is not None else None
        self._text_processor: UnicodeProcessor | None = None

        # Resolve config
        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = json.load(f)
        if config is None:
            config = {}

        self.config = config
        ae = config.get("ae", {})
        ttl = config.get("ttl", {})
        dp_cfg = config.get("dp", {})

        self.sample_rate = ae.get("sample_rate", sample_rate)
        self.base_chunk_size = ae.get("base_chunk_size", base_chunk_size)
        self.ldim = ttl.get("latent_dim", ldim)
        self.chunk_compress_factor = ttl.get("chunk_compress_factor", chunk_compress_factor)

        self.duration_predictor = DurationPredictor(vocab_size=vocab_size, config=dp_cfg)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, config=ttl)
        self.vector_field = VectorField(config=ttl.get("vector_field"))
        self.vocoder = Vocoder(config=ae.get("decoder"))
        self.audio_encoder = AudioEncoder(config=config)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        # Update _device when a device argument is provided (not dtype-only).
        target = kwargs.get("device") or (args[0] if args else None)
        if target is not None and not isinstance(target, torch.dtype):
            ret._device = torch.device(target)
        return ret

    def sample_noisy_latent(self, duration: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = duration.shape[0]
        wav_len_max = int(duration.max().item() * self.sample_rate)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = int(math.ceil(wav_len_max / chunk_size))
        latent_dim = self.ldim * self.chunk_compress_factor

        noisy_latent = torch.randn(bsz, latent_dim, latent_len, device=self.device)

        wav_lengths = (duration * self.sample_rate).long()
        latent_lengths = (wav_lengths + chunk_size - 1) // chunk_size
        max_len = latent_len
        ids = torch.arange(max_len, device=self.device).unsqueeze(0)
        latent_mask = (ids < latent_lengths.unsqueeze(1)).float().unsqueeze(1)
        noisy_latent = noisy_latent * latent_mask
        return noisy_latent, latent_mask

    @property
    def text_processor(self) -> UnicodeProcessor:
        """Lazily-initialized :class:`UnicodeProcessor` for the model's unicode indexer."""
        if self._text_processor is None:
            if self.unicode_indexer is None:
                raise ValueError(
                    "unicode_indexer not set on model. Pass it to __init__ or "
                    "set model.unicode_indexer to a path to unicode_indexer.json."
                )
            self._text_processor = UnicodeProcessor(str(self.unicode_indexer))
        return self._text_processor

    def forward(
        self,
        text_ids: torch.Tensor,
        style_ttl: torch.Tensor,
        style_dp: torch.Tensor,
        text_mask: torch.Tensor,
        total_steps: int = 8,
        speed: float = 1.05,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dur = self.duration_predictor(text_ids, style_dp, text_mask)
        text_emb = self.text_encoder(text_ids, style_ttl, text_mask)
        dur_scaled = dur / speed
        xt, latent_mask = self.sample_noisy_latent(dur_scaled)

        for step in range(total_steps):
            cs = torch.full((text_ids.shape[0],), step, device=self.device, dtype=torch.float32)
            ts = torch.full((text_ids.shape[0],), total_steps, device=self.device, dtype=torch.float32)
            # VectorField output includes Euler step internally (matches ONNX)
            xt = self.vector_field(xt, text_emb, style_ttl, latent_mask, text_mask, cs, ts)

        wav = self.vocoder(xt)
        return wav, dur_scaled

    # ------------------------------------------------------------------
    # High-level synthesis API (mirrors pipeline.TTS)
    # ------------------------------------------------------------------

    def get_voice_style(self, voice_name: str) -> Style:
        """Load a voice style by name from the model directory.

        Args:
            voice_name: Name of the voice style (e.g. ``'M1'..'M5'``, ``'F1'..'F5'``).

        Returns:
            Style object containing voice style vectors.
        """
        if self.model_dir is None:
            raise ValueError("model_dir not set on model.")
        return load_voice_style_from_name(self.model_dir, voice_name)

    @staticmethod
    def get_voice_style_from_path(voice_style_path: str | Path) -> Style:
        """Load a voice style from a JSON file path.

        Args:
            voice_style_path: Path to the voice style JSON file.

        Returns:
            Style object containing voice style vectors.
        """
        return load_voice_style_from_json_file(voice_style_path)

    @property
    def voice_style_names(self) -> list[str]:
        """List available built-in voice style names (e.g. M1-M5, F1-F5)."""
        if self.model_dir is None:
            return []
        return list_available_voice_style_names(self.model_dir)

    # ── AudioEncoder support ─────────────────────────────────────────────

    def load_encoder(self, checkpoint_path: str | Path) -> None:
        """Load trained AudioEncoder weights from a checkpoint.

        Args:
            checkpoint_path: Path to a ``.pt`` checkpoint saved by
                ``scripts/train_encoder.py``.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("encoder", checkpoint)
        self.audio_encoder.load_state_dict(state, strict=False)
        self.audio_encoder.eval()

    def has_encoder(self) -> bool:
        """Check if AudioEncoder has been trained (weights are not random)."""
        # Heuristic: if the first Conv1d weight has non-trivial variance,
        # the encoder has been trained. Random init has very low variance.
        w = self.audio_encoder.proj_in.weight
        return bool(w.var().item() > 1e-4)

    def _load_voice_ref(self, voice_ref: str | Path | np.ndarray) -> Style:
        """Convert a voice_ref (path or waveform) into a Style via the encoder.

        Args:
            voice_ref: Path to an audio file, or a numpy waveform array
                of shape (samples,) or (1, samples) at 44100 Hz.

        Returns:
            Style object with extracted style_ttl and style_dp.
        """
        if not self.has_encoder():
            raise RuntimeError(
                "AudioEncoder has not been trained yet. "
                "Train with scripts/train_encoder.py, then call load_encoder()."
            )

        # Convert to waveform tensor (1, 1, T)
        if isinstance(voice_ref, (str, Path)):
            waveform, sr = torchaudio.load(str(voice_ref))
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        elif isinstance(voice_ref, np.ndarray):
            waveform = torch.from_numpy(voice_ref.astype(np.float32))
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(1)  # (B, 1, T)
        else:
            raise TypeError(f"voice_ref must be str, Path, or np.ndarray, got {type(voice_ref)}")

        waveform = waveform.to(self.device)

        with torch.no_grad():
            style_ttl, style_dp = self.audio_encoder(waveform)

        return Style(
            style_ttl_onnx=style_ttl.cpu().numpy(),
            style_dp_onnx=style_dp.cpu().numpy(),
        )

    # ── Synthesis ────────────────────────────────────────────────────────

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        voice_style: Style,
        total_steps: int = 8,
        speed: float = 1.05,
        silence_duration: float = 0.3,
        max_chunk_length: Optional[int] = None,
        lang: Optional[str] = "na",
        voice_ref: str | Path | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize.
            voice_style: Voice style object containing ttl and dp vectors.
                Ignored if ``voice_ref`` is provided.
            total_steps: Number of diffusion steps (default: 8).
            speed: Speech speed multiplier (default: 1.05).
            silence_duration: Seconds of silence between chunks (default: 0.3).
            max_chunk_length: Max characters per chunk. If None, auto-detected
                (120 for Korean, 300 otherwise).
            lang: Language code. Default ``"na"`` for multilingual models.
                Set to ``None`` for English-only models (v1).
            voice_ref: Optional reference audio for voice cloning. Can be:
                - str/Path: path to an audio file
                - np.ndarray: waveform (samples,) or (1, samples) at 44100 Hz
                Requires :meth:`load_encoder` to have been called first.
                Overrides ``voice_style`` when provided.

        Returns:
            Tuple of (waveform, duration):
                - waveform: float32 array of shape (1, num_samples)
                - duration: Total duration in seconds
        """
        if voice_ref is not None:
            voice_style = self._load_voice_ref(voice_ref)

        if self.unicode_indexer is None:
            raise ValueError(
                "unicode_indexer not set on model. Pass it to __init__ or "
                "set model.unicode_indexer to a path to unicode_indexer.json."
            )
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if max_chunk_length is None:
            max_chunk_length = 120 if lang == "ko" else 300

        # Preprocess text (unicode norm, cleaning) but defer language token
        # wrapping to per-chunk tokenization — matches ONNX pipeline behavior
        # where language tags are applied per chunk, not on the full text.
        pp_text = self.text_processor._preprocess_text(text, None)
        text_chunks = self._chunk_text(pp_text, max_chunk_length)
        silence_samples = int(silence_duration * self.sample_rate)

        wav_list = []
        dur_list = []

        for text_chunk in text_chunks:
            # Tokenize each chunk with per-chunk language wrapping
            text_ids_np, text_mask_np = self.text_processor([text_chunk], lang)

            wav_t, dur_t = self.forward(
                text_ids=torch.from_numpy(text_ids_np).to(self.device),
                style_ttl=torch.from_numpy(voice_style.ttl).to(self.device),
                style_dp=torch.from_numpy(voice_style.dp).to(self.device),
                text_mask=torch.from_numpy(text_mask_np).to(self.device),
                total_steps=total_steps,
                speed=speed,
            )
            wav_list.append(wav_t.cpu().numpy())
            dur_list.append(dur_t.cpu().numpy().item())

        # Concatenate with silence between chunks
        silence = np.zeros((1, silence_samples), dtype=np.float32)
        arrays = []
        for i, wav in enumerate(wav_list):
            arrays.append(wav)
            if i < len(wav_list) - 1:
                arrays.append(silence)

        wav_cat = np.concatenate(arrays, axis=1)
        dur_cat = np.array([sum(dur_list) + silence_duration * (len(wav_list) - 1)])

        return wav_cat, dur_cat

    @torch.inference_mode()
    def synthesize_generator(
        self,
        text: str,
        voice_style: Style,
        total_steps: int = 8,
        speed: float = 1.05,
        silence_duration: float = 0.3,
        max_chunk_length: Optional[int] = None,
        lang: Optional[str] = "na",
        voice_ref: str | Path | np.ndarray | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Synthesize speech from text, yielding audio chunks on the fly.

        Args:
            text: Text to synthesize.
            voice_style: Voice style object containing ttl and dp vectors.
                Ignored if ``voice_ref`` is provided.
            total_steps: Number of diffusion steps (default: 8).
            speed: Speech speed multiplier (default: 1.05).
            silence_duration: Seconds of silence between chunks (default: 0.3).
            max_chunk_length: Max characters per chunk. If None, auto-detected
                (120 for Korean, 300 otherwise).
            lang: Language code. Default ``"na"`` for multilingual models.
                Set to ``None`` for English-only models (v1).
            voice_ref: Optional reference audio for voice cloning (see
                :meth:`synthesize`). Overrides ``voice_style`` when provided.

        Yields:
            Tuple of (waveform, duration) for each processed chunk.
        """
        if voice_ref is not None:
            voice_style = self._load_voice_ref(voice_ref)

        if self.unicode_indexer is None:
            raise ValueError(
                "unicode_indexer not set on model. Pass it to __init__ or "
                "set model.unicode_indexer to a path to unicode_indexer.json."
            )
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if max_chunk_length is None:
            max_chunk_length = 120 if lang == "ko" else 300

        # Preprocess text (unicode norm, cleaning) but defer language token
        # wrapping to per-chunk tokenization — matches ONNX pipeline behavior.
        pp_text = self.text_processor._preprocess_text(text, None)
        text_chunks = self._chunk_text(pp_text, max_chunk_length)

        silence_wav = None
        silence_wav_half = None
        if silence_duration > 0:
            silence_wav = np.zeros((1, int(silence_duration * self.sample_rate)), dtype=np.float32)
            silence_wav_half = np.zeros((1, int(silence_duration * self.sample_rate / 2.0)), dtype=np.float32)

        for i, text_chunk in enumerate(text_chunks):
            # Tokenize each chunk with per-chunk language wrapping
            text_ids_np, text_mask_np = self.text_processor([text_chunk], lang)

            wav_t, dur_t = self.forward(
                text_ids=torch.from_numpy(text_ids_np).to(self.device),
                style_ttl=torch.from_numpy(voice_style.ttl).to(self.device),
                style_dp=torch.from_numpy(voice_style.dp).to(self.device),
                text_mask=torch.from_numpy(text_mask_np).to(self.device),
                total_steps=total_steps,
                speed=speed,
            )
            wav = wav_t.cpu().numpy()
            dur = dur_t.cpu().numpy()

            # Trim leading/trailing silence from the waveform
            audio = wav[0]
            active_speech = np.where(np.abs(audio) > 0.002)[0]
            if len(active_speech) > 0:
                margin = int(self.sample_rate * 0.04)
                start = max(0, active_speech[0] - margin)
                end = min(len(audio), active_speech[-1] + margin)
                wav = wav[:, start:end]

            chunk_wav = wav
            chunk_dur = dur

            if silence_wav is not None and i < len(text_chunks) - 1:
                if re.search(r'[,，]["\')\]]*$', text_chunk.strip()):
                    chunk_wav = np.concatenate([wav, silence_wav_half], axis=1)
                    chunk_dur = dur + silence_duration / 2.0
                else:
                    chunk_wav = np.concatenate([wav, silence_wav], axis=1)
                    chunk_dur = dur + silence_duration

            yield chunk_wav, chunk_dur

    @staticmethod
    def _chunk_text(text: str, max_len: int) -> list[str]:
        """Split text into chunks respecting sentence boundaries."""
        return _chunk_text_util(text, max_len)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_pretrained(self, save_dir: str | Path, config: dict | str | Path | None = None) -> None:
        """Save model config, weights, and unicode indexer to a directory.

        Args:
            save_dir: Output directory path.
            config: Optional full model config dict, or path to an existing
                    tts.json file. If None, a minimal config is reconstructed
                    from model attributes.

        Creates:
            {save_dir}/tts.json              – model configuration
            {save_dir}/model.safetensors     – weights in safetensors format
            {save_dir}/unicode_indexer.json  – unicode indexer (if set on model)
        """
        import shutil

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Resolve config
        if config is None:
            config = self._build_config()
        elif isinstance(config, (str, Path)):
            with open(config) as f:
                config = json.load(f)

        with open(save_dir / "tts.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save weights
        state = self.state_dict()
        # Convert non-tensor buffers to tensors, ensure contiguous for safetensors
        for k in list(state.keys()):
            if not isinstance(state[k], torch.Tensor):
                state[k] = torch.tensor(state[k])
            elif not state[k].is_contiguous():
                state[k] = state[k].contiguous()
        safetensors.torch.save_file(state, str(save_dir / "model.safetensors"))

        # Save unicode indexer if set (skip if same path)
        if self.unicode_indexer is not None:
            dst = save_dir / "unicode_indexer.json"
            src = Path(self.unicode_indexer).resolve()
            if dst.resolve() != src:
                shutil.copy(src, dst)

        # Copy voice styles if available (skip if same path)
        if self.model_dir is not None:
            styles_src = self.model_dir / "voice_styles"
            styles_dst = save_dir / "voice_styles"
            if styles_src.is_dir() and not styles_dst.exists():
                shutil.copytree(styles_src, styles_dst)

    def _build_config(self) -> dict:
        """Stored config from tts.json."""
        return self.config

    @classmethod
    def from_pretrained(
        cls, model_dir: str | Path, device: str | torch.device | None = None,
    ) -> "SupertonicModel":
        """Load model from a directory created by save_pretrained.

        Args:
            model_dir: Path to directory containing tts.json and model.safetensors.
            device: Torch device to place the model on (e.g. ``"cuda"``, ``"cpu"``).
                Default ``None`` keeps weights on CPU.

        Returns:
            SupertonicModel with pretrained weights loaded.
        """

        model_dir = Path(model_dir)

        # Load config and construct model
        with open(model_dir / "tts.json") as f:
            config = json.load(f)

        # Resolve unicode_indexer if present
        indexer_path = model_dir / "unicode_indexer.json"
        unicode_indexer = indexer_path if indexer_path.exists() else None

        model = cls(config=config, unicode_indexer=unicode_indexer, model_dir=model_dir, device=device)

        # Load weights
        state = safetensors.torch.load_file(str(model_dir / "model.safetensors"))
        model.load_state_dict(state, strict=False)

        # Warn if AudioEncoder hasn't been trained yet
        if not model.has_encoder():
            warnings.warn(
                "AudioEncoder has not been trained yet. "
                "Voice cloning (voice_ref) is disabled until you train with "
                "scripts/train_encoder.py and call model.load_encoder()."
            )

        if device is not None:
            model = model.to(device)

        model.eval()
        return model


def length_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Convert lengths to binary mask."""
    max_len = max_len or int(lengths.max().item())
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).float()
    return mask.unsqueeze(1)
