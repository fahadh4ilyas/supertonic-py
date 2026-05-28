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

import math
from pathlib import Path
from typing import Optional

import torch
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
)
from .duration_predictor import DurationPredictor
from .text_encoder import TextEncoder
from .vector_field import VectorField
from .vocoder import Vocoder


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
    # Sub-models
    "DurationPredictor",
    "TextEncoder",
    "VectorField",
    "Vocoder",
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
    ):
        super().__init__()

        # Resolve config
        if isinstance(config, (str, Path)):
            import json
            with open(config) as f:
                config = json.load(f)
        if config is None:
            config = {}

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

    def sample_noisy_latent(self, duration: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = duration.shape[0]
        wav_len_max = int(duration.max().item() * self.sample_rate)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = int(math.ceil(wav_len_max / chunk_size))
        latent_dim = self.ldim * self.chunk_compress_factor

        noisy_latent = torch.randn(bsz, latent_dim, latent_len, device=duration.device)

        wav_lengths = (duration * self.sample_rate).long()
        latent_lengths = (wav_lengths + chunk_size - 1) // chunk_size
        max_len = latent_len
        ids = torch.arange(max_len, device=duration.device).unsqueeze(0)
        latent_mask = (ids < latent_lengths.unsqueeze(1)).float().unsqueeze(1)
        noisy_latent = noisy_latent * latent_mask
        return noisy_latent, latent_mask

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
            cs = torch.full((text_ids.shape[0],), step, device=text_ids.device, dtype=torch.float32)
            ts = torch.full((text_ids.shape[0],), total_steps, device=text_ids.device, dtype=torch.float32)
            # VectorField output includes Euler step internally (matches ONNX)
            xt = self.vector_field(xt, text_emb, style_ttl, latent_mask, text_mask, cs, ts)

        wav = self.vocoder(xt)
        return wav, dur_scaled

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_pretrained(self, save_dir: str | Path, config: dict | str | Path | None = None) -> None:
        """Save model config and weights to a directory.

        Args:
            save_dir: Output directory path.
            config: Optional full model config dict, or path to an existing
                    tts.json file. If None, a minimal config is reconstructed
                    from model attributes.

        Creates:
            {save_dir}/tts.json          – model configuration
            {save_dir}/model.safetensors – weights in safetensors format
        """
        import json
        import safetensors.torch

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

    def _build_config(self) -> dict:
        """Reconstruct a tts.json-compatible config from model architecture."""
        return {
            "ae": {
                "sample_rate": self.sample_rate,
                "base_chunk_size": self.base_chunk_size,
                "decoder": {
                    "num_layers": 10,
                    "dilation_lst": [1, 2, 4, 1, 2, 4, 1, 1, 1, 1],
                    "intermediate_dim": 2048,
                    "idim": 24,
                    "hdim": 512,
                    "ksz": 7,
                    "head": {"idim": 512, "hdim": 2048, "odim": 512, "ksz": 3},
                },
            },
            "ttl": {
                "latent_dim": self.ldim,
                "chunk_compress_factor": self.chunk_compress_factor,
                "normalizer": {"scale": 0.25},
                "text_encoder": {
                    "text_embedder": {"char_emb_dim": 256},
                    "convnext": {
                        "idim": 256, "ksz": 5, "intermediate_dim": 1024,
                        "num_layers": 6, "dilation_lst": [1, 1, 2, 2, 4, 4],
                    },
                    "attn_encoder": {
                        "hidden_channels": 256, "filter_channels": 1024,
                        "n_heads": 4, "n_layers": 4,
                    },
                },
                "style_encoder": {
                    "style_token_layer": {
                        "n_style": 50,
                        "style_value_dim": 256,
                    },
                },
                "speech_prompted_text_encoder": {
                    "text_dim": 256, "style_dim": 256,
                    "n_units": 256, "n_heads": 2,
                },
                "vector_field": {
                    "proj_in": {"ldim": 24, "chunk_compress_factor": 6, "odim": 512},
                    "time_encoder": {"time_dim": 64, "hdim": 256},
                    "main_blocks": {
                        "n_blocks": 4,
                        "text_cond_layer": {
                            "idim": 512, "text_dim": 256,
                            "n_heads": 8, "n_units": 512,
                        },
                        "style_cond_layer": {"idim": 512, "style_dim": 256},
                        "convnext_0": {
                            "idim": 512, "ksz": 5, "intermediate_dim": 2048,
                            "num_layers": 4, "dilation_lst": [1, 2, 4, 8],
                        },
                        "convnext_1": {
                            "idim": 512, "ksz": 5, "intermediate_dim": 2048,
                            "num_layers": 1, "dilation_lst": [1],
                        },
                        "convnext_2": {
                            "idim": 512, "ksz": 5, "intermediate_dim": 2048,
                            "num_layers": 1, "dilation_lst": [1],
                        },
                    },
                    "last_convnext": {
                        "idim": 512, "ksz": 5, "intermediate_dim": 2048,
                        "num_layers": 4, "dilation_lst": [1, 1, 1, 1],
                    },
                    "proj_out": {"idim": 512, "chunk_compress_factor": 6, "ldim": 24},
                },
            },
            "dp": {
                "sentence_encoder": {
                    "char_emb_dim": 64,
                    "convnext": {
                        "idim": 64, "ksz": 5, "intermediate_dim": 256,
                        "num_layers": 6, "dilation_lst": [1, 1, 1, 1, 1, 1],
                    },
                    "attn_encoder": {
                        "hidden_channels": 64, "filter_channels": 256,
                        "n_heads": 2, "n_layers": 2,
                    },
                },
                "style_encoder": {
                    "style_token_layer": {
                        "n_style": 8,
                        "style_value_dim": 16,
                    },
                },
                "predictor": {
                    "hdim": 128, "n_layer": 2,
                },
            },
        }

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "SupertonicModel":
        """Load model from a directory created by save_pretrained.

        Args:
            model_dir: Path to directory containing tts.json and model.safetensors.

        Returns:
            SupertonicModel with pretrained weights loaded.
        """
        import json
        import safetensors.torch

        model_dir = Path(model_dir)

        # Load config and construct model
        with open(model_dir / "tts.json") as f:
            config = json.load(f)

        model = cls(config=config)

        # Load weights
        state = safetensors.torch.load_file(str(model_dir / "model.safetensors"))
        model.load_state_dict(state, strict=False)

        return model


def length_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Convert lengths to binary mask."""
    max_len = max_len or int(lengths.max().item())
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).float()
    return mask.unsqueeze(1)
