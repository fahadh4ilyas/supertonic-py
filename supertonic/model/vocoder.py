"""Vocoder sub-model.

Neural audio decoder: latent representation → waveform.
Uses causal padding throughout (matches ONNX vocoder).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import CausalPad1d, ConvNeXtStack


class Vocoder(nn.Module):
    """Neural audio decoder: latent → waveform.

    Uses causal padding throughout (matches ONNX vocoder).

    Args:
        config: Optional dict matching tts.json ``ae.decoder`` section.
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        if config is None:
            config = {}
        head_cfg = config.get("head", {})

        latent_channels = config.get("idim", 24)
        hidden_dim = config.get("hdim", 512)
        self.temporal_factor = 6  # fixed: chunk_compress_factor

        # Normalization
        self.register_buffer("normalizer_scale", torch.tensor(0.25))
        self.latent_mean = nn.Parameter(torch.zeros(1, latent_channels, 1))
        self.latent_std = nn.Parameter(torch.ones(1, latent_channels, 1))

        ksz = config.get("ksz", 7)
        dilations = config.get("dilation_lst", [1, 2, 4, 1, 2, 4, 1, 1, 1, 1])
        num_layers = config.get("num_layers", len(dilations))
        intermediate_dim = config.get("intermediate_dim", 2048)

        # Input embedding with causal padding
        self.embed_pad = CausalPad1d(ksz, dilation=1)
        self.embed = nn.Conv1d(latent_channels, hidden_dim, kernel_size=ksz)

        # ConvNeXt decoder
        self.convnext = ConvNeXtStack(hidden_dim, intermediate_dim, kernel_size=ksz,
                                       num_layers=num_layers, dilations=dilations, causal=True)

        self.final_norm = nn.BatchNorm1d(hidden_dim)

        # Output head
        head_ksz = head_cfg.get("ksz", 3)
        head_hidden = head_cfg.get("hdim", 2048)
        head_output = head_cfg.get("odim", 512)
        self.head_pad = CausalPad1d(head_ksz, dilation=1)
        self.head_layer1 = nn.Conv1d(hidden_dim, head_hidden, kernel_size=head_ksz)
        self.head_act = nn.PReLU(num_parameters=1)
        self.head_layer2 = nn.Conv1d(head_hidden, head_output, kernel_size=1, bias=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B, C, L = latent.shape

        # Denormalize: x = latent / scale (with clamp for stability)
        x = latent.clamp(-2.0, 2.0) / self.normalizer_scale

        # Reshape: (B, 144, L) → (B, 24, 6, L) → permute → (B, 24, L, 6) → (B, 24, L*6)
        x = x.view(B, 24, self.temporal_factor, L)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(B, 24, -1)

        # Apply learned mean/std
        x = x * self.latent_std + self.latent_mean

        # Embed
        x = self.embed_pad(x)
        x = self.embed(x)

        # ConvNeXt
        for block in self.convnext.layers:
            x = block(x)

        # Final norm and head
        x = self.final_norm(x)
        x = self.head_pad(x)
        x = self.head_layer1(x)
        x = self.head_act(x)
        x = self.head_layer2(x)

        # Output: transpose and flatten
        x = x.transpose(1, 2).reshape(B, -1)
        return x
