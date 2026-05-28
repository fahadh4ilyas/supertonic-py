"""Common modules shared across sub-models.

Padding utilities, LayerNorm, and ConvNeXt block implementations
that match the ONNX architecture exactly.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Padding utilities (match ONNX symmetric / causal pad with replicate mode)
# ---------------------------------------------------------------------------


class SymmetricPad1d(nn.Module):
    """Symmetric (edge-replicate) padding for 1D convolutions.

    ONNX uses symmetric padding: pads equally on both sides.
    """

    def __init__(self, kernel_size: int, dilation: int = 1):
        super().__init__()
        total_pad = (kernel_size - 1) * dilation
        self.pad_left = total_pad // 2
        self.pad_right = total_pad - self.pad_left

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (self.pad_left, self.pad_right), mode="replicate")


class CausalPad1d(nn.Module):
    """Causal (left-only, edge-replicate) padding for 1D convolutions.

    ONNX vocoder uses causal padding: all padding on the left side.
    """

    def __init__(self, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad_left = (kernel_size - 1) * dilation
        self.pad_right = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (self.pad_left, self.pad_right), mode="replicate")


# ---------------------------------------------------------------------------
# LayerNorm for (B, C, T) layout
# ---------------------------------------------------------------------------


class LayerNorm(nn.Module):
    """LayerNorm along the channel dimension for (B, C, T) inputs."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


# ---------------------------------------------------------------------------
# ConvNeXt block
# ---------------------------------------------------------------------------


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with explicit padding (matches ONNX).

    Flow: Pad → DepthwiseConv → LayerNorm → PWConv1 → GELU → PWConv2 → gamma*out + residual
    """

    def __init__(
        self,
        channels: int,
        intermediate: int,
        kernel_size: int = 5,
        dilation: int = 1,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal
        self.pad = CausalPad1d(kernel_size, dilation) if causal else SymmetricPad1d(kernel_size, dilation)
        self.dwconv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, groups=channels, bias=True)
        self.norm = LayerNorm(channels)
        self.pwconv1 = nn.Conv1d(channels, intermediate, 1)
        self.pwconv2 = nn.Conv1d(intermediate, channels, 1)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1) * 1e-6)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.pad(x)
        x = self.dwconv(x)
        if mask is not None:
            x = x * mask
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x, approximate='none')
        x = self.pwconv2(x)
        x = residual + self.gamma * x
        if mask is not None:
            x = x * mask
        return x


class ConvNeXtStack(nn.Module):
    """Stack of ConvNeXt blocks."""

    def __init__(
        self,
        channels: int,
        intermediate: int,
        kernel_size: int = 5,
        num_layers: int = 4,
        dilations: Optional[list[int]] = None,
        causal: bool = False,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1] * num_layers
        assert len(dilations) == num_layers
        self.layers = nn.ModuleList([
            ConvNeXtBlock(channels, intermediate, kernel_size, dilation=d, causal=causal)
            for d in dilations
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
