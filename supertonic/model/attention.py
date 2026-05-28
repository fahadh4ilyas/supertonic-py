"""Attention modules used across sub-models.

- RelPosAttention: self-attention with relative position embeddings
- FFN: feed-forward network for transformer blocks
- RoPECrossAttention: cross-attention with Rotary Position Embedding (VectorField)
- StyleCrossAttention: cross-attention with split key/value (VectorField)
- SpeechPromptedAttention: cross-attention with style key buffer (TextEncoder)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Relative position self-attention
# ---------------------------------------------------------------------------


class RelPosAttention(nn.Module):
    """Self-attention with relative position embeddings (windowed).

    Uses Conv1d projections and relative position bias within a local window.
    """

    def __init__(self, channels: int, num_heads: int, rel_window: int = 9, dropout: float = 0.0):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.rel_window = rel_window
        self.half_window = rel_window // 2

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)

        self.emb_rel_k = nn.Parameter(torch.randn(1, rel_window, self.head_dim) * 0.02)
        self.emb_rel_v = nn.Parameter(torch.randn(1, rel_window, self.head_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def _get_relative_embeddings(self, emb: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Build (seq_len, seq_len, head_dim) relative embedding matrix.

        Uses pure tensor operations (no Python range) for ONNX export compatibility.
        """
        L = seq_len
        device, dtype = emb.device, emb.dtype

        # rel_pos[i,j] = j - i,  shape (L, L)
        rel_pos = torch.arange(L, device=device).unsqueeze(0) - torch.arange(L, device=device).unsqueeze(1)

        idx = (rel_pos + self.half_window).clamp(0, self.rel_window - 1)
        result = emb[0, idx]  # (L, L, head_dim)

        in_window = (rel_pos >= -self.half_window) & (rel_pos <= self.half_window)  # (L, L)
        result = result * in_window.unsqueeze(-1).to(dtype)

        return result

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, L = x.shape

        q = self.conv_q(x).view(B, self.num_heads, self.head_dim, L).transpose(2, 3)  # (B, H, L, D)
        k = self.conv_k(x).view(B, self.num_heads, self.head_dim, L).transpose(2, 3)
        v = self.conv_v(x).view(B, self.num_heads, self.head_dim, L).transpose(2, 3)

        scale = self.head_dim ** -0.5
        q_s = q * scale

        scores_content = torch.matmul(q_s, k.transpose(-2, -1))  # (B, H, L, L)

        rel_k = self._get_relative_embeddings(self.emb_rel_k, L)  # (L, L, D)
        scores_rel = torch.einsum("bhld,lcd->bhlc", q_s, rel_k)

        scores = scores_content + scores_rel

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out_content = torch.matmul(attn, v)  # (B, H, L, D)

        rel_v = self._get_relative_embeddings(self.emb_rel_v, L)
        out_rel = torch.einsum("bhlc,lcd->bhld", attn, rel_v)

        out = (out_content + out_rel).transpose(2, 3).contiguous().view(B, C, L)
        out = self.conv_o(out)
        return out


# ---------------------------------------------------------------------------
# FFN (used in transformer encoder layers)
# ---------------------------------------------------------------------------


class FFN(nn.Module):
    """Feed-forward network: conv1d → ReLU → conv1d."""

    def __init__(self, channels: int, filter_channels: int):
        super().__init__()
        self.conv_1 = nn.Conv1d(channels, filter_channels, 1)
        self.conv_2 = nn.Conv1d(filter_channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_2(F.relu(self.conv_1(x)))


# ---------------------------------------------------------------------------
# RoPE-based cross-attention
# ---------------------------------------------------------------------------


class RoPECrossAttention(nn.Module):
    """Cross-attention with Rotary Position Embedding.

    Used in VectorField for latent-to-text cross-attention.
    """

    def __init__(
        self,
        query_dim: int = 512,
        context_dim: int = 256,
        attn_dim: int = 256,
        head_dim: int = 64,
        max_seq_len: int = 1000,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.attn_dim = attn_dim
        self.head_dim = head_dim
        self.n_heads = attn_dim // head_dim
        self.scale = 1.0 / 16.0  # matches ONNX constant (16.0 divisor)

        self.W_query = nn.Linear(query_dim, attn_dim)
        self.W_key = nn.Linear(context_dim, attn_dim)
        self.W_value = nn.Linear(context_dim, attn_dim)
        self.out_fc = nn.Linear(attn_dim, query_dim)

        theta = 10000.0 ** (-torch.arange(0, head_dim // 2).float() / (head_dim // 2))
        self.register_buffer("theta", theta)
        increments = torch.arange(max_seq_len).unsqueeze(0).unsqueeze(-1)
        self.register_buffer("increments", increments)

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        B, H, L, D = x.shape
        pos = self.increments[:, :seq_len, :].float() / seq_len
        angles = pos * self.theta
        x1, x2 = x[..., :D // 2], x[..., D // 2:]
        cos, sin = angles.cos().unsqueeze(0), angles.sin().unsqueeze(0)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, L = x.shape
        x_t = x.transpose(1, 2)  # (B, L, C)

        q = self.W_query(x_t).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        q = self._apply_rope(q, L)
        k = self._apply_rope(k, context.shape[1])

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if context_mask is not None:
            scores = scores.masked_fill(context_mask.unsqueeze(1) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.attn_dim)
        out = self.out_fc(out).transpose(1, 2)  # (B, C, L)

        if mask is not None:
            out = out * mask
        return out


# ---------------------------------------------------------------------------
# Style cross-attention (VectorField)
# ---------------------------------------------------------------------------


class StyleCrossAttention(nn.Module):
    """Cross-attention for style conditioning in VectorField.

    Key insight from ONNX: W_key operates on a precomputed k_context constant,
    NOT on style_ttl directly. W_value uses style_ttl. tanh is applied to keys.
    """

    def __init__(self, query_dim: int = 512, context_dim: int = 256, attn_dim: int = 256):
        super().__init__()
        self.half_dim = attn_dim // 2  # 128
        self.scale = attn_dim ** -0.5
        self.attn_dim = attn_dim

        self.W_query = nn.Linear(query_dim, attn_dim)
        self.W_key = nn.Linear(context_dim, attn_dim)
        self.W_value = nn.Linear(context_dim, attn_dim)
        self.out_fc = nn.Linear(attn_dim, query_dim)

        # Precomputed key context (loaded from ONNX, shared across blocks)
        self.register_buffer("k_context", torch.zeros(1, 50, context_dim))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, L = x.shape
        x_t = x.transpose(1, 2)  # (B, L, C)

        q = self.W_query(x_t)
        q0, q1 = q.split(self.half_dim, dim=-1)
        q = torch.stack([q0, q1], dim=0)  # (2, B, L, 128)

        # Key from precomputed k_context (or key_context for CFG)
        if key_context is not None:
            k_input = key_context
        else:
            k_input = self.k_context.expand(B, -1, -1)
        k = self.W_key(k_input)
        k0, k1 = k.split(self.half_dim, dim=-1)
        k = torch.stack([k0, k1], dim=0).transpose(2, 3)  # (2, B, 128, L_ctx)
        k = torch.tanh(k)

        # Value from style_ttl
        v = self.W_value(context)
        v0, v1 = v.split(self.half_dim, dim=-1)
        v = torch.stack([v0, v1], dim=0)  # (2, B, L_ctx, 128)

        scores = torch.matmul(q, k) * self.scale  # (2, B, L, L_ctx)
        attn = F.softmax(scores, dim=-1)

        if mask is not None:
            attn = attn * mask.transpose(1, 2).unsqueeze(0)

        out = torch.matmul(attn, v)  # (2, B, L, 128)
        out0, out1 = out[0], out[1]
        out = torch.cat([out0, out1], dim=-1)  # (B, L, 256)
        out = self.out_fc(out).transpose(1, 2)  # (B, 512, L)
        return out


# ---------------------------------------------------------------------------
# Speech-prompted cross-attention (TextEncoder)
# ---------------------------------------------------------------------------


class SpeechPromptedAttention(nn.Module):
    """Cross-attention for TextEncoder: text queries attend to style.

    Key insight from ONNX: W_key operates on a precomputed style_key buffer
    (tanh applied), W_value operates on style_ttl context. Uses 2-way split.
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2  # 128

        self.W_query = nn.Linear(dim, dim)
        self.W_key = nn.Linear(dim, dim)    # Key from style_key buffer
        self.W_value = nn.Linear(dim, dim)  # Value from style_ttl context
        self.out_fc = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        style_key: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, C = x.shape

        # Query: split into 2 halves, stack → (2, B, L, 128)
        q = self.W_query(x)
        q0, q1 = q.split(self.half_dim, dim=-1)
        q = torch.stack([q0, q1], dim=0)

        # Key from style_key buffer (precomputed), tanh + transpose
        k_input = style_key.expand(B, -1, -1) if style_key is not None else context
        k = self.W_key(k_input)
        k0, k1 = k.split(self.half_dim, dim=-1)
        k = torch.stack([k0, k1], dim=0)  # (2, B, L_ctx, 128)
        k = k.transpose(2, 3)  # (2, B, 128, L_ctx)
        k = torch.tanh(k)

        # Value from context (style_ttl)
        v = self.W_value(context)
        v0, v1 = v.split(self.half_dim, dim=-1)
        v = torch.stack([v0, v1], dim=0)  # (2, B, L_ctx, 128)

        # Attention
        scale = self.dim ** -0.5
        scores = torch.matmul(q, k) * scale  # (2, B, L, L_ctx)
        attn = F.softmax(scores, dim=-1)

        if mask is not None:
            attn = attn * mask.transpose(1, 2).unsqueeze(0)

        out = torch.matmul(attn, v)  # (2, B, L, 128)
        out0, out1 = out[0], out[1]
        out = torch.cat([out0, out1], dim=-1)  # (B, L, 256)
        return self.out_fc(out)
