"""Duration Predictor sub-model.

Predicts total utterance duration from text tokens + style vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvNeXtStack, LayerNorm
from .attention import FFN


class DurationPredictor(nn.Module):
    """Predicts total utterance duration from text + style.

    Architecture:
        Embedding + prepend sentence_token → ConvNeXt(6) → SelfAttn(2) + skip
        → extract CLS token → proj_out → concat style → MLP → exp

    Args:
        vocab_size: Vocabulary size (default 8322 for BPE tokenizer).
        config: Optional dict matching tts.json ``dp`` section.
    """

    def __init__(self, vocab_size: int = 8322, config: dict | None = None):
        super().__init__()
        if config is None:
            config = {}
        se = config.get("sentence_encoder", {})
        st = config.get("style_encoder", {}).get("style_token_layer", {})
        pr = config.get("predictor", {})

        embed_dim = se.get("char_emb_dim", 64)
        hidden_dim = se.get("convnext", {}).get("intermediate_dim", 256)
        style_dim = st.get("n_style", 8) * st.get("style_value_dim", 16)
        ksz = se.get("convnext", {}).get("ksz", 5)
        n_convnext = se.get("convnext", {}).get("num_layers", 6)
        dilations = se.get("convnext", {}).get("dilation_lst", [1] * n_convnext)
        n_attn_heads = se.get("attn_encoder", {}).get("n_heads", 2)
        n_attn_layers = se.get("attn_encoder", {}).get("n_layers", 2)
        predictor_hidden = pr.get("hdim", 128)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sentence_token = nn.Parameter(torch.randn(1, embed_dim, 1))

        self.convnext = ConvNeXtStack(embed_dim, hidden_dim, kernel_size=ksz, num_layers=n_convnext, dilations=dilations)

        self.n_heads = n_attn_heads
        self.head_dim = embed_dim // self.n_heads
        self.window_size = 9

        self.attn_layers = nn.ModuleList()
        self.attn_norms1 = nn.ModuleList()
        self.attn_ffn = nn.ModuleList()
        self.attn_norms2 = nn.ModuleList()
        self.attn_emb_rel_k = nn.ParameterList()
        self.attn_emb_rel_v = nn.ParameterList()

        for _ in range(n_attn_layers):
            self.attn_layers.append(nn.ModuleDict({
                'conv_q': nn.Conv1d(embed_dim, embed_dim, 1),
                'conv_k': nn.Conv1d(embed_dim, embed_dim, 1),
                'conv_v': nn.Conv1d(embed_dim, embed_dim, 1),
                'conv_o': nn.Conv1d(embed_dim, embed_dim, 1),
            }))
            self.attn_emb_rel_k.append(nn.Parameter(torch.randn(1, self.window_size, self.head_dim) * 0.02))
            self.attn_emb_rel_v.append(nn.Parameter(torch.randn(1, self.window_size, self.head_dim) * 0.02))
            self.attn_norms1.append(LayerNorm(embed_dim))
            self.attn_ffn.append(FFN(embed_dim, hidden_dim))
            self.attn_norms2.append(LayerNorm(embed_dim))

        self.proj_out = nn.Conv1d(embed_dim, embed_dim, 1, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + style_dim, predictor_hidden),
            nn.PReLU(num_parameters=1),
            nn.Linear(predictor_hidden, 1),
        )

    def _get_relative_embeddings(self, emb: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Build (seq_len, seq_len, head_dim) relative embedding matrix.

        Uses pure tensor operations (no Python range) for ONNX export compatibility.
        """
        half = self.window_size // 2
        L = seq_len
        device, dtype = emb.device, emb.dtype

        # rel_pos[i,j] = j - i,  shape (L, L)
        rel_pos = torch.arange(L, device=device).unsqueeze(0) - torch.arange(L, device=device).unsqueeze(1)

        # Index into the embedding: shift by half, clamp to valid range
        idx = (rel_pos + half).clamp(0, self.window_size - 1)

        # Gather: emb is (1, window_size, head_dim)
        result = emb[0, idx]  # (L, L, head_dim)

        # Zero out positions outside the window
        in_window = (rel_pos >= -half) & (rel_pos <= half)  # (L, L)
        result = result * in_window.unsqueeze(-1).to(dtype)

        return result

    def _self_attention(self, x: torch.Tensor, attn_mod: nn.ModuleDict,
                         emb_rel_k: torch.Tensor, emb_rel_v: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        n_heads = self.n_heads
        head_dim = self.head_dim

        q = attn_mod['conv_q'](x).view(B, n_heads, head_dim, L).transpose(2, 3)
        k = attn_mod['conv_k'](x).view(B, n_heads, head_dim, L).transpose(2, 3)
        v = attn_mod['conv_v'](x).view(B, n_heads, head_dim, L).transpose(2, 3)

        scale = head_dim ** -0.5
        q_s = q * scale

        scores_content = torch.matmul(q_s, k.transpose(-2, -1))
        rel_k = self._get_relative_embeddings(emb_rel_k, L)
        scores_rel = torch.einsum("bhld,lcd->bhlc", q_s, rel_k)

        scores = scores_content + scores_rel
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, L, D)

        rel_v = self._get_relative_embeddings(emb_rel_v, L)
        out_rel = torch.einsum("bhlc,lcd->bhld", attn, rel_v)
        out = (out + out_rel).transpose(2, 3).contiguous().view(B, C, L)
        out = attn_mod['conv_o'](out)
        return out * mask if mask is not None else out

    def forward(self, text_ids: torch.Tensor, style_dp: torch.Tensor,
                text_mask: torch.Tensor) -> torch.Tensor:
        B = text_ids.shape[0]

        # Embed and prepend sentence token
        x = self.embedding(text_ids).transpose(1, 2) * text_mask
        token = self.sentence_token.expand(B, -1, -1)
        x = torch.cat([token, x], dim=2)
        mask_padded = F.pad(text_mask, (1, 0), value=1.0)

        # ConvNeXt
        for block in self.convnext.layers:
            x = block(x) * mask_padded

        convnext_out = x

        # Self-attention
        for i, (attn_mod, norm1, ffn, norm2) in enumerate(zip(
            self.attn_layers, self.attn_norms1, self.attn_ffn, self.attn_norms2
        )):
            residual = x
            x = self._self_attention(x, attn_mod, self.attn_emb_rel_k[i], self.attn_emb_rel_v[i], mask_padded)
            x = residual + x
            x = norm1(x) * mask_padded

            residual = x
            x = ffn(x) * mask_padded
            x = residual + x
            x = norm2(x) * mask_padded

        # Skip connection from convnext
        x = x + convnext_out

        # Extract CLS token (position 0)
        sentence_repr = self.proj_out(x[:, :, :1])
        x_flat = sentence_repr.view(B, -1)
        style_flat = style_dp.reshape(B, -1)

        combined = torch.cat([x_flat, style_flat], dim=-1)
        return torch.exp(self.mlp(combined)).squeeze(-1)
