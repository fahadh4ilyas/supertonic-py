"""Text Encoder sub-model.

Encodes text tokens to style-conditioned embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import ConvNeXtStack, LayerNorm
from .attention import RelPosAttention, FFN, SpeechPromptedAttention


class TextEncoder(nn.Module):
    """Encodes text to style-conditioned embeddings.

    Embedding → ConvNeXt(6) → SelfAttn(4) + skip → SpeechPromptedAttn(2) → norm

    Args:
        vocab_size: Vocabulary size (default 8322 for BPE tokenizer).
        config: Optional dict matching tts.json ``ttl`` section.
    """

    def __init__(self, vocab_size: int = 8322, config: dict | None = None):
        super().__init__()
        if config is None:
            config = {}
        te = config.get("text_encoder", {})
        st = config.get("style_encoder", {}).get("style_token_layer", {})
        sp = config.get("speech_prompted_text_encoder", {})

        embed_dim = te.get("text_embedder", {}).get("char_emb_dim", 256)
        hidden_dim = te.get("convnext", {}).get("intermediate_dim", 1024)
        ksz = te.get("convnext", {}).get("ksz", 5)
        n_convnext = te.get("convnext", {}).get("num_layers", 6)
        dilations = te.get("convnext", {}).get("dilation_lst", [1, 1, 2, 2, 4, 4])
        n_attn_heads = te.get("attn_encoder", {}).get("n_heads", 4)
        n_attn_layers = te.get("attn_encoder", {}).get("n_layers", 4)
        speech_attn_heads = sp.get("n_heads", 2)
        n_style_tokens = st.get("n_style", 50)
        style_dim = st.get("style_value_dim", 256)

        self.text_embedder = nn.Embedding(vocab_size, embed_dim)

        self.convnext = ConvNeXtStack(embed_dim, hidden_dim, kernel_size=ksz, num_layers=n_convnext, dilations=dilations)

        # Self-attention layers with relative position
        self.attn_layers = nn.ModuleList([
            RelPosAttention(embed_dim, num_heads=n_attn_heads, rel_window=9)
            for _ in range(n_attn_layers)
        ])
        self.attn_norms1 = nn.ModuleList([LayerNorm(embed_dim) for _ in range(n_attn_layers)])
        self.attn_ffn = nn.ModuleList([FFN(embed_dim, hidden_dim) for _ in range(n_attn_layers)])
        self.attn_norms2 = nn.ModuleList([LayerNorm(embed_dim) for _ in range(n_attn_layers)])

        # Speech-prompted cross-attention layers
        self.speech_prompted_attn = nn.ModuleList([
            SpeechPromptedAttention(embed_dim) for _ in range(speech_attn_heads)
        ])
        self.out_norm = LayerNorm(embed_dim)

        # Precomputed style key buffer (loaded from ONNX)
        self.register_buffer("style_key", torch.zeros(1, n_style_tokens, style_dim))

    def forward(self, text_ids: torch.Tensor, style_ttl: torch.Tensor,
                text_mask: torch.Tensor) -> torch.Tensor:
        x = self.text_embedder(text_ids).transpose(1, 2) * text_mask  # (B, 256, L)

        # ConvNeXt
        for block in self.convnext.layers:
            x = block(x) * text_mask

        convnext_out = x

        # Self-attention
        for attn, norm1, ffn, norm2 in zip(
            self.attn_layers, self.attn_norms1, self.attn_ffn, self.attn_norms2
        ):
            residual = x
            x = attn(x, mask=text_mask)
            x = residual + x
            x = norm1(x) * text_mask

            residual = x
            x = ffn(x) * text_mask
            x = residual + x
            x = norm2(x) * text_mask

        # Skip connection from convnext
        x = x + convnext_out
        x = x * text_mask

        # Speech-prompted cross-attention
        x_t = x.transpose(1, 2)  # (B, L, 256)
        shared_residual = x_t

        attn1_out = self.speech_prompted_attn[0](x_t, context=style_ttl, style_key=self.style_key, mask=text_mask)
        attn1_out = attn1_out * text_mask.transpose(1, 2)
        x_t = shared_residual + attn1_out

        attn2_out = self.speech_prompted_attn[1](x_t, context=style_ttl, style_key=self.style_key, mask=text_mask)
        attn2_out = attn2_out * text_mask.transpose(1, 2)
        x_t = shared_residual + attn2_out

        x = x_t.transpose(1, 2)  # (B, 256, L)
        x = self.out_norm(x) * text_mask
        return x
