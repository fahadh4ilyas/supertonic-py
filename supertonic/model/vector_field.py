"""Vector Field (Flow Matching Denoiser) sub-model.

CFG-augmented diffusion denoiser with ConvNeXt, RoPE cross-attention,
and style cross-attention blocks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvNeXtStack, ConvNeXtBlock, LayerNorm
from .attention import RoPECrossAttention, StyleCrossAttention


class VectorField(nn.Module):
    """Flow-matching vector field denoiser.

    Noisy latent → proj_in → 4×[ConvNeXt + time + ConvNeXt + RoPE(attn_text) + ConvNeXt + StyleAttn]
    → last_convnext → proj_out

    Includes Euler integration step in the output (matches ONNX),
    and classifier-free guidance with cfg_scale=4.0.

    Args:
        config: Optional dict matching tts.json ``ttl.vector_field`` section.
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        if config is None:
            config = {}
        st = config.get("style_cond_layer", {})
        st_parent = config.get("main_blocks", {})
        st2 = st_parent.get("style_cond_layer", {}) if st_parent else {}
        tx = st_parent.get("text_cond_layer", {}) if st_parent else {}
        c0 = st_parent.get("convnext_0", {}) if st_parent else {}
        c1 = st_parent.get("convnext_1", {}) if st_parent else {}
        c2 = st_parent.get("convnext_2", {}) if st_parent else {}
        lc = config.get("last_convnext", {})
        pi = config.get("proj_in", {})
        po = config.get("proj_out", {})
        te_cfg = config.get("time_encoder", {})

        latent_dim = (pi.get("ldim", 24)) * (pi.get("chunk_compress_factor", 6))  # 144
        hidden_dim = pi.get("odim", 512)
        text_dim = tx.get("text_dim", 256)
        style_dim = st2.get("style_dim", 256) or st.get("style_dim", 256)
        intermediate_dim = c0.get("intermediate_dim", 2048)
        time_dim = te_cfg.get("time_dim", 64)
        time_hidden = te_cfg.get("hdim", 256)
        n_blocks = st_parent.get("n_blocks", 4) if st_parent else 4
        ksz = c0.get("ksz", 5)
        c0_dilations = c0.get("dilation_lst", [1, 2, 4, 8])

        self.proj_in = nn.Conv1d(latent_dim, hidden_dim, 1, bias=False)

        # Sinusoidal time encoder
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, time_hidden),
            nn.Mish() if hasattr(F, 'mish') else nn.GELU(),
            nn.Linear(time_hidden, time_dim),
        )
        self.time_proj = nn.ModuleList([
            nn.Linear(time_dim, hidden_dim) for _ in range(n_blocks)
        ])

        # Main blocks
        self.main_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.main_blocks.append(ConvNeXtStack(hidden_dim, intermediate_dim, kernel_size=ksz, num_layers=len(c0_dilations), dilations=c0_dilations))
            self.main_blocks.append(ConvNeXtBlock(hidden_dim, intermediate_dim, kernel_size=ksz))
            self.main_blocks.append(RoPECrossAttention(query_dim=hidden_dim, context_dim=text_dim, attn_dim=hidden_dim, head_dim=64))
            self.main_blocks.append(LayerNorm(hidden_dim))
            self.main_blocks.append(ConvNeXtBlock(hidden_dim, intermediate_dim, kernel_size=ksz))
            self.main_blocks.append(StyleCrossAttention(query_dim=hidden_dim, context_dim=style_dim, attn_dim=style_dim))
            self.main_blocks.append(LayerNorm(hidden_dim))

        self.last_convnext = ConvNeXtStack(hidden_dim, intermediate_dim, kernel_size=lc.get("ksz", 5),
                                            num_layers=lc.get("num_layers", 4),
                                            dilations=lc.get("dilation_lst", [1, 1, 1, 1]))
        self.proj_out = nn.Conv1d(hidden_dim, latent_dim, 1, bias=False)

        # CFG uncond tokens (loaded from ONNX)
        n_style_tokens = 50
        self.register_buffer('uncond_text_token', torch.zeros(1, text_dim, 1))
        self.register_buffer('uncond_style_value_token', torch.zeros(1, n_style_tokens, style_dim))
        self.register_buffer('uncond_style_key_token', torch.zeros(1, n_style_tokens, style_dim))

    def _sinusoidal(self, t: torch.Tensor, dim: int = 64) -> torch.Tensor:
        half = dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = (t[:, None] * 1000) * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        text_emb: torch.Tensor,
        style_ttl: torch.Tensor,
        latent_mask: torch.Tensor,
        text_mask: torch.Tensor,
        current_step: torch.Tensor,
        total_step: torch.Tensor,
    ) -> torch.Tensor:
        """Returns denoised latent (includes Euler step), matching ONNX output."""
        v_pred_cfg = self._forward_cfg(noisy_latent, text_emb, style_ttl, latent_mask, text_mask, current_step, total_step, cfg_scale=4.0)
        dt = 1.0 / total_step.clamp(min=1)
        return (noisy_latent + v_pred_cfg * dt[:, None, None]) * latent_mask

    def _forward_cfg(
        self,
        noisy_latent: torch.Tensor,
        text_emb: torch.Tensor,
        style_ttl: torch.Tensor,
        latent_mask: torch.Tensor,
        text_mask: torch.Tensor,
        current_step: torch.Tensor,
        total_step: torch.Tensor,
        cfg_scale: float = 4.0,
    ) -> torch.Tensor:
        B = noisy_latent.shape[0]

        # --- CFG batch doubling ---
        noisy_latent_2 = noisy_latent.repeat(2, 1, 1)

        # Concat text_emb with uncond_text_token
        uncond_text = self.uncond_text_token.expand(B, -1, text_emb.shape[-1])
        text_emb_2 = torch.cat([text_emb, uncond_text], dim=0)

        # Concat style_ttl with uncond_style_value_token
        uncond_style_val = self.uncond_style_value_token.expand(B, -1, -1)
        style_ttl_2 = torch.cat([style_ttl, uncond_style_val], dim=0)

        # Build doubled key context for StyleCrossAttention
        k_context_cond = self.main_blocks[5].k_context.expand(B, -1, -1)
        uncond_style_key = self.uncond_style_key_token.expand(B, -1, -1)
        key_context_2 = torch.cat([k_context_cond, uncond_style_key], dim=0)

        # Double masks and timesteps
        latent_mask_2 = latent_mask.repeat(2, 1, 1)
        text_mask_2 = text_mask.repeat(2, 1, 1)
        current_step_2 = current_step.repeat(2)
        total_step_2 = total_step.repeat(2)

        # --- Main computation ---
        t = (current_step_2 / total_step_2.clamp(min=1)).float()
        t_emb = self._sinusoidal(t)
        t_emb = self.time_encoder(t_emb)

        x = self.proj_in(noisy_latent_2) * latent_mask_2
        text_context = text_emb_2.transpose(1, 2)

        time_idx = 0
        block_idx = 0

        for _ in range(len(self.time_proj)):
            # 4x dilated ConvNeXt
            for conv_block in self.main_blocks[block_idx].layers:
                x = conv_block(x, mask=latent_mask_2)
            block_idx += 1

            # Time injection
            x = x + self.time_proj[time_idx](t_emb).unsqueeze(-1)
            time_idx += 1

            # Single ConvNeXt
            x = self.main_blocks[block_idx](x, mask=latent_mask_2)
            block_idx += 1

            # RoPE cross-attention to text + norm
            x = x + self.main_blocks[block_idx](x, context=text_context, mask=latent_mask_2, context_mask=text_mask_2)
            x = self.main_blocks[block_idx + 1](x) * latent_mask_2
            block_idx += 2

            # Single ConvNeXt
            x = self.main_blocks[block_idx](x, mask=latent_mask_2)
            block_idx += 1

            # Style cross-attention + norm (with CFG key_context)
            x = x + self.main_blocks[block_idx](x, context=style_ttl_2, mask=latent_mask_2, key_context=key_context_2)
            x = self.main_blocks[block_idx + 1](x) * latent_mask_2
            block_idx += 2

        # Last convnext
        for conv_block in self.last_convnext.layers:
            x = conv_block(x, mask=latent_mask_2)

        v_pred_2 = self.proj_out(x) * latent_mask_2

        # --- CFG combination ---
        v_cond = v_pred_2[:B]
        v_uncond = v_pred_2[B:]
        v_pred_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)

        return v_pred_cfg
