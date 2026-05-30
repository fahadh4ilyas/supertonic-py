"""Audio Encoder for voice style extraction.

Converts a reference audio waveform into style_ttl (1, 50, 256) and
style_dp (1, 8, 16) vectors that can be fed into the frozen SupertonicModel.

Architecture mirrors the original training config (tts.json style_encoder
sections) but replaces the missing AE encoder (mel→latent) with a
mel-spectrogram + shared ConvNeXt backbone.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvNeXtStack, SymmetricPad1d, LayerNorm
from .attention import StyleTokenLayer


class MelSpectrogram(nn.Module):
    """Mel-spectrogram extraction matching the AE encoder config.

    Config from tts.json ae.encoder.spec_processor:
        n_fft=2048, win_length=2048, hop_length=512, n_mels=228,
        sample_rate=44100, norm_mean=0.0, norm_std=1.0
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        n_mels: int = 228,
        f_min: float = 0.0,
        f_max: float | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        # STFT
        self.register_buffer(
            "window",
            torch.hann_window(win_length),
        )

        # Mel filter bank
        f_max = f_max or sample_rate / 2
        mel_fb = self._mel_filterbank(n_mels, n_fft, sample_rate, f_min, f_max)
        self.register_buffer("mel_filterbank", mel_fb)

    @staticmethod
    def _mel_filterbank(
        n_mels: int, n_fft: int, sample_rate: int, f_min: float, f_max: float
    ) -> torch.Tensor:
        """Build mel filterbank matrix."""
        n_freqs = n_fft // 2 + 1

        # Convert Hz to mel
        def hz_to_mel(hz):
            return 2595.0 * torch.log10(1.0 + hz / 700.0)

        mel_min = hz_to_mel(torch.tensor(f_min))
        mel_max = hz_to_mel(torch.tensor(f_max))
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        bin_indices = torch.floor((n_fft + 1) * hz_points / sample_rate).long()
        bin_indices = bin_indices.clamp(0, n_freqs - 1)

        filterbank = torch.zeros(n_mels, n_freqs)
        for i in range(n_mels):
            start, center, end = bin_indices[i], bin_indices[i + 1], bin_indices[i + 2]
            for j in range(start, center):
                filterbank[i, j] = (j - start) / max(center - start, 1)
            for j in range(center, end):
                filterbank[i, j] = (end - j) / max(end - center, 1)

        return filterbank

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel spectrogram.

        Args:
            waveform: (B, 1, T) or (B, T) mono audio at self.sample_rate.

        Returns:
            Mel spectrogram of shape (B, n_mels, T_mel).
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        # STFT
        stft = torch.stft(
            waveform.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            return_complex=True,
        )

        # Power spectrogram
        power_spec = stft.abs() ** 2  # (B, n_fft//2+1, T_mel)

        # Apply mel filterbank
        mel_spec = torch.matmul(
            self.mel_filterbank.to(power_spec.device), power_spec
        )  # (B, n_mels, T_mel)

        # Log scale with epsilon
        mel_spec = torch.log(mel_spec.clamp(min=1e-5))

        return mel_spec


class AudioEncoder(nn.Module):
    """Voice style encoder: audio → style_ttl + style_dp.

    Takes a reference audio waveform (15-30 seconds recommended) and
    produces style vectors compatible with the frozen SupertonicModel.

    Architecture:
        waveform
          → MelSpectrogram (228 mel bands)
          → Conv1d(228 → 512, ksz=7) + LayerNorm
          → ConvNeXt backbone (4 layers, 512→1024 intermediate)
          ──→ TTL head: Conv1d(512→256) → ConvNeXt(6 layers, 256→1024)
          │                 → StyleTokenLayer(50 tokens, 256-dim)
          │                 → style_ttl (1, 50, 256)
          ──→ DP head:  Conv1d(512→64)  → ConvNeXt(4 layers, 64→256)
                            → StyleTokenLayer(8 tokens, 16-dim)
                            → style_dp (1, 8, 16)

    Args:
        config: Optional dict matching tts.json structure.
        sample_rate: Audio sample rate (default 44100).
    """

    def __init__(
        self,
        config: dict | None = None,
        sample_rate: int = 44100,
    ):
        super().__init__()
        if config is None:
            config = {}

        ae = config.get("ae", {})
        ttl_cfg = config.get("ttl", {})
        dp_cfg = config.get("dp", {})

        spec_cfg = ae.get("encoder", {}).get("spec_processor", {})

        self.sample_rate = spec_cfg.get("sample_rate", sample_rate)

        # Mel spectrogram
        self.mel = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=spec_cfg.get("n_fft", 2048),
            win_length=spec_cfg.get("win_length", 2048),
            hop_length=spec_cfg.get("hop_length", 512),
            n_mels=spec_cfg.get("n_mels", 228),
        )

        # Shared backbone: mel (228) → 512 dim
        backbone_ksz = 7
        self.proj_in_pad = SymmetricPad1d(backbone_ksz)
        self.proj_in = nn.Conv1d(228, 512, kernel_size=backbone_ksz, bias=False)
        self.proj_in_norm = LayerNorm(512)

        self.backbone = ConvNeXtStack(
            channels=512,
            intermediate=1024,
            kernel_size=backbone_ksz,
            num_layers=4,
            dilations=[1, 1, 1, 1],
        )

        # ── TTL head (matches ttl.style_encoder from tts.json) ──
        ttl_se = ttl_cfg.get("style_encoder", {})
        self.ttl_proj = nn.Conv1d(512, 256, 1, bias=False)

        ttl_cnv_cfg = ttl_se.get("convnext", {})
        self.ttl_convnext = ConvNeXtStack(
            channels=256,
            intermediate=ttl_cnv_cfg.get("intermediate_dim", 1024),
            kernel_size=ttl_cnv_cfg.get("ksz", 5),
            num_layers=ttl_cnv_cfg.get("num_layers", 6),
            dilations=ttl_cnv_cfg.get("dilation_lst", [1, 1, 1, 1, 1, 1]),
        )

        ttl_st = ttl_se.get("style_token_layer", {})
        self.ttl_style_tokens = StyleTokenLayer(
            input_dim=256,
            n_style=ttl_st.get("n_style", 50),
            style_value_dim=ttl_st.get("style_value_dim", 256),
            n_heads=ttl_st.get("n_heads", 2),
            n_units=ttl_st.get("n_units", 256),
        )

        # ── DP head (matches dp.style_encoder from tts.json) ──
        dp_se = dp_cfg.get("style_encoder", {})
        self.dp_proj = nn.Conv1d(512, 64, 1, bias=False)

        dp_cnv_cfg = dp_se.get("convnext", {})
        self.dp_convnext = ConvNeXtStack(
            channels=64,
            intermediate=dp_cnv_cfg.get("intermediate_dim", 256),
            kernel_size=dp_cnv_cfg.get("ksz", 5),
            num_layers=dp_cnv_cfg.get("num_layers", 4),
            dilations=dp_cnv_cfg.get("dilation_lst", [1, 1, 1, 1]),
        )

        dp_st = dp_se.get("style_token_layer", {})
        self.dp_style_tokens = StyleTokenLayer(
            input_dim=64,
            n_style=dp_st.get("n_style", 8),
            style_value_dim=dp_st.get("style_value_dim", 16),
            n_heads=dp_st.get("n_heads", 2),
            n_units=dp_st.get("n_units", 64),
        )

    def forward(
        self,
        waveform: torch.Tensor,
        return_mel: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract style vectors from a reference audio waveform.

        Args:
            waveform: (B, 1, T) or (B, T) mono audio at self.sample_rate.
            return_mel: If True, also return the mel spectrogram.

        Returns:
            Tuple of (style_ttl, style_dp) or (style_ttl, style_dp, mel_spec)
            if return_mel is True.
            - style_ttl: (B, 50, 256)
            - style_dp: (B, 8, 16)
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        B = waveform.shape[0]

        # Mel spectrogram: (B, 228, T_mel)
        mel_spec = self.mel(waveform)

        # Build mask (all ones for now — mel has uniform frames)
        mel_mask = torch.ones(B, 1, mel_spec.shape[-1], device=mel_spec.device)

        # Shared backbone
        x = self.proj_in_pad(mel_spec)
        x = self.proj_in(x)
        x = self.proj_in_norm(x) * mel_mask

        for block in self.backbone.layers:
            x = block(x, mask=mel_mask)

        # ── TTL branch ──
        ttl = self.ttl_proj(x) * mel_mask
        for block in self.ttl_convnext.layers:
            ttl = block(ttl, mask=mel_mask)
        style_ttl = self.ttl_style_tokens(ttl, mask=mel_mask)  # (B, 50, 256)

        # ── DP branch ──
        dp = self.dp_proj(x) * mel_mask
        for block in self.dp_convnext.layers:
            dp = block(dp, mask=mel_mask)
        style_dp = self.dp_style_tokens(dp, mask=mel_mask)  # (B, 8, 16)

        if return_mel:
            return style_ttl, style_dp, mel_spec
        return style_ttl, style_dp

    @torch.no_grad()
    def encode_from_path(
        self,
        audio_path: str,
        target_sample_rate: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract style vectors from an audio file path.

        Requires torchaudio for file loading.

        Args:
            audio_path: Path to audio file.
            target_sample_rate: If set, resample audio to this rate.
                                Defaults to self.sample_rate.

        Returns:
            Tuple of (style_ttl, style_dp), each (1, N, D).
        """
        import torchaudio

        sr = target_sample_rate or self.sample_rate
        waveform, orig_sr = torchaudio.load(audio_path)
        if orig_sr != sr:
            resampler = torchaudio.transforms.Resample(orig_sr, sr)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Add batch dim if needed
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        device = next(self.parameters()).device
        waveform = waveform.to(device)

        return self.forward(waveform)
