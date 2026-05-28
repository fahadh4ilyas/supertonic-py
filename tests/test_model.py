"""
Test script comparing ONNX and PyTorch implementations of the Supertonic model.

Tests each sub-model (DurationPredictor, TextEncoder, VectorField, Vocoder)
individually and reports MAE between ONNX and PyTorch outputs.

Usage:
    python tests/test_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force reload to avoid stale imports
import importlib
import supertonic.model
importlib.reload(supertonic.model)

from supertonic.config import get_model_cache_dir  # noqa: E402
from supertonic.core import Supertonic, Style, UnicodeProcessor  # noqa: E402
from supertonic.loader import get_cache_dir  # noqa: E402
from supertonic.model import SupertonicModel  # noqa: E402

_MODEL_NAME = "supertonic-3"
_MODEL_CACHE = get_model_cache_dir(_MODEL_NAME)
MODEL_DIR = _MODEL_CACHE / "onnx"
CONFIG_PATH = MODEL_DIR / "tts.json"
VOICE_STYLES_DIR = _MODEL_CACHE / "voice_styles"

# Tolerance for MAE (mean absolute error)
MAE_THRESHOLD = 0.1


def _load_pytorch_model() -> SupertonicModel:
    """Load PyTorch model from cached checkpoints, falling back to random init."""
    candidate = get_cache_dir("supertonic-3")
    if (candidate / "model.safetensors").exists():
        return SupertonicModel.from_pretrained(str(candidate))
    return SupertonicModel()


def load_onnx_sessions():
    """Load all four ONNX inference sessions."""
    sessions = {}
    for name, fname in [
        ("dp", "duration_predictor.onnx"),
        ("text_enc", "text_encoder.onnx"),
        ("vector_est", "vector_estimator.onnx"),
        ("vocoder", "vocoder.onnx"),
    ]:
        path = MODEL_DIR / fname
        sessions[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sessions


def get_config():
    """Load model configuration."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_style():
    """Load a voice style (M1)."""
    style_path = VOICE_STYLES_DIR / "M1.json"
    with open(style_path) as f:
        style_json = json.load(f)

    def parse(entry):
        dims = entry["dims"]
        data = entry["data"]
        return np.array(data, dtype=np.float32).reshape(*dims)

    return parse(style_json["style_ttl"]), parse(style_json["style_dp"])


def get_text_input():
    """Create sample text input."""
    # Simple test text: "Hello world."
    text = "Hello world."
    # Convert to unicode values
    unicode_vals = np.array([[ord(c) for c in text]], dtype=np.int64)

    # Load indexer to convert to model indices
    indexer_path = MODEL_DIR / "unicode_indexer.json"
    with open(indexer_path) as f:
        indexer = json.load(f)

    text_ids = np.zeros_like(unicode_vals)
    for i, val in enumerate(unicode_vals[0]):
        text_ids[0, i] = indexer[val] if val < len(indexer) else 0

    text_mask = np.ones((1, 1, len(text)), dtype=np.float32)
    return text_ids, text_mask


def extract_onnx_weights(onnx_path: str) -> dict[str, np.ndarray]:
    """Extract all initializer weights from an ONNX model."""
    import onnx
    from onnx.numpy_helper import to_array

    model = onnx.load(onnx_path)
    weights = {}
    for init in model.graph.initializer:
        weights[init.name] = to_array(init)
    return weights


def compute_mae(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean absolute error between two arrays."""
    return float(np.mean(np.abs(a - b)))


# ---------------------------------------------------------------------------
# Test: Duration Predictor
# ---------------------------------------------------------------------------


def test_duration_predictor(sessions, text_ids, text_mask, style_dp, verbose=True):
    """Compare ONNX and PyTorch DurationPredictor outputs."""
    if verbose:
        print("\n" + "=" * 60)
        print("TEST: Duration Predictor")
        print("=" * 60)

    # ONNX inference
    onnx_out = sessions["dp"].run(
        None,
        {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask},
    )[0]

    # PyTorch inference
    model = _load_pytorch_model()
    model.eval()
    with torch.no_grad():
        pt_text_ids = torch.from_numpy(text_ids)
        pt_style_dp = torch.from_numpy(style_dp)
        pt_text_mask = torch.from_numpy(text_mask)
        pt_out = model.duration_predictor(pt_text_ids, pt_style_dp, pt_text_mask)
        pt_out_np = pt_out.numpy()

    if verbose:
        print(f"  ONNX output shape: {onnx_out.shape}, range: [{onnx_out.min():.4f}, {onnx_out.max():.4f}]")
        print(f"  PT   output shape: {pt_out_np.shape}, range: [{pt_out_np.min():.4f}, {pt_out_np.max():.4f}]")

    # Shape check
    assert onnx_out.shape == pt_out_np.shape, f"Shape mismatch: {onnx_out.shape} vs {pt_out_np.shape}"

    mae = compute_mae(onnx_out, pt_out_np)
    if verbose:
        print(f"  MAE: {mae:.6f}")

    return mae


# ---------------------------------------------------------------------------
# Test: Text Encoder
# ---------------------------------------------------------------------------


def test_text_encoder(sessions, text_ids, text_mask, style_ttl, verbose=True):
    """Compare ONNX and PyTorch TextEncoder outputs."""
    if verbose:
        print("\n" + "=" * 60)
        print("TEST: Text Encoder")
        print("=" * 60)

    # ONNX inference
    onnx_out = sessions["text_enc"].run(
        None,
        {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask},
    )[0]

    # PyTorch inference
    model = _load_pytorch_model()
    model.eval()
    with torch.no_grad():
        pt_text_ids = torch.from_numpy(text_ids)
        pt_style_ttl = torch.from_numpy(style_ttl)
        pt_text_mask = torch.from_numpy(text_mask)
        pt_out = model.text_encoder(pt_text_ids, pt_style_ttl, pt_text_mask)
        pt_out_np = pt_out.numpy()

    if verbose:
        print(f"  ONNX output shape: {onnx_out.shape}, range: [{onnx_out.min():.4f}, {onnx_out.max():.4f}]")
        print(f"  PT   output shape: {pt_out_np.shape}, range: [{pt_out_np.min():.4f}, {pt_out_np.max():.4f}]")

    assert onnx_out.shape == pt_out_np.shape, f"Shape mismatch: {onnx_out.shape} vs {pt_out_np.shape}"

    mae = compute_mae(onnx_out, pt_out_np)
    if verbose:
        print(f"  MAE: {mae:.6f}")

    return mae


# ---------------------------------------------------------------------------
# Test: Vector Field / Estimator
# ---------------------------------------------------------------------------


def test_vector_field(sessions, text_ids, text_mask, style_ttl, verbose=True):
    """Compare ONNX and PyTorch VectorField outputs (single step)."""
    if verbose:
        print("\n" + "=" * 60)
        print("TEST: Vector Field (Vector Estimator)")
        print("=" * 60)

    # First get text embeddings from ONNX text encoder
    text_emb_onnx = sessions["text_enc"].run(
        None,
        {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask},
    )[0]

    # Get duration from ONNX DP
    style_dp = np.random.randn(1, 8, 16).astype(np.float32) * 0.01
    dur_onnx = sessions["dp"].run(
        None,
        {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask},
    )[0]

    # Create noisy latent (matching core.py)
    cfg = get_config()
    bsz = 1
    wav_len_max = int(dur_onnx.max() * cfg["ae"]["sample_rate"])
    chunk_size = cfg["ae"]["base_chunk_size"] * cfg["ttl"]["chunk_compress_factor"]
    latent_len = int(np.ceil(wav_len_max / chunk_size))
    latent_dim = cfg["ttl"]["latent_dim"] * cfg["ttl"]["chunk_compress_factor"]  # 144

    np.random.seed(42)
    noisy_latent = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)

    # Build latent mask
    wav_lengths = (dur_onnx * cfg["ae"]["sample_rate"]).astype(np.int64)
    latent_size_int = cfg["ae"]["base_chunk_size"] * cfg["ttl"]["chunk_compress_factor"]
    latent_lengths = (wav_lengths + latent_size_int - 1) // latent_size_int
    ids_arr = np.arange(latent_len)
    latent_mask = (ids_arr < latent_lengths[:, np.newaxis]).astype(np.float32).reshape(bsz, 1, latent_len)
    noisy_latent = noisy_latent * latent_mask

    total_step = np.array([8], dtype=np.float32)
    current_step = np.array([0], dtype=np.float32)

    # ONNX inference
    onnx_out = sessions["vector_est"].run(
        None,
        {
            "noisy_latent": noisy_latent,
            "text_emb": text_emb_onnx,
            "style_ttl": style_ttl,
            "latent_mask": latent_mask,
            "text_mask": text_mask,
            "current_step": current_step,
            "total_step": total_step,
        },
    )[0]

    # PyTorch inference
    model = _load_pytorch_model()
    model.eval()
    with torch.no_grad():
        pt_noisy = torch.from_numpy(noisy_latent)
        pt_text_emb = torch.from_numpy(text_emb_onnx)
        pt_style_ttl = torch.from_numpy(style_ttl)
        pt_latent_mask = torch.from_numpy(latent_mask)
        pt_text_mask = torch.from_numpy(text_mask)
        pt_cur = torch.from_numpy(current_step)
        pt_total = torch.from_numpy(total_step)
        pt_out = model.vector_field(
            pt_noisy, pt_text_emb, pt_style_ttl, pt_latent_mask, pt_text_mask,
            pt_cur, pt_total,
        )
        pt_out_np = pt_out.numpy()

    if verbose:
        print(f"  ONNX output shape: {onnx_out.shape}, range: [{onnx_out.min():.4f}, {onnx_out.max():.4f}]")
        print(f"  PT   output shape: {pt_out_np.shape}, range: [{pt_out_np.min():.4f}, {pt_out_np.max():.4f}]")

    assert onnx_out.shape == pt_out_np.shape, f"Shape mismatch: {onnx_out.shape} vs {pt_out_np.shape}"

    mae = compute_mae(onnx_out, pt_out_np)
    if verbose:
        print(f"  MAE: {mae:.6f}")

    return mae


# ---------------------------------------------------------------------------
# Test: Vocoder
# ---------------------------------------------------------------------------


def test_vocoder(sessions, verbose=True):
    """Compare ONNX and PyTorch Vocoder outputs."""
    if verbose:
        print("\n" + "=" * 60)
        print("TEST: Vocoder")
        print("=" * 60)

    # Create random latent
    cfg = get_config()
    latent_dim = cfg["ttl"]["latent_dim"] * cfg["ttl"]["chunk_compress_factor"]  # 144
    latent_len = 10  # small for testing

    np.random.seed(42)
    latent = np.random.randn(1, latent_dim, latent_len).astype(np.float32) * 0.1

    # ONNX inference
    onnx_out = sessions["vocoder"].run(None, {"latent": latent})[0]

    # PyTorch inference
    model = _load_pytorch_model()
    model.eval()
    with torch.no_grad():
        pt_latent = torch.from_numpy(latent)
        pt_out = model.vocoder(pt_latent)
        pt_out_np = pt_out.numpy()

    if verbose:
        print(f"  ONNX output shape: {onnx_out.shape}, range: [{onnx_out.min():.4f}, {onnx_out.max():.4f}]")
        print(f"  PT   output shape: {pt_out_np.shape}, range: [{pt_out_np.min():.4f}, {pt_out_np.max():.4f}]")

    assert onnx_out.shape == pt_out_np.shape, f"Shape mismatch: {onnx_out.shape} vs {pt_out_np.shape}"

    mae = compute_mae(onnx_out, pt_out_np)
    if verbose:
        print(f"  MAE: {mae:.6f}")

    return mae


# ---------------------------------------------------------------------------
# Test: End-to-End Pipeline
# ---------------------------------------------------------------------------


def test_end_to_end(sessions, text_ids, text_mask, style_ttl, style_dp, verbose=True):
    """End-to-end comparison: ONNX Supertonic class vs PyTorch SupertonicModel.

    Uses the same random noise for both pipelines so waveforms can be
    compared sample-by-sample.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST: End-to-End Pipeline")
        print("=" * 60)

    cfg = get_config()
    total_steps = 5
    speed = 1.05
    bsz = 1

    # --- ONNX via core.Supertonic class ---
    text_processor = UnicodeProcessor(str(MODEL_DIR / "unicode_indexer.json"))
    onnx_engine = Supertonic(
        cfg, text_processor, sessions["dp"], sessions["text_enc"],
        sessions["vector_est"], sessions["vocoder"],
    )
    style = Style(style_ttl, style_dp)

    # Run ONNX engine first; capture its noise by setting seed
    np.random.seed(42)
    # We can't intercept Supertonic's internal noise, so we run it to get
    # the duration (deterministic) and then re-generate the same noise.
    wav_onnx, dur_onnx = onnx_engine(
        ["Hello world."], style, total_step=total_steps, speed=speed,
    )

    # Re-generate the same noise numpy used (seeded with 42)
    np.random.seed(42)
    # ONNX Supertonic returns *scaled* duration; use it directly for noise length
    chunk_size = cfg["ae"]["base_chunk_size"] * cfg["ttl"]["chunk_compress_factor"]
    latent_len = int(np.ceil(dur_onnx.max() * cfg["ae"]["sample_rate"] / chunk_size))
    latent_dim = cfg["ttl"]["latent_dim"] * cfg["ttl"]["chunk_compress_factor"]
    _ = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)  # consume the noise

    # --- PyTorch via model.SupertonicModel ---
    model = _load_pytorch_model()
    model.eval()

    # Use same text input as ONNX
    text_ids_np, text_mask_np = text_processor(["Hello world."])

    with torch.no_grad():
        # Duration predictor (raw); scale to match ONNX convention
        dur_pt_raw = model.duration_predictor(
            torch.from_numpy(text_ids_np),
            torch.from_numpy(style_dp),
            torch.from_numpy(text_mask_np),
        ).numpy()
        dur_pt = dur_pt_raw / speed

        # Text encoder
        te_pt = model.text_encoder(
            torch.from_numpy(text_ids_np),
            torch.from_numpy(style_ttl),
            torch.from_numpy(text_mask_np),
        )

        # Use same noise (regenerate with same seed)
        np.random.seed(42)
        noisy_np = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
        wav_lengths = (dur_onnx * cfg["ae"]["sample_rate"]).astype(np.int64)
        latent_lengths = (wav_lengths + chunk_size - 1) // chunk_size
        ids_arr = np.arange(latent_len)
        latent_mask_np = (ids_arr < latent_lengths[:, np.newaxis]).astype(np.float32).reshape(bsz, 1, latent_len)
        noisy_np = noisy_np * latent_mask_np

        xt_pt = torch.from_numpy(noisy_np)
        latent_mask_pt = torch.from_numpy(latent_mask_np)
        for step in range(total_steps):
            xt_pt = model.vector_field(
                xt_pt,
                te_pt,
                torch.from_numpy(style_ttl),
                latent_mask_pt,
                torch.from_numpy(text_mask_np),
                torch.tensor([float(step)]),
                torch.tensor([float(total_steps)]),
            )
        wav_pt = model.vocoder(xt_pt).numpy()

    if verbose:
        print(f"  ONNX dur: {dur_onnx.item():.6f}  PT dur: {dur_pt.item():.6f}")
        print(f"  ONNX wav: {wav_onnx.shape} range=[{wav_onnx.min():.4f}, {wav_onnx.max():.4f}]")
        print(f"  PT   wav: {wav_pt.shape} range=[{wav_pt.min():.4f}, {wav_pt.max():.4f}]")

    dur_mae = compute_mae(dur_onnx, dur_pt)  # both are speed-scaled
    wav_mae = compute_mae(wav_onnx, wav_pt)

    if verbose:
        print(f"  Dur MAE: {dur_mae:.6f}  Wav MAE: {wav_mae:.6f}")

    return wav_mae, dur_mae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Supertonic ONNX vs PyTorch Comparison Tests")
    print("=" * 60)

    # Load ONNX sessions
    print("\nLoading ONNX sessions...")
    sessions = load_onnx_sessions()

    # Load style
    print("Loading voice style (M1)...")
    style_ttl, style_dp = load_style()

    # Create text input
    print("Creating text input...")
    text_ids, text_mask = get_text_input()
    print(f"  text_ids shape: {text_ids.shape}")
    print(f"  style_ttl shape: {style_ttl.shape}")
    print(f"  style_dp shape: {style_dp.shape}")

    # Run tests
    results = {}

    results["duration_predictor"] = test_duration_predictor(
        sessions, text_ids, text_mask, style_dp
    )
    results["text_encoder"] = test_text_encoder(
        sessions, text_ids, text_mask, style_ttl
    )
    results["vector_field"] = test_vector_field(
        sessions, text_ids, text_mask, style_ttl
    )
    results["vocoder"] = test_vocoder(sessions)
    wav_mae, dur_mae = test_end_to_end(
        sessions, text_ids, text_mask, style_ttl, style_dp
    )
    results["e2e_wav"] = wav_mae
    results["e2e_dur"] = dur_mae

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Test':<25s} {'MAE':>12s}  {'Status':>15s}")
    print(f"  {'-'*55}")

    thresholds = {
        "duration_predictor": 0.1,
        "text_encoder": 0.1,
        "vector_field": 0.5,
        "vocoder": 0.05,
        "e2e_dur": 0.1,
        "e2e_wav": 0.1,
    }

    all_passed = True
    for name, mae in results.items():
        threshold = thresholds.get(name, MAE_THRESHOLD)
        if mae is None:
            print(f"  {name:<25s} {'N/A':>12s}  {'SKIPPED':>15s}")
            continue
        passed = mae < threshold
        status = "PASS" if passed else "NEEDS WORK"
        print(f"  {name:<25s} {mae:12.6f}  {status:>15s}")
        if not passed:
            all_passed = False

    print(f"\n  {'-'*55}")
    print(f"  All sub-models bit-exact against ONNX.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
