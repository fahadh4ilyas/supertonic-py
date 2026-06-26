"""
Export PyTorch SupertonicModel sub-models to ONNX.

Usage:
    python scripts/export_onnx.py --input-dir INPUT_DIR [--output-dir ./exported_onnx]

Matches the input/output names of the original supertonic-3 ONNX models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supertonic.model import SupertonicModel


def export_sub_model(model, dummy_inputs, input_names, output_names, path, dynamic_axes=None):
    """Export a single sub-model to ONNX."""
    torch.onnx.export(
        model,
        dummy_inputs,
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=19,
    )
    print(f"  Exported: {path}")


def main(output_dir: str = "exported_onnx", input_dir: str | None = None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading PyTorch model...")
    if input_dir:
        pt_model = SupertonicModel.from_pretrained(input_dir)
        print(f"  Loaded from: {input_dir}")
    else:
        pt_model = SupertonicModel()
        ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "model.safetensors"
        if ckpt_path.exists():
            pt_model = SupertonicModel.from_pretrained(str(ckpt_path.parent))
            print(f"  Loaded from: {ckpt_path.parent}")
        else:
            print("  No checkpoint found, using random weights")
    pt_model.eval()

    # ------------------------------------------------------------------
    # 1. DurationPredictor
    # ------------------------------------------------------------------
    print("\n[1/4] DurationPredictor...")
    dp = pt_model.duration_predictor
    text_ids = torch.randint(0, 8322, (1, 10), dtype=torch.long)
    style_dp = torch.randn(1, 8, 16)
    text_mask = torch.ones(1, 1, 10)

    export_sub_model(
        dp,
        (text_ids, style_dp, text_mask),
        input_names=["text_ids", "style_dp", "text_mask"],
        output_names=["duration"],
        path=output_path / "duration_predictor.onnx",
        dynamic_axes={
            "text_ids": {0: "batch", 1: "seq_len"},
            "style_dp": {0: "batch"},
            "text_mask": {0: "batch", 2: "seq_len"},
            "duration": {0: "batch"},
        },
    )

    # ------------------------------------------------------------------
    # 2. TextEncoder
    # ------------------------------------------------------------------
    print("\n[2/4] TextEncoder...")
    te = pt_model.text_encoder
    style_ttl = torch.randn(1, 50, 256)

    export_sub_model(
        te,
        (text_ids, style_ttl, text_mask),
        input_names=["text_ids", "style_ttl", "text_mask"],
        output_names=["text_emb"],
        path=output_path / "text_encoder.onnx",
        dynamic_axes={
            "text_ids": {0: "batch", 1: "seq_len"},
            "style_ttl": {0: "batch"},
            "text_mask": {0: "batch", 2: "seq_len"},
            "text_emb": {0: "batch", 2: "seq_len"},
        },
    )

    # ------------------------------------------------------------------
    # 3. VectorField
    # ------------------------------------------------------------------
    print("\n[3/4] VectorField...")
    vf = pt_model.vector_field
    noisy_latent = torch.randn(1, 144, 20)
    latent_mask = torch.ones(1, 1, 20)
    current_step = torch.tensor([0.0])
    total_step = torch.tensor([5.0])

    # Text encoder output needed as input
    te_out = te(text_ids, style_ttl, text_mask)

    export_sub_model(
        vf,
        (noisy_latent, te_out, style_ttl, latent_mask, text_mask, current_step, total_step),
        input_names=["noisy_latent", "text_emb", "style_ttl", "latent_mask", "text_mask",
                      "current_step", "total_step"],
        output_names=["denoised_latent"],
        path=output_path / "vector_estimator.onnx",
        dynamic_axes={
            "noisy_latent": {0: "batch", 2: "latent_len"},
            "text_emb": {0: "batch", 2: "seq_len"},
            "style_ttl": {0: "batch"},
            "latent_mask": {0: "batch", 2: "latent_len"},
            "text_mask": {0: "batch", 2: "seq_len"},
            "current_step": {0: "batch"},
            "total_step": {0: "batch"},
            "denoised_latent": {0: "batch", 2: "latent_len"},
        },
    )

    # ------------------------------------------------------------------
    # 4. Vocoder
    # ------------------------------------------------------------------
    print("\n[4/4] Vocoder...")
    vo = pt_model.vocoder
    latent = torch.randn(1, 144, 20)

    export_sub_model(
        vo,
        (latent,),
        input_names=["latent"],
        output_names=["waveform"],
        path=output_path / "vocoder.onnx",
        dynamic_axes={
            "latent": {0: "batch", 2: "latent_len"},
            "waveform": {0: "batch", 1: "num_samples"},
        },
    )

    print("\nDone. Exported to:", output_path.resolve())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--input-dir", required=True, help="Directory with tts.json + model.safetensors")
    parser.add_argument("--output-dir", default="exported_onnx", help="Output directory for ONNX files")
    args = parser.parse_args()
    main(output_dir=args.output_dir, input_dir=args.input_dir)
