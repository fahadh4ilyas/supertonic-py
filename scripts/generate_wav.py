"""
Generate speech using PyTorch SupertonicModel and ONNX pipeline.TTS side-by-side.

Usage:
    python scripts/generate_wav.py "Hello world!" --voice M1
    → saves output_pytorch.wav and output_onnx.wav

    python scripts/generate_wav.py "Hello world!" --voice M1 --pytorch-only
    python scripts/generate_wav.py "Hello world!" --voice M1 --onnx-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supertonic import TTS
from supertonic.loader import get_cache_dir
from supertonic.model import SupertonicModel


def main():
    parser = argparse.ArgumentParser(description="Generate speech with PyTorch vs ONNX Supertonic")
    parser.add_argument("text", type=str, help="Text to synthesize")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Model directory. Defaults to supertonic-3 cache.")
    parser.add_argument("--voice", type=str, default="M1", help="Voice style name (M1-5, F1-5)")
    parser.add_argument("--output", type=str, default="output", help="Output filename prefix")
    parser.add_argument("--steps", type=int, default=8, help="Diffusion steps (4-32)")
    parser.add_argument("--speed", type=float, default=1.05, help="Speed multiplier")
    parser.add_argument("--lang", type=str, default="na", help="Language code")
    parser.add_argument("--pytorch-only", action="store_true")
    parser.add_argument("--onnx-only", action="store_true")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device for PyTorch model (e.g. 'cuda', 'cpu'). Default: CPU.")
    args = parser.parse_args()

    input_dir: Path = args.input_dir if args.input_dir is not None else get_cache_dir("supertonic-3")

    print(f"Model directory: {input_dir}")

    # ---- PyTorch ----
    if not args.onnx_only:
        print("\n" + "=" * 55)
        print("PyTorch Pipeline (SupertonicModel.synthesize)")
        print("=" * 55)

        model = SupertonicModel.from_pretrained(str(input_dir), device=args.device)
        model.eval()
        pt_style = model.get_voice_style(args.voice)

        t0 = time.perf_counter()
        wav_pt, dur_pt = model.synthesize(
            text=args.text,
            voice_style=pt_style,
            total_steps=args.steps,
            speed=args.speed,
        )
        t_pt = time.perf_counter() - t0

        pt_path = Path(f"{args.output}_pytorch.wav")
        sf.write(str(pt_path), wav_pt.squeeze(), model.sample_rate)
        print(f"  Duration : {dur_pt[0]:.2f}s  |  samples: {wav_pt.shape[1]:,}")
        print(f"  Range   : [{wav_pt.min():.4f}, {wav_pt.max():.4f}]")
        print(f"  Time    : {t_pt:.1f}s")
        print(f"  Saved   : {pt_path}")

    # ---- ONNX ----
    if not args.pytorch_only:
        print("\n" + "=" * 55)
        print("ONNX Pipeline (pipeline.TTS.synthesize)")
        print("=" * 55)

        tts = TTS(model="supertonic-3", model_dir=input_dir, auto_download=False)
        onnx_style = tts.get_voice_style(args.voice)

        t0 = time.perf_counter()
        wav_onnx, dur_onnx = tts.synthesize(
            text=args.text,
            voice_style=onnx_style,
            total_steps=args.steps,
            speed=args.speed,
            lang=args.lang,
        )
        t_onnx = time.perf_counter() - t0

        onnx_path = Path(f"{args.output}_onnx.wav")
        sf.write(str(onnx_path), wav_onnx.squeeze(), tts.sample_rate)
        print(f"  Duration : {dur_onnx[0]:.2f}s  |  samples: {wav_onnx.shape[1]:,}")
        print(f"  Range   : [{wav_onnx.min():.4f}, {wav_onnx.max():.4f}]")
        print(f"  Time    : {t_onnx:.1f}s")
        print(f"  Saved   : {onnx_path}")

    # ---- Comparison ----
    if not args.pytorch_only and not args.onnx_only:
        import numpy as np

        print("\n" + "=" * 55)
        print("Comparison")
        print("=" * 55)
        print(f"  {'':<18s} {'PyTorch':>12s}  {'ONNX':>12s}  {'Δ':>10s}")
        print(f"  {'Duration':<18s} {dur_pt[0]:12.4f}  {dur_onnx[0]:12.4f}  {abs(dur_pt[0] - dur_onnx[0]):10.4f}s")
        print(f"  {'Samples':<18s} {wav_pt.shape[1]:12,}  {wav_onnx.shape[1]:12,}  {abs(wav_pt.shape[1] - wav_onnx.shape[1]):10,}")
        print(f"  {'Time':<18s} {t_pt:11.1f}s  {t_onnx:11.1f}s")
        if wav_pt.shape[1] == wav_onnx.shape[1]:
            mae = float(np.mean(np.abs(wav_pt - wav_onnx)))
            print(f"  {'Wav MAE':<18s} {'':>12s}  {'':>12s}  {mae:10.6f}")
        else:
            min_len = min(wav_pt.shape[1], wav_onnx.shape[1])
            mae = float(np.mean(np.abs(wav_pt[0, :min_len] - wav_onnx[0, :min_len])))
            print(f"  {'Wav MAE (first':<18s} {min_len} samples)  {'':>12s}  {mae:10.6f}")


if __name__ == "__main__":
    main()
