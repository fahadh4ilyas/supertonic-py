#!/usr/bin/env python3
"""Merge trained AudioEncoder into a SupertonicModel safetensors file.

Usage:
    python scripts/merge_encoder.py --checkpoint checkpoints/best_model.pt
    python scripts/merge_encoder.py --checkpoint checkpoints/best_model.pt --output_dir ./my_model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supertonic.loader import get_cache_dir
from supertonic.model import SupertonicModel


def main():
    parser = argparse.ArgumentParser(description="Merge trained encoder into model safetensors")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, default=get_cache_dir("supertonic-3"))
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Output dir. Defaults to model_dir (overwrites in-place).")
    args = parser.parse_args()

    output_dir = args.output_dir or args.model_dir

    model = SupertonicModel.from_pretrained(str(args.model_dir))
    model.load_encoder(args.checkpoint)
    model.model_dir = output_dir  # save_pretrained uses this
    model.save_pretrained(output_dir)

    print(f"Saved: {output_dir / 'model.safetensors'}")


if __name__ == "__main__":
    main()
