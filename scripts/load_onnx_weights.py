"""
Extract ONNX weights and load them into the PyTorch SupertonicModel.

Usage:
    python scripts/load_onnx_weights.py [--input-dir <onnx_dir>] [--output-dir <output_dir>]

Both arguments default to the supertonic-3 cache directory (~/.cache/supertonic3).
If --input-dir is not set and the model is not cached, it is auto-downloaded.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supertonic.loader import download_model, get_cache_dir, has_all_onnx_modules  # noqa: E402
from supertonic.model import SupertonicModel  # noqa: E402


def extract_onnx_weights(onnx_path: str) -> dict[str, np.ndarray]:
    import onnx
    from onnx.numpy_helper import to_array
    model = onnx.load(onnx_path)
    return {init.name: to_array(init) for init in model.graph.initializer}


def load_all_onnx_weights(input_dir: Path) -> dict[str, np.ndarray]:
    all_weights = {}
    for fname in ["duration_predictor.onnx", "text_encoder.onnx", "vector_estimator.onnx", "vocoder.onnx"]:
        for k, v in extract_onnx_weights(str(input_dir / fname)).items():
            all_weights[f"{fname}/{k}"] = v
    return all_weights


def map_onnx_to_torch(onnx_name: str) -> Optional[str]:
    name = onnx_name.split("/", 1)[1] if "/" in onnx_name else onnx_name

    # ---- Duration Predictor ----
    if name == "tts.dp.sentence_encoder.sentence_token":
        return "duration_predictor.sentence_token"
    if name == "tts.dp.sentence_encoder.text_embedder.char_embedder.weight":
        return "duration_predictor.embedding.weight"

    m = re.match(r"tts\.dp\.sentence_encoder\.convnext\.convnext\.(\d+)\.(.+)", name)
    if m:
        return f"duration_predictor.convnext.layers.{m.group(1)}.{m.group(2)}"

    m = re.match(r"tts\.dp\.sentence_encoder\.attn_encoder\.attn_layers\.(\d+)\.(.+)", name)
    if m:
        idx, rest = m.group(1), m.group(2)
        # emb_rel_k / emb_rel_v live in separate ParameterLists in model.py
        if rest in ("emb_rel_k", "emb_rel_v"):
            return f"duration_predictor.attn_{rest}.{idx}"
        return f"duration_predictor.attn_layers.{idx}.{rest}"

    m = re.match(r"tts\.dp\.sentence_encoder\.attn_encoder\.norm_layers_1\.(\d+)\.norm\.(.+)", name)
    if m:
        return f"duration_predictor.attn_norms1.{m.group(1)}.norm.{m.group(2)}"

    m = re.match(r"tts\.dp\.sentence_encoder\.attn_encoder\.norm_layers_2\.(\d+)\.norm\.(.+)", name)
    if m:
        return f"duration_predictor.attn_norms2.{m.group(1)}.norm.{m.group(2)}"

    m = re.match(r"tts\.dp\.sentence_encoder\.attn_encoder\.ffn_layers\.(\d+)\.(.+)", name)
    if m:
        return f"duration_predictor.attn_ffn.{m.group(1)}.{m.group(2)}"

    if name == "tts.dp.sentence_encoder.proj_out.net.weight":
        return "duration_predictor.proj_out.weight"

    if name == "tts.dp.predictor.layers.0.weight":
        return "duration_predictor.mlp.0.weight"
    if name == "tts.dp.predictor.layers.0.bias":
        return "duration_predictor.mlp.0.bias"
    if name == "tts.dp.predictor.layers.1.weight":
        return "duration_predictor.mlp.2.weight"
    if name == "tts.dp.predictor.layers.1.bias":
        return "duration_predictor.mlp.2.bias"
    if name == "tts.dp.predictor.activation.weight":
        return "duration_predictor.mlp.1.weight"

    # ---- Text Encoder ----
    if name == "tts.ttl.text_encoder.text_embedder.char_embedder.weight":
        return "text_encoder.text_embedder.weight"
    if name == "tts.ttl.style_encoder.style_token_layer.style_key":
        return "text_encoder.style_key"

    m = re.match(r"tts\.ttl\.text_encoder\.convnext\.convnext\.(\d+)\.(.+)", name)
    if m:
        return f"text_encoder.convnext.layers.{m.group(1)}.{m.group(2)}"

    m = re.match(r"tts\.ttl\.text_encoder\.attn_encoder\.attn_layers\.(\d+)\.(.+)", name)
    if m:
        return f"text_encoder.attn_layers.{m.group(1)}.{m.group(2)}"

    m = re.match(r"tts\.ttl\.text_encoder\.attn_encoder\.norm_layers_1\.(\d+)\.norm\.(.+)", name)
    if m:
        return f"text_encoder.attn_norms1.{m.group(1)}.norm.{m.group(2)}"

    m = re.match(r"tts\.ttl\.text_encoder\.attn_encoder\.norm_layers_2\.(\d+)\.norm\.(.+)", name)
    if m:
        return f"text_encoder.attn_norms2.{m.group(1)}.norm.{m.group(2)}"

    m = re.match(r"tts\.ttl\.text_encoder\.attn_encoder\.ffn_layers\.(\d+)\.(.+)", name)
    if m:
        return f"text_encoder.attn_ffn.{m.group(1)}.{m.group(2)}"

    m = re.match(r"tts\.ttl\.speech_prompted_text_encoder\.attention1\.(.+)", name)
    if m:
        rest = m.group(1)
        # Strip .linear from ONNX bias naming (e.g. W_key.linear.bias → W_key.bias)
        rest = rest.replace(".linear.", ".")
        if "W_key" in rest:
            return f"text_encoder.speech_prompted_attn.0.{rest}"
        return f"text_encoder.speech_prompted_attn.0.{rest}"

    m = re.match(r"tts\.ttl\.speech_prompted_text_encoder\.attention2\.(.+)", name)
    if m:
        rest = m.group(1).replace(".linear.", ".")
        return f"text_encoder.speech_prompted_attn.1.{rest}"

    m = re.match(r"tts\.ttl\.speech_prompted_text_encoder\.norm\.norm\.(.+)", name)
    if m:
        return f"text_encoder.out_norm.norm.{m.group(1)}"

    matmul_map = {
        "onnx::MatMul_3680": "text_encoder.speech_prompted_attn.0.W_query.weight",
        "onnx::MatMul_3681": "text_encoder.speech_prompted_attn.0.W_key.weight",
        "onnx::MatMul_3682": "text_encoder.speech_prompted_attn.0.W_value.weight",
        "onnx::MatMul_3683": "text_encoder.speech_prompted_attn.0.out_fc.weight",
        "onnx::MatMul_3684": "text_encoder.speech_prompted_attn.1.W_query.weight",
        "onnx::MatMul_3685": "text_encoder.speech_prompted_attn.1.W_key.weight",
        "onnx::MatMul_3686": "text_encoder.speech_prompted_attn.1.W_value.weight",
        "onnx::MatMul_3687": "text_encoder.speech_prompted_attn.1.out_fc.weight",
    }

    # ---- Vector Estimator MatMul mapping ----
    # Per group: time_proj, RoPE(W_q,W_k,W_v,out), Style(W_q,W_k,W_v,out)
    _vf_matmul_groups = [
        (3384, 3390, 3391, 3392, 3399, 3405, 3406, 3407, 3408),
        (3429, 3435, 3436, 3437, 3444, 3450, 3451, 3452, 3453),
        (3474, 3480, 3481, 3482, 3489, 3495, 3496, 3497, 3498),
        (3519, 3525, 3526, 3527, 3534, 3540, 3541, 3542, 3543),
    ]
    for g, (tp, rq, rk, rv, ro, sq, sk, sv, so) in enumerate(_vf_matmul_groups):
        fb = g * 7
        matmul_map.update({
            f"onnx::MatMul_{tp}": f"vector_field.time_proj.{g}.weight",
            f"onnx::MatMul_{rq}": f"vector_field.main_blocks.{fb + 2}.W_query.weight",
            f"onnx::MatMul_{rk}": f"vector_field.main_blocks.{fb + 2}.W_key.weight",
            f"onnx::MatMul_{rv}": f"vector_field.main_blocks.{fb + 2}.W_value.weight",
            f"onnx::MatMul_{ro}": f"vector_field.main_blocks.{fb + 2}.out_fc.weight",
            f"onnx::MatMul_{sq}": f"vector_field.main_blocks.{fb + 5}.W_query.weight",
            f"onnx::MatMul_{sk}": f"vector_field.main_blocks.{fb + 5}.W_key.weight",
            f"onnx::MatMul_{sv}": f"vector_field.main_blocks.{fb + 5}.W_value.weight",
            f"onnx::MatMul_{so}": f"vector_field.main_blocks.{fb + 5}.out_fc.weight",
        })
    if name in matmul_map:
        return matmul_map[name]

    # ---- Vector Estimator ----
    if name == "vector_estimator.tts.ttl.vector_field.proj_in.net.weight":
        return "vector_field.proj_in.weight"
    if name == "vector_estimator.tts.ttl.vector_field.proj_out.net.weight":
        return "vector_field.proj_out.weight"

    # CFG uncond tokens
    if name == "vector_estimator.tts.ttl.uncond_masker.text_special_token":
        return "vector_field.uncond_text_token"
    if name == "vector_estimator.tts.ttl.uncond_masker.style_value_special_token":
        return "vector_field.uncond_style_value_token"
    if name == "vector_estimator.tts.ttl.uncond_masker.style_key_special_token":
        return "vector_field.uncond_style_key_token"
    # Shared k_context (one ONNX buffer, used by all 4 CrossAttention blocks)
    if name == "/vector_estimator/Expand_output_0":
        return "vector_field.main_blocks.5.k_context"

    if name == "vector_estimator.tts.ttl.vector_field.time_encoder.mlp.0.linear.weight":
        return "vector_field.time_encoder.0.weight"
    if name == "vector_estimator.tts.ttl.vector_field.time_encoder.mlp.0.linear.bias":
        return "vector_field.time_encoder.0.bias"
    if name == "vector_estimator.tts.ttl.vector_field.time_encoder.mlp.2.linear.weight":
        return "vector_field.time_encoder.2.weight"
    if name == "vector_estimator.tts.ttl.vector_field.time_encoder.mlp.2.linear.bias":
        return "vector_field.time_encoder.2.bias"

    m = re.match(r"vector_estimator\.tts\.ttl\.vector_field\.main_blocks\.(\d+)\.(.+)", name)
    if m:
        return _map_vf_block(m.group(1), m.group(2))

    m = re.match(r"vector_estimator\.tts\.ttl\.vector_field\.last_convnext\.convnext\.(\d+)\.(.+)", name)
    if m:
        return f"vector_field.last_convnext.layers.{m.group(1)}.{m.group(2)}"

    # ---- Vocoder ----
    if name == "tts.ae.latent_mean":
        return "vocoder.latent_mean"
    if name == "tts.ae.latent_std":
        return "vocoder.latent_std"
    if name == "tts.ttl.normalizer.scale":
        return "vocoder.normalizer_scale"
    if name == "onnx::Conv_1441":
        return "vocoder.embed.weight"
    if name == "onnx::Conv_1442":
        return "vocoder.embed.bias"
    if name == "onnx::PRelu_1506":
        return "vocoder.head_act.weight"

    m = re.match(r"tts\.ae\.decoder\.convnext\.(\d+)\.(.+)", name)
    if m:
        rest = m.group(2).replace("dwconv.net.", "dwconv.")
        return f"vocoder.convnext.layers.{m.group(1)}.{rest}"

    if name == "tts.ae.decoder.final_norm.norm.weight":
        return "vocoder.final_norm.weight"
    if name == "tts.ae.decoder.final_norm.norm.bias":
        return "vocoder.final_norm.bias"
    if name == "tts.ae.decoder.final_norm.norm.running_mean":
        return "vocoder.final_norm.running_mean"
    if name == "tts.ae.decoder.final_norm.norm.running_var":
        return "vocoder.final_norm.running_var"
    if name == "tts.ae.decoder.head.layer1.net.weight":
        return "vocoder.head_layer1.weight"
    if name == "tts.ae.decoder.head.layer1.net.bias":
        return "vocoder.head_layer1.bias"
    if name == "tts.ae.decoder.head.layer2.weight":
        return "vocoder.head_layer2.weight"

    return None


def _map_vf_block(block_idx: str, rest: str) -> Optional[str]:
    """Map ONNX VF main_block sub-block to model.py flat ModuleList index.

    ONNX has 24 sub-blocks (0-23), 6 per logical group.
    model.py stores 7 items per group in a flat ModuleList (28 total).

    Flat mapping per logical group g (flat_base = g * 7):
      sub 0: dilated ConvNeXt (4 layers)  → main_blocks.{flat_base}.layers.{i}.X
      sub 1: time proj Linear             → time_proj.{g}.X
      sub 2: first single ConvNeXt        → main_blocks.{flat_base + 1}.X
      sub 3: RoPE attn                    → main_blocks.{flat_base + 2}.X
      sub 3: LayerNorm                    → main_blocks.{flat_base + 3}.norm.X
      sub 4: second single ConvNeXt       → main_blocks.{flat_base + 4}.X
      sub 5: Style cross-attn             → main_blocks.{flat_base + 5}.X
      sub 5: LayerNorm                    → main_blocks.{flat_base + 6}.norm.X
    """
    bn = int(block_idx)
    logical = bn // 6
    sub = bn % 6
    flat_base = logical * 7

    # Strip .linear. from ONNX bias naming (e.g. W_key.linear.bias → W_key.bias)
    rest = rest.replace(".linear.", ".")

    if sub == 0:
        # Dilated ConvNeXt: convnext.{i}.{attr}
        m = re.match(r"convnext\.(\d+)\.(.+)", rest)
        if m:
            return f"vector_field.main_blocks.{flat_base}.layers.{m.group(1)}.{m.group(2)}"
    elif sub == 1:
        # Time proj linear
        m = re.match(r"linear\.(.+)", rest)
        if m:
            return f"vector_field.time_proj.{logical}.{m.group(1)}"
    elif sub == 2:
        # First single ConvNeXt
        m = re.match(r"convnext\.\d+\.(.+)", rest)
        if m:
            return f"vector_field.main_blocks.{flat_base + 1}.{m.group(1)}"
    elif sub == 3:
        # RoPE attention or its LayerNorm
        if rest.startswith("attn."):
            return f"vector_field.main_blocks.{flat_base + 2}.{rest[5:]}"
        if rest.startswith("norm.norm."):
            return f"vector_field.main_blocks.{flat_base + 3}.norm.{rest[10:]}"
    elif sub == 4:
        # Second single ConvNeXt
        m = re.match(r"convnext\.\d+\.(.+)", rest)
        if m:
            return f"vector_field.main_blocks.{flat_base + 4}.{m.group(1)}"
    elif sub == 5:
        # Style cross-attention or its LayerNorm
        if rest.startswith("attention."):
            return f"vector_field.main_blocks.{flat_base + 5}.{rest[10:]}"
        if rest.startswith("norm.norm."):
            return f"vector_field.main_blocks.{flat_base + 6}.norm.{rest[10:]}"

    return None


def load_weights_to_model(onnx_weights: dict, model: SupertonicModel, verbose: bool = True) -> dict:
    state_dict = model.state_dict()
    loaded = {}
    unmatched = []
    skipped = []

    # MatMul weight names that need transpose (ONNX stores them transposed)
    force_transpose = {k for k in onnx_weights if "onnx::MatMul" in k}

    for onnx_name, arr in onnx_weights.items():
        torch_key = map_onnx_to_torch(onnx_name)
        if torch_key is None:
            skipped.append(onnx_name)
            continue
        if torch_key not in state_dict:
            unmatched.append(f"{onnx_name} → {torch_key}")
            continue

        target = state_dict[torch_key]
        tensor = torch.from_numpy(arr.copy())

        # Force transpose for MatMul weights
        if onnx_name in force_transpose and len(tensor.shape) == 2:
            tensor = tensor.T

        if tensor.shape != target.shape:
            if tensor.numel() == target.numel():
                tensor = tensor.reshape(target.shape)
            elif len(tensor.shape) == 2 and len(target.shape) == 2 and tensor.shape == target.shape[::-1]:
                tensor = tensor.T
            else:
                unmatched.append(f"{onnx_name} → {torch_key} (shape {tensor.shape} vs {target.shape})")
                continue

        loaded[torch_key] = tensor.to(target.dtype)

    if verbose:
        print(f"Loaded: {len(loaded)}, Unmatched: {len(unmatched)}, Skipped: {len(skipped)}")

    return loaded


def main():
    parser = argparse.ArgumentParser(description="Extract ONNX weights and load into PyTorch SupertonicModel.")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Directory containing ONNX model files. If not set, downloads to cache.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to save converted tts.json and model.safetensors. Defaults to cache.")
    args = parser.parse_args()

    cache_dir = get_cache_dir("supertonic-3")

    output_dir: Path = args.output_dir if args.output_dir is not None else cache_dir

    if args.input_dir is not None:
        input_dir: Path = args.input_dir
    else:
        if not has_all_onnx_modules(cache_dir):
            print("ONNX model not found in cache. Downloading supertonic-3 ...")
            download_model(cache_dir, "supertonic-3")
        input_dir = cache_dir / "onnx"

    print("Extracting ONNX weights...")
    onnx_weights = load_all_onnx_weights(input_dir)
    print(f"Total ONNX parameters: {len(onnx_weights)}")

    print("Creating PyTorch model...")
    model = SupertonicModel(
        config=str(input_dir / "tts.json"),
        unicode_indexer=str(input_dir / "unicode_indexer.json"),
    )
    model.eval()
    state_dict = model.state_dict()
    print(f"Total PyTorch parameters: {len(state_dict)}")

    print("Mapping and loading...")
    loaded = load_weights_to_model(onnx_weights, model)
    state_dict.update(loaded)

    # Distribute shared k_context to all 4 StyleCrossAttention blocks
    kctx_key = "vector_field.main_blocks.5.k_context"
    if kctx_key in state_dict:
        kctx_val = state_dict[kctx_key]
        for idx in (12, 19, 26):
            other_key = f"vector_field.main_blocks.{idx}.k_context"
            if other_key in state_dict:
                state_dict[other_key] = kctx_val.clone()

    # Distribute shared increments and theta to all 4 RoPECrossAttention blocks
    for buf_name in ("increments", "theta"):
        src_key = f"vector_field.main_blocks.2.{buf_name}"
        if src_key in state_dict:
            val = state_dict[src_key]
            for idx in (9, 16, 23):
                dst_key = f"vector_field.main_blocks.{idx}.{buf_name}"
                if dst_key in state_dict:
                    state_dict[dst_key] = val.clone()

    model.load_state_dict(state_dict, strict=False)

    total = sum(p.numel() for p in model.parameters())
    loaded_n = sum(v.numel() for v in loaded.values())
    print(f"Total: {total:,} | Loaded: {loaded_n:,} ({100*loaded_n/total:.1f}%)")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy original tts.json and unicode_indexer.json from ONNX cache
    import shutil
    config_src = input_dir / "tts.json"
    config_dst = output_dir / "tts.json"
    shutil.copy(config_src, config_dst)
    indexer_src = input_dir / "unicode_indexer.json"
    shutil.copy(indexer_src, output_dir / "unicode_indexer.json")

    # Save model in safetensors format
    import safetensors.torch
    safe_state = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
    safetensors.torch.save_file(safe_state, str(output_dir / "model.safetensors"))
    print(f"Saved: {output_dir / 'tts.json'}")
    print(f"Saved: {output_dir / 'unicode_indexer.json'}")
    print(f"Saved: {output_dir / 'model.safetensors'}")
    # Remove old .pt file if it exists
    old_pt = output_dir / "supertonic_pytorch.pt"
    if old_pt.exists():
        old_pt.unlink()
        print(f"Removed old: {old_pt}")


if __name__ == "__main__":
    main()
