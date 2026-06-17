#!/usr/bin/env python3
"""Validate SD3.5 split Core ML stages against the PyTorch transformer.

The checks here are intentionally small and deterministic:
- PyTorch full transformer vs. split wrapper parity.
- Core ML stage stability for fp16 and optional int8 packages.

This does not require the iOS app. Set TMPDIR to a writable directory if local
Core ML compilation refuses the default macOS temp directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from convert_sd35_diffusers_split_coreml import (
    SD35BlockStage,
    SD35Conditioning,
    SD35InputStage,
)


def build_transformer(ckpt_path: Path):
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

    model = SD3Transformer2DModel(
        sample_size=128,
        patch_size=2,
        in_channels=16,
        num_layers=24,
        attention_head_dim=64,
        num_attention_heads=24,
        joint_attention_dim=4096,
        caption_projection_dim=1536,
        pooled_projection_dim=2048,
        out_channels=16,
        pos_embed_max_size=384,
        dual_attention_layers=tuple(range(13)),
        qk_norm="rms_norm",
    ).eval()
    model.load_state_dict(load_file(str(ckpt_path), device="cpu"), strict=True)
    return model


def stats(array: np.ndarray) -> str:
    return (
        f"shape={tuple(array.shape)} "
        f"mean={float(np.nanmean(array)):.6g} "
        f"std={float(np.nanstd(array)):.6g} "
        f"maxAbs={float(np.nanmax(np.abs(array))):.6g} "
        f"nan={int(np.isnan(array).sum())} "
        f"inf={int(np.isinf(array).sum())}"
    )


def ensure_finite(name: str, array: np.ndarray) -> None:
    if np.isnan(array).any() or np.isinf(array).any():
        raise RuntimeError(f"{name} produced non-finite output: {stats(array)}")


def parity_check(model) -> None:
    torch.manual_seed(0)
    latent = torch.randn(1, 16, 64, 64)
    tokens = torch.randn(1, 154, 4096)
    pooled = torch.randn(1, 2048)
    timestep = torch.tensor([1.0])

    with torch.no_grad():
        full = model(
            hidden_states=latent,
            encoder_hidden_states=tokens,
            pooled_projections=pooled,
            timestep=timestep,
            return_dict=False,
        )[0]

        modulation = SD35Conditioning(model)(pooled.reshape(1, 2048, 1, 1), timestep)[0]
        hidden, context = SD35InputStage(model)(
            latent,
            tokens.transpose(1, 2).unsqueeze(2),
            modulation,
        )
        for block_index in range(24):
            output = SD35BlockStage(model, block_index, block_index + 1, 64, 64)(
                hidden,
                context,
                modulation,
            )
            if block_index < 23:
                hidden, context = output
            else:
                split = output[0]

    diff = (full - split).abs()
    print(
        f"[parity] maxdiff={float(diff.max()):.6g} "
        f"meandiff={float(diff.mean()):.6g}"
    )
    if float(diff.max()) != 0.0:
        raise RuntimeError("PyTorch split wrapper does not match full transformer")


def coreml_stage_check(model, root: Path, scenario: str) -> None:
    import coremltools as ct

    if scenario == "zero":
        torch.manual_seed(2)
        pooled = torch.zeros(1, 2048, 1, 1)
        latent = torch.randn(1, 16, 64, 64) * 0.1
        tokens = torch.zeros(1, 4096, 1, 154)
    elif scenario == "random":
        torch.manual_seed(1)
        pooled = torch.randn(1, 2048, 1, 1)
        latent = torch.randn(1, 16, 64, 64)
        tokens = torch.randn(1, 4096, 1, 154)
    else:
        raise ValueError(scenario)

    timestep = torch.tensor([1.0])
    with torch.no_grad():
        modulation = SD35Conditioning(model)(pooled, timestep)[0]

    def predict_package(name: str, inputs: dict[str, torch.Tensor | np.ndarray]) -> dict[str, np.ndarray]:
        package = root / f"{name}.mlpackage"
        if not package.exists():
            raise FileNotFoundError(package)
        mlmodel = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
        coreml_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().numpy()
            coreml_inputs[key] = value.astype(np.float32)
        return {
            key: np.asarray(value).astype(np.float32)
            for key, value in mlmodel.predict(coreml_inputs).items()
        }

    conditioning = predict_package(
        "MultiModalDiffusionTransformerConditioning",
        {"pooled_text_embeddings": pooled, "timestep": timestep},
    )
    for name, value in conditioning.items():
        print(f"[{root.name}:{scenario}] Conditioning.{name} {stats(value)}")
        ensure_finite(f"{root}/Conditioning.{name}", value)

    stage0 = predict_package(
        "MultiModalDiffusionTransformerStage0",
        {
            "latent_image_embeddings": latent,
            "token_level_text_embeddings": tokens,
            "modulation_inputs": conditioning["modulation_inputs"],
        },
    )
    hidden = stage0["latent_image_embeddings_out"]
    context = stage0["token_level_text_embeddings_out"]
    for name, value in stage0.items():
        print(f"[{root.name}:{scenario}] Stage0.{name} {stats(value)}")
        ensure_finite(f"{root}/Stage0.{name}", value)

    for stage_index in range(1, 25):
        stage = predict_package(
            f"MultiModalDiffusionTransformerStage{stage_index}",
            {
                "latent_image_embeddings": hidden,
                "token_level_text_embeddings": context,
                "modulation_inputs": conditioning["modulation_inputs"],
            },
        )
        if "noise_pred" in stage:
            value = stage["noise_pred"]
            print(f"[{root.name}:{scenario}] Stage{stage_index}.noise_pred {stats(value)}")
            ensure_finite(f"{root}/Stage{stage_index}.noise_pred", value)
            break

        hidden = stage["latent_image_embeddings_out"]
        context = stage["token_level_text_embeddings_out"]
        for name, value in stage.items():
            print(f"[{root.name}:{scenario}] Stage{stage_index}.{name} {stats(value)}")
            ensure_finite(f"{root}/Stage{stage_index}.{name}", value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("checkpoints/diffusion_pytorch_model.safetensors"),
    )
    parser.add_argument("--fp16-root", type=Path, default=Path("sd35_build_split_512"))
    parser.add_argument("--int8-root", type=Path, default=Path("sd35_build_split_512/int8"))
    parser.add_argument(
        "--scenario",
        choices=("zero", "random"),
        default="zero",
        help="Use zero for stability smoke test; random intentionally stresses conditioning range.",
    )
    args = parser.parse_args()

    model = build_transformer(args.ckpt)
    parity_check(model)
    coreml_stage_check(model, args.fp16_root, args.scenario)
    if args.int8_root.exists():
        coreml_stage_check(model, args.int8_root, args.scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
