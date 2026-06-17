#!/usr/bin/env python3
"""
Convert an SD3 Medium diffusers-format transformer safetensors file into split
Core ML MMDiT stages compatible with the app's StableDiffusion3Pipeline runner.

The input model is the transformer component only. Text encoders, tokenizer, and
VAE decoder are copied from the existing SD3 resources by the prepare script.
"""

from __future__ import annotations

import argparse
import gc
import logging
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

LOG = logging.getLogger("sd3-diffusers-coreml")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def save_mlpackage(model, out_path: Path) -> None:
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))


def cleanup_stale_packages(out_dir: Path) -> None:
    for path in out_dir.glob("MultiModalDiffusionTransformer*.mlpackage"):
        shutil.rmtree(path)


class SD3Conditioning(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.time_text_embed = transformer.time_text_embed

    def forward(self, pooled_text_embeddings, timestep):
        pooled_text_embeddings = pooled_text_embeddings.reshape(pooled_text_embeddings.shape[0], -1)
        return (self.time_text_embed(timestep, pooled_text_embeddings),)


class SD3InputStage(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.pos_embed = transformer.pos_embed
        self.context_embedder = transformer.context_embedder

    def forward(self, latent_image_embeddings, token_level_text_embeddings, modulation_inputs):
        token_level_text_embeddings = token_level_text_embeddings.squeeze(2).transpose(1, 2)
        latent_image_embeddings = self.pos_embed(latent_image_embeddings)
        token_level_text_embeddings = self.context_embedder(token_level_text_embeddings)
        return latent_image_embeddings, token_level_text_embeddings


class SD3BlockStage(nn.Module):
    def __init__(self, transformer, start_block: int, end_block: int, latent_h: int, latent_w: int):
        super().__init__()
        self.blocks = nn.ModuleList([transformer.transformer_blocks[i] for i in range(start_block, end_block)])
        self.norm_out = transformer.norm_out if end_block == transformer.config.num_layers else None
        self.proj_out = transformer.proj_out if end_block == transformer.config.num_layers else None
        self.patch_size = transformer.config.patch_size
        self.out_channels = transformer.out_channels
        self.latent_h = latent_h
        self.latent_w = latent_w

    def forward(self, latent_image_embeddings, token_level_text_embeddings, modulation_inputs):
        for block in self.blocks:
            token_level_text_embeddings, latent_image_embeddings = block(
                hidden_states=latent_image_embeddings,
                encoder_hidden_states=token_level_text_embeddings,
                temb=modulation_inputs,
            )

        if self.norm_out is None:
            return latent_image_embeddings, token_level_text_embeddings

        latent_image_embeddings = self.norm_out(latent_image_embeddings, modulation_inputs)
        latent_image_embeddings = self.proj_out(latent_image_embeddings)

        patch_h = self.latent_h // self.patch_size
        patch_w = self.latent_w // self.patch_size
        latent_image_embeddings = latent_image_embeddings.reshape(
            latent_image_embeddings.shape[0],
            patch_h,
            patch_w,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        latent_image_embeddings = torch.einsum("nhwpqc->nchpwq", latent_image_embeddings)
        return (latent_image_embeddings.reshape(
            latent_image_embeddings.shape[0],
            self.out_channels,
            self.latent_h,
            self.latent_w,
        ),)


def convert_module(module: nn.Module, inputs: dict[str, torch.Tensor], output_names: list[str], out_path: Path, target):
    import coremltools as ct

    LOG.info("Tracing %s", out_path.name)
    module = module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, example_inputs=tuple(inputs.values()), strict=False)

    coreml_inputs = [
        ct.TensorType(name=name, shape=tensor.shape, dtype=tensor.numpy().dtype)
        for name, tensor in inputs.items()
    ]
    outputs = [ct.TensorType(name=name) for name in output_names]

    LOG.info("Converting %s", out_path.name)
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=coreml_inputs,
        outputs=outputs,
        minimum_deployment_target=target,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
        convert_to="mlprogram",
    )
    LOG.info("Converted %s in %.1fs", out_path.name, time.time() - t0)
    save_mlpackage(mlmodel, out_path)
    del traced, mlmodel
    gc.collect()


def parse_stage_sizes(stage_sizes: str, depth: int) -> list[tuple[int, int]]:
    sizes = [int(part) for part in stage_sizes.split(",") if part.strip()]
    if not sizes or sum(sizes) != depth:
        raise ValueError(f"--stage-sizes must sum to {depth}; got {sizes} (sum={sum(sizes) if sizes else 0})")
    ranges = []
    start = 0
    for size in sizes:
        ranges.append((start, start + size))
        start += size
    return ranges


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True, type=Path)
    parser.add_argument(
        "--model-family",
        choices=("sd3-medium",),
        default="sd3-medium",
        help="Transformer architecture. Only SD3 Medium is supported by this release script.",
    )
    parser.add_argument("--latent-h", type=int, default=64)
    parser.add_argument("--latent-w", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--stage-sizes",
        default="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
        help="Comma-separated block counts; must sum to 24.",
    )
    parser.add_argument("--ios-target", choices=("iOS17", "iOS18"), default="iOS18")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("sd3_build_split_512"))
    args = parser.parse_args()

    if not args.ckpt_path.exists():
        raise FileNotFoundError(args.ckpt_path)

    import coremltools as ct
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

    deployment_target = ct.target.iOS18 if args.ios_target == "iOS18" else ct.target.iOS17
    stage_ranges = parse_stage_sizes(args.stage_sizes, 24)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_packages(args.output_dir)

    transformer_kwargs = dict(
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
        pos_embed_max_size=192,
        dual_attention_layers=(),
        qk_norm=None,
    )

    LOG.info("Building %s transformer", args.model_family)
    transformer = SD3Transformer2DModel(**transformer_kwargs).to("cpu").to(torch.float32).eval()

    LOG.info("Loading weights: %s", args.ckpt_path)
    state = load_file(str(args.ckpt_path), device="cpu")
    missing, unexpected = transformer.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch: missing={missing}, unexpected={unexpected}")
    del state
    gc.collect()

    batch = args.batch_size
    hidden_tokens = (args.latent_h // 2) * (args.latent_w // 2)

    convert_module(
        SD3Conditioning(transformer),
        {
            "pooled_text_embeddings": torch.randn(batch, 2048, 1, 1),
            "timestep": torch.randn(batch),
        },
        ["modulation_inputs"],
        args.output_dir / "MultiModalDiffusionTransformerConditioning.mlpackage",
        deployment_target,
    )

    convert_module(
        SD3InputStage(transformer),
        {
            "latent_image_embeddings": torch.randn(batch, 16, args.latent_h, args.latent_w),
            "token_level_text_embeddings": torch.randn(batch, 4096, 1, 154),
            "modulation_inputs": torch.randn(batch, 1536),
        },
        ["latent_image_embeddings_out", "token_level_text_embeddings_out"],
        args.output_dir / "MultiModalDiffusionTransformerStage0.mlpackage",
        deployment_target,
    )

    for stage_index, (start, end) in enumerate(stage_ranges, start=1):
        is_final = end == 24
        outputs = ["noise_pred"] if is_final else [
            "latent_image_embeddings_out",
            "token_level_text_embeddings_out",
        ]
        convert_module(
            SD3BlockStage(transformer, start, end, args.latent_h, args.latent_w),
            {
                "latent_image_embeddings": torch.randn(batch, hidden_tokens, 1536),
                "token_level_text_embeddings": torch.randn(batch, 154, 1536),
                "modulation_inputs": torch.randn(batch, 1536),
            },
            outputs,
            args.output_dir / f"MultiModalDiffusionTransformerStage{stage_index}.mlpackage",
            deployment_target,
        )

    LOG.info("Done: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
