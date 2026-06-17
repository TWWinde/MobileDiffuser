#!/usr/bin/env python3
"""Run a local SD3 Medium two-step transformer checkpoint through diffusers.

This mirrors the iOS app path: CLIP-L + CLIP-G text encoders, no T5 encoder
(diffusers fills zero T5 embeddings), 512x512 latent geometry, CFG disabled.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from safetensors.torch import load_file


def build_sd3_medium_transformer() -> SD3Transformer2DModel:
    return SD3Transformer2DModel(
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("checkpoints/diffusion_pytorch_model.safetensors"),
        help="Path to a local SD3 Medium distilled transformer checkpoint.",
    )
    parser.add_argument(
        "--repo",
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="Diffusers repo that provides tokenizer, CLIP text encoders, VAE, scheduler.",
    )
    parser.add_argument("--prompt", default="A cinematic portrait of a robot in a tailored suit")
    parser.add_argument("--output", type=Path, default=Path("gen/sd3_medium_two_step_mac.png"))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(args.ckpt)

    print(f"[mac] loading transformer: {args.ckpt}")
    transformer = build_sd3_medium_transformer().eval()
    state = load_file(str(args.ckpt), device="cpu")
    missing, unexpected = transformer.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: missing={missing}, unexpected={unexpected}")
    del state

    print(f"[mac] loading pipeline components from {args.repo}")
    dtype = torch.float32
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.repo,
        transformer=transformer,
        text_encoder_3=None,
        tokenizer_3=None,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=args.shift)
    pipe = pipe.to("cpu")
    pipe.set_progress_bar_config(disable=False)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[mac] generating 512x512 steps={args.steps} cfg={args.guidance_scale} "
        f"shift={args.shift} seed={args.seed}"
    )
    start = time.time()
    image = pipe(
        prompt=args.prompt,
        height=512,
        width=512,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        max_sequence_length=77,
    ).images[0]
    elapsed = time.time() - start
    image.save(args.output)
    print(f"[mac] saved {args.output} in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
