#!/usr/bin/env python3
"""
Low-memory MMDiT (SD3 medium) -> CoreML mlpackage converter.

Bypasses diffusionkit's unittest-based pipeline. Key optimizations vs
ml-stable-diffusion's default path:

  1. compute_precision = FLOAT16 (default path hardcodes FLOAT32, ~2x memory)
  2. Skips the post-convert "first load time" specialization (biggest peak)
  3. Skips PyTorch-vs-CoreML PSNR correctness test (skips a 2nd forward pass)
  4. Drops the torch model right after tracing, before ct.convert()

Result on a 24 GB Mac: peak memory ~16-18 GB instead of 30+ GB.

Run AFTER torch2coreml has already produced VAE/TextEncoder/TextEncoder2
mlpackages. This script only does the MMDiT.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("convert_mmdit_low_mem")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Local path to sd3_medium*.safetensors. If set, skips HF download.",
    )
    parser.add_argument(
        "--sd3-repo",
        default="stabilityai/stable-diffusion-3-medium",
        help="HF repo containing sd3_medium.safetensors (used only when --ckpt-path absent)",
    )
    parser.add_argument(
        "--ckpt-file",
        default="sd3_medium.safetensors",
        help="Filename in the HF repo (used only when --ckpt-path absent)",
    )
    parser.add_argument(
        "--latent-h", type=int, default=96, help="Latent height (96 -> 768px image)"
    )
    parser.add_argument(
        "--latent-w", type=int, default=96, help="Latent width  (96 -> 768px image)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        choices=(1, 2),
        help=(
            "MMDiT batch dim: 2 = classic CFG (cond+uncond), 1 = single-conditional "
            "(distilled models that don't use CFG). Halves MMDiT cost on device."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("sd3_build"),
        help="Output directory for the mlpackage",
    )
    parser.add_argument(
        "--name",
        default="Stable_Diffusion_version_stabilityai_stable-diffusion-3-medium_mmdit",
        help="Filename stem (without .mlpackage) — keep default to match ml-stable-diffusion layout",
    )
    parser.add_argument(
        "--ios-target",
        choices=("iOS17", "iOS18"),
        default="iOS18",
        help=(
            "Minimum CoreML deployment target. iOS18 is required for "
            "per_grouped_channel palettization (where the iPhone GPU/ANE "
            "consumes the LUT directly without dequantizing to fp16)."
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the canonical name so palettize_sd3.py's glob (`*_mmdit.mlpackage`)
    # still picks it up. We rely on `--output-dir` to isolate batch variants.
    out_path = args.output_dir / f"{args.name}.mlpackage"
    if out_path.exists():
        logger.warning(f"Output already exists, will overwrite: {out_path}")

    logger.info("Importing heavy deps (torch / coremltools / diffusionkit)...")
    import torch
    import coremltools as ct
    from huggingface_hub import hf_hub_download
    from diffusionkit.torch import mmdit
    from diffusionkit.torch.model_io import _load_mmdit_weights

    torch.set_grad_enabled(False)

    if args.ckpt_path:
        ckpt_path = args.ckpt_path
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(ckpt_path)
        logger.info(f"Using local checkpoint: {ckpt_path}")
    else:
        logger.info("Downloading / locating SD3 checkpoint from HF...")
        ckpt_path = hf_hub_download(args.sd3_repo, args.ckpt_file)
        logger.info(f"Checkpoint at: {ckpt_path}")

    logger.info("Building MMDiT (SD3_2b) PyTorch module on CPU (fp32)...")
    cfg = mmdit.SD3_2b
    model = mmdit.MMDiT(cfg).to("cpu").to(torch.float32).eval()

    logger.info("Loading MMDiT weights from safetensors...")
    _load_mmdit_weights(model, ckpt_path)
    logger.info("Weights loaded.")

    batch_size = args.batch_size
    assert args.latent_h <= cfg.max_latent_resolution
    assert args.latent_w <= cfg.max_latent_resolution
    logger.info(f"Tracing with batch_size={batch_size}")
    inputs = {
        "latent_image_embeddings": torch.randn(
            batch_size, cfg.vae_latent_dim, args.latent_h, args.latent_w
        ),
        "token_level_text_embeddings": torch.randn(
            batch_size, cfg.token_level_text_embed_dim, 1, cfg.text_seq_len
        ),
        "pooled_text_embeddings": torch.randn(
            batch_size, cfg.pooled_text_embed_dim, 1, 1
        ),
        "timestep": torch.randn(batch_size),
    }

    logger.info("Tracing MMDiT (this allocates a few GB temporarily)...")
    t0 = time.time()
    traced = torch.jit.trace(model, example_kwarg_inputs=inputs)
    logger.info(f"Traced in {time.time()-t0:.1f}s; releasing torch model...")
    del model
    gc.collect()

    coreml_inputs = [
        ct.TensorType(name=k, shape=v.shape, dtype=v.numpy().dtype)
        for k, v in inputs.items()
    ]

    deployment_target = (
        ct.target.iOS18 if args.ios_target == "iOS18" else ct.target.iOS17
    )
    logger.info(
        f"Running ct.convert (precision=FLOAT16, target={args.ios_target}, "
        "skip_model_load=True)..."
    )
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=coreml_inputs,
        outputs=[ct.TensorType(name="denoiser_output")],
        minimum_deployment_target=deployment_target,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
        convert_to="mlprogram",
    )
    logger.info(f"ct.convert finished in {time.time()-t0:.1f}s; releasing trace...")
    del traced
    gc.collect()

    mlmodel.author = f"Please refer to the Model Card available at huggingface.co/{args.sd3_repo}"
    mlmodel.license = (
        "Stability AI Community License "
        "(https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE.md)"
    )
    mlmodel.short_description = (
        "Stable Diffusion 3 MMDiT (low-memory conversion, fp16). "
        "Quantize separately with scripts/palettize_sd3.py."
    )
    mlmodel.input_description["latent_image_embeddings"] = (
        "The low resolution latent feature maps being denoised through reverse diffusion"
    )
    mlmodel.input_description["token_level_text_embeddings"] = (
        "Output embeddings from the associated text_encoder model"
    )
    mlmodel.input_description["pooled_text_embeddings"] = (
        "Additional pooled embeddings from the text encoders"
    )
    mlmodel.input_description["timestep"] = (
        "A scheduler timestep value to condition the model on the noise schedule"
    )
    mlmodel.output_description["denoiser_output"] = (
        "Same shape and dtype as latent_image_embeddings; predicted noise"
    )

    logger.info(f"Saving mlpackage to {out_path} ...")
    t0 = time.time()
    if out_path.exists():
        import shutil

        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    logger.info(f"Saved in {time.time()-t0:.1f}s.")

    size_gb = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file()) / 1e9
    logger.info(f"Final mlpackage size: {size_gb:.2f} GB")
    logger.info(
        "Next: python scripts/palettize_sd3.py --mlpackage-dir sd3_build --only mmdit --compile --output-dir sd3_build/Resources"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
