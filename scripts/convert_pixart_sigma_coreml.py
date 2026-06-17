#!/usr/bin/env python3
"""
Download and convert PixArt-Sigma 512 resources for the iOS app.

This is a first-pass converter for the PixArt path. It deliberately writes to
`coremlpixart/`, separate from the existing SD3 resources, so the app can offer
model selection without mixing incompatible pipelines.

The iOS runtime does not require an on-device T5 model. PixArtTransformer still
accepts `encoder_hidden_states`; produce those prompt embeddings offline or with
another runtime and feed them into the denoising loop.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSFORMER_REPO = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
DEFAULT_SHARED_REPO = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
DEFAULT_HF_DIR = REPO_ROOT / "models" / "pixart_sigma_512_hf"
DEFAULT_SHARED_DIR = REPO_ROOT / "models" / "pixart_sigma_shared_hf"
DEFAULT_BUILD_DIR = REPO_ROOT / "pixart_build_512"
DEFAULT_COREML_DIR = REPO_ROOT / "coremlpixart"

LOG = logging.getLogger("pixart-coreml")


class PixArtTransformerWrapper(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer.eval()
        self.latent_channels = transformer.config.in_channels

    def forward(self, latent_model_input, timestep, encoder_hidden_states):
        output = self.transformer(
            latent_model_input,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            timestep=timestep,
            added_cond_kwargs={"resolution": None, "aspect_ratio": None},
            return_dict=False,
        )[0]
        if output.shape[1] == self.latent_channels * 2:
            output = output[:, : self.latent_channels]
        return output


class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv.eval()
        self.decoder = vae.decoder.eval()
        self.scaling_factor = float(vae.config.scaling_factor)

    def forward(self, z):
        z = z / self.scaling_factor
        z = self.post_quant_conv(z)
        return self.decoder(z)


class T5EncoderWrapper(torch.nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder.eval()

    def forward(self, input_ids, attention_mask):
        return self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-repo", default=DEFAULT_TRANSFORMER_REPO)
    parser.add_argument("--shared-repo", default=DEFAULT_SHARED_REPO)
    parser.add_argument("--hf-dir", type=Path, default=DEFAULT_HF_DIR)
    parser.add_argument("--shared-dir", type=Path, default=DEFAULT_SHARED_DIR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--compile-into", type=Path, default=DEFAULT_COREML_DIR)
    parser.add_argument("--latent-h", type=int, default=64)
    parser.add_argument("--latent-w", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=300)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--download-text-encoder",
        action="store_true",
        help="Also download T5-XXL weights. This is very large and usually not suitable for iPhone bundles.",
    )
    parser.add_argument("--convert-transformer", action="store_true")
    parser.add_argument("--convert-vae-decoder", action="store_true")
    parser.add_argument("--convert-text-encoder", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--quantize-int8", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def download(args: argparse.Namespace) -> None:
    LOG.info("Downloading PixArt transformer repo: %s", args.transformer_repo)
    snapshot_download(
        repo_id=args.transformer_repo,
        local_dir=args.hf_dir,
        allow_patterns=["README.md", "transformer/*"],
    )

    shared_patterns = [
        "model_index.json",
        "scheduler/*",
        "vae/*",
        "tokenizer/*",
        "text_encoder/config.json",
    ]
    if args.download_text_encoder or args.convert_text_encoder:
        shared_patterns.append("text_encoder/*")

    LOG.info("Downloading PixArt shared VAE/tokenizer repo: %s", args.shared_repo)
    snapshot_download(
        repo_id=args.shared_repo,
        local_dir=args.shared_dir,
        allow_patterns=shared_patterns,
    )


def convert_transformer(args: argparse.Namespace) -> Path:
    from diffusers import Transformer2DModel

    LOG.info("Loading PixArt Transformer2DModel")
    transformer = Transformer2DModel.from_pretrained(
        args.hf_dir,
        subfolder="transformer",
        torch_dtype=torch.float32,
    ).eval()

    wrapper = PixArtTransformerWrapper(transformer).eval()
    latent = torch.zeros(1, 4, args.latent_h, args.latent_w, dtype=torch.float32)
    timestep = torch.ones(1, dtype=torch.float32)
    text = torch.zeros(1, args.sequence_length, 4096, dtype=torch.float32)

    LOG.info("Tracing PixArt transformer")
    traced = torch.jit.trace(wrapper, (latent, timestep, text), strict=False)

    LOG.info("Converting PixArt transformer to Core ML")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        inputs=[
            ct.TensorType("latent_model_input", shape=latent.shape, dtype=np.float32),
            ct.TensorType("timestep", shape=timestep.shape, dtype=np.float32),
            ct.TensorType("encoder_hidden_states", shape=text.shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType("noise_pred", dtype=np.float32)],
        skip_model_load=True,
    )
    out = args.build_dir / "PixArtTransformer.mlpackage"
    save_model(mlmodel, out)
    return out


def convert_vae_decoder(args: argparse.Namespace) -> Path:
    from diffusers import AutoencoderKL

    LOG.info("Loading PixArt/SDXL VAE")
    vae = AutoencoderKL.from_pretrained(
        args.shared_dir,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).eval()

    wrapper = VAEDecoderWrapper(vae).eval()
    z = torch.zeros(1, 4, args.latent_h, args.latent_w, dtype=torch.float32)

    LOG.info("Tracing VAE decoder")
    traced = torch.jit.trace(wrapper, (z,), strict=False)

    LOG.info("Converting VAE decoder to Core ML")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        inputs=[ct.TensorType("z", shape=z.shape, dtype=np.float32)],
        outputs=[ct.TensorType("image", dtype=np.float32)],
        skip_model_load=True,
    )
    out = args.build_dir / "PixArtVAEDecoder.mlpackage"
    save_model(mlmodel, out)
    return out


def convert_text_encoder(args: argparse.Namespace) -> Path:
    from transformers import T5EncoderModel

    LOG.warning(
        "Converting T5-XXL is very memory intensive. Prefer a preconverted or "
        "quantized TextEncoderT5 if one is available."
    )
    text_encoder = T5EncoderModel.from_pretrained(
        args.shared_dir,
        subfolder="text_encoder",
        torch_dtype=torch.float32,
    ).eval()

    wrapper = T5EncoderWrapper(text_encoder).eval()
    input_ids = torch.zeros(1, args.sequence_length, dtype=torch.int32)
    attention_mask = torch.ones(1, args.sequence_length, dtype=torch.int32)

    LOG.info("Tracing T5 text encoder")
    traced = torch.jit.trace(wrapper, (input_ids, attention_mask), strict=False)

    LOG.info("Converting T5 text encoder to Core ML")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        inputs=[
            ct.TensorType("input_ids", shape=input_ids.shape, dtype=np.int32),
            ct.TensorType("attention_mask", shape=attention_mask.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType("last_hidden_state", dtype=np.float32)],
        skip_model_load=True,
    )
    out = args.build_dir / "TextEncoderT5.mlpackage"
    save_model(mlmodel, out)
    return out


def save_model(mlmodel, out_path: Path) -> None:
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    LOG.info("Saved %s", out_path)


def quantize_if_requested(package: Path, enabled: bool) -> Path:
    if not enabled:
        return package
    LOG.info("INT8 quantizing %s", package.name)
    model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    config = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=np.int8,
        )
    )
    quantized = ct.optimize.coreml.linear_quantize_weights(model, config=config)
    out = package.with_name(package.stem + ".int8.mlpackage")
    save_model(quantized, out)
    return out


def compile_package(package: Path, destination: Path) -> None:
    LOG.info("Compiling %s", package.name)
    tmp_dir = destination / f".{package.stem}.compile-tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    compiled_path = ct.models.utils.compile_model(
        str(package),
        destination_path=str(tmp_dir / f"{package.stem}.mlmodelc"),
    )
    target = destination / f"{package.stem.removesuffix('.int8')}.mlmodelc"
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(compiled_path), target)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    LOG.info("Installed %s", target)


def copy_runtime_files(args: argparse.Namespace) -> None:
    args.compile_into.mkdir(parents=True, exist_ok=True)
    scheduler = args.shared_dir / "scheduler" / "scheduler_config.json"
    if scheduler.exists():
        shutil.copy2(scheduler, args.compile_into / "scheduler_config.json")

    # Tokenizer files are optional for the no-on-device-T5 runtime, but keeping
    # them in coremlpixart makes it easier to generate/debug prompt embeddings
    # from the same source vocabulary during development.
    try:
        from transformers import T5TokenizerFast

        tokenizer = T5TokenizerFast.from_pretrained(args.shared_dir / "tokenizer")
        tokenizer.save_pretrained(args.compile_into)
    except Exception as exc:  # pragma: no cover - best-effort helper
        LOG.warning("Could not export tokenizer.json: %s", exc)
        for name in ["tokenizer_config.json", "spiece.model", "special_tokens_map.json"]:
            src = args.shared_dir / "tokenizer" / name
            if src.exists():
                shutil.copy2(src, args.compile_into / name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if args.clean and args.build_dir.exists():
        shutil.rmtree(args.build_dir)
    args.build_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        download(args)

    converted: list[Path] = []
    if args.convert_transformer:
        converted.append(convert_transformer(args))
    if args.convert_vae_decoder:
        converted.append(convert_vae_decoder(args))
    if args.convert_text_encoder:
        converted.append(convert_text_encoder(args))

    if args.compile:
        args.compile_into.mkdir(parents=True, exist_ok=True)
        copy_runtime_files(args)
        for package in converted:
            compile_package(quantize_if_requested(package, args.quantize_int8), args.compile_into)

    LOG.info("Done")


if __name__ == "__main__":
    main()
