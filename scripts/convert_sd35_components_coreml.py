#!/usr/bin/env python3
"""Convert SD3.5 CLIP text encoders and VAE decoder to app-compatible Core ML.

Outputs mlpackages with app-facing names:
- TextEncoder.mlpackage
- TextEncoder2.mlpackage
- VAEDecoder.mlpackage

T5 / text_encoder_3 is intentionally not downloaded or converted here.

The compiled .mlmodelc files are optionally installed into coremlsd35.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn

LOG = logging.getLogger("sd35-components")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class CLIPTextEncoderWrapper(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder.eval()

    def forward(self, input_ids):
        output = self.text_encoder(
            input_ids.to(torch.int32),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_embeds = output.hidden_states[-2]
        pooled_outputs = getattr(output, "text_embeds", None)
        if pooled_outputs is None:
            pooled_outputs = output.pooler_output
        return hidden_embeds, pooled_outputs


class VAEDecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        post_quant_conv = getattr(vae, "post_quant_conv", None)
        self.post_quant_conv = post_quant_conv.eval() if post_quant_conv is not None else None
        self.decoder = vae.decoder.eval()

    def forward(self, z):
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        return self.decoder(z)


def clean(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def save_mlpackage(mlmodel, path: Path) -> None:
    clean(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(path))


def compile_mlpackage(package: Path, destination: Path, deployment_target: str) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    tmp = destination / f".{package.stem}.compile-tmp"
    clean(tmp)
    tmp.mkdir(parents=True)
    subprocess.run(
        [
            "xcrun",
            "coremlcompiler",
            "compile",
            str(package),
            str(tmp),
            "--platform",
            "iOS",
            "--deployment-target",
            deployment_target,
        ],
        check=True,
    )
    compiled = tmp / f"{package.stem}.mlmodelc"
    if not compiled.exists():
        candidates = list(tmp.glob("*.mlmodelc"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"No compiled model found in {tmp}")
        compiled = candidates[0]
    target = destination / f"{package.stem}.mlmodelc"
    clean(target)
    compiled.rename(target)
    clean(tmp)
    return target


def convert_clip(repo: str, subfolder: str, output: Path, deployment_target):
    import coremltools as ct
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    LOG.info("Loading %s/%s", repo, subfolder)
    tokenizer = CLIPTokenizer.from_pretrained(repo, subfolder=subfolder.replace("text_encoder", "tokenizer"))
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        repo,
        subfolder=subfolder,
        variant="fp16",
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).eval()
    wrapper = CLIPTextEncoderWrapper(text_encoder).eval()
    sequence_length = int(tokenizer.model_max_length)
    sample = torch.randint(
        text_encoder.config.vocab_size,
        (1, sequence_length),
        dtype=torch.float32,
    )

    LOG.info("Tracing %s", output.name)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (sample,), strict=False)

    LOG.info("Converting %s", output.name)
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=sample.shape, dtype=sample.numpy().dtype)],
        outputs=[
            ct.TensorType(name="hidden_embeds"),
            ct.TensorType(name="pooled_outputs"),
        ],
        minimum_deployment_target=deployment_target,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
        convert_to="mlprogram",
    )
    LOG.info("Converted %s in %.1fs", output.name, time.time() - t0)
    save_mlpackage(mlmodel, output)
    return output


def convert_vae(repo: str, output: Path, latent_h: int, latent_w: int, deployment_target):
    import coremltools as ct
    from diffusers import AutoencoderKL

    LOG.info("Loading %s/vae", repo)
    vae = AutoencoderKL.from_pretrained(
        repo,
        subfolder="vae",
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).eval()
    wrapper = VAEDecoderWrapper(vae).eval()
    sample = torch.randn(1, int(vae.config.latent_channels), latent_h, latent_w, dtype=torch.float32)

    LOG.info("Tracing %s", output.name)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (sample,), strict=False)

    LOG.info("Converting %s", output.name)
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="z", shape=sample.shape, dtype=sample.numpy().dtype)],
        outputs=[ct.TensorType(name="image")],
        minimum_deployment_target=deployment_target,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
        convert_to="mlprogram",
    )
    LOG.info("Converted %s in %.1fs", output.name, time.time() - t0)
    save_mlpackage(mlmodel, output)
    return output


def copy_tokenizer(repo: str, destination: Path) -> None:
    from huggingface_hub import snapshot_download

    LOG.info("Downloading tokenizer files")
    snapshot = Path(
        snapshot_download(
            repo,
            allow_patterns=[
                "tokenizer/vocab.json",
                "tokenizer/merges.txt",
                "tokenizer_2/vocab.json",
                "tokenizer_2/merges.txt",
            ],
        )
    )
    for source_name, target_name in [
        ("tokenizer/vocab.json", "vocab.json"),
        ("tokenizer/merges.txt", "merges.txt"),
    ]:
        target = destination / target_name
        clean(target)
        shutil.copy2(snapshot / source_name, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--build-dir", type=Path, default=Path("sd35_build_components_512"))
    parser.add_argument("--compile-into", type=Path, default=Path("coremlsd35"))
    parser.add_argument("--latent-h", type=int, default=64)
    parser.add_argument("--latent-w", type=int, default=64)
    parser.add_argument("--ios-target", choices=("iOS17", "iOS18"), default="iOS18")
    parser.add_argument("--ios-deployment-target", default="18.2")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-vae", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()

    import coremltools as ct

    deployment_target = ct.target.iOS18 if args.ios_target == "iOS18" else ct.target.iOS17
    args.build_dir.mkdir(parents=True, exist_ok=True)
    args.compile_into.mkdir(parents=True, exist_ok=True)

    packages: list[Path] = []
    if not args.skip_text:
        packages.append(convert_clip(args.repo, "text_encoder", args.build_dir / "TextEncoder.mlpackage", deployment_target))
        packages.append(convert_clip(args.repo, "text_encoder_2", args.build_dir / "TextEncoder2.mlpackage", deployment_target))
        copy_tokenizer(args.repo, args.compile_into)
    if not args.skip_vae:
        packages.append(convert_vae(args.repo, args.build_dir / "VAEDecoder.mlpackage", args.latent_h, args.latent_w, deployment_target))

    if not args.skip_compile:
        for package in packages:
            LOG.info("Compiling %s into %s", package.name, args.compile_into)
            target = compile_mlpackage(package, args.compile_into, args.ios_deployment_target)
            LOG.info("Installed %s", target)

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
