#!/usr/bin/env python3
"""FLUX.2 VAE decoder -> Core ML (fixed 64x64x32 latent tile -> 512x512 RGB), ANE friendly.

The mid-block self-attention (4096 tokens) is rewritten as exact query-chunked attention (no stock SDPA).
Input `latent` is the *denormalised, unpatchified* VAE latent [1,32,64,64]; output `image` is in [-1,1].
Also writes vae_bn.json with the batch-norm mean/std (128 values each) used to denormalise packed latents.

With --check LATENT.npy (packed [1,N,128] from the denoiser) it decodes with 3x3 feathered tiles through
Core ML and compares against the PyTorch full decode.
"""
import argparse, json, math, subprocess, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ChunkedVAEAttn(nn.Module):
    def __init__(self, attn, n_chunks):
        super().__init__()
        self.a, self.n = attn, n_chunks

    def forward(self, x, *args, **kw):
        b, c, h, w = x.shape
        res = x
        t = self.a.group_norm(x).view(b, c, h * w).transpose(1, 2)
        q, k, v = self.a.to_q(t), self.a.to_k(t), self.a.to_v(t)
        kt = k.transpose(1, 2)
        size = math.ceil(q.shape[1] / self.n)
        o = torch.cat([torch.matmul(torch.softmax(torch.matmul(qc * c ** -0.5, kt), dim=-1), v)
                       for qc in q.split(size, dim=1)], dim=1)
        o = self.a.to_out[0](o).transpose(1, 2).reshape(b, c, h, w)
        return o + res


class Dec(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent, return_dict=False)[0]


def unpack(lat, mean, std):
    """packed [1,N,128] -> denormalised, unpatchified [1,32,2s,2s]"""
    n = lat.shape[1]
    s = int(n ** 0.5)
    x = lat.transpose(0, 2, 1).reshape(1, 128, s, s) * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)
    return x.reshape(1, 32, 2, 2, s, s).transpose(0, 1, 4, 2, 5, 3).reshape(1, 32, 2 * s, 2 * s)


def tiled(decode, z, tile=64, stride=32, scale=8):
    """feather-blended tiled decode; z [1,32,H,W] -> [3, H*8, W*8]"""
    H, W = z.shape[2:]
    out = np.zeros((3, H * scale, W * scale), np.float64)
    wsum = np.zeros((1, H * scale, W * scale), np.float64)
    ramp = np.minimum(np.arange(tile * scale) + 0.5, tile * scale - np.arange(tile * scale) - 0.5)
    w2 = np.outer(ramp, ramp)[None]
    for y in range(0, H - tile + 1, stride):
        for x in range(0, W - tile + 1, stride):
            im = decode(np.ascontiguousarray(z[:, :, y:y + tile, x:x + tile], dtype=np.float32))[0]
            out[:, y * scale:(y + tile) * scale, x * scale:(x + tile) * scale] += im * w2
            wsum[:, y * scale:(y + tile) * scale, x * scale:(x + tile) * scale] += w2
    return out / wsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--compile-into", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, default=Path("vae_build"))
    ap.add_argument("--q-chunks", type=int, default=16)
    ap.add_argument("--check", type=Path)
    ap.add_argument("--save-image", type=Path)
    args = ap.parse_args()
    import coremltools as ct
    from diffusers import AutoencoderKLFlux2
    vae = AutoencoderKLFlux2.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32).eval()
    mean = vae.bn.running_mean.numpy().astype(np.float64)
    std = np.sqrt(vae.bn.running_var.numpy().astype(np.float64) + vae.config.batch_norm_eps)
    args.compile_into.mkdir(parents=True, exist_ok=True)
    (args.compile_into / "vae_bn.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}))
    target = args.compile_into / "KleinVAEDecoder.mlmodelc"
    if not target.exists():
        mid = vae.decoder.mid_block
        mid.attentions[0] = ChunkedVAEAttn(mid.attentions[0], args.q_chunks)
        ex = torch.randn(1, 32, 64, 64)
        with torch.no_grad():
            traced = torch.jit.trace(Dec(vae), ex, strict=False)
        ml = ct.convert(traced, inputs=[ct.TensorType(name="latent", shape=ex.shape)],
                        outputs=[ct.TensorType(name="image")], minimum_deployment_target=ct.target.iOS18,
                        compute_precision=ct.precision.FLOAT16, convert_to="mlprogram", skip_model_load=True)
        args.out.mkdir(parents=True, exist_ok=True)
        pkg = args.out / "KleinVAEDecoder.mlpackage"
        if pkg.exists():
            shutil.rmtree(pkg)
        ml.save(str(pkg))
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(args.compile_into)], check=True,
                       stdout=subprocess.DEVNULL)
        shutil.rmtree(pkg)
        print("[vae] converted", flush=True)
    if args.check:
        import time
        from PIL import Image
        m = ct.models.CompiledMLModel(str(target), ct.ComputeUnit.CPU_AND_NE)
        z = unpack(np.load(args.check).astype(np.float64), mean, std)
        ts = []

        def dec(t):
            t0 = time.time()
            r = m.predict({"latent": t})["image"]
            ts.append(time.time() - t0)
            return r
        img = tiled(dec, z)
        print(f"[vae] tiles {len(ts)}  first {ts[0]:.2f}s  avg later {np.mean(ts[1:]):.2f}s  finite={np.isfinite(img).all()}", flush=True)
        vae2 = AutoencoderKLFlux2.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32).eval()
        with torch.no_grad():
            ref = vae2.decode(torch.from_numpy(z).float(), return_dict=False)[0][0].numpy()
        a, b = np.clip(img, -1, 1), np.clip(ref, -1, 1)
        print(f"[vae] vs PyTorch full decode: mean abs diff {np.abs(a - b).mean():.4f} (range 2)  PSNR "
              f"{10 * np.log10(4 / np.mean((a - b) ** 2)):.1f} dB", flush=True)
        if args.save_image:
            Image.fromarray(((a.transpose(1, 2, 0) + 1) * 127.5).round().astype(np.uint8)).save(args.save_image)


if __name__ == "__main__":
    main()
