#!/usr/bin/env python3
"""Build 1024 single-block speed-probe variants (random weights, int8 weights):
  ane_q{N}        ANE layout (B,C,1,S), conv1x1, query chunks N
  ane_sh_q{N}     same + per-head split einsum
  *_w8a8          + int8 activation quantization (calibrated on random inputs; speed only)
  base_q16_w8a8   original (B,S,C) layout qchunk16 + W8A8
Usage: convert_probe_variants.py OUT_DIR variant [variant ...]
"""
import subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from ane_block import ANEJointBlock  # noqa: E402

TOK, TXT, C = 4096, 154, 1536


def build(variant: str):
    from diffusers.models.attention import JointTransformerBlock
    torch.manual_seed(0)
    ref = JointTransformerBlock(dim=C, num_attention_heads=24, attention_head_dim=64).eval()
    name = variant.removesuffix("_w8a8")
    if name.startswith("ane"):
        split = name.startswith("ane_sh")
        q = int(name.rsplit("_q", 1)[1])
        mod = ANEJointBlock(ref, q_chunks=q, split_heads=split).eval()
        ex = (torch.randn(1, C, 1, TOK), torch.randn(1, C, 1, TXT), torch.randn(1, C))
    elif name == "base_q16":
        from convert_block import OneBlock
        mod = OneBlock("qchunk16").eval()
        ex = (torch.randn(1, TOK, C), torch.randn(1, TXT, C), torch.randn(1, C))
    else:
        raise ValueError(variant)
    return mod, ex


def main():
    import coremltools as ct
    import coremltools.optimize.coreml as cto
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for variant in sys.argv[2:]:
        t0 = time.time()
        mod, ex = build(variant)
        names = ["latent_image_embeddings", "token_level_text_embeddings", "modulation_inputs"]
        with torch.no_grad():
            traced = torch.jit.trace(mod, ex, strict=False)
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name=n, shape=t.shape) for n, t in zip(names, ex)],
            outputs=[ct.TensorType(name="latent_image_embeddings_out"),
                     ct.TensorType(name="token_level_text_embeddings_out")],
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16,
            convert_to="mlprogram",
        )
        if variant.endswith("_w8a8"):
            rng = np.random.default_rng(0)
            samples = [{n: rng.standard_normal(t.shape).astype(np.float32) for n, t in zip(names, ex)}
                       for _ in range(4)]
            acfg = cto.OptimizationConfig(global_config=cto.experimental.OpActivationLinearQuantizerConfig(
                mode="linear_symmetric"))
            ml = cto.experimental.linear_quantize_activations(ml, acfg, samples)
        wcfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
        ml = cto.linear_quantize_weights(ml, config=wcfg)
        pkg = out / f"r1024_{variant}.mlpackage"
        ml.save(str(pkg))
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(out)], check=True, stdout=subprocess.DEVNULL)
        print(f"[variant] {variant} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
