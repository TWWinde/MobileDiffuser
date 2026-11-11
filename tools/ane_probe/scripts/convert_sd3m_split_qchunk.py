#!/usr/bin/env python3
"""Convert an SD3-Medium (distilled) transformer into split Core ML MMDiT stages with
ANE-friendly query-chunked attention, for any latent size (e.g. 128x128 = 1024px).

Same stage contract as the MCL app / scripts/convert_sd3_medium_split_coreml.py:
  Stage0          : pos_embed + context_embedder
  Stage1..StageN  : groups of transformer blocks; last one adds norm_out/proj_out/unpatchify
  (Conditioning is resolution independent -> reuse the 512 folder's compiled model.)

Differences from the 512 release script:
  * attention = exact attention with queries split into --q-chunks pieces (stock SDPA fails
    ANE compile at 1024 on iPhone 15 Pro and falls back to CPU -> jetsam OOM)
  * optional int8 linear-symmetric weight quantization + compile into a resource folder
  * --random-weights for structure/compile testing without a checkpoint

Example (1024, 4-step):
  python convert_sd3m_split_qchunk.py --ckpt-path sd3-medium-4step.safetensors \
     --latent 128 --stage-sizes 4,4,4,4,4,4 --q-chunks 16 \
     -o build_1024 --int8 --compile-into coremlsd3_4step_1024
"""
from __future__ import annotations

import argparse
import gc
import math
import shutil
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TXT_TOKENS = 154


def chunked_attention(q, k, v, n_chunks: int):
    # q,k,v: [B, H, S, D]; exact softmax attention, queries processed in chunks
    scale = q.shape[-1] ** -0.5
    if n_chunks <= 1:
        return torch.matmul(torch.softmax(torch.matmul(q * scale, k.transpose(-1, -2)), dim=-1), v)
    size = math.ceil(q.shape[2] / n_chunks)
    kt = k.transpose(-1, -2)
    outs = [torch.matmul(torch.softmax(torch.matmul(qc * scale, kt), dim=-1), v) for qc in q.split(size, dim=2)]
    return torch.cat(outs, dim=2)


class QChunkJointProc:
    def __init__(self, n_chunks: int):
        self.n_chunks = n_chunks

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, *a, **kw):
        b, n_img = hidden_states.shape[0], hidden_states.shape[1]
        heads = attn.heads

        def split_heads(x):
            return x.view(b, -1, heads, x.shape[-1] // heads).transpose(1, 2)

        q = split_heads(attn.to_q(hidden_states))
        k = split_heads(attn.to_k(hidden_states))
        v = split_heads(attn.to_v(hidden_states))
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        cq = split_heads(attn.add_q_proj(encoder_hidden_states))
        ck = split_heads(attn.add_k_proj(encoder_hidden_states))
        cv = split_heads(attn.add_v_proj(encoder_hidden_states))
        if attn.norm_added_q is not None:
            cq = attn.norm_added_q(cq)
        if attn.norm_added_k is not None:
            ck = attn.norm_added_k(ck)
        q = torch.cat([q, cq], dim=2)
        k = torch.cat([k, ck], dim=2)
        v = torch.cat([v, cv], dim=2)

        o = chunked_attention(q, k, v, self.n_chunks).transpose(1, 2).reshape(b, -1, q.shape[1] * q.shape[3])
        img, txt = o[:, :n_img], o[:, n_img:]
        if not attn.context_pre_only:
            txt = attn.to_add_out(txt)
        img = attn.to_out[1](attn.to_out[0](img))
        return img, txt


class SD3InputStage(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.pos_embed = transformer.pos_embed
        self.context_embedder = transformer.context_embedder

    def forward(self, latent_image_embeddings, token_level_text_embeddings, modulation_inputs):
        token_level_text_embeddings = token_level_text_embeddings.squeeze(2).transpose(1, 2)
        return self.pos_embed(latent_image_embeddings), self.context_embedder(token_level_text_embeddings)


class SD3BlockStage(nn.Module):
    def __init__(self, transformer, start: int, end: int, latent: int):
        super().__init__()
        self.blocks = nn.ModuleList([transformer.transformer_blocks[i] for i in range(start, end)])
        final = end == transformer.config.num_layers
        self.norm_out = transformer.norm_out if final else None
        self.proj_out = transformer.proj_out if final else None
        self.patch = transformer.config.patch_size
        self.out_channels = transformer.out_channels
        self.latent = latent

    def forward(self, latent_image_embeddings, token_level_text_embeddings, modulation_inputs):
        for block in self.blocks:
            txt, img = block(
                hidden_states=latent_image_embeddings,
                encoder_hidden_states=token_level_text_embeddings,
                temb=modulation_inputs,
            )
            latent_image_embeddings = img
            if txt is not None:
                token_level_text_embeddings = txt
        if self.norm_out is None:
            return latent_image_embeddings, token_level_text_embeddings
        x = self.proj_out(self.norm_out(latent_image_embeddings, modulation_inputs))
        g = self.latent // self.patch
        x = x.reshape(x.shape[0], g, g, self.patch, self.patch, self.out_channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return (x.reshape(x.shape[0], self.out_channels, self.latent, self.latent),)


def convert(module, inputs: dict, output_names: list[str], out_path: Path, int8: bool):
    import coremltools as ct

    module = module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, tuple(inputs.values()), strict=False)
    t0 = time.time()
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name=n, shape=t.shape) for n, t in inputs.items()],
        outputs=[ct.TensorType(name=n) for n in output_names],
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        skip_model_load=True,
    )
    if int8:
        cfg = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
        ml = ct.optimize.coreml.linear_quantize_weights(ml, config=cfg)
    if out_path.exists():
        shutil.rmtree(out_path)
    ml.save(str(out_path))
    print(f"[convert] {out_path.name} {time.time() - t0:.1f}s", flush=True)
    del traced, ml
    gc.collect()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", type=Path)
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--latent", type=int, default=128, help="latent side (128 = 1024px, 64 = 512px)")
    ap.add_argument("--stage-sizes", default="4,4,4,4,4,4")
    ap.add_argument("--q-chunks", type=int, default=16)
    ap.add_argument("--only-stages", default="", help="comma list of stage indices to (re)convert, e.g. 0,1,6")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("-o", "--output-dir", type=Path, required=True)
    ap.add_argument("--compile-into", type=Path, help="xcrun coremlcompiler each package into this folder")
    args = ap.parse_args()

    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
    from safetensors.torch import load_file

    sizes = [int(s) for s in args.stage_sizes.split(",") if s.strip()]
    assert sum(sizes) == 24, f"--stage-sizes must sum to 24, got {sum(sizes)}"
    ranges, s = [], 0
    for n in sizes:
        ranges.append((s, s + n)); s += n
    only = {int(x) for x in args.only_stages.split(",") if x.strip()}

    transformer = SD3Transformer2DModel(
        sample_size=128, patch_size=2, in_channels=16, num_layers=24, attention_head_dim=64,
        num_attention_heads=24, joint_attention_dim=4096, caption_projection_dim=1536,
        pooled_projection_dim=2048, out_channels=16, pos_embed_max_size=192,
        dual_attention_layers=(), qk_norm=None,
    ).float().eval()
    if args.random_weights:
        torch.manual_seed(0)
    else:
        assert args.ckpt_path and args.ckpt_path.exists(), "--ckpt-path required"
        state = load_file(str(args.ckpt_path), device="cpu")
        state = {k: v.float() for k, v in state.items()}
        transformer.load_state_dict(state, strict=True)
        del state; gc.collect()
    for blk in transformer.transformer_blocks:
        blk.attn.set_processor(QChunkJointProc(args.q_chunks))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens = (args.latent // 2) ** 2
    jobs = [(0, SD3InputStage(transformer),
             {"latent_image_embeddings": torch.randn(1, 16, args.latent, args.latent),
              "token_level_text_embeddings": torch.randn(1, 4096, 1, TXT_TOKENS),
              "modulation_inputs": torch.randn(1, 1536)},
             ["latent_image_embeddings_out", "token_level_text_embeddings_out"])]
    for i, (a, b) in enumerate(ranges, start=1):
        outs = ["noise_pred"] if b == 24 else ["latent_image_embeddings_out", "token_level_text_embeddings_out"]
        jobs.append((i, SD3BlockStage(transformer, a, b, args.latent),
                     {"latent_image_embeddings": torch.randn(1, tokens, 1536),
                      "token_level_text_embeddings": torch.randn(1, TXT_TOKENS, 1536),
                      "modulation_inputs": torch.randn(1, 1536)}, outs))

    for idx, module, inputs, outs in jobs:
        if only and idx not in only:
            continue
        pkg = args.output_dir / f"MultiModalDiffusionTransformerStage{idx}.mlpackage"
        convert(module, inputs, outs, pkg, args.int8)
        if args.compile_into:
            args.compile_into.mkdir(parents=True, exist_ok=True)
            dst = args.compile_into / f"MultiModalDiffusionTransformerStage{idx}.mlmodelc"
            if dst.exists():
                shutil.rmtree(dst)
            subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(args.compile_into)],
                           check=True, stdout=subprocess.DEVNULL)
    # remove stale stages beyond this split
    if args.compile_into and not only:
        for p in args.compile_into.glob("MultiModalDiffusionTransformerStage*.mlmodelc"):
            if int(p.stem.removeprefix("MultiModalDiffusionTransformerStage")) > len(ranges):
                shutil.rmtree(p)
    print("[convert] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
