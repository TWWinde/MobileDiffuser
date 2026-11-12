#!/usr/bin/env python3
"""FLUX.2 Klein (4B) single-block ANE probe: one double-stream block or one single-stream block,
random weights, query-chunked attention (no stock SDPA op), RoPE cos/sin as model inputs,
int8 weights. Klein-4B: dim 3072 (24 heads x 128), 5 double + 20 single blocks, text 512 tokens.

Usage: flux2_probe.py OUT_DIR --kind double|single --res 512|1024 --q-chunks N [--int8] [--check]
"""
import argparse, math, subprocess, time
from pathlib import Path

import torch
import torch.nn as nn

DIM, HEADS, HEAD_DIM, TXT = 3072, 24, 128, 512


def chunked_attention(q, k, v, n_chunks):
    # q,k,v: [B, S, H, D] (diffusers layout) -> [B, S, H*D]; n_chunks == 0 -> attention removed
    if n_chunks == 0:
        return v.flatten(2, 3)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, S, D]
    scale = q.shape[-1] ** -0.5
    kt = k.transpose(-1, -2)
    size = math.ceil(q.shape[2] / n_chunks)
    o = torch.cat([torch.matmul(torch.softmax(torch.matmul(qc * scale, kt), dim=-1), v)
                   for qc in q.split(size, dim=2)], dim=2)
    return o.transpose(1, 2).flatten(2, 3)


class DoubleProc:
    def __init__(self, n):
        self.n = n

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        from diffusers.models.embeddings import apply_rotary_emb
        q = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1)))
        k = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1)))
        v = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        eq = attn.norm_added_q(attn.add_q_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1)))
        ek = attn.norm_added_k(attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1)))
        ev = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        q, k, v = torch.cat([eq, q], 1), torch.cat([ek, k], 1), torch.cat([ev, v], 1)
        q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
        k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
        o = chunked_attention(q, k, v, self.n)
        t = encoder_hidden_states.shape[1]
        txt, img = o[:, :t], o[:, t:]
        return attn.to_out[0](img), attn.to_add_out(txt)


class SingleProc:
    def __init__(self, n):
        self.n = n

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        from diffusers.models.embeddings import apply_rotary_emb
        h = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp = torch.split(h, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1)
        q, k, v = (t.unflatten(-1, (attn.heads, -1)) for t in qkv.chunk(3, dim=-1))
        q, k = attn.norm_q(q), attn.norm_k(k)
        q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
        k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
        o = chunked_attention(q, k, v, self.n)
        return attn.to_out(torch.cat([o, attn.mlp_act_fn(mlp)], dim=-1))


class DoubleProbe(nn.Module):
    def __init__(self, n):
        super().__init__()
        from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock
        self.block = Flux2TransformerBlock(DIM, HEADS, HEAD_DIM, mlp_ratio=3.0)
        self.block.attn.set_processor(DoubleProc(n))

    def forward(self, img, txt, mod_img, mod_txt, cos, sin):
        txt, img = self.block(img, txt, mod_img, mod_txt, image_rotary_emb=(cos, sin))
        return img, txt


class SingleProbe(nn.Module):
    def __init__(self, n):
        super().__init__()
        from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock
        self.block = Flux2SingleTransformerBlock(DIM, HEADS, HEAD_DIM, mlp_ratio=3.0)
        self.block.attn.set_processor(SingleProc(n))

    def forward(self, hs, mod, cos, sin):
        return self.block(hs, None, mod, image_rotary_emb=(cos, sin))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--kind", choices=["double", "single"], required=True)
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--q-chunks", type=int, required=True)
    ap.add_argument("--txt", type=int, default=512, help="text tokens (Klein default 512)")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--check", action="store_true", help="compare traced module vs stock diffusers block (fp32)")
    args = ap.parse_args()

    import coremltools as ct
    import coremltools.optimize.coreml as cto
    torch.manual_seed(0)
    global TXT
    TXT = args.txt
    n_img = (args.res // 16) ** 2
    s = n_img + TXT
    cos, sin = torch.randn(s, HEAD_DIM), torch.randn(s, HEAD_DIM)
    if args.kind == "double":
        mod = DoubleProbe(args.q_chunks).eval()
        ex = (torch.randn(1, n_img, DIM), torch.randn(1, TXT, DIM), torch.randn(1, 1, 6 * DIM) * 0.1,
              torch.randn(1, 1, 6 * DIM) * 0.1, cos, sin)
        names = ["img", "txt", "mod_img", "mod_txt", "rope_cos", "rope_sin"]
        outs = ["img_out", "txt_out"]
    else:
        mod = SingleProbe(args.q_chunks).eval()
        ex = (torch.randn(1, s, DIM), torch.randn(1, 1, 3 * DIM) * 0.1, cos, sin)
        names = ["hidden", "mod", "rope_cos", "rope_sin"]
        outs = ["hidden_out"]

    if args.check and args.q_chunks > 0:
        from diffusers.models.transformers.transformer_flux2 import (Flux2AttnProcessor,
                                                                     Flux2ParallelSelfAttnProcessor)
        with torch.no_grad():
            a = mod(*ex)
            mod.block.attn.set_processor(Flux2AttnProcessor() if args.kind == "double" else Flux2ParallelSelfAttnProcessor())
            b = mod(*ex)
            mod.block.attn.set_processor(DoubleProc(args.q_chunks) if args.kind == "double" else SingleProc(args.q_chunks))
        a, b = (a if isinstance(a, tuple) else (a,)), (b if isinstance(b, tuple) else (b,))
        for x, y in zip(a, b):
            print(f"[check] rel={((x - y).norm() / y.norm()).item():.2e}")

    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(mod, ex, strict=False)
    ml = ct.convert(traced, inputs=[ct.TensorType(name=n, shape=t.shape) for n, t in zip(names, ex)],
                    outputs=[ct.TensorType(name=o) for o in outs], minimum_deployment_target=ct.target.iOS18,
                    compute_precision=ct.precision.FLOAT16, convert_to="mlprogram", skip_model_load=True)
    if args.int8:
        ml = cto.linear_quantize_weights(ml, config=cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")))
    name = (f"flux2_{args.kind}_r{args.res}_t{args.txt}_" + ("noattn" if args.q_chunks == 0 else f"q{args.q_chunks}")
            + ("_int8" if args.int8 else ""))
    args.out.mkdir(parents=True, exist_ok=True)
    pkg = args.out / f"{name}.mlpackage"
    ml.save(str(pkg))
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(args.out)], check=True, stdout=subprocess.DEVNULL)
    print(f"[flux2] {name} tokens={s} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
