#!/usr/bin/env python3
"""Convert ONE SD3-Medium MMDiT JointTransformerBlock to Core ML for ANE placement probing.

Random weights (op placement does not depend on weight values). Optional int8
linear-symmetric weight quantization to mirror the release pipeline.

Attention variants:
  sdpa     : F.scaled_dot_product_attention (what the 512 release uses via diffusers)
  manual   : explicit QK^T -> softmax -> @V with full score matrix
  qchunkN  : exact attention, queries split into N chunks (scores per chunk: H x T/N x T)
  splitkvN : MCL-style split-KV online-softmax (scores per chunk: H x T x T/N)
"""
import argparse, gc, math, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

H, D, DIM, TXT = 24, 64, 1536, 154


def attn_core(q, k, v, mode: str):
    # q,k,v: [B, H, S, D]
    if mode == "none":  # attention removed: measures linear/MLP/norm cost only
        return v
    if mode == "sdpa":
        return F.scaled_dot_product_attention(q, k, v)
    scale = D ** -0.5
    if mode == "manual":
        w = torch.softmax(torch.matmul(q * scale, k.transpose(-1, -2)), dim=-1)
        return torch.matmul(w, v)
    n_img = q.shape[2] - TXT
    if mode.startswith("kvpool"):
        # ToDo-style: average-pool IMAGE keys/values p x p on the latent token grid; text keys
        # untouched; queries unchanged (chunked n ways). mode = kvpool{p}q{n}
        p, n = (int(x) for x in mode[len("kvpool"):].split("q"))
        b, h, _, d = k.shape
        g = int(math.isqrt(n_img))

        def pool(x):
            img = x[:, :, :n_img].transpose(2, 3).reshape(b, h * d, g, g)
            img = F.avg_pool2d(img, p).reshape(b, h, d, -1).transpose(2, 3)
            return torch.cat([img, x[:, :, n_img:]], dim=2)

        kp, vp = pool(k), pool(v)
        kt = kp.transpose(-1, -2)
        size = math.ceil(q.shape[2] / n)
        return torch.cat([torch.matmul(torch.softmax(torch.matmul(qc * scale, kt), dim=-1), vp)
                          for qc in q.split(size, dim=2)], dim=2)
    if mode.startswith("qalign"):
        # query chunks aligned to the image/text boundary: n equal image chunks + 1 text chunk
        n = int(mode[len("qalign"):])
        kt = k.transpose(-1, -2)
        q_chunks = list(q[:, :, :n_img].split(n_img // n, dim=2)) + [q[:, :, n_img:]]
        return torch.cat([torch.matmul(torch.softmax(torch.matmul(qc * scale, kt), dim=-1), v)
                          for qc in q_chunks], dim=2)
    if mode.startswith("kv2q"):
        # keys split into aligned image (4096) and text (154) parts, exact 2-way online softmax;
        # queries chunked as in qalign
        n = int(mode[len("kv2q"):])
        ki, kt_ = k[:, :, :n_img].transpose(-1, -2), k[:, :, n_img:].transpose(-1, -2)
        vi, vt = v[:, :, :n_img], v[:, :, n_img:]
        outs = []
        for qc in list(q[:, :, :n_img].split(n_img // n, dim=2)) + [q[:, :, n_img:]]:
            qc = qc * scale
            si, st = torch.matmul(qc, ki), torch.matmul(qc, kt_)
            m = torch.maximum(si.max(dim=-1, keepdim=True)[0], st.max(dim=-1, keepdim=True)[0])
            ei, et = (si - m).exp(), (st - m).exp()
            outs.append((torch.matmul(ei, vi) + torch.matmul(et, vt)) /
                        (ei.sum(dim=-1, keepdim=True) + et.sum(dim=-1, keepdim=True)))
        return torch.cat(outs, dim=2)
    if mode.startswith("padq"):
        # keys/values zero-padded to a multiple of 64 with a -1e4 score bias on the pad keys
        n = int(mode[len("padq"):])
        s = k.shape[2]
        pad = (-s) % 64
        kp = F.pad(k, (0, 0, 0, pad)).transpose(-1, -2)
        vp = F.pad(v, (0, 0, 0, pad))
        bias = torch.cat([torch.zeros(s), torch.full((pad,), -1e4)]).to(q.dtype)
        size = math.ceil(q.shape[2] / n)
        return torch.cat([torch.matmul(torch.softmax(torch.matmul(qc * scale, kp) + bias, dim=-1), vp)
                          for qc in q.split(size, dim=2)], dim=2)
    if mode.startswith("qchunk"):
        n = int(mode[len("qchunk"):])
        size = math.ceil(q.shape[2] / n)
        outs = []
        for qc in q.split(size, dim=2):
            w = torch.softmax(torch.matmul(qc * scale, k.transpose(-1, -2)), dim=-1)
            outs.append(torch.matmul(w, v))
        return torch.cat(outs, dim=2)
    if mode.startswith("splitkv"):
        n = int(mode[len("splitkv"):])
        size = math.ceil(k.shape[2] / n)
        qs = q * scale
        maxes, sums, vals = [], [], []
        for kc, vc in zip(k.split(size, dim=2), v.split(size, dim=2)):
            w = torch.matmul(qs, kc.transpose(-1, -2))
            m = w.max(dim=-1, keepdim=True)[0]
            e = (w - m).exp()
            maxes.append(m); sums.append(e.sum(dim=-1, keepdim=True)); vals.append(torch.matmul(e, vc))
        g = torch.cat(maxes, dim=-1).max(dim=-1, keepdim=True)[0]
        acc_v = acc_s = None
        for m, s, val in zip(maxes, sums, vals):
            c = (m - g).exp()
            acc_v = val * c if acc_v is None else acc_v + val * c
            acc_s = s * c if acc_s is None else acc_s + s * c
        return acc_v / acc_s
    raise ValueError(mode)


class JointProc:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, *a, **kw):
        b, n_img = hidden_states.shape[0], hidden_states.shape[1]

        def heads(x):
            return x.view(b, -1, H, D).transpose(1, 2)

        q = torch.cat([heads(attn.to_q(hidden_states)), heads(attn.add_q_proj(encoder_hidden_states))], dim=2)
        k = torch.cat([heads(attn.to_k(hidden_states)), heads(attn.add_k_proj(encoder_hidden_states))], dim=2)
        v = torch.cat([heads(attn.to_v(hidden_states)), heads(attn.add_v_proj(encoder_hidden_states))], dim=2)
        o = attn_core(q, k, v, self.mode).transpose(1, 2).reshape(b, -1, H * D)
        img, txt = o[:, :n_img], o[:, n_img:]
        if not attn.context_pre_only:
            txt = attn.to_add_out(txt)
        return attn.to_out[0](img), txt


class OneBlock(nn.Module):
    def __init__(self, mode):
        super().__init__()
        from diffusers.models.attention import JointTransformerBlock
        self.block = JointTransformerBlock(dim=DIM, num_attention_heads=H, attention_head_dim=D)
        self.block.attn.set_processor(JointProc(mode))

    def forward(self, latent_image_embeddings, token_level_text_embeddings, modulation_inputs):
        txt, img = self.block(latent_image_embeddings, token_level_text_embeddings, modulation_inputs)
        return img, txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import coremltools as ct
    torch.manual_seed(0)
    tokens = (args.res // 16) ** 2
    model = OneBlock(args.mode).eval().float()
    ex = (torch.randn(1, tokens, DIM), torch.randn(1, TXT, DIM), torch.randn(1, DIM))
    with torch.no_grad():
        traced = torch.jit.trace(model, ex, strict=False)
    t0 = time.time()
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latent_image_embeddings", shape=ex[0].shape),
                ct.TensorType(name="token_level_text_embeddings", shape=ex[1].shape),
                ct.TensorType(name="modulation_inputs", shape=ex[2].shape)],
        outputs=[ct.TensorType(name="latent_image_embeddings_out"),
                 ct.TensorType(name="token_level_text_embeddings_out")],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        skip_model_load=True,
    )
    if args.int8:
        cfg = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
        ml = ct.optimize.coreml.linear_quantize_weights(ml, config=cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ml.save(str(args.out))
    print(f"saved {args.out} tokens={tokens} mode={args.mode} int8={args.int8} in {time.time()-t0:.1f}s")
    del ml, traced; gc.collect()


if __name__ == "__main__":
    main()
