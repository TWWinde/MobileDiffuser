#!/usr/bin/env python3
"""ANE-native-layout SD3 MMDiT joint block.

Activations live in Apple's ANE-preferred layout (B, C, 1, S); every nn.Linear becomes a 1x1
Conv2d, LayerNorm reduces over the channel axis, and attention is computed per query chunk
(and optionally per head, Apple ml-ane-transformers "split einsum" style).

Built from a diffusers JointTransformerBlock and numerically equivalent to it (fp32).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_from_linear(lin: nn.Linear) -> nn.Conv2d:
    conv = nn.Conv2d(lin.in_features, lin.out_features, 1, bias=lin.bias is not None)
    with torch.no_grad():
        conv.weight.copy_(lin.weight[:, :, None, None])
        if lin.bias is not None:
            conv.bias.copy_(lin.bias)
    return conv


def layer_norm_c(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: (B, C, 1, S); normalize over C, no affine
    mean = x.mean(dim=1, keepdim=True)
    xc = x - mean
    var = (xc * xc).mean(dim=1, keepdim=True)
    return xc * torch.rsqrt(var + eps)


class ANEAttention(nn.Module):
    def __init__(self, attn, heads: int, q_chunks: int, split_heads: bool):
        super().__init__()
        self.heads, self.q_chunks, self.split_heads = heads, q_chunks, split_heads
        self.to_q, self.to_k, self.to_v = (conv_from_linear(attn.to_q), conv_from_linear(attn.to_k),
                                           conv_from_linear(attn.to_v))
        self.add_q, self.add_k, self.add_v = (conv_from_linear(attn.add_q_proj), conv_from_linear(attn.add_k_proj),
                                              conv_from_linear(attn.add_v_proj))
        self.to_out = conv_from_linear(attn.to_out[0])
        self.to_add_out = None if attn.context_pre_only else conv_from_linear(attn.to_add_out)

    def _attend(self, q, k, v):
        # q,k,v: (B, C, 1, S)
        b, c, _, s = q.shape
        h = self.heads
        d = c // h
        scale = d ** -0.5
        q = q * scale
        chunk = math.ceil(s / self.q_chunks)
        if not self.split_heads:
            qh, kh, vh = (t.reshape(b, h, d, s) for t in (q, k, v))
            outs = []
            for qc in qh.split(chunk, dim=3):
                w = torch.softmax(torch.einsum("bhdq,bhdk->bhqk", qc, kh), dim=-1)
                outs.append(torch.einsum("bhqk,bhdk->bhdq", w, vh))
            return torch.cat(outs, dim=3).reshape(b, c, 1, s)
        head_outs = []
        for qh, kh, vh in zip(q.split(d, dim=1), k.split(d, dim=1), v.split(d, dim=1)):
            # (B, d, 1, S) per head
            outs = []
            for qc in qh.split(chunk, dim=3):
                w = torch.softmax(torch.einsum("bdiq,bdik->biqk", qc, kh), dim=-1)  # (B,1,q,k)
                outs.append(torch.einsum("biqk,bdik->bdiq", w, vh))
            head_outs.append(torch.cat(outs, dim=3))
        return torch.cat(head_outs, dim=1)

    def forward(self, x, cx):
        n_img = x.shape[3]
        q = torch.cat([self.to_q(x), self.add_q(cx)], dim=3)
        k = torch.cat([self.to_k(x), self.add_k(cx)], dim=3)
        v = torch.cat([self.to_v(x), self.add_v(cx)], dim=3)
        o = self._attend(q, k, v)
        img, txt = o[:, :, :, :n_img], o[:, :, :, n_img:]
        img = self.to_out(img)
        txt = self.to_add_out(txt) if self.to_add_out is not None else None
        return img, txt


class ANEFeedForward(nn.Module):
    def __init__(self, ff):
        super().__init__()
        self.fc1 = conv_from_linear(ff.net[0].proj)
        self.fc2 = conv_from_linear(ff.net[2])

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class ANEJointBlock(nn.Module):
    """Equivalent of diffusers JointTransformerBlock (context_pre_only=False only)."""

    def __init__(self, block, heads: int = 24, q_chunks: int = 16, split_heads: bool = False):
        super().__init__()
        assert not block.context_pre_only, "final block not supported in the probe"
        self.mod = block.norm1.linear          # temb -> 6C (image)
        self.cmod = block.norm1_context.linear  # temb -> 6C (text)
        self.attn = ANEAttention(block.attn, heads, q_chunks, split_heads)
        self.ff = ANEFeedForward(block.ff)
        self.ff_c = ANEFeedForward(block.ff_context)

    @staticmethod
    def _chunks(lin, temb):
        m = lin(F.silu(temb))[:, :, None, None]  # (B, 6C, 1, 1)
        return m.chunk(6, dim=1)

    def forward(self, x, cx, temb):
        # x: (B,C,1,Si) image tokens, cx: (B,C,1,St) text tokens, temb: (B,C)
        sh, sc, g, sh2, sc2, g2 = self._chunks(self.mod, temb)
        csh, csc, cg, csh2, csc2, cg2 = self._chunks(self.cmod, temb)
        xn = layer_norm_c(x) * (1 + sc) + sh
        cn = layer_norm_c(cx) * (1 + csc) + csh
        a, ca = self.attn(xn, cn)
        x = x + g * a
        x = x + g2 * self.ff(layer_norm_c(x) * (1 + sc2) + sh2)
        cx = cx + cg * ca
        cx = cx + cg2 * self.ff_c(layer_norm_c(cx) * (1 + csc2) + csh2)
        return x, cx


if __name__ == "__main__":
    # Equivalence check vs diffusers (fp32, random weights)
    from diffusers.models.attention import JointTransformerBlock

    torch.manual_seed(0)
    ref = JointTransformerBlock(dim=1536, num_attention_heads=24, attention_head_dim=64).eval()
    for split in (False, True):
        ane = ANEJointBlock(ref, q_chunks=4, split_heads=split).eval()
        x, c, t = torch.randn(1, 256, 1536), torch.randn(1, 154, 1536), torch.randn(1, 1536)
        with torch.no_grad():
            rc, rx = ref(x, c, t)
            ax, ac = ane(x.transpose(1, 2)[:, :, None], c.transpose(1, 2)[:, :, None], t)
        ax, ac = ax[:, :, 0].transpose(1, 2), ac[:, :, 0].transpose(1, 2)
        rel = lambda a, b: ((a - b).norm() / b.norm()).item()
        print(f"split_heads={split}: img rel={rel(ax, rx):.2e} txt rel={rel(ac, rc):.2e}")
