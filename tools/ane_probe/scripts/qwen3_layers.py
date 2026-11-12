"""Minimal, trace-friendly Qwen3 decoder layers for the FLUX.2 Klein text encoder (hidden states after
layers 9/18/27, concatenated -> [T, 7680]). Weights are read per layer from the sharded safetensors."""
import json, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm that stays inside fp16: Qwen3 carries ~1.6e4 activations on the first token, whose squares
    overflow half precision. Dividing by the per-token max |x| first keeps every intermediate <= 1."""
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps, self.dim = eps, dim

    def forward(self, x):
        m = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-3)
        y = x / m
        return y * torch.rsqrt((y * y).mean(dim=-1, keepdim=True) + 1e-6) * self.weight


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, self.h, self.kv, self.hd = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.q_proj = nn.Linear(d, self.h * self.hd, bias=False)
        self.k_proj = nn.Linear(d, self.kv * self.hd, bias=False)
        self.v_proj = nn.Linear(d, self.kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.h * self.hd, d, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.hd, c["rms_norm_eps"]), RMSNorm(self.hd, c["rms_norm_eps"])

    @staticmethod
    def rope(x, cos, sin):
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin

    def forward(self, x, mask, cos, sin):
        T = x.shape[1]
        q = self.q_norm(self.q_proj(x).view(1, T, self.h, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(1, T, self.kv, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(1, T, self.kv, self.hd).transpose(1, 2)
        q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
        rep = self.h // self.kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        a = torch.softmax(torch.matmul(q * self.hd ** -0.5, k.transpose(-1, -2)) + mask, dim=-1)
        return self.o_proj(torch.matmul(a, v).transpose(1, 2).reshape(1, T, self.h * self.hd))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, m = c["hidden_size"], c["intermediate_size"]
        self.gate_proj, self.up_proj, self.down_proj = (nn.Linear(d, m, bias=False), nn.Linear(d, m, bias=False),
                                                        nn.Linear(m, d, bias=False))

    def forward(self, x):
        # Layer 6 writes a ~1.6e4 value on the first token; on ANE the fp16 accumulation inside down_proj
        # overflows. Scaling the product down and the result back up keeps partial sums in range.
        return self.down_proj(F.silu(self.gate_proj(x)) * (self.up_proj(x) * 0.03125)) * 32.0


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn, self.mlp = Attn(c), MLP(c)
        self.input_layernorm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])

    def forward(self, x, mask, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), mask, cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Stage(nn.Module):
    def __init__(self, c, n):
        super().__init__()
        self.layers = nn.ModuleList([Layer(c) for _ in range(n)])

    def forward(self, x, mask, cos, sin):
        for l in self.layers:
            x = l(x, mask, cos, sin)
        return x


class Weights:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.cfg = json.loads((self.folder / "config.json").read_text())
        self.index = json.loads((self.folder / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, key):
        from safetensors import safe_open
        with safe_open(str(self.folder / self.index[key]), "pt") as f:
            return f.get_tensor(key).float()

    def stage(self, first, n):
        m = Stage(self.cfg, n)
        sd = {}
        for j in range(n):
            pre = f"model.layers.{first + j}."
            for k in self.index:
                if k.startswith(pre):
                    sd[f"layers.{j}." + k[len(pre):]] = self.get(k)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing[:4], unexpected[:4])
        return m.eval()


def rope_tables(cfg, T):
    hd = cfg["head_dim"]
    inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    f = torch.outer(torch.arange(T, dtype=torch.float64), inv)
    e = torch.cat([f, f], dim=-1)
    return e.cos().float(), e.sin().float()


def attn_mask(n_real, T, neg=-1e4):
    """Additive mask [1,1,T,T]: causal, and padded keys (index >= n_real) are masked."""
    m = torch.full((T, T), neg)
    m = torch.triu(m, diagonal=1)
    m[:, n_real:] = neg
    return m[None, None]


TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
