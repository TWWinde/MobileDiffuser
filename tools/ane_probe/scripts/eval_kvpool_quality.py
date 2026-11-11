#!/usr/bin/env python3
"""Quality comparison of ToDo-style image-KV pooling for the distilled 4-step SD3-Medium at 1024.

Pipeline mirrors the app: Core ML CLIP-L/G text encoders (no T5, zeros), PyTorch MMDiT (MPS, fp32),
flow-matching 4-step schedule (shift 3), Core ML 512 VAE decoder with the same 3x3 tiled
feather-blend decode the app uses. Variants only change the MMDiT attention.

Outputs: <out>/<variant>/<idx>.png, metrics.csv (LPIPS/PSNR vs exact attention, CLIP score), grid.jpg
"""
import argparse, csv, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROMPTS = [
    "A photo of a red fox in a snowy forest, highly detailed",
    "Close-up portrait of an elderly woman with freckles, soft window light, 85mm photo",
    "A neon sign that says \"OPEN 24 HOURS\" on a rainy city street at night",
    "Aerial view of a winding river through autumn mountains, golden hour",
    "A modern glass skyscraper with a geometric facade, architectural photography",
    "A bowl of ramen with a soft-boiled egg, chopsticks, food photography",
]
TXT = 154

VARIANTS = {  # name -> (pool size, set of pooled blocks, mass correction)
    "exact": (1, set(), False),
    "kv2_all": (2, set(range(24)), False),
    "kv4_all": (4, set(range(24)), False),
    "kv2_first12": (2, set(range(12)), False),
    "kv2_last12": (2, set(range(12, 24)), False),
    # mass-corrected: each pooled image key stands for p*p original keys -> +log(p*p) logit bias
    "kv2c_all": (2, set(range(24)), True),
    "kv4c_all": (4, set(range(24)), True),
    "kv2c_first12": (2, set(range(12)), True),
}


class PoolProc:
    def __init__(self):
        self.pool = 1
        self.correct = False

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, *a, **kw):
        b, n_img = hidden_states.shape[0], hidden_states.shape[1]
        h = attn.heads

        def heads(x):
            return x.view(b, -1, h, x.shape[-1] // h).transpose(1, 2)

        q = torch.cat([heads(attn.to_q(hidden_states)), heads(attn.add_q_proj(encoder_hidden_states))], 2)
        k = torch.cat([heads(attn.to_k(hidden_states)), heads(attn.add_k_proj(encoder_hidden_states))], 2)
        v = torch.cat([heads(attn.to_v(hidden_states)), heads(attn.add_v_proj(encoder_hidden_states))], 2)
        if self.pool > 1:
            g = int(math.isqrt(n_img))
            d = k.shape[-1]

            def pool(x):
                img = x[:, :, :n_img].transpose(2, 3).reshape(b, h * d, g, g)
                img = F.avg_pool2d(img, self.pool).reshape(b, h, d, -1).transpose(2, 3)
                return torch.cat([img, x[:, :, n_img:]], 2)

            k, v = pool(k), pool(v)
        mask = None
        if self.pool > 1 and self.correct:
            n_pooled = k.shape[2] - (q.shape[2] - n_img)
            mask = torch.zeros(k.shape[2], device=q.device, dtype=q.dtype)
            mask[:n_pooled] = math.log(self.pool * self.pool)
            mask = mask.view(1, 1, 1, -1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(b, -1, h * q.shape[-1])
        img, txt = o[:, :n_img], o[:, n_img:]
        if not attn.context_pre_only:
            txt = attn.to_add_out(txt)
        return attn.to_out[0](img), txt


def encode_prompt(prompt, res_dir):
    import coremltools as ct
    from transformers import CLIPTokenizer
    outs = []
    for model, pad in (("TextEncoder", "<|endoftext|>"), ("TextEncoder2", "!")):
        tok = CLIPTokenizer(f"{res_dir}/vocab.json", f"{res_dir}/merges.txt", pad_token=pad)
        ids = tok(prompt, padding="max_length", max_length=77, truncation=True)["input_ids"]
        m = ct.models.CompiledMLModel(f"{res_dir}/{model}.mlmodelc", compute_units=ct.ComputeUnit.CPU_ONLY)
        r = m.predict({"input_ids": np.array([ids], dtype=np.float32)})
        outs.append((r["hidden_embeds"], r["pooled_outputs"]))
    (h1, p1), (h2, p2) = outs
    h = np.concatenate([h1, h2, np.zeros((1, 77, 2048), np.float32)], 2)
    h = np.concatenate([h, np.zeros((1, 77, 4096), np.float32)], 1)
    return torch.from_numpy(h), torch.from_numpy(np.concatenate([p1, p2], 1))


def tiled_decode(latent, vae, tile=64):
    # latent: (16, H, W) numpy, already in model scale; mirrors Decoder.decodeTiled in Swift
    c, H, W = latent.shape

    def starts(n):
        if n <= tile:
            return [0]
        cnt = math.ceil((n - tile) / (tile // 2)) + 1
        return [round(i * (n - tile) / (cnt - 1)) for i in range(cnt)]

    ys, xs = starts(H), starts(W)
    z = latent / 1.5305 + 0.0609
    acc = ws = None
    for iy, y0 in enumerate(ys):
        for ix, x0 in enumerate(xs):
            out = vae.predict({"z": z[None, :, y0:y0 + tile, x0:x0 + tile].astype(np.float16)})["image"][0].astype(np.float32)
            ph = out.shape[1]; s = ph // tile
            if acc is None:
                acc = np.zeros((3, H * s, W * s), np.float32); ws = np.zeros((H * s, W * s), np.float32)
            ramp_y = (tile - (ys[1] - ys[0])) * s if len(ys) > 1 else 1
            ramp_x = (tile - (xs[1] - xs[0])) * s if len(xs) > 1 else 1

            def w1(n, ramp, lo, hi):
                x = np.arange(n, dtype=np.float32); w = np.ones(n, np.float32)
                if lo: w = np.minimum(w, (x + 0.5) / ramp)
                if hi: w = np.minimum(w, (n - 1 - x + 0.5) / ramp)
                return np.maximum(w, 1e-4)

            w = np.outer(w1(ph, ramp_y, iy > 0, iy < len(ys) - 1), w1(ph, ramp_x, ix > 0, ix < len(xs) - 1))
            acc[:, y0 * s:y0 * s + ph, x0 * s:x0 * s + ph] += out * w
            ws[y0 * s:y0 * s + ph, x0 * s:x0 * s + ph] += w
    img = np.clip((acc / ws + 1) / 2, 0, 1)
    return Image.fromarray((img.transpose(1, 2, 0) * 255).round().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--res-dir", required=True, help="Core ML resource folder (text encoders, VAE, tokenizer)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import coremltools as ct
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
    from safetensors.torch import load_file

    dev = torch.device("mps")
    tr = SD3Transformer2DModel(
        sample_size=128, patch_size=2, in_channels=16, num_layers=24, attention_head_dim=64,
        num_attention_heads=24, joint_attention_dim=4096, caption_projection_dim=1536,
        pooled_projection_dim=2048, out_channels=16, pos_embed_max_size=192,
        dual_attention_layers=(), qk_norm=None).float().eval()
    tr.load_state_dict({k: v.float() for k, v in load_file(args.ckpt_path).items()}, strict=True)
    tr.to(dev)
    procs = [PoolProc() for _ in tr.transformer_blocks]
    for blk, p in zip(tr.transformer_blocks, procs):
        blk.attn.set_processor(p)
    vae = ct.models.CompiledMLModel(f"{args.res_dir}/VAEDecoder.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE)

    sigmas = [1.0, 0.9, 0.75, 0.5, 0.0]
    conds = [encode_prompt(p, args.res_dir) for p in PROMPTS]
    for name, (pool, blocks, correct) in VARIANTS.items():
        (args.out / name).mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(procs):
            p.pool = pool if i in blocks else 1
            p.correct = correct
        for idx, (txt, pooled) in enumerate(conds):
            if (args.out / name / f"{idx}.png").exists():
                continue
            g = torch.Generator().manual_seed(args.seed + idx)
            x = torch.randn(1, 16, args.latent, args.latent, generator=g).to(dev)
            txt_d, pooled_d = txt.to(dev), pooled.to(dev)
            with torch.no_grad():
                for s, s_next in zip(sigmas[:-1], sigmas[1:]):
                    v = tr(hidden_states=x, encoder_hidden_states=txt_d, pooled_projections=pooled_d,
                           timestep=torch.tensor([s * 1000.0], device=dev)).sample
                    x = x + (s_next - s) * v
            tiled_decode(x[0].float().cpu().numpy(), vae).save(args.out / name / f"{idx}.png")
            print(f"[eval] {name} prompt{idx} done", flush=True)

    # Metrics vs exact
    import lpips
    from transformers import CLIPModel, CLIPProcessor
    lp = lpips.LPIPS(net="alex", verbose=False)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def to_t(im):
        return torch.from_numpy(np.asarray(im, np.float32) / 127.5 - 1).permute(2, 0, 1)[None]

    rows = []
    for name in VARIANTS:
        for idx, prompt in enumerate(PROMPTS):
            im = Image.open(args.out / name / f"{idx}.png").convert("RGB")
            ref = Image.open(args.out / "exact" / f"{idx}.png").convert("RGB")
            a, b = np.asarray(im, np.float32), np.asarray(ref, np.float32)
            mse = ((a - b) ** 2).mean()
            psnr = 99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse)
            with torch.no_grad():
                d = lp(to_t(im), to_t(ref)).item()
                inp = proc(text=[prompt], images=im, return_tensors="pt", padding=True)
                cs = clip(**inp).logits_per_image.item()
            rows.append((name, idx, d, psnr, cs))
    with open(args.out / "metrics.csv", "w", newline="") as f:
        csv.writer(f).writerows([("variant", "prompt", "lpips_vs_exact", "psnr_vs_exact", "clip_score")] + rows)
    print("\nvariant        LPIPS↓  PSNR↑  CLIP")
    for name in VARIANTS:
        r = [x for x in rows if x[0] == name]
        print(f"{name:13s} {np.mean([x[2] for x in r]):.3f}  {np.mean([x[3] for x in r]):5.1f}  {np.mean([x[4] for x in r]):.2f}")

    # Grid: rows = prompts, cols = variants (downscaled to 384)
    cell = 384
    grid = Image.new("RGB", (cell * len(VARIANTS), cell * len(PROMPTS)), "white")
    for c, name in enumerate(VARIANTS):
        for r in range(len(PROMPTS)):
            grid.paste(Image.open(args.out / name / f"{r}.png").resize((cell, cell)), (c * cell, r * cell))
    grid.save(args.out / "grid.jpg", quality=90)


if __name__ == "__main__":
    main()
