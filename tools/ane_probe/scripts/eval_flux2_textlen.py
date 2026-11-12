#!/usr/bin/env python3
"""FLUX.2 Klein 4B: image quality vs text sequence length (max_sequence_length 512 / 256 / 128).

Shorter text sequences make every transformer block cheaper on ANE (static shapes). This checks
how much the output changes. Two phases to fit 24 GB unified memory: (1) Qwen3 encodes all prompts
at every length, (2) transformer + VAE generate with the cached embeddings. 4 steps, 512x512, bf16 MPS.

Outputs: <out>/t<L>/<idx>.png, metrics.csv (LPIPS vs t512, CLIP score), grid.jpg
"""
import argparse, csv, gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROMPTS = [
    "A photo of a red fox in a snowy forest, highly detailed",
    "Close-up portrait of an elderly woman with freckles, soft window light, 85mm photo",
    "A neon sign that says \"OPEN 24 HOURS\" on a rainy city street at night",
    "Aerial view of a winding river through autumn mountains, golden hour",
    "A modern glass skyscraper with a geometric facade, architectural photography",
    "A bowl of ramen with a soft-boiled egg, chopsticks, food photography",
    # a long prompt (~100+ tokens) to see truncation effects at 128
    "A cozy reading nook inside an old stone cottage: a deep green velvet armchair with a knitted "
    "blanket, a stack of leather-bound books on a wooden side table, a steaming cup of tea, a sleeping "
    "orange tabby cat, warm light from a brass floor lamp, rain on the leaded window, a fireplace "
    "with glowing embers, potted ferns, a worn Persian rug, dust motes in the light, cinematic, "
    "highly detailed, 35mm film photograph",
]
LENGTHS = [512, 256, 128]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    from diffusers import Flux2KleinPipeline

    dev = torch.device("mps")
    dt = torch.bfloat16
    emb_path = args.out / "embeds.pt"
    args.out.mkdir(parents=True, exist_ok=True)

    if not emb_path.exists():
        pipe = Flux2KleinPipeline.from_pretrained(args.model, transformer=None, vae=None, torch_dtype=dt).to(dev)
        embeds = {}
        for L in LENGTHS:
            for i, p in enumerate(PROMPTS):
                with torch.no_grad():
                    e, _ = pipe.encode_prompt(p, device=dev, max_sequence_length=L)
                embeds[(L, i)] = e.cpu()
            n_tok = [len(pipe.tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=True,
                                                            add_generation_prompt=True, enable_thinking=False))
                     for p in PROMPTS]
        print("[eval] prompt token counts (chat template):", n_tok, flush=True)
        torch.save(embeds, emb_path)
        del pipe; gc.collect(); torch.mps.empty_cache()
    embeds = torch.load(emb_path)

    pipe = Flux2KleinPipeline.from_pretrained(args.model, text_encoder=None, tokenizer=None, torch_dtype=dt).to(dev)
    for L in LENGTHS:
        (args.out / f"t{L}").mkdir(exist_ok=True)
        for i in range(len(PROMPTS)):
            f = args.out / f"t{L}" / f"{i}.png"
            if f.exists():
                continue
            g = torch.Generator(device="cpu").manual_seed(args.seed + i)
            img = pipe(prompt_embeds=embeds[(L, i)].to(dev, dt), height=args.size, width=args.size,
                       num_inference_steps=4, generator=g).images[0]
            img.save(f)
            print(f"[eval] t{L} prompt{i} done", flush=True)
    del pipe; gc.collect(); torch.mps.empty_cache()

    import lpips
    from transformers import CLIPModel, CLIPProcessor
    lp = lpips.LPIPS(net="alex", verbose=False)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def to_t(im):
        return torch.from_numpy(np.asarray(im, np.float32) / 127.5 - 1).permute(2, 0, 1)[None]

    rows = []
    for L in LENGTHS:
        for i, prompt in enumerate(PROMPTS):
            im = Image.open(args.out / f"t{L}" / f"{i}.png").convert("RGB")
            ref = Image.open(args.out / "t512" / f"{i}.png").convert("RGB")
            with torch.no_grad():
                d = lp(to_t(im), to_t(ref)).item()
                cs = clip(**proc(text=[prompt], images=im, return_tensors="pt", padding=True,
                                 truncation=True)).logits_per_image.item()
            rows.append((L, i, d, cs))
    with open(args.out / "metrics.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("text_len", "prompt", "lpips_vs_t512", "clip_score")] + rows)
    print("\ntext_len  LPIPS_vs_512(mean/max)  CLIP")
    for L in LENGTHS:
        r = [x for x in rows if x[0] == L]
        print(f"{L:8d}  {np.mean([x[2] for x in r]):.3f} / {max(x[2] for x in r):.3f}        {np.mean([x[3] for x in r]):.2f}")
    print("per-prompt LPIPS t256:", [round(x[2], 3) for x in rows if x[0] == 256])
    print("per-prompt LPIPS t128:", [round(x[2], 3) for x in rows if x[0] == 128])

    cell = 320
    grid = Image.new("RGB", (cell * len(LENGTHS), cell * len(PROMPTS)), "white")
    for c, L in enumerate(LENGTHS):
        for r in range(len(PROMPTS)):
            grid.paste(Image.open(args.out / f"t{L}" / f"{r}.png").resize((cell, cell)), (c * cell, r * cell))
    grid.save(args.out / "grid.jpg", quality=90)


if __name__ == "__main__":
    main()
