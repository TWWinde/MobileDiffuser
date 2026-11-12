#!/usr/bin/env python3
"""Encode prompts with the FLUX.2 Klein text encoder (Qwen3) into a cache usable by run_flux2_klein_lowmem.py.
Runs alone in its own process with a memory watchdog. Output: torch file {(text_len, idx): [1, L, 7680]}.

Usage: encode_flux2_prompts.py --model CKPT --out embeds.pt --txt 256 --start 100 "prompt a" "prompt b" ...
"""
import argparse, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from run_flux2_klein_lowmem import avail_gb, start_watchdog  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--txt", type=int, default=256)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--min-free-gb", type=float, default=2.0)
ap.add_argument("prompts", nargs="+")
a = ap.parse_args()
start_watchdog(a.min_free_gb)
from diffusers import Flux2KleinPipeline
dev = torch.device("mps")
pipe = Flux2KleinPipeline.from_pretrained(a.model, transformer=None, vae=None, torch_dtype=torch.bfloat16).to(dev)
print(f"[encode] loaded, available {avail_gb():.1f} GB", flush=True)
emb = torch.load(a.out) if a.out.exists() else {}
for i, p in enumerate(a.prompts):
    with torch.no_grad():
        e, _ = pipe.encode_prompt(p, device=dev, max_sequence_length=a.txt)
    emb[(a.txt, a.start + i)] = e.cpu()
    print(f"[encode] {a.start + i}: {tuple(e.shape)} {p[:60]}", flush=True)
a.out.parent.mkdir(parents=True, exist_ok=True)
torch.save(emb, a.out)
