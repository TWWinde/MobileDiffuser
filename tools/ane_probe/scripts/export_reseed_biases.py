#!/usr/bin/env python3
"""Export a ReSeed SD3 residual-bias adapter (residual_biases.pt, sd3-block-residual-bias-v1) to
ReSeedBiases.json for the Core ML runtime: {"image": [[...] x blocks], "context": [[...] | null]}.

Usage: export_reseed_biases.py ADAPTER_DIR OUT.json
"""
import json, sys
from pathlib import Path

import torch

adapter_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
payload = torch.load(adapter_dir / "residual_biases.pt", map_location="cpu", weights_only=True)
assert payload.get("adapter_type") == "sd3-block-residual-bias-v1", payload.get("adapter_type")
blocks = [int(b) for b in payload["blocks"]]
assert blocks == list(range(len(blocks))), "runtime expects biases for consecutive blocks 0..N-1"
sd = payload["state_dict"]
image = [sd[f"image_biases.{b}"].float().tolist() for b in blocks]
context = [sd[f"context_biases.{b}"].float().tolist() if f"context_biases.{b}" in sd else None for b in blocks]
out.write_text(json.dumps({"source": str(adapter_dir), "blocks": len(blocks), "image": image, "context": context}))
n = sum(len(x) for x in image) + sum(len(x) for x in context if x)
print(f"exported {len(blocks)} blocks, {n} parameters -> {out}")
