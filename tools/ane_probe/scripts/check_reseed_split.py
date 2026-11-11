#!/usr/bin/env python3
"""Check a ReSeed one-block-per-stage Core ML folder against PyTorch (first interval, t=1000).

Reference: diffusers SD3 transformer (fp32) + the paper's SD3BlockResidualBias forward hooks.
Core ML: compiled Stage0..Stage24 run in sequence; biases for blocks 0..N-2 added between stages
(as the Swift runtime does) and the last block's image bias fed as `reseed_image_bias`.
Reports rel-L2 vs reference for ReSeed on and off, and the size of the ReSeed effect itself.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", type=Path, required=True)
    ap.add_argument("--resources", type=Path, required=True)
    ap.add_argument("--cond", type=Path, required=True, help="npz txt/pooled from real CLIP encoders")
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    import coremltools as ct
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
    from safetensors.torch import load_file

    tr = SD3Transformer2DModel(
        sample_size=128, patch_size=2, in_channels=16, num_layers=24, attention_head_dim=64,
        num_attention_heads=24, joint_attention_dim=4096, caption_projection_dim=1536,
        pooled_projection_dim=2048, out_channels=16, pos_embed_max_size=192,
        dual_attention_layers=(), qk_norm=None).float().eval()
    state = {}
    for f in sorted(args.student.glob("*.safetensors")):
        state.update({k: v.float() for k, v in load_file(str(f)).items()})
    tr.load_state_dict(state, strict=True); del state

    biases = json.loads((args.resources / "ReSeedBiases.json").read_text())
    img_b = [torch.tensor(b) for b in biases["image"]]
    ctx_b = [torch.tensor(b) if b is not None else None for b in biases["context"]]
    active = {"on": False}

    def hook(i):
        def f(_m, _a, out):
            if not active["on"]:
                return out
            c, x = out
            x = x + args.scale * img_b[i]
            if c is not None and ctx_b[i] is not None:
                c = c + args.scale * ctx_b[i]
            return (c, x)
        return f

    for i, blk in enumerate(tr.transformer_blocks):
        blk.register_forward_hook(hook(i))

    c = np.load(args.cond)
    txt, pooled = torch.from_numpy(c["txt"]).float(), torch.from_numpy(c["pooled"]).float()
    lat = torch.randn(1, 16, args.latent, args.latent, generator=torch.Generator().manual_seed(0))
    t = torch.tensor([1000.0])
    refs = {}
    with torch.no_grad():
        for on in (False, True):
            active["on"] = on
            refs[on] = tr(hidden_states=lat, encoder_hidden_states=txt, pooled_projections=pooled,
                          timestep=t).sample.numpy()

    units = ct.ComputeUnit.CPU_AND_NE
    cond = ct.models.CompiledMLModel(str(args.resources / "MultiModalDiffusionTransformerConditioning.mlmodelc"),
                                     compute_units=units)
    mod = cond.predict({"pooled_text_embeddings": pooled.numpy().reshape(1, 2048, 1, 1),
                        "timestep": np.array([1000.0], np.float32)})["modulation_inputs"].astype(np.float32)
    n_stages = len(list(args.resources.glob("MultiModalDiffusionTransformerStage*.mlmodelc")))
    outs = {}
    for on in (False, True):
        feats = {"latent_image_embeddings": lat.numpy(),
                 "token_level_text_embeddings": txt.transpose(1, 2).unsqueeze(2).numpy(),
                 "modulation_inputs": mod,
                 "reseed_image_bias": (args.scale * img_b[-1].numpy()[None] if on else np.zeros((1, 1536))).astype(np.float32)}
        for k in range(n_stages):
            m = ct.models.CompiledMLModel(str(args.resources / f"MultiModalDiffusionTransformerStage{k}.mlmodelc"),
                                          compute_units=units)
            names = {i.name for i in m.get_spec().description.input} if hasattr(m, "get_spec") else None
            inp = {n: v for n, v in feats.items() if names is None or n in names}
            try:
                r = m.predict(inp)
            except Exception:
                inp.pop("reseed_image_bias", None)
                r = m.predict(inp)
            if "noise_pred" in r:
                outs[on] = r["noise_pred"].astype(np.float32)
                break
            x = r["latent_image_embeddings_out"].astype(np.float32)
            cx = r["token_level_text_embeddings_out"].astype(np.float32)
            if on and k >= 1:
                x = x + args.scale * img_b[k - 1].numpy()
                if ctx_b[k - 1] is not None:
                    cx = cx + args.scale * ctx_b[k - 1].numpy()
            feats["latent_image_embeddings"], feats["token_level_text_embeddings"] = x, cx
            del m

    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    print(f"ReSeed effect in reference  rel(on, off) = {rel(refs[True], refs[False]):.3e}")
    print(f"Core ML vs ref, ReSeed off   rel = {rel(outs[False], refs[False]):.3e}")
    print(f"Core ML vs ref, ReSeed on    rel = {rel(outs[True], refs[True]):.3e}")
    print(f"Core ML ReSeed effect vs ref effect: rel = "
          f"{rel(outs[True] - outs[False], refs[True] - refs[False]):.3e}")


if __name__ == "__main__":
    main()
