#!/usr/bin/env python3
"""Run the split Core ML MMDiT stages (Stage0..N mlpackages) in sequence and compare the
final noise_pred against the PyTorch SD3 transformer (stock attention, fp32).

Modulation inputs come from PyTorch's time_text_embed so only the split stages are tested.
"""
import argparse, glob, re
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", type=Path, required=True)
    ap.add_argument("--pkg-dir", type=Path, required=True)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--timestep", type=float, default=1000.0)
    ap.add_argument("--cond", type=Path, help="npz with txt [1,154,4096] and pooled [1,2048] (real prompt)")
    ap.add_argument("--units", default="CPU_AND_NE", choices=["CPU_AND_NE", "CPU_ONLY", "CPU_AND_GPU", "ALL"])
    args = ap.parse_args()

    import coremltools as ct
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
    from safetensors.torch import load_file

    tr = SD3Transformer2DModel(
        sample_size=128, patch_size=2, in_channels=16, num_layers=24, attention_head_dim=64,
        num_attention_heads=24, joint_attention_dim=4096, caption_projection_dim=1536,
        pooled_projection_dim=2048, out_channels=16, pos_embed_max_size=192,
        dual_attention_layers=(), qk_norm=None).float().eval()
    tr.load_state_dict({k: v.float() for k, v in load_file(str(args.ckpt_path)).items()}, strict=True)

    g = torch.Generator().manual_seed(0)
    lat = torch.randn(1, 16, args.latent, args.latent, generator=g)
    txt = torch.randn(1, 154, 4096, generator=g)
    pooled = torch.randn(1, 2048, generator=g)
    if args.cond:  # random N(0,1) text overflows fp16 inside SD3; real CLIP conditioning does not
        c = np.load(args.cond)
        txt, pooled = torch.from_numpy(c["txt"]).float(), torch.from_numpy(c["pooled"]).float()
    t = torch.tensor([args.timestep])
    with torch.no_grad():
        ref = tr(hidden_states=lat, encoder_hidden_states=txt, pooled_projections=pooled, timestep=t).sample.numpy()
        mod = tr.time_text_embed(t, pooled).numpy()

    pkgs = sorted(glob.glob(str(args.pkg_dir / "MultiModalDiffusionTransformerStage*.mlpackage")),
                  key=lambda p: int(re.search(r"Stage(\d+)", p).group(1)))
    units = getattr(ct.ComputeUnit, args.units)
    feats = {
        "latent_image_embeddings": lat.numpy(),
        "token_level_text_embeddings": txt.transpose(1, 2).unsqueeze(2).numpy(),
        "modulation_inputs": mod,
    }
    out = None
    for p in pkgs:
        m = ct.models.MLModel(p, compute_units=units)
        res = m.predict(feats)
        if "noise_pred" in res:
            out = res["noise_pred"]
        else:
            feats["latent_image_embeddings"] = res["latent_image_embeddings_out"].astype(np.float32)
            feats["token_level_text_embeddings"] = res["token_level_text_embeddings_out"].astype(np.float32)
        del m
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    cos = float((out * ref).sum() / (np.linalg.norm(out) * np.linalg.norm(ref)))
    print(f"stages={len(pkgs)} units={args.units} t={args.timestep} relL2={rel:.3e} cos={cos:.6f} "
          f"maxabs={np.abs(out - ref).max():.3e} nan={int(np.isnan(out).sum())} ref_std={ref.std():.3f}")


if __name__ == "__main__":
    main()
