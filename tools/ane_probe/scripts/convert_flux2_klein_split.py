#!/usr/bin/env python3
"""FLUX.2 Klein (4B) -> split Core ML stages for ANE (static shapes, one transformer block per stage).

Stages (dim 3072, 5 double-stream + 20 single-stream blocks):
  Embed     latent[1,N,128], text[1,T,7680], timestep[1] (sigma in 0..1)
            -> img, txt, mod_img, mod_txt, mod_single, temb
  Double{i} img, txt, mod_img, mod_txt, rope_cos, rope_sin -> img_out, txt_out
  Single{i} hidden[1,T+N,3072] ([txt, img]), mod, rope_cos, rope_sin -> hidden_out
  Final     hidden, temb -> noise[1,N,128]

Attention is exact, with the queries split into chunks (no stock SDPA op). RoPE cos/sin are constant for
a fixed resolution / text length and are written to rope_cos.npy / rope_sin.npy ([T+N, 128], text first).
Weights are read per stage straight from the safetensors file, so the 4B model is never fully in RAM.

Usage:
  convert_flux2_klein_split.py --model checkpoints/flux2-klein-4b --res 1024 --txt 256 --q-chunks 16 \
      --int8 -o build_dir --compile-into coreml_flux2_klein_1024
"""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from flux2_probe import DoubleProc, SingleProc  # noqa: E402


class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from diffusers.models.transformers.transformer_flux2 import (Flux2Modulation,
                                                                     Flux2TimestepGuidanceEmbeddings)
        dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=cfg["timestep_guidance_channels"], embedding_dim=dim, bias=False,
            guidance_embeds=cfg["guidance_embeds"])
        self.double_stream_modulation_img = Flux2Modulation(dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(dim, mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(dim, mod_param_sets=1, bias=False)
        self.x_embedder = nn.Linear(cfg["in_channels"], dim, bias=False)
        self.context_embedder = nn.Linear(cfg["joint_attention_dim"], dim, bias=False)

    def forward(self, latent, text, timestep):
        temb = self.time_guidance_embed(timestep * 1000, None)
        return (self.x_embedder(latent), self.context_embedder(text),
                self.double_stream_modulation_img(temb).unsqueeze(1),
                self.double_stream_modulation_txt(temb).unsqueeze(1),
                self.single_stream_modulation(temb).unsqueeze(1), temb)


class Double(nn.Module):
    def __init__(self, cfg, n):
        super().__init__()
        from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock
        self.block = Flux2TransformerBlock(cfg["num_attention_heads"] * cfg["attention_head_dim"],
                                           cfg["num_attention_heads"], cfg["attention_head_dim"],
                                           mlp_ratio=cfg["mlp_ratio"], eps=cfg["eps"])
        self.block.attn.set_processor(DoubleProc(n))

    def forward(self, img, txt, mod_img, mod_txt, cos, sin):
        txt, img = self.block(img, txt, mod_img, mod_txt, image_rotary_emb=(cos, sin))
        return img, txt


class Single(nn.Module):
    def __init__(self, cfg, n):
        super().__init__()
        from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock
        self.block = Flux2SingleTransformerBlock(cfg["num_attention_heads"] * cfg["attention_head_dim"],
                                                 cfg["num_attention_heads"], cfg["attention_head_dim"],
                                                 mlp_ratio=cfg["mlp_ratio"], eps=cfg["eps"])
        self.block.attn.set_processor(SingleProc(n))

    def forward(self, hs, mod, cos, sin):
        return self.block(hs, None, mod, image_rotary_emb=(cos, sin))


class Final(nn.Module):
    def __init__(self, cfg, n_txt):
        super().__init__()
        from diffusers.models.normalization import AdaLayerNormContinuous
        dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        self.n_txt = n_txt
        self.norm_out = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=cfg["eps"], bias=False)
        self.proj_out = nn.Linear(dim, cfg["in_channels"], bias=False)

    def forward(self, hidden, temb):
        return self.proj_out(self.norm_out(hidden[:, self.n_txt:], temb))


def load_prefixed(module, st_path, mapping):
    """mapping: {checkpoint key prefix: module attribute prefix}"""
    from safetensors import safe_open
    sd = {}
    with safe_open(str(st_path), "pt") as f:
        for k in f.keys():
            for src, dst in mapping.items():
                if k.startswith(src):
                    sd[dst + k[len(src):]] = f.get_tensor(k).float()
    missing, unexpected = module.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return module


def rope(cfg, res, n_txt):
    from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline as P
    side = res // 16
    pe = Flux2PosEmbed(theta=cfg["rope_theta"], axes_dim=cfg["axes_dims_rope"])
    tc, ts = pe(P._prepare_text_ids(torch.zeros(1, n_txt, 1))[0])
    ic, isn = pe(P._prepare_latent_ids(torch.zeros(1, 1, side, side))[0])
    return torch.cat([tc, ic]).float(), torch.cat([ts, isn]).float()


def build(name, cfg, args, st, cos, sin):
    dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    n_img, n_txt = (args.res // 16) ** 2, args.txt
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    if name == "Embed":
        m = load_prefixed(Embed(cfg), st, {p: p for p in (
            "time_guidance_embed.", "double_stream_modulation_img.", "double_stream_modulation_txt.",
            "single_stream_modulation.", "x_embedder.", "context_embedder.")})
        ex = (r(1, n_img, cfg["in_channels"]), r(1, n_txt, cfg["joint_attention_dim"]), torch.tensor([0.5]))
        return m, ex, ["latent", "text", "timestep"], ["img", "txt", "mod_img", "mod_txt", "mod_single", "temb"]
    if name.startswith("Double"):
        i = int(name[6:])
        m = load_prefixed(Double(cfg, args.q_chunks), st, {f"transformer_blocks.{i}.": "block."})
        ex = (r(1, n_img, dim), r(1, n_txt, dim), r(1, 1, 6 * dim) * 0.1, r(1, 1, 6 * dim) * 0.1, cos, sin)
        return m, ex, ["img", "txt", "mod_img", "mod_txt", "rope_cos", "rope_sin"], ["img_out", "txt_out"]
    if name.startswith("Single"):
        i = int(name[6:])
        m = load_prefixed(Single(cfg, args.q_chunks), st, {f"single_transformer_blocks.{i}.": "block."})
        ex = (r(1, n_txt + n_img, dim), r(1, 1, 3 * dim) * 0.1, cos, sin)
        return m, ex, ["hidden", "mod", "rope_cos", "rope_sin"], ["hidden_out"]
    m = load_prefixed(Final(cfg, n_txt), st, {"norm_out.": "norm_out.", "proj_out.": "proj_out."})
    ex = (r(1, n_txt + n_img, dim), r(1, dim))
    return m, ex, ["hidden", "temb"], ["noise"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="diffusers FLUX.2 Klein folder")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--txt", type=int, default=256)
    ap.add_argument("--q-chunks", type=int, default=16)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("-o", "--out", type=Path, required=True, help="scratch dir for .mlpackage files")
    ap.add_argument("--compile-into", type=Path, required=True)
    ap.add_argument("--only", default="", help="comma-separated stage names")
    ap.add_argument("--keep-packages", action="store_true")
    args = ap.parse_args()

    import coremltools as ct
    import coremltools.optimize.coreml as cto
    cfg = json.loads((args.model / "transformer" / "config.json").read_text())
    st = args.model / "transformer" / "diffusion_pytorch_model.safetensors"
    names = (["Embed"] + [f"Double{i}" for i in range(cfg["num_layers"])]
             + [f"Single{i}" for i in range(cfg["num_single_layers"])] + ["Final"])
    if args.only:
        names = [n for n in names if n in args.only.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    args.compile_into.mkdir(parents=True, exist_ok=True)
    cos, sin = rope(cfg, args.res, args.txt)
    np.save(args.compile_into / "rope_cos.npy", cos.numpy().astype(np.float32))
    np.save(args.compile_into / "rope_sin.npy", sin.numpy().astype(np.float32))
    cos.numpy().astype("<f4").tofile(args.compile_into / "rope_cos.bin")  # raw float32 for the Swift pipeline
    sin.numpy().astype("<f4").tofile(args.compile_into / "rope_sin.bin")
    (args.compile_into / "flux2_klein.json").write_text(json.dumps({
        "resolution": args.res, "text_tokens": args.txt, "image_tokens": (args.res // 16) ** 2,
        "q_chunks": args.q_chunks, "int8": args.int8, "stages": names if not args.only else None,
        "double_blocks": cfg["num_layers"], "single_blocks": cfg["num_single_layers"]}, indent=1))

    for name in names:
        if (args.compile_into / f"{name}.mlmodelc").exists():
            print(f"[klein] {name} exists, skip", flush=True)
            continue
        t0 = time.time()
        mod, ex, ins, outs = build(name, cfg, args, st, cos, sin)
        mod.eval()
        with torch.no_grad():
            traced = torch.jit.trace(mod, ex, strict=False)
        ml = ct.convert(traced, inputs=[ct.TensorType(name=n, shape=t.shape) for n, t in zip(ins, ex)],
                        outputs=[ct.TensorType(name=o) for o in outs],
                        minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16,
                        convert_to="mlprogram", skip_model_load=True)
        if args.int8:
            ml = cto.linear_quantize_weights(ml, config=cto.OptimizationConfig(
                global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")))
        pkg = args.out / f"{name}.mlpackage"
        if pkg.exists():
            shutil.rmtree(pkg)
        ml.save(str(pkg))
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(args.compile_into)], check=True,
                       stdout=subprocess.DEVNULL)
        if not args.keep_packages:
            shutil.rmtree(pkg)
        del mod, traced, ml
        print(f"[klein] {name} done {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
