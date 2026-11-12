#!/usr/bin/env python3
"""Qwen3-4B (FLUX.2 Klein text encoder) -> split Core ML stages for ANE.

Only the first 27 of 36 layers are needed (Klein concatenates hidden states after layers 9/18/27).
Stage{i}: hidden[1,T,2560], mask[1,1,T,T] (additive), rope_cos/rope_sin[T,128] -> hidden_out.
Also writes qwen3_rope_cos.bin / qwen3_rope_sin.bin (float32) and, with --embeddings, the token embedding
table as float16 (embed_tokens.f16, [151936, 2560]) for lookup on the host side.

Usage: convert_qwen3_split.py --model checkpoints/flux2-klein-4b --txt 256 --layers-per-stage 3 --int8 \
           -o build_dir --compile-into coreml_flux2_klein_1024 [--embeddings]
"""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from qwen3_layers import Weights, rope_tables  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--txt", type=int, default=256)
    ap.add_argument("--layers", type=int, default=27)
    ap.add_argument("--layers-per-stage", type=int, default=3)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--compile-into", type=Path, required=True)
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--prefix", default="TextStage")
    args = ap.parse_args()
    import coremltools as ct
    import coremltools.optimize.coreml as cto
    W = Weights(args.model / "text_encoder")
    T, d = args.txt, W.cfg["hidden_size"]
    cos, sin = rope_tables(W.cfg, T)
    args.out.mkdir(parents=True, exist_ok=True)
    args.compile_into.mkdir(parents=True, exist_ok=True)
    cos.numpy().astype("<f4").tofile(args.compile_into / "qwen3_rope_cos.bin")
    sin.numpy().astype("<f4").tofile(args.compile_into / "qwen3_rope_sin.bin")
    n_stages = args.layers // args.layers_per_stage
    assert n_stages * args.layers_per_stage == args.layers
    (args.compile_into / "qwen3.json").write_text(json.dumps({
        "text_tokens": T, "hidden": d, "stages": n_stages, "layers_per_stage": args.layers_per_stage,
        "taps_after_layer": [9, 18, 27], "vocab": W.cfg["vocab_size"], "pad_token_id": 151643,
        "template": "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"}, indent=1))
    if args.embeddings:
        e = W.get("model.embed_tokens.weight").numpy().astype("<f2")
        e.tofile(args.compile_into / "embed_tokens.f16")
        print(f"[qwen3] embeddings {e.shape} written", flush=True)
        del e
    g = torch.Generator().manual_seed(0)
    ex = (torch.randn(1, T, d, generator=g), torch.zeros(1, 1, T, T), cos, sin)
    for s in range(n_stages):
        name = f"{args.prefix}{s}"
        if (args.compile_into / f"{name}.mlmodelc").exists():
            print(f"[qwen3] {name} exists, skip", flush=True)
            continue
        t0 = time.time()
        mod = W.stage(s * args.layers_per_stage, args.layers_per_stage)
        with torch.no_grad():
            traced = torch.jit.trace(mod, ex, strict=False)
        ml = ct.convert(traced, inputs=[ct.TensorType(name=n, shape=t.shape) for n, t in
                                        zip(["hidden", "mask", "rope_cos", "rope_sin"], ex)],
                        outputs=[ct.TensorType(name="hidden_out")], minimum_deployment_target=ct.target.iOS18,
                        compute_precision=ct.precision.FLOAT16, convert_to="mlprogram", skip_model_load=True)
        if args.int8:
            ml = cto.linear_quantize_weights(ml, config=cto.OptimizationConfig(
                global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")))
        pkg = args.out / f"{name}.mlpackage"
        if pkg.exists():
            shutil.rmtree(pkg)
        ml.save(str(pkg))
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(args.compile_into)], check=True,
                       stdout=subprocess.DEVNULL)
        shutil.rmtree(pkg)
        del mod, traced, ml
        print(f"[qwen3] {name} done {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
