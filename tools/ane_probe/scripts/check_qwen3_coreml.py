#!/usr/bin/env python3
"""Run the split Qwen3 Core ML stages (one resident at a time) on a prompt and compare with a PyTorch fp32
reference saved by the reference script (dict with ids, n, fp32 [1,T,7680]). Optionally save the embeddings.
Usage: check_qwen3_coreml.py --coreml DIR [--ref ref.pt | --prompt "..." --tokenizer DIR] [--save out.npy]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_flux2_klein_lowmem import Stages, start_watchdog  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--coreml", type=Path, required=True)
ap.add_argument("--ref")
ap.add_argument("--prompt")
ap.add_argument("--tokenizer")
ap.add_argument("--save")
ap.add_argument("--units", default="ne")
a = ap.parse_args()
start_watchdog(3.0)
meta = json.loads((a.coreml / "qwen3.json").read_text())
T, d = meta["text_tokens"], meta["hidden"]
ref = None
if a.ref:
    import torch
    r = torch.load(a.ref)
    ids, n, ref = r["ids"].numpy(), int(r["n"]), r["fp32"].numpy()
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    e = tok(meta["template"].format(a.prompt), padding="max_length", truncation=True, max_length=T)
    ids, n = np.array(e["input_ids"]), int(sum(e["attention_mask"]))
emb = np.memmap(a.coreml / "embed_tokens.f16", dtype="<f2", mode="r").reshape(meta["vocab"], d)
x = emb[ids].astype(np.float32)[None]
mask = np.triu(np.full((T, T), -1e4, np.float32), 1)
mask[:, n:] = -1e4
mask = mask[None, None]
cos = np.fromfile(a.coreml / "qwen3_rope_cos.bin", "<f4").reshape(T, -1)
sin = np.fromfile(a.coreml / "qwen3_rope_sin.bin", "<f4").reshape(T, -1)
st, taps = Stages(a.coreml, a.units), []
per = meta["layers_per_stage"]
for s in range(meta["stages"]):
    x = st.run(f"TextStage{s}", {"hidden": x, "mask": mask, "rope_cos": cos, "rope_sin": sin})["hidden_out"]
    if (s + 1) * per in meta["taps_after_layer"]:
        taps.append(x.astype(np.float32))
out = np.stack(taps, 1).transpose(0, 2, 1, 3).reshape(1, T, -1)
print(f"[qwen3] tokens={n} load {sum(l[1] for l in st.log):.1f}s predict {sum(l[2] for l in st.log):.2f}s "
      f"finite={np.isfinite(out).all()} absmax={np.abs(out).max():.0f}", flush=True)
if ref is not None:
    big = np.abs(ref).max(axis=(0, 1)) > 1000
    for tag, sl in (("real", slice(0, n)), ("pad", slice(n, T))):
        for nm, m in (("all dims", slice(None)), ("excl massive", ~big)):
            p, q = out[0, sl][:, m].ravel().astype(np.float64), ref[0, sl][:, m].ravel().astype(np.float64)
            print(f"[qwen3] {tag:4s} {nm:12s} rel {np.linalg.norm(p - q) / np.linalg.norm(q):.4f} "
                  f"cos {p @ q / np.linalg.norm(p) / np.linalg.norm(q):.5f}", flush=True)
if a.save:
    np.save(a.save, out)
