#!/usr/bin/env python3
"""Memory-safe Mac runner for the split FLUX.2 Klein Core ML stages.

Only one stage is loaded at a time (load -> predict -> release), nothing from PyTorch but the scheduler is
in the denoising process, and a watchdog kills the process when system-available memory drops too low.
VAE decoding is a separate subcommand (separate process).

  probe   --coreml DIR --stage Double0          load one stage, run it on random inputs, report memory
  denoise --model CKPT --coreml DIR --embeds embeds.pt --prompt 0 --out DIR   -> DIR/latent_<i>.npy
  decode  --model CKPT --out DIR --prompt 0 [--ref DIR/torch_0.png]           -> DIR/coreml_<i>.png
"""
import argparse, gc, json, os, sys, threading, time
from pathlib import Path

import numpy as np


def avail_gb():
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


def start_watchdog(min_gb):
    def loop():
        while True:
            a = avail_gb()
            if a < min_gb:
                sys.stderr.write(f"\n[watchdog] available memory {a:.1f} GB < {min_gb} GB, aborting\n")
                sys.stderr.flush()
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=loop, daemon=True).start()


class Stages:
    def __init__(self, folder, units="ne"):
        import coremltools as ct
        self.ct, self.folder = ct, Path(folder)
        self.cu = {"ne": ct.ComputeUnit.CPU_AND_NE, "gpu": ct.ComputeUnit.CPU_AND_GPU,
                   "cpu": ct.ComputeUnit.CPU_ONLY}[units]
        self.log = []  # (stage, load_s, predict_s, avail_gb)

    def run(self, name, feed):
        t0 = time.time()
        m = self.ct.models.CompiledMLModel(str(self.folder / f"{name}.mlmodelc"), self.cu)
        t1 = time.time()
        out = m.predict(feed)
        t2 = time.time()
        del m
        gc.collect()
        self.log.append((name, t1 - t0, t2 - t1, avail_gb()))
        return out


def cmd_probe(a):
    meta = json.loads((a.coreml / "flux2_klein.json").read_text())
    n, t, d = meta["image_tokens"], meta["text_tokens"], 3072
    r = np.random.default_rng(0)
    f = lambda *s: (r.standard_normal(s) * 0.1).astype(np.float32)  # noqa: E731
    cos, sin = np.load(a.coreml / "rope_cos.npy"), np.load(a.coreml / "rope_sin.npy")
    if a.stage.startswith("Double"):
        feed = {"img": f(1, n, d), "txt": f(1, t, d), "mod_img": f(1, 1, 6 * d), "mod_txt": f(1, 1, 6 * d),
                "rope_cos": cos, "rope_sin": sin}
    else:
        feed = {"hidden": f(1, n + t, d), "mod": f(1, 1, 3 * d), "rope_cos": cos, "rope_sin": sin}
    st = Stages(a.coreml, a.units)
    print(f"[probe] available before: {avail_gb():.1f} GB", flush=True)
    for k in range(3):
        st.run(a.stage, feed)
        name, ld, pr, av = st.log[-1]
        print(f"[probe] {name} run{k}: load {ld:.1f}s predict {pr * 1000:.0f} ms, available after {av:.1f} GB", flush=True)


def cmd_denoise(a):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
    meta = json.loads((a.coreml / "flux2_klein.json").read_text())
    n, T, side = meta["image_tokens"], meta["text_tokens"], meta["resolution"] // 16
    text = torch.load(a.embeds)[(T, a.prompt)].float().numpy()
    cos, sin = np.load(a.coreml / "rope_cos.npy"), np.load(a.coreml / "rope_sin.npy")
    g = torch.Generator(device="cpu").manual_seed(a.seed + a.prompt)
    lat = torch.randn((1, 128, side, side), generator=g, dtype=torch.bfloat16).float()
    lat = lat.reshape(1, 128, n).permute(0, 2, 1).numpy().copy()
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(a.model, subfolder="scheduler")
    sch.set_timesteps(sigmas=np.linspace(1.0, 1 / a.steps, a.steps), mu=compute_empirical_mu(n, a.steps))
    sig = sch.sigmas.numpy().astype(np.float64)
    print("[denoise] sigmas:", np.round(sig, 5).tolist(), f"available {avail_gb():.1f} GB", flush=True)
    st = Stages(a.coreml, a.units)
    a.out.mkdir(parents=True, exist_ok=True)
    for s in range(a.steps):
        t0, k0 = time.time(), len(st.log)
        o = st.run("Embed", {"latent": lat.astype(np.float32), "text": text,
                             "timestep": np.array([sig[s]], np.float32)})
        img, txt = o["img"], o["txt"]
        for i in range(meta["double_blocks"]):
            d = st.run(f"Double{i}", {"img": img, "txt": txt, "mod_img": o["mod_img"], "mod_txt": o["mod_txt"],
                                      "rope_cos": cos, "rope_sin": sin})
            img, txt = d["img_out"], d["txt_out"]
        hidden = np.concatenate([txt, img], axis=1)
        for i in range(meta["single_blocks"]):
            hidden = st.run(f"Single{i}", {"hidden": hidden, "mod": o["mod_single"], "rope_cos": cos,
                                           "rope_sin": sin})["hidden_out"]
        noise = st.run("Final", {"hidden": hidden, "temb": o["temb"]})["noise"]
        assert np.isfinite(noise).all(), "non-finite noise prediction"
        lat = lat + (sig[s + 1] - sig[s]) * noise.astype(np.float64)
        L = st.log[k0:]
        pd = lambda p: np.mean([x[2] for x in L if x[0].startswith(p)]) * 1000  # noqa: E731
        print(f"[denoise] step {s + 1}/{a.steps}: {time.time() - t0:.1f}s  load {sum(x[1] for x in L):.1f}s  "
              f"predict {sum(x[2] for x in L):.1f}s (Double {pd('Double'):.0f} ms, Single {pd('Single'):.0f} ms)  "
              f"min available {min(x[3] for x in L):.1f} GB", flush=True)
    np.save(a.out / f"latent_{a.prompt}.npy", lat.astype(np.float32))
    print("[denoise] saved latent", flush=True)


def klein_sigmas(n_tokens, steps):
    """FLUX.2 schedule (empirical mu, exponential shift) -> steps + 1 sigmas ending in 0."""
    a1, b1, a2, b2 = 8.73809524e-05, 1.89833333, 0.00016927, 0.45666666
    if n_tokens > 4300:
        mu = a2 * n_tokens + b2
    else:
        m200, m10 = a2 * n_tokens + b2, a1 * n_tokens + b1
        k = (m200 - m10) / 190.0
        mu = k * steps + (m200 - 200.0 * k)
    s = np.linspace(1.0, 1.0 / steps, steps)
    return np.append(np.exp(mu) / (np.exp(mu) + (1.0 / s - 1.0)), 0.0)


def run_steps(st, meta, lat, text, cos, sin, sig, x0_at_end=False):
    """Euler steps over consecutive sigmas `sig`; returns the latent at sig[-1] (and the x0 prediction of the
    last step when x0_at_end)."""
    lat = lat.astype(np.float64)
    x0 = None
    for s in range(len(sig) - 1):
        o = st.run("Embed", {"latent": lat.astype(np.float32), "text": text, "timestep": np.array([sig[s]], np.float32)})
        img, txt = o["img"], o["txt"]
        for i in range(meta["double_blocks"]):
            d = st.run(f"Double{i}", {"img": img, "txt": txt, "mod_img": o["mod_img"], "mod_txt": o["mod_txt"],
                                      "rope_cos": cos, "rope_sin": sin})
            img, txt = d["img_out"], d["txt_out"]
        hidden = np.concatenate([txt, img], axis=1)
        for i in range(meta["single_blocks"]):
            hidden = st.run(f"Single{i}", {"hidden": hidden, "mod": o["mod_single"], "rope_cos": cos,
                                           "rope_sin": sin})["hidden_out"]
        v = st.run("Final", {"hidden": hidden, "temb": o["temb"]})["noise"].astype(np.float64)
        assert np.isfinite(v).all()
        x0 = lat - sig[s] * v
        lat = lat + (sig[s + 1] - sig[s]) * v
    return (lat, x0) if x0_at_end else lat


def upsample_packed(lat, bn, mode):
    """packed normalised [1, s*s, 128] -> x2 in the unpatchified 32-channel VAE space -> packed [1, 4*s*s, 128]"""
    import torch
    import torch.nn.functional as F
    mean, std = np.array(bn["mean"]), np.array(bn["std"])
    s = int(lat.shape[1] ** 0.5)
    x = lat.transpose(0, 2, 1).reshape(1, 128, s, s) * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)
    z = x.reshape(1, 32, 2, 2, s, s).transpose(0, 1, 4, 2, 5, 3).reshape(1, 32, 2 * s, 2 * s)
    kw = {} if mode == "nearest" else {"align_corners": False}
    z = F.interpolate(torch.from_numpy(z), scale_factor=2, mode=mode, **kw).numpy()
    S = 2 * s
    x = z.reshape(1, 32, S, 2, S, 2).transpose(0, 1, 3, 5, 2, 4).reshape(1, 128, S, S)
    x = (x - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)
    return x.reshape(1, 128, S * S).transpose(0, 2, 1)


def cmd_hires(a):
    """Low-resolution start, high-resolution finish: `--lo-steps` Euler steps of the 512 schedule, upsample the
    x0 estimate x2, re-noise to the sigma where the last `--hi-steps` steps of the 1024 schedule begin, finish
    at 1024. --lo-steps 4 denoises fully at 512 first."""
    import torch
    lo_meta = json.loads((a.coreml_lo / "flux2_klein.json").read_text())
    hi_meta = json.loads((a.coreml / "flux2_klein.json").read_text())
    T = hi_meta["text_tokens"]
    text = torch.load(a.embeds)[(T, a.prompt)].float().numpy()
    bn = json.loads((a.coreml / "vae_bn.json").read_text())
    n_lo, n_hi = lo_meta["image_tokens"], hi_meta["image_tokens"]
    side = int(n_lo ** 0.5)
    g = torch.Generator(device="cpu").manual_seed(a.seed + a.prompt)
    lat = torch.randn((1, 128, side, side), generator=g).reshape(1, 128, n_lo).permute(0, 2, 1).numpy().copy()
    sig_lo = klein_sigmas(n_lo, a.steps)[: a.lo_steps + 1]
    sig_hi = klein_sigmas(n_hi, a.steps)[a.steps - a.hi_steps:]
    t0 = time.time()
    st = Stages(a.coreml_lo, a.units)
    _, x0 = run_steps(st, lo_meta, lat, text, np.load(a.coreml_lo / "rope_cos.npy") if (a.coreml_lo / "rope_cos.npy").exists()
                      else np.fromfile(a.coreml_lo / "rope_cos.bin", "<f4").reshape(T + n_lo, -1),
                      np.fromfile(a.coreml_lo / "rope_sin.bin", "<f4").reshape(T + n_lo, -1), sig_lo, x0_at_end=True)
    t_lo = sum(x[2] for x in st.log)
    up = upsample_packed(x0, bn, a.upsample)
    side_hi = int(n_hi ** 0.5)
    eps = torch.randn((1, 128, side_hi, side_hi), generator=g).reshape(1, 128, n_hi).permute(0, 2, 1).numpy()
    lat_hi = (1 - sig_hi[0]) * up + sig_hi[0] * eps
    st2 = Stages(a.coreml, a.units)
    out = run_steps(st2, hi_meta, lat_hi, text, np.fromfile(a.coreml / "rope_cos.bin", "<f4").reshape(T + n_hi, -1),
                    np.fromfile(a.coreml / "rope_sin.bin", "<f4").reshape(T + n_hi, -1), sig_hi)
    t_hi = sum(x[2] for x in st2.log)
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / f"latent_{a.tag}_{a.prompt}.npy", out.astype(np.float32))
    print(f"[hires] {a.tag} prompt{a.prompt}: lo sigmas {np.round(sig_lo, 3).tolist()} -> renoise {sig_hi[0]:.3f} -> "
          f"hi sigmas {np.round(sig_hi, 3).tolist()}  predict lo {t_lo:.1f}s hi {t_hi:.1f}s  wall {time.time() - t0:.1f}s",
          flush=True)


def cmd_decode(a):
    import torch
    from diffusers import AutoencoderKLFlux2
    from PIL import Image
    name = f"{a.tag}_{a.prompt}" if a.tag else f"{a.prompt}"
    lat = torch.from_numpy(np.load(a.out / f"latent_{name}.npy"))  # [1, N, 128]
    side = int(lat.shape[1] ** 0.5)
    dev, dt = torch.device("mps"), torch.bfloat16
    vae = AutoencoderKLFlux2.from_pretrained(a.model, subfolder="vae", torch_dtype=dt).to(dev)
    x = lat.permute(0, 2, 1).reshape(1, 128, side, side).to(dev, dt)
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(dev, dt)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(dev, dt)
    x = x * std + mean
    x = x.reshape(1, 32, 2, 2, side, side).permute(0, 1, 4, 2, 5, 3).reshape(1, 32, side * 2, side * 2)
    with torch.no_grad():
        im = vae.decode(x, return_dict=False)[0]
    arr = ((im[0].float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    p = a.out / f"coreml_{name}.png"
    Image.fromarray(arr).save(p)
    print("[decode] saved", p, flush=True)
    if a.ref and Path(a.ref).exists():
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False)
        to_t = lambda q: torch.from_numpy(np.asarray(Image.open(q).convert("RGB"), np.float32) / 127.5 - 1).permute(2, 0, 1)[None]  # noqa: E731
        with torch.no_grad():
            print(f"[decode] LPIPS vs {a.ref}: {lp(to_t(p), to_t(a.ref)).item():.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "denoise", "decode", "hires"])
    ap.add_argument("--coreml-lo", type=Path, help="hires: low-resolution stage folder")
    ap.add_argument("--lo-steps", type=int, default=4)
    ap.add_argument("--hi-steps", type=int, default=1)
    ap.add_argument("--upsample", default="bicubic", choices=["bicubic", "bilinear", "nearest"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    start_watchdog(a.min_free_gb)
    {"probe": cmd_probe, "denoise": cmd_denoise, "decode": cmd_decode, "hires": cmd_hires}[a.cmd](a)


if __name__ == "__main__":
    main()
