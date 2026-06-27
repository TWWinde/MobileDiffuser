# FLUX.2 1024-on-iPhone block-streaming — status & validation

**The streaming engine AND the app router are fully built, compile green on iOS + macOS, AND the 512
parity gate PASSES bit-identically on Mac** — including the forced per-step streaming path. The only
thing left needs the phone: the **on-device 1024** thermal/memory run.

The streaming path is the resident path's twin — it reuses the same flux-mlx units
(`KleinTextEncoder`, `LatentUtils`, the VAE). Validated:

```
swift run flux2-demo --parity            # resident MLXEngine vs facade  → maxPixelDiff 0, PSNR ∞  ✅
swift run flux2-demo --parity --stream   # per-step block-STREAMING vs facade → maxPixelDiff 0, PSNR ∞  ✅
```

Both are **bit-identical** to the resident facade at 512. The forced-streaming run exercises the exact
load→run→release→clearCache path the iPhone uses at 1024, so the on-device *mechanics* are validated;
only the *physics* (thermal at 1024) remain to test on-device.

---

## ✅ 1024 MEMORY IS SOLVED (2026-06-27) — end-to-end peak 3.83 GB, parity bit-identical

`swift run flux2-demo --parity --size 1024 --stream` → **MLX peak 3.83 GB, `maxPixelDiff=0 PSNR=inf PASS ✅`.**
That 3.83 GB is *lower* than the 4.3 GB the working 512 path peaks at on-device (512 runs resident and
holds all weights; 1024 streams and releases blocks as it goes). So **native 1024 should fit an 8 GB
iPhone** — pending only the on-device confirm.

The earlier "1024 VAE decode is a 10 GB hard wall" was a **measurement artifact**: the profiler held the
full resident transformer (~2.5 GB) + text encoder (~2 GB) alive *through* the decode. The real path
frees both first. True decode-only peak: **full-frame 5.62 GB** (just over the ~5.5 GB iPhone budget —
*this is the "crashed at the last step" the user saw*) → **conv-striped 4.28 GB** (under budget).

**The fix — conv striping** (`flux-2-swift-mlx@main`, `VAEDecoder.decodeConv`/`stripedConv`, ON by
default). The decode peak is a single high-res 3×3 conv (im2col), not accumulation, so eval granularity
couldn't shrink it. The heavy (≥512²) decoder convs now run in ~128-row horizontal **strips with a
1-row halo** — EXACT (the 3×3 receptive field reaches exactly 1 row out; true image edges keep the
conv's own zero-pad → bit-identical bar fp16 rounding, measured maxΔ 1/255). **Seam-free** because
GroupNorm stays full-frame (its global spatial stats are what spatial *tiling* seams on; striping only
splits the spatially-local convs). Both facade and streaming decode share the striped `VAEDecoder`, so
parity stays bit-identical. `swift run flux2-demo --tile` measures it (full-frame vs striped peak +
exactness). Spatial tiling (`decodeWithTiling`) is **abandoned** — 6.5 GB *and* 24 dB seams; striping
supersedes it.

---

## What's wired (done, committed, compile-green)

- **Core un-gate** — `MLXDiffusionEngine.capabilities` is memory-driven for FLUX.2 now (no more
  "macOS only"). `swift-diffusion-core@main`.
- **App router** — `AppModel`: `fluxUsesStreaming = isPhone && size > 512`. 512 → resident
  `Flux2FacadeEngine`; 1024 → `MLXDiffusionEngine(architecture: Flux2Architecture(...))` with a
  transformer-only `Flux2ComponentSource.openKlein4BStreaming()`. The loaded streaming flag is in the
  reload key, so crossing 512↔1024 rebuilds the engine. Mac is unaffected (`isPhone` false → facade).
- **The engine** — `Flux2Architecture` / `Flux2Denoiser` / `Flux2StreamableBlock` / `Flux2Weights` /
  `Flux2Sigmas` / `Flux2ComponentSource` in `flux2-diffusion-engine@main`; the streaming decomposition
  + `Flux2StreamingSupport` in `flux-2-swift-mlx@main`.

To build the app, bump the app's two remote pins (`swift-diffusion-core`, `flux-2-swift-mlx`) to
latest `main`; the local-path deps follow automatically. Already done in this branch's (gitignored)
`Package.resolved`. **Never commit a local-path dep into a shared package.**

---

## 1. THE GATE — 512 parity (PASSED on Mac ✅)

Already run and bit-identical (see the two commands above). If you want to reproduce: `cd
flux2-diffusion-engine && swift run flux2-demo --parity [--stream]` (downloads the 4-bit Klein weights
on first run; writes `parity-resident.png` / `parity-streamed.png`). A `--diag` mode compares weights,
the streaming forward, and the encode/init/decode glue tensor-by-tensor.

The one bug found and fixed during validation: this package had pinned an old `swift-diffusion-core`
that predated the architecture-owned sigma hook, so the engine silently fell back to the fixed-shift
sampler schedule (step-3 σ ~0.001 vs FLUX's ~0.717) and produced a coherent-but-different image. The
pin is bumped; keep shared-package pins current.

---

## 2. Then on-device 1024 (iPhone) — the only step left

Once 512 parity is green, build the app to your iPhone and render at 1024 — the router selects the
streaming engine automatically. Instrument: `thermalState` transitions, per-step wall-clock,
`MemoryProbe` peak. Expect slower + hotter than 512 (intrinsic 4× FLOPs + ~8.7 GB pread/image); the
win is it **completes or auto-pauses to cool** (Wave 1's ThermalGovernor) instead of restarting.
Budget for ~a few on-device fixes (the Z-Image streaming bring-up needed 4).

Tunables if tight: `Flux2StreamableBlock.approximateBytes` (residency planning), the streaming
`cacheLimit` (384 MB in `MLXDiffusionEngine.load`), the VAE tile overlap (bump `.aggressive` 4 → 8 if
seams show).

---

## What's proven offline (no checkpoint) vs to validate

| Piece | Status | Test |
|---|---|---|
| Block-streaming decomposition == monolithic forward | ✅ proven 1e-4 | `Flux2Core/StreamDecompositionTests` |
| FLUX sigma schedule (exact values) | ✅ proven | `Flux2DiffusionEngine/Flux2SigmasTests` |
| Per-block + shared disk↔module key bijection | ✅ proven | `Flux2DiffusionEngine/Flux2WeightsTests` |
| Denoiser wiring (holder/adapter) == monolithic | ✅ proven 1e-4 | `Flux2DiffusionEngine/Flux2DenoiserTests` |
| Component-source routing | ✅ proven | `Flux2DiffusionEngine/Flux2ComponentSourceTests` |
| App router + un-gate | ✅ compile-green iOS + macOS | (build) |
| encode / initialLatent / decode end-to-end parity | ✅ **bit-identical on Mac** | `flux2-demo --parity` |
| per-step streaming load/release parity | ✅ **bit-identical on Mac** | `flux2-demo --parity --stream` |
| 1024 thermal/memory survival | ⏳ on-device | §2 |

`reuse-shell` (the ~100-quantize-passes optimization) stays deferred per the audit until 512 parity is
green and a dedicated bit-exact parity test covers it.
