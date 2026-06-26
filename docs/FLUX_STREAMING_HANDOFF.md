# FLUX.2 1024-on-iPhone block-streaming — status & validation

**The streaming engine AND the app router are fully built and compile green on iOS + macOS.** The
core numerics are proven offline. The only things left need real weights on your hardware: run the
**512 parity gate** on your Mac, then the **on-device 1024** test on the iPhone.

The streaming path is the resident path's twin — it reuses the same flux-mlx units
(`KleinTextEncoder`, `LatentUtils`, the VAE), so parity is expected; the gate confirms it.

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

## 1. THE GATE — 512 parity (run on your Mac, real weights)

One command (downloads the 4-bit Klein weights on first run):

```
cd flux2-diffusion-engine
swift run flux2-demo --parity
```

It generates the same 512 image two ways — the resident facade and the streaming
`MLXDiffusionEngine + Flux2Architecture`, same seed — writes `parity-resident.png` /
`parity-streamed.png`, and prints **max pixel diff + PSNR**. PASS ≈ visually identical (PSNR > ~35 dB).

On a Mac the streaming engine loads resident (plenty of memory) but runs the SAME architecture code
(encode → streamEmbed → 25 blocks → streamUnembed → decode), so passing here validates the whole path;
the only iPhone difference is per-step block load/release (memory management, not math).

**If it diverges**, suspects in order: (1) the encoder call — `Flux2Architecture.encode` uses
`encode(prompt, upsample: false)`; confirm the resident path matches (no extra prompt enrichment).
(2) position-id parity — `initialLatent` uses `LatentUtils.combinePositionIDs` exactly as the resident
T2I path. (3) VAE variant must match on both sides (the CLI uses `.small`/`.smallDecoder` for both).
The transformer decomposition, sigma schedule, and per-block key mapping are already proven, so a
divergence is almost certainly in the encode/decode glue, not the transformer.

---

## 2. Then on-device 1024 (iPhone)

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
| encode / initialLatent / decode end-to-end parity | ⏳ **`swift run flux2-demo --parity`** | §1 |
| 1024 thermal/memory survival | ⏳ on-device | §2 |

`reuse-shell` (the ~100-quantize-passes optimization) stays deferred per the audit until 512 parity is
green and a dedicated bit-exact parity test covers it.
