# FLUX.2 block-streaming on iPhone — status & validation

**The streaming engine AND the app router are fully built, compile green on iOS + macOS, the 512 parity
gate PASSES bit-identically on Mac (incl. the forced per-step streaming path), AND both on-device
streaming workloads are now VALIDATED on an iPhone 16 Pro (8 GB):**

- **1024 text-to-image** — block-streaming transformer, on-device peak **3.83 GB**, **~4m22s**, completes
  clean (no restart). *(validated 2026-06-27)*
- **512 image-to-image** (reference-context) — block-streaming, on-device peak **~3.45 GB**, **1m49s**,
  no restart, good quality. *(validated 2026-06-29 — see §3)*

The streaming path is the resident path's twin — it reuses the same flux-mlx units
(`KleinTextEncoder`, `LatentUtils`, the VAE). Validated:

```
swift run flux2-demo --parity            # resident MLXEngine vs facade  → maxPixelDiff 0, PSNR ∞  ✅
swift run flux2-demo --parity --stream   # per-step block-STREAMING vs facade → maxPixelDiff 0, PSNR ∞  ✅
swift run flux2-demo --parity --i2i      # streamed i2i vs resident facade → maxPixelDiff 0, PSNR ∞  ✅
```

All three are **bit-identical** to the resident facade at 512. The forced-streaming runs exercise the
exact load→run→release→clearCache path the iPhone uses, so the on-device *mechanics* were validated
ahead of the phone; the *physics* (thermal + jetsam) are now confirmed on-device for both workloads.

---

## ✅ 1024 ON IPHONE — SOLVED & ON-DEVICE VALIDATED (2026-06-27) — peak 3.83 GB, ~4m22s

`swift run flux2-demo --parity --size 1024 --stream` → **MLX peak 3.83 GB, `maxPixelDiff=0 PSNR=inf PASS ✅`.**
That 3.83 GB is *lower* than the 4.3 GB the working 512 path peaks at on-device (512 runs resident and
holds all weights; 1024 streams and releases blocks as it goes). **Confirmed on an iPhone 16 Pro (8 GB):
a 1024 T2I render streamed to completion at 3.83 GB peak in ~4m22s, no restart** — the offline 3.83 GB
prediction held exactly.

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
- **App router** — `AppModel`: `fluxUsesStreaming = isPhone && (size > 512 || !referenceImages.isEmpty)`,
  i.e. streaming kicks in for **1024 T2I** *and* for **any i2i**. The plain text-to-image 512 path stays
  on the resident `Flux2FacadeEngine`; everything streamed goes to
  `MLXDiffusionEngine(architecture: Flux2Architecture(...))` with a transformer-only
  `Flux2ComponentSource.openKlein4BStreaming()`. Because both i2i (~2048 tokens) and T2I-1024 (4096)
  stream, the engine is built with an explicit `targetImageSeqLen: streamingImageSeqLen` (output tokens +
  the 512²-capped reference budget) so `load()`'s residency plan sizes the activation working set to the
  *actual* render; both that flag and the streaming boolean are in the reload key, so crossing
  512↔1024 *or* toggling references rebuilds the engine. iPhone i2i is also capped — `fluxEffectiveSize`
  forces 512 output and `maxReferenceImages` is 1 on the phone (3 on Mac). Mac is unaffected (`isPhone`
  false → facade, 1–3 references up to the chosen size).
- **The engine** — `Flux2Architecture` / `Flux2Denoiser` / `Flux2StreamableBlock` / `Flux2Weights` /
  `Flux2Sigmas` / `Flux2ComponentSource` in `flux2-diffusion-engine@main`; the streaming decomposition
  + `Flux2StreamingSupport` (incl. the i2i reference-token path) in `flux-2-swift-mlx@main`.

To build the app, bump the app's two remote pins (`swift-diffusion-core`, `flux-2-swift-mlx`) to
latest `main`; the local-path deps follow automatically. The validated i2i wiring is at
`flux-2-swift-mlx` **`bbae617`** (streaming-i2i reference-token support), `swift-diffusion-core`
**`c0e8f43`**, local `flux2-diffusion-engine` **`6d29c85`**, and app commit **`8210253`**. Already
resolved in this branch's `Package.resolved`. **Never commit a local-path dep into a shared package.**

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

## 2. On-device 1024 (iPhone) — DONE ✅

Build the app to your iPhone and render at 1024 — the router selects the streaming engine automatically.
On an **iPhone 16 Pro (8 GB)** this completed at **3.83 GB peak in ~4m22s with no restart**, matching the
offline prediction. It is slower + hotter than 512 (intrinsic 4× FLOPs + ~8.7 GB pread/image), but it
**completes or auto-pauses to cool** (the per-step `ThermalGovernor` — pace-down at `.serious`, a visible
recoverable "Cooling…" PAUSE at `.critical`) instead of restarting. The thermal *start*-gate was wired
then removed: a silent start-refusal just made the Generate button look dead on a warm phone.

Tunables if tight: `Flux2StreamableBlock.approximateBytes` (residency planning), the streaming
`cacheLimit` (384 MB in `MLXDiffusionEngine.load`), the VAE conv-strip row height.

---

## 3. On-device 512 image-to-image (iPhone) — DONE ✅ (2026-06-29)

i2i here is **FLUX.2 reference-context**, not a strength slider: 1–3 reference images are VAE-encoded and
concatenated into the transformer sequence as conditioning; the output denoises from **pure noise**
attending to the refs, so **strength is always 1.0**.

The resident facade OOM'd the phone on i2i. Each reference is ~4096 tokens at `maxImageArea` 1024², so
even a "512 i2i" ran **5120 tokens RESIDENT ≈ 5.75 GB**, over the ~5.5 GB jetsam line → a whole-phone
respring. JetsamEvent logs confirmed this is a **MEMORY** limit, **not thermal**. So i2i was first
*disabled* on iPhone, then **re-enabled via the streaming path**:

- The streamed transformer carries the packed image sequence as **`[output ; reference]`** (output
  first). Only the **output** tokens are denoised and decoded — a new `outputSeqLen` slice in
  `streamUnembed` (`Flux2Transformer.swift`) takes `[textSeqLen ..< textSeqLen + outputSeqLen]`;
  `outputSeqLen == nil` (T2I) falls back to the original byte-for-byte slice.
- The reference is **capped to 512²** (~1024 tokens, a **single** reference on iPhone), so the streamed
  sequence is **≤2048 tokens** — *lighter* than the already-shipped T2I-1024 path (4096 tokens). The app
  enforces this: `fluxEffectiveSize` forces 512 output, `streamingImageSeqLen = output + ref-budget`
  plans the worst case, and `maxReferenceImages` is 1 on the phone.
- The reference VAE-encode is **freed before the transformer streams** — no encoder↔transformer
  co-residency.

**Validation:**
- `swift run flux2-demo --parity --i2i` → the streamed i2i output is **pixel-identical** to the resident
  facade (**maxPixelDiff=0, PSNR=inf**), both resident and forced-block-streaming.
- An 8-agent adversarial code audit found the token order, position IDs, and `outputSeqLen` slice correct
  and the memory lifecycle clean; the streamed-decomposition unit tests pass.
- **On-device (iPhone 16 Pro, 8 GB): 512 i2i streamed to completion, peak ~3.45 GB, 1m49s, no restart,
  good quality.** (Mac forced-stream measured 3.78 GB.)

**Still open for i2i:** multi-reference streaming on iPhone (single ref today), and higher-than-512 i2i
output on iPhone (1024-i2i resident facade peaks ~6.9 GB on Mac — fine there, over budget on a phone).
Z-Image classic (strength-based) i2i is also not built.

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
| streamed i2i (reference-token) parity | ✅ **bit-identical on Mac** | `flux2-demo --parity --i2i` |
| 1024 T2I thermal/memory survival | ✅ **on-device (3.83 GB, ~4m22s)** | §2 |
| 512 i2i thermal/memory survival | ✅ **on-device (~3.45 GB, 1m49s)** | §3 |

`reuse-shell` (the ~100-quantize-passes optimization) stays deferred per the audit until 512 parity is
green and a dedicated bit-exact parity test covers it.
