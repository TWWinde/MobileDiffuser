# FLUX.2 1024-on-iPhone block-streaming — hand-off

This is the Mac-in-the-loop final phase. The **streaming engine is fully built, compiles, and its
core numerics are proven offline** (see "What's proven" below). What remains needs real weights on
your hardware: wire the app router, run the **512 parity gate**, then enable 1024.

Everything below is the *resident* path's twin — the streaming path reuses the same flux-mlx units
(`KleinTextEncoder`, `LatentUtils`, the VAE), so parity is expected; the gate confirms it.

---

## 0. Pull + pin

All engine work is on `main` of the three packages; the app is on `rebuild/mlx-foundation`.

```
swift-diffusion-core   main  (≥ 92badb9  architecture-owned sigma hook)
flux-2-swift-mlx       main  (≥ f80f26a… i.e. its latest: decomposition, public mapper, streaming support)
flux2-diffusion-engine main  (≥ f80f26a  ComponentSource/Weights/Denoiser/Architecture/Sigmas + openKlein4BStreaming)
```

In the app, bump the two remote pins (`swift package update` or edit
`MobileDiffuser.xcodeproj/.../Package.resolved`) to the latest `main` of `swift-diffusion-core` and
`flux-2-swift-mlx`. The local-path deps (`flux2-diffusion-engine`, `z-image`, `AppEngines`) pick up
automatically. **Do not commit any local-path dep into a shared package.**

---

## 1. (optional) Un-gate the core capabilities badge

`MLXDiffusionEngine.load()` is already un-gated — the streaming path runs without this. The only
remaining gate is in `capabilities()` (`swift-diffusion-core/Sources/DiffusionCore/Engine/MLXDiffusionEngine.swift`,
~line 51), and the app's FLUX badge uses `Flux2FacadeEngine.capabilities`, not this. Touch it only if
you want `MLXDiffusionEngine.capabilities(flux2, phone)` to report a streaming plan:

```swift
// remove the `if model.family == .flux2 && device.isPhone { return … "macOS only" }` early-return
return MemoryGovernor.plan(variant: variant, device: device, externalSSDAvailable: true)
```

Then update `CapabilitiesTests.testFluxUnsupportedOnPhone` (it currently asserts "macOS only").

---

## 2. App router (the real wiring) — `MobileDiffuser/AppModel.swift`

Mirror the Z-Image streaming path exactly. **Route 1024 → streaming engine, keep 512 on the validated
resident facade.**

**a. A streaming predicate** (next to `zImageUsesStreaming`, ~line 961):

```swift
/// 1024 streams the transformer (fits the phone); 512 stays on the fast, validated resident facade.
private var fluxUsesStreaming: Bool { device.isPhone && size > 512 }
```

**b. `makeEngine(for:)` flux2 branch** (~line 991):

```swift
case .flux2:
    if fluxUsesStreaming {
        return MLXDiffusionEngine(architecture: Flux2Architecture(vaeVariant: fluxDecoder.vae),
                                  device: device)
    }
    return Flux2FacadeEngine(transformer: fluxTransformer, encoder: fluxEncoder, decoder: fluxDecoder)
```

(`fluxDecoder.vae` is the `ModelRegistry.VAEVariant` the facade already maps to. iPhone caps the
standard decoder to ≤512, so a 1024 streaming run uses `.smallDecoder` — fine.)

**c. The weight source** (~line 752, the `let source: WeightSource = …`): the streaming engine needs a
real source; the facade ignores it.

```swift
let source: WeightSource
if model.family == .zImage && zImageUsesStreaming {
    source = try zImageSource(for: model, streaming: true)
} else if model.family == .flux2 && fluxUsesStreaming {
    source = try Flux2ComponentSource.openKlein4BStreaming()   // transformer-only; enc/VAE load from cache
} else {
    source = SafetensorsWeightSource(tensors: [:])
}
```

**d. Rebuild when crossing the 512↔1024 boundary** (the streaming and resident engines differ). Add a
stored `private var loadedFluxStreaming: Bool?`, set it after a successful load (`loadedFluxStreaming =
fluxUsesStreaming`), clear it on unload, and extend the reload check (~line 739):

```swift
var needsReload = engine == nil || loadedID != model.id
if model.family == .flux2 {
    if loadedRecipe != fluxRecipeLabel { needsReload = true }
    if loadedFluxStreaming != fluxUsesStreaming { needsReload = true }   // 512↔1024 switches engines
}
```

**e. Capabilities badge** (`capabilities(for:)`, ~line 842): optional — for an accurate 1024 badge,
return `MLXDiffusionEngine.capabilities(...)` when `fluxUsesStreaming`, else the facade's.

That's the whole app change. The `import` for `Flux2Architecture`/`Flux2ComponentSource`/
`MLXDiffusionEngine` comes through `AppEngines` → `Flux2DiffusionEngine` + `DiffusionCore` (already
imported).

---

## 3. THE GATE — 512 parity (run on your Mac, real weights)

Before enabling 1024, prove the streamed path equals the resident path at 512 with the same seed.
Easiest as a tiny CLI (extend `flux2-demo`, or a scratch `main.swift`):

```swift
let model = /* the Klein 4B catalog model */
let seed: UInt64 = 42, prompt = "a red panda, studio light"

// RESIDENT (oracle): the facade / Flux2Pipeline at 512.
let facade = Flux2FacadeEngine(transformer: .bit4, encoder: .bit4, decoder: .small)
try await facade.load(model, variant: model.variants[0], source: SafetensorsWeightSource(tensors: [:])) { _ in }
let imgResident = try await facade.generate(
    GenerationRequest(prompt: prompt, steps: 4, seed: seed, size: .square512)) { _ in }

// STREAMED: MLXDiffusionEngine + Flux2Architecture at 512.
let streamed = MLXDiffusionEngine(architecture: Flux2Architecture(vaeVariant: .smallDecoder), device: .current)
try await streamed.load(model, variant: model.variants[0],
                        source: try Flux2ComponentSource.openKlein4BStreaming()) { _ in }
let imgStreamed = try await streamed.generate(
    GenerationRequest(prompt: prompt, steps: 4, seed: seed, size: .square512)) { _ in }

// Compare imgResident vs imgStreamed (per-pixel max diff / PSNR). PASS ≈ visually identical.
```

If it diverges, the suspects (in order) are: (1) the encoder call — confirm the resident path uses
`encode(prompt, upsample: false)` with no extra enrichment (match `Flux2Architecture.encode`);
(2) position-id parity — `Flux2Architecture.initialLatent` uses `LatentUtils.combinePositionIDs`
exactly as the resident T2I path; (3) the VAE variant must match on both sides. The decomposition,
sigma schedule, and per-block key mapping are already proven, so divergence is almost certainly in
encode/decode glue, not the transformer.

---

## 4. Then enable + on-device 1024

Once 512 parity is green, run 1024 on the iPhone (the streaming engine is selected automatically by
`fluxUsesStreaming`). Instrument: `thermalState` transitions, per-step wall-clock, `MemoryProbe` peak.
Expect it to be slower and hotter than 512 (intrinsic 4× FLOPs + ~8.7 GB pread/image) — the win is it
**completes or auto-pauses to cool** instead of restarting (Wave 1's ThermalGovernor). Budget for ~a
few on-device fixes (the Z-Image streaming bring-up needed 4).

Tunables if it's tight: `Flux2StreamableBlock.approximateBytes` (residency planning), the streaming
`cacheLimit` (384 MB in `MLXDiffusionEngine.load`), and the VAE tile overlap (bump `.aggressive`
overlap 4 → 8 if seams show).

---

## What's proven offline (no checkpoint needed) vs to validate

| Piece | Status | Test |
|---|---|---|
| Block-streaming decomposition == monolithic forward | ✅ proven 1e-4 | `Flux2Core/StreamDecompositionTests` |
| FLUX sigma schedule (exact values) | ✅ proven | `Flux2DiffusionEngine/Flux2SigmasTests` |
| Per-block + shared disk↔module key bijection | ✅ proven | `Flux2DiffusionEngine/Flux2WeightsTests` |
| Denoiser wiring (holder/adapter) == monolithic | ✅ proven 1e-4 | `Flux2DiffusionEngine/Flux2DenoiserTests` |
| Component-source routing | ✅ proven | `Flux2DiffusionEngine/Flux2ComponentSourceTests` |
| encode / initialLatent / decode end-to-end parity | ⏳ **512 gate** | §3 above (real weights) |
| 1024 thermal/memory survival | ⏳ on-device | §4 |

`reuse-shell` (the ~100-quantize-passes optimization) stays deferred per the audit until 512 parity is
green and a dedicated bit-exact parity test covers it.
