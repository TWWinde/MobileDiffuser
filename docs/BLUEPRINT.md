# Universal Diffusion App — Rebuild Blueprint

> A universal (macOS + iOS) Swift app for on-device image generation, built entirely on
> **Apple MLX + Swift**, with first-class model management, smart per-hardware
> load/unload, and support for many open-weight diffusion models at multiple precisions —
> all downloadable in-app.

This document is the canonical plan for rebuilding this repository. It supersedes the
original CoreML iPhone app: the UI and logic are torn down and rebuilt from scratch.

---

## 1. North star

- **One engine, everywhere.** A single **MLX/Swift** inference stack runs on both Mac and
  iPhone. No CoreML. The cost (MLX on a phone is heavier/slower than ANE) is accepted in
  exchange for one elegant, unified codebase and the freedom to add any open MLX model.
- **Partial load is how big models fit a phone.** Rather than a second engine, we make the
  one engine memory-frugal through a streaming/partial-load ladder (below).
- **Model management is the product.** Picking a model + precision, seeing whether it fits
  *this* device, downloading it (resumably, from public sources or an external SSD), and
  running it — that whole loop is the core experience.
- **Open source, open weights.** Only public, openly-licensed models and dependencies.

---

## 2. Engine strategy — pure MLX, made to fit via partial load

MLX-on-GPU uses more peak memory than ANE for the same model, but the gap is mostly
*optimization effort* (4-bit quantization + per-submodule streaming), not an inherent
framework penalty. We close it with a ladder the engine climbs only as far as a given
model/device requires:

| Rung | Technique | Effect | Prior art |
|---|---|---|---|
| 1 | 4-bit (down to 2/3-bit) quantization | halve, then halve again | community MLX weights |
| 2 | **Two-phase staging** — load text encoder → encode → release → load transformer + VAE → denoise → decode | encoder and transformer never co-reside | mflux `--low-ram` |
| 3 | **Block streaming** — load each denoiser block from mmap'd / ranged-read weights, run, release | transformer residency drops from GB to hundreds of MB | SwiftLM SSD streaming on mlx-swift; split-stage CoreML |
| 4 | Memory-efficient attention (flash / chunked SDPA) | bounds activation memory at high resolution | MLX `scaledDotProductAttention` |
| 5 | Device gating — `MemoryProbe.availableBytes()` + `MLX.GPU.set(cacheLimit:)` per device | runtime resident-vs-stream decision | jetsam-accurate probe |

### Feasibility (verified sizes, two-phase peak ≈ max(encoder, transformer+VAE) + working set)

| Model (MLX) | Disk | Peak (Q4) | 8 GB iPhone (~4 GB budget) | 12 GB iPhone (~6 GB) | Mac |
|---|---|---|---|---|---|
| FLUX.2 Klein 4B | 4.6 GB | ~2.6–3 GB | runs (staging alone) | runs | runs great |
| Z-Image Turbo (6B) | 5.9 GB | ~4 GB | needs block streaming (or 3-bit) | runs resident | runs great |
| FLUX.2 Klein 9B | 9.5 GB | ~6 GB | external-SSD stream only | tight | runs great |
| Qwen-Image 2512 | 25.9 GB | ~16 GB | — | — | Mac (24 GB+) |

> The iPhone path is genuine R&D — no one has shipped MLX diffusion this way on a phone.
> Klein 4B 4-bit (two-phase) is the fastest proof point; Z-Image Turbo validates streaming.

### FLUX on iOS — implemented

**Decision (2026-06-24): EXTEND `flux-2-swift-mlx` to be cross-platform rather than reimplement FLUX
in the Z-Image streaming framework** — one shared forward keeps Mac and iOS consistent and preserves
everything the package already supports (dev / klein-4b / klein-9b, multiple precisions, reference-image
conditioning, LoRA, training). This has now been **done** (the cross-platform port shipped 2026-06-24);
FLUX.2 Klein 4B builds and is wired up on iPhone alongside Z-Image.

**What shipped (across three repos, pushed to `main` / `rebuild/mlx-foundation`):**

1. **`flux-2-swift-mlx` → cross-platform** (`platforms += .iOS(.v17)`). The AppKit surfaces are
   concentrated at image boundaries and are macOS-only features, so they're guarded, not ported:
   the Pixtral/Mistral VLM (`ImageProcessor`, `analyzeImage(NSImage)`, `loadVLMModel`) behind
   `#if canImport(AppKit)`, and LoRA-training image loading behind `#if os(macOS)`. The Klein /
   Qwen3 text2img path (CGImage-based) stays unguarded. `homeDirectoryForCurrentUser` (unavailable
   on iOS) is replaced with a caches-dir fallback. The iOS per-app memory reserve is shrunk (jetsam
   caps RAM well below total).
2. **Pre-quantized 4-bit load path** — `flux-2-swift-mlx`'s legacy "4-bit" downloads the 7.2 GB bf16
   and quantizes on load (would OOM a phone). **`mlx-community/flux2-klein-4b-4bit`** is a clean
   PRE-quantized checkpoint (mflux 0.17.5, group size 64, transformer 2.18 GB, 387 tensors). The
   pipeline now detects the MLX-quantized format (a `.scales` sibling on each linear), quantizes the
   bf16 shell to `QuantizedLinear` **first**, then loads the packed weight/scales/biases straight from
   disk — **no float16 intermediate, ~2.2 GB resident** instead of the ~7 GB spike. A dedicated
   Diffusers→Swift 4-bit key mapper nests the time embedder correctly
   (`time_guidance_embed.linear_1` → `timeGuidanceEmbed.timestepEmbedder.linear1`) and applies no
   adaLN half-swap (mflux `norm_out.linear` is already `[scale|shift]`); the load verifies
   `notFound == 0` and that every quantized layer is filled. A unit test asserts the mapping against
   the real 387-key layout. The Klein 4B arch is hardcoded, so the loader accepts the
   `model.safetensors.index.json` sharded layout without a `config.json`.
3. **Phone-aware facade** (`flux2-diffusion-engine`, also `+= .iOS`): `capabilities()` returns a
   two-phase estimate on a phone (text encoder unloaded before the transformer + VAE denoise) gated
   against the device budget; a `transformerVariantOverride` seam selects the pre-quantized 4-bit
   checkpoint. **4-bit uses the pre-quantized checkpoint on *both* platforms** (smaller download,
   no load spike); Mac's **16-bit and 8-bit are unchanged** (16-bit bf16, 8-bit pre-quantized int8).
4. **App un-gated** (`AppEngines` drops the macOS-only product condition + export gate; `Catalog`
   ships FLUX on both platforms pointed at the 4-bit repo; `AppModel` un-gates the whole FLUX
   surface). iPhone defaults to **4-bit transformer + 4-bit encoder**; Mac keeps 8-bit, and saved
   precision prefs survive (shared persisted keys).

**Memory:** two-phase resident — **512 fits** (≈ max(encoder ~1.9 GB, transformer 2.18 + VAE 0.58) +
working set ≈ 3.3 GB, under an 8 GB phone's ~4 GB budget); **resident 1024 is tight** (double-stream
activations push toward ~4.3 GB), so iPhone 1024 runs on the **block-streaming** path instead
(validated; see *Block streaming for FLUX* above).

**Validation — done, both platforms.** Mac (2026-06-24): downloaded the 2.18 GB checkpoint, loaded
with zero `notFound`/OOM, generated a clean 512 text2img in 4 steps — proving the 4-bit key mapping +
quantize-shell-then-update numerics. **iPhone 16 Pro (2026-06-25): PASSED** — FLUX.2 Klein 4B, 512px,
4 steps, small decoder, **two-phase, peak 4.3 GB, ~1m11s, clean image**. Two empirical notes: the real
4.3 GB peak runs **higher than the facade's `capabilities()` estimate (~3.3 GB)** — the working-set
constant under-predicts; and 4.3 GB exceeded the conservative ~4 GB (50%-of-RAM) budget yet iOS did not
jetsam, so there is real foreground headroom beyond that budget. **512 resident is confirmed safe;**
1024 was later landed on iPhone via **block streaming** (conv-striped VAE decode, peak 3.83 GB — see
above), rather than the resident facade. (Mac CLI note: `swift run` can't find MLX's `default.metallib`
— copy `mlx-swift_Cmlx.bundle` next to the binary.)

**Block streaming for FLUX — done (2026-06-27/29).** The per-block ladder (one transformer block
resident at a time, the rest streamed from disk) now backs the headroom-hungry FLUX paths and is
bit-exact with the resident facade:

- **1024 text2img** streams the transformer (peak **3.83 GB, ~4m22s** on iPhone 16 Pro). The "10 GB
  decode wall" was a measurement artifact; **conv-striping** (seam-free, bit-exact) bounds the 1024
  VAE decode to ~4.28 GB. A cheap **latent preview** (x0-pred latent→RGB, no VAE) shows the image
  forming during the denoise.
- **512 image-to-image** (reference-context, below) streams the transformer carrying the reference
  tokens (peak **~3.45 GB, on-device-validated 1m49s**, no respring). Still needed for larger
  variants (Klein 9B), where peak would drop to ~1 resident block + base.

### Image-to-image — FLUX.2 reference-context (shipped, iPhone included)

img2img here is **not** a strength/noise-injection slider. It is FLUX.2 **reference-context**: 1–3
reference images are VAE-encoded and concatenated into the transformer sequence as conditioning; the
output denoises from **pure noise** while attending to the refs (strength is always 1.0). Shipped
2026-06-29 (app `8210253`, `swift-diffusion-core` `c0e8f43`, `flux-2-swift-mlx` `bbae617`,
`flux2-diffusion-engine` `6d29c85`).

- **macOS** — resident facade, 1–3 references, up to the chosen size (1024-i2i facade peaks ~6.9 GB).
- **iPhone** — the resident facade OOM'd the phone: each reference is ~4096 tokens at the 1024²
  max-image-area, so even "512 i2i" ran ~5120 tokens **resident** ≈ 5.75 GB, over the ~5.5 GB jetsam
  line → whole-phone respring (a **memory** limit confirmed via JetsamEvent logs, not thermal). So
  i2i was first disabled on iPhone, then **re-enabled via block streaming**: the streamed transformer
  carries the sequence as `[output ; reference]` (output first); only the **output** tokens are
  denoised/decoded (a new `outputSeqLen` slice in `streamUnembed`); the reference is capped to 512²
  (~1024 tokens, single reference) so the streamed sequence (≤2048 tokens) is **lighter** than the
  already-shipped 1024 text2img (4096 tokens); the reference VAE is freed before the transformer
  streams (no encoder↔transformer co-residency). **On-device validated** on iPhone 16 Pro: 512 i2i,
  peak **~3.45 GB**, 1m49s, no restart, good quality.
- **Validation** — a 512 i2i parity gate (`flux2-demo --parity --i2i`) proved the streamed output is
  **pixel-identical** to the resident facade (`maxPixelDiff=0`, PSNR=inf), both resident and
  forced-block-streaming; an adversarial code audit found the order/pos-id/slice correct and the
  memory lifecycle clean; the streamed-decomposition unit tests pass.
- **Not yet:** multi-reference streaming i2i on iPhone (single ref today) and higher-than-512 i2i
  output on iPhone; Z-Image classic (strength-based) i2i.

---

## 3. Architecture

### Package topology (all public)

```
App (this repo — rebuilt)         depends on ↓
  ├── swift-diffusion-core   (NEW public repo: nanguoyu/swift-diffusion-core)
  │     engine protocol · streaming partial-loader · WeightSource · samplers ·
  │     memory governor · catalog + download
  ├── flux-2-swift-mlx       (existing public, MIT — use main branch)
  └── z-image-swift-mlx      (NEW public repo: nanguoyu/z-image-swift-mlx)
```

`swift-diffusion-core` and `z-image-swift-mlx` are standalone public repos (siblings of the
app, like `flux-2-swift-mlx`); the app consumes them as local path dependencies during
development and as versioned git dependencies in release.

### The boundary — `DiffusionEngine`

The app talks only to `DiffusionEngine`. There are **two engine shapes** behind it:

- **`MLXDiffusionEngine`** (in core, iOS + macOS) — drives any *block-streamable*
  `DiffusionArchitecture` and applies the partial-load ladder. This is the path for Z-Image
  and the iPhone.
- **Whole-pipeline facade engines** — wrap monolithic pipelines that own their own denoise
  loop. FLUX.2 is this shape: it exposes `Flux2Pipeline.generateTextToImage(...)` with no
  per-block access, so it cannot be block-streamed by the generic engine. It is now
  cross-platform after the guarded AppKit port, and the iPhone path uses a phone-aware two-phase
  4-bit load plan. Z-Image also has a resident facade for macOS, while iPhone uses the generic
  block-streaming path.

`MLXDiffusionEngine` consumes the `DiffusionArchitecture` seam each model package implements:

```
DiffusionEngine        load / generate(progress) / unload / capabilities
   └─ drives ─▶ DiffusionArchitecture   encode() · denoiserBlocks() · decode()
                   each block is a StreamableBlock  (load → run → release)
                   reads weights via WeightSource   (mmap | ranged SSD | hybrid)
```

### `WeightSource` — internal storage *and* external SSD, transparently

Weights are read as byte ranges, not bound to mmap, so the same streaming engine runs from
internal storage or a USB-C external SSD:

- `MmapWeightSource` — mmap the safetensors file (internal, fastest)
- `RangedFileWeightSource` — `pread` byte ranges on demand (external SSD; avoids
  mmap-on-external limits)
- `HybridWeightSource` — hot tensors resident, cold tensors streamed + prefetched

This unlocks running Mac-class models (Klein 9B, even Qwen-Image) off an external SSD on a
phone, at an I/O cost (~1 GB/s over USB 3).

### Memory governor

`DeviceTier` detects chip + `ProcessInfo.physicalMemory` → `(defaultPrecision, cacheLimit,
residentVsStream)`. On iPhone it reads `MemoryProbe.availableBytes()` before building the
pipeline and gates which rung of the ladder is used.

---

## 4. Model catalog

Per-variant layout descriptors (the on-disk layouts differ across families — verified at
least four schemes), so the weight loader dispatches per layout:

| Family | Source (public) | Precisions | License |
|---|---|---|---|
| Z-Image Turbo / Base (6B, S3-DiT, Qwen3-4B encoder) | community MLX repos | 8/4/2-bit | Apache-2.0 |
| FLUX.2 Klein 4B / 9B | mlx-community | 8/4-bit | Apache-2.0 |
| Qwen-Image 2512 | mlx-community | 8/6/5/4/3-bit | Apache-2.0 |

Licenses are encoded per variant and enforced (e.g. FLUX.2 **dev** is non-commercial and is
excluded). Default endpoints: public HuggingFace + user-configurable mirrors. Downloads are
**byte-range resumable** (multi-GB transformers), SHA-verified, with smoothed progress.

---

## 5. UI / UX — dark creative studio

Design language: near-black studio surface, a single violet generative accent, hairline
borders, Tabler outline icons. Two things are first-class because they are what makes this
app special:

1. **Precision is a first-class input** — switching precision live updates size, the
   transformer/encoder/VAE component breakdown, and the hardware-fit badge.
2. **Hardware awareness is everywhere** — a fit badge = `device × model × precision`:
   green *runs great* (resident) · amber *two-phase* / *streams from SSD* · gray *needs more*.

### Four tabs (Mac sidebar / iPhone tab bar)

- **Models** — download center: family-grouped cards, recommended-for-device, precision
  chips, fit badges, install/progress. Model detail drawer: variant table, component
  breakdown, storage location (internal / external SSD), resumable download.
- **Create** — generation workspace: prompt, full-bleed canvas, steps/seed/size, 1–3 reference
  images (img2img = FLUX.2 reference-context, not a strength slider), a memory governor pill
  (resident / streaming), latent preview + per-step progress.
- **Library** — your generated images: grid by day, tap for detail (prompt + params),
  **reuse settings** to iterate, favorite, export.
- **Settings** — storage & external SSD: default download location, *stream large models
  from SSD* toggle, per-model location; on iOS the SSD is granted via Files (security-scoped
  bookmark) and must stay connected while generating.

Mac uses `NavigationSplitView`; iOS uses `TabView` + `NavigationStack`; both render the same
shared components.

---

## 6. Open-source / privacy rules

- Use only public dependencies and public, openly-licensed model weights.
- The download/catalog/MLX-abstraction code is adapted from the author's own prior work, but
  everything that lands here is **scrubbed of private identifiers**: no private repo names,
  no private CDN/infra/hostnames, no store/billing IDs, no keys. Endpoints default to public
  HuggingFace + a user-configurable mirror field.

---

## 7. Roadmap

- **Phase 0 — spike & de-risk.** Stand up `swift-diffusion-core` + the `MLXDiffusionEngine`.
  On Mac, run FLUX.2 Klein 4B (via `flux-2-swift-mlx`) and Z-Image Turbo (new package)
  end-to-end. Then on device: measure Klein 4B 4-bit two-phase peak on an 8 GB iPhone and
  Z-Image block-streaming. Prove/disprove MLX-on-iPhone.
  **Status (2026-06-23):** core engine landed and unit-tested — `MLXDiffusionEngine`
  (streaming denoise loop), `FlowMatchEulerSampler`, `SafetensorsWeightSource`,
  `ImageConversion`, `MemoryGovernor`/`DeviceTier`/`MemoryProbe`. Pure-logic tests
  (governor decisions, sampler schedule) pass in CI; MLX-eval tests pass in Xcode (a headless
  CI box has no Metal lib).
  **Status (2026-06-25):** Z-Image S3-DiT, Qwen3-4B, VAE, ranged weight source, streaming
  block lifecycle, and resident-vs-streaming parity gates have landed. Z-Image Turbo runs
  on iPhone 16 Pro via block streaming at about 2.2 GB peak. FLUX.2 Klein 4B runs on iPhone
  through the cross-platform facade and pre-quantized 4-bit checkpoint at about 4.3 GB peak
  for 512px / 4 steps / small decoder.
- **Phase 1 — Mac app.** Dark-studio shell, Models gallery + detail/download (resumable),
  Create workspace, Library, memory governor, persisted image cache.
- **Phase 2 — iPhone.** Same shell adapted (TabView), `MemoryProbe` gating,
  increased-memory-limit entitlement, internal-storage streaming.
  **Status:** the shell builds for iPhone and the Z-Image / FLUX paths are validated on device —
  512 text2img (~4.3 GB), **1024 text2img via block streaming** (peak 3.83 GB, conv-striped VAE
  decode), and **512 image-to-image** (reference-context, block-streamed, peak ~3.45 GB), all on
  iPhone 16 Pro. Still cautious: the **resident** 1024 facade and larger variants (Klein 9B) on
  iPhone, plus multi-reference / >512 i2i — VAE decode and attention activations dominate peak
  memory, which is exactly why the streaming ladder, not the facade, carries those paths.
- **Phase 3 — external SSD + breadth.** `WeightSource` ranged-read path (USB-C SSD), more
  model packages (one public repo each), generation queue, downloader hardening.

---

## 8. Migration — what leaves this repo

The original CoreML app has been removed from the running app. Historical Core ML notes remain
only as attribution/background; current implementation work should start from the MLX app and
packages:

- `MobileDiffuser/ContentView.swift`, `SD3PipelineLoader.swift`, `ModelResourceManager.swift`
  — replaced by the new shell + `swift-diffusion-core`.
- `ml-stable-diffusion/` (vendored CoreML fork) — dropped.
- `scripts/*.py` (CoreML conversion/quantization) — dropped.
- **Kept (ported):** `MemoryProbe` (already in `DiffusionCore`), the partial-load *concept*,
  the increased-memory-limit entitlement, the lazy-build/unload lifecycle.

### Phase 0 open questions

1. Real per-architecture Swift cost for a non-FLUX model (Z-Image S3-DiT + Qwen3-4B encoder).
2. mmap-on-external feasibility on iOS (fall back to `pread` ranged reads).
3. Sustained USB-3 throughput + security-scoped resource lifetime during a long generation.
4. Exact community MLX weight layouts per family (the loader dispatches on layout).
