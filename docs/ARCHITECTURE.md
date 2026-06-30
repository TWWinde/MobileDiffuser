# Architecture

This document describes the current MobileDiffuser architecture: a universal
macOS + iOS image-generation app built in SwiftUI, Swift, and MLX. The old
Core ML / SD3 pipeline has been removed from the running app.

## Top-Level Shape

```text
SwiftUI app
  -> AppModel
  -> DiffusionEngine
      -> ZImageFacadeEngine on macOS
      -> MLXDiffusionEngine + ZImageArchitecture on iPhone
      -> Flux2FacadeEngine on macOS and iPhone
```

The app does not talk to a concrete model implementation from the UI. It
selects a catalog model, builds a `DiffusionEngine`, loads the chosen recipe,
and sends a `GenerationRequest`.

## Package Topology

The development checkout uses local package references:

```text
MobileDiffuser
  -> ../z-image-swift-mlx                 (ZImageMLX)
  -> AppEngines
       -> ../../flux2-diffusion-engine    (Flux2DiffusionEngine)

z-image-swift-mlx
  -> swift-diffusion-core                 (remote git dependency)
  -> mlx-swift
  -> swift-transformers

flux2-diffusion-engine
  -> swift-diffusion-core                 (remote git dependency)
  -> flux-2-swift-mlx
  -> mlx-swift
```

Important development detail: edits to `z-image-swift-mlx`,
`flux2-diffusion-engine`, and `AppEngines` are picked up by the app directly
because they are local path dependencies. Edits to `swift-diffusion-core` are
not picked up by the app until the remote dependency is updated or a local
override is configured.

## App Layer

`AppModel` owns the user-facing state:

- selected model and persisted selected model id,
- prompt, size, steps, seed,
- FLUX recipe axes (transformer precision, encoder precision, decoder),
- model/component download state,
- engine load/generation phase,
- peak resident-memory readout,
- in-session generation history.

The UI is a studio shell:

- Create: prompt, canvas, size/steps/seed controls, model bar, and (for FLUX.2)
  a 1-to-3 reference-image picker for img2img.
- Models: model cards, fit badges, recipe axes, component downloads.
- Library: in-session generated images, settings reuse, export.
- Settings: appearance, storage, device memory, model management.

## Engine Boundary

`DiffusionEngine` is the app boundary:

```swift
load(model, variant, source, progress)
generate(request, progress) -> CGImage
unload()
capabilities(...)
```

Two engine shapes exist.

### Generic Block-Streaming Engine

`MLXDiffusionEngine` drives any `DiffusionArchitecture`. The architecture
provides:

- `encode`: text encoder phase,
- `releaseTextEncoder`: free text encoder resources after conditioning is
  materialized,
- `makeDenoiser`: build the denoiser with independently loadable blocks,
- `initialLatent`: seeded latent creation,
- `decode`: VAE decode.

For streaming residency, the denoise loop is:

```text
for each step:
  embed latent
  for each transformer block:
    load block weights from WeightSource
    run block
    materialize hidden
    release block
    clear MLX GPU cache
  unembed velocity
  Euler step
```

This keeps transformer residency near one block instead of the full model.

### Whole-Pipeline Facades

Some model packages own their entire denoise loop and do not expose per-block
access. Those are wrapped as facade engines.

- `ZImageFacadeEngine`: resident macOS wrapper over `ZImagePipeline`.
- `Flux2FacadeEngine`: wrapper over `Flux2Pipeline` from `flux-2-swift-mlx`.

FLUX.2 is cross-platform now. At the app boundary it remains a whole-pipeline
facade rather than a block-streamed `DiffusionArchitecture`, but the FLUX package
owns its own internal block-streaming path: the facade dispatches the heavy
iPhone cases (1024px text-to-image and image-to-image) to a streamed transformer
that keeps one block resident at a time, while macOS and the lighter iPhone cases
stay resident.

## Model Paths

### Z-Image Turbo

Z-Image Turbo is a 6B S3-DiT model with a Qwen3-4B text encoder and FLUX-family
VAE. The shipped app uses the 4-bit MLX checkpoint:

```text
deepsweet/Z-Image-Turbo-6B-MLX-Q4
```

On macOS, Z-Image runs through `ZImageFacadeEngine` and loads the pipeline
resident.

On iPhone, Z-Image runs through `MLXDiffusionEngine` and `ZImageArchitecture`.
The downloaded model directory has separate component folders:

```text
text_encoder/
transformer/
vae/
tokenizer/
```

Those components have colliding tensor key spaces, so the app opens them through
`ZImageComponentSource`. It is a composite `WeightSource` that routes
component-prefixed keys to the right sub-source and routes bare transformer keys
to the transformer source for the generic streaming engine.

Only the transformer is opened with `RangedFileWeightSource` on the streaming
path. Text encoder and VAE are loaded phase-wise. The text-encoder source is
dropped after `encode` so the encoder and transformer do not co-reside.

Validated state:

- macOS: coherent 1024px images.
- iPhone 16 Pro: fully on-device block streaming, about 2.2 GB peak resident
  memory in the measured path.

### FLUX.2 Klein

FLUX.2 Klein 4B runs through `Flux2FacadeEngine`. The app exposes recipe axes:

- transformer: 16-bit, 8-bit, 4-bit,
- text encoder: 8-bit, 4-bit,
- decoder: small decoder, standard VAE.

The 4-bit transformer uses:

```text
mlx-community/flux2-klein-4b-4bit
```

This is pre-quantized and loads directly into quantized layers. It avoids the
large load-time spike that would happen if the bf16 transformer were downloaded
and quantized in memory.

On iPhone, FLUX runs as a two-phase pipeline. The Qwen3 text encoder is unloaded
before the transformer/VAE phase. iPhone builds also trim the MLX GPU cache each
denoise step. The standard VAE is gated to 512px in the app because its wider
decoder activations are risky at larger sizes.

Validated state (text-to-image):

- macOS: 4-bit pre-quantized path generates clean 512px images.
- iPhone 16 Pro, 512px: resident facade, 4 steps, small decoder, about 4.3 GB
  peak resident memory and about 1m11s runtime.
- iPhone 16 Pro, 1024px: the transformer runs as a block-streaming pipeline (one
  block resident at a time) instead of the resident facade. Peak about 3.83 GB,
  about 4m22s. The earlier "10 GB decode wall" was a measurement artifact; the
  1024px VAE decode is bounded to about 4.28 GB by conv striping (seam-free,
  bit-exact). A cheap x0-pred latent-to-RGB preview shows the image forming
  without running the VAE.

### FLUX.2 image-to-image (reference-context)

img2img in MobileDiffuser is FLUX.2 reference-context, not a strength/denoise
slider. One to three reference images are VAE-encoded and concatenated into the
transformer sequence as conditioning; the output always denoises from pure noise
while attending to the references, so strength is effectively always 1.0. The
reference latents are packed by the same `LatentUtils` path the facade uses, and
each reference is tagged on a separate RoPE T-coordinate so the transformer keeps
reference tokens distinct from the T=0 output tokens.

It flows through two residencies, mirroring text-to-image:

- macOS: the resident facade, 1 to 3 references, up to the chosen size. A 1024px
  i2i run peaks around 6.9 GB resident.
- iPhone: the resident facade OOM'd the phone. Each reference is about 4096
  tokens at the resident `maxImageArea` of 1024², so even a "512 i2i" ran roughly
  5120 tokens resident (about 5.75 GB), over the phone's jetsam line and causing a
  whole-device respring. This was confirmed a memory limit via JetsamEvent logs,
  not thermal. i2i was first disabled on iPhone, then re-enabled through the
  block-streaming path.

The streaming i2i path is shaped to stay lighter than the already-shipped
1024px text-to-image:

- The streamed transformer carries the sequence as `[output ; reference]`
  (output tokens first). Only the output tokens are denoised and decoded; a new
  `outputSeqLen` slice in `streamUnembed` selects them.
- The reference is capped to 512² (about 1024 tokens, a single reference on
  iPhone), so the streamed sequence is at most about 2048 tokens — lighter than
  the 4096-token 1024px text-to-image path. On macOS the facade keeps the
  resident 1024² reference encoding.
- The reference VAE encoder is freed before the transformer streams, so the
  encoder and transformer never co-reside.

Validated state (image-to-image):

- iPhone 16 Pro, 512px output, single reference: block-streaming, peak about
  3.45 GB on-device, about 1m49s, no restart, good quality. (A Mac forced-stream
  run measured about 3.78 GB.)
- A 512px i2i parity gate (`flux2-demo --parity --i2i`) proves the streamed
  output is pixel-identical to the resident facade (max pixel diff 0, PSNR inf),
  both resident and forced-block-streaming. The streamed-decomposition unit
  tests pass.

## Memory Governance

`DeviceTier` records physical memory and whether the device is a phone. The
iPhone budget is intentionally conservative at about 50 percent of physical RAM.

`MemoryGovernor` estimates:

- two-phase peak: `max(textEncoder, transformer + VAE) + workingSet`,
- streaming peak: `max(textEncoder, one-block-estimate) + workingSet`.

The generic engine refuses a streaming plan unless the `WeightSource` reports
`freesOnRelease == true`. This prevents a false streaming badge when releasing a
block would not actually free its buffers.

The app samples `MemoryProbe.residentBytes()` during generation and displays the
peak vs the device budget after a run.

Known calibration note: FLUX.2 Klein measured about 4.3 GB peak on iPhone while
the facade estimate was lower. The current budget is conservative; actual
foreground jetsam headroom on an 8 GB iPhone can be higher than the displayed
budget.

## Downloads and Storage

Z-Image downloads through `ModelDownloader`, which wraps Hugging Face snapshot
download into Application Support. It verifies that component index files and
all referenced shards exist, and rejects empty or incomplete downloads.

FLUX downloads are managed by `Flux2FacadeEngine` and the underlying FLUX
package. The app presents each transformer, encoder, and VAE component as an
individual row and can install or delete them independently.

Downloaded models live under Application Support. Settings shows the app's
storage location and the current model bytes on disk.

## Testing Strategy

The package tests lock down the high-risk seams:

- sampler schedule and Euler step,
- memory governor decisions,
- resident vs streaming block lifecycle,
- ranged safetensors reads,
- Z-Image component-source routing,
- per-block streaming load equivalence,
- optional real-checkpoint resident-vs-streaming parity.

MLX tests should be run with Xcode/xcodebuild where possible, because plain
`swift test` may not provide MLX's compiled Metal library.

## Current Limitations

- img2img is FLUX.2 reference-context only. Z-Image classic (strength-based)
  img2img is not implemented.
- iPhone img2img is a single reference at 512px output. Multi-reference streaming
  i2i and higher-than-512 i2i output on iPhone are not done; macOS still allows
  1 to 3 references up to the chosen size.
- Z-Image iPhone streaming is proven, but still the highest-risk path for
  future model changes because it depends on exact key routing and block parity.
- FLUX.2 at 1024px on iPhone is validated through the streaming path but stays
  cautious. VAE decode and attention activations dominate peak memory.
- External SSD streaming is designed into `WeightSource`, but the app does not
  yet expose external model locations or security-scoped bookmarks. It is a
  deferred challenge experiment (no-go as a product, go as an experiment).
