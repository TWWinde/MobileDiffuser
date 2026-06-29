# Reproducing Model Resources

This guide explains how to reproduce the current MobileDiffuser model setup.
The app no longer uses generated Core ML folders. Runtime weights are public
MLX/Hugging Face checkpoints downloaded by the app or by the engine packages.

Model weights are not committed to this repository.

## Repository Layout

During development the app expects sibling checkouts:

```text
~/code/
  MobileDiffuser/
  z-image-swift-mlx/
  flux2-diffusion-engine/
  swift-diffusion-core/      optional local clone; app normally sees remote git revision
  flux-2-swift-mlx/          optional, resolved by flux2-diffusion-engine
```

Clone the public repos:

```bash
git clone https://github.com/nanguoyu/MobileDiffuser
git clone https://github.com/nanguoyu/z-image-swift-mlx
git clone https://github.com/nanguoyu/flux2-diffusion-engine
```

Open `MobileDiffuser.xcodeproj`. The app links `../z-image-swift-mlx` and
`AppEngines`; `AppEngines` links `../../flux2-diffusion-engine`.

## App Download Path

The normal user flow is in-app download:

1. Run the app.
2. Open Models.
3. Select a model and recipe.
4. Download the missing components.
5. Generate.

Z-Image downloads into the app's Application Support directory through
`ModelDownloader`. FLUX.2 components are managed by `Flux2FacadeEngine` and the
underlying FLUX package.

Settings shows the exact storage location used by the app.

## Model Sources

### Z-Image Turbo

```text
repo: deepsweet/Z-Image-Turbo-6B-MLX-Q4
layout:
  text_encoder/*.safetensors
  transformer/*.safetensors
  vae/*.safetensors
  tokenizer/*
approx size: 5.9 GB
```

The downloader considers the model installed only when every component index
file exists, every shard referenced by each index exists, and no
`*.incomplete` markers remain.

### FLUX.2 Klein

The app manages FLUX as separate recipe components:

```text
transformer 16-bit: black-forest-labs/FLUX.2-klein-4B
transformer 8-bit:  black-forest-labs/FLUX.2-klein-4B
transformer 4-bit:  mlx-community/flux2-klein-4b-4bit
Qwen3 encoder 8-bit: lmstudio-community/Qwen3-4B-MLX-8bit
Qwen3 encoder 4-bit: lmstudio-community/Qwen3-4B-MLX-4bit
small decoder:       black-forest-labs/FLUX.2-small-decoder
standard VAE:        black-forest-labs/FLUX.2-klein-4B
```

The 4-bit transformer path uses the pre-quantized checkpoint. It must load
packed 4-bit weights directly, not bf16 weights followed by load-time
quantization.

The decoder component is also the VAE *encoder* used by image-to-image:
reference images are VAE-encoded into conditioning tokens, and that encode runs
on whichever decoder checkpoint is installed. On iPhone the small decoder file
supplies that encoder, so no extra download is needed for i2i.

## Building the App

Set local signing without committing your Team ID:

```bash
cp Signing.xcconfig.example Signing.xcconfig
# edit Signing.xcconfig and set DEVELOPMENT_TEAM
```

Build for macOS:

```bash
xcodebuild \
  -project MobileDiffuser.xcodeproj \
  -scheme MobileDiffuser \
  -configuration Debug \
  -destination 'platform=macOS' \
  build
```

Build for iOS without signing:

```bash
xcodebuild \
  -project MobileDiffuser.xcodeproj \
  -scheme MobileDiffuser \
  -configuration Debug \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

For device deployment, use Xcode with a valid signing team and an installed iOS
SDK compatible with the device OS.

## CLI Validation

The engine repos include small CLIs for model-level validation.

### Z-Image

```bash
cd ../z-image-swift-mlx
swift run zimage-demo <model-dir> "a red panda on a mossy rock" 1024 8 out.png
```

`<model-dir>` is a local checkout/download of
`deepsweet/Z-Image-Turbo-6B-MLX-Q4`.

### FLUX.2

```bash
cd ../flux2-diffusion-engine
swift run flux2-demo "a red panda on a mossy rock"
```

For `swift run`, MLX may not find its Metal library. The most reliable path is
to run through Xcode. If using CLI, copy an Xcode-built MLX
`default.metallib` to `mlx.metallib` next to the built executable and ensure the
executable has `/usr/lib` on its rpath. The demo targets already include the
rpath fix.

## Tests

Use Xcode/xcodebuild for MLX package tests when possible. Plain `swift test`
often fails in MLX packages because the CLI test runner does not automatically
bundle `default.metallib`.

Useful test groups:

```text
swift-diffusion-core:
  sampler schedule
  memory governor
  MLXDiffusionEngine resident vs streaming lifecycle
  RangedFileWeightSource

z-image-swift-mlx:
  component source routing
  per-block load equivalence
  streaming block lifecycle
  optional real-checkpoint streaming parity
```

The real-checkpoint Z-Image parity test is opt-in. Set:

```bash
ZIMAGE_CHECKPOINT=/path/to/Z-Image-Turbo-6B-MLX-Q4
```

It compares resident and streamed denoise results on the same checkpoint.

## Device Validation

For iPhone runs, collect:

```text
device model:
iOS version:
Xcode version:
model:
recipe:
size:
steps:
seed:
generation time:
peak resident memory shown in Create:
thermal state, if known:
jetsam log, if any:
```

Expected current validated paths:

```text
Z-Image Turbo:
  iPhone 16 Pro
  block streaming
  512px default
  about 2.2 GB measured peak in the validated run

FLUX.2 Klein:
  iPhone 16 Pro
  4-bit transformer, 4-bit encoder, small decoder

  text-to-image 512px, 4 steps
    resident facade
    about 4.3 GB peak, about 1m11s

  text-to-image 1024px
    block-streaming transformer (one block resident at a time)
    conv-striped VAE decode (seam-free, bit-exact)
    about 3.83 GB peak, about 4m22s

  image-to-image 512px (reference-context, single 512x512 reference)
    block-streaming transformer
    about 3.45 GB peak, about 1m49s
```

Image-to-image on iPhone is reference-context: 1-3 references on macOS, a single
512x512 reference on iPhone. There is no strength slider; the output denoises
from pure noise while attending to the reference. The iPhone path streams the
transformer so the reference VAE is freed before the transformer runs; higher
i2i output resolution and multi-reference i2i on iPhone are not yet supported.

1024px on iPhone with the FLUX standard VAE should still be treated as
experimental; the validated 1024 path uses the small decoder with conv-striped
decode.

## Cleanup

Delete downloaded weights from the app's Models screen or Settings -> Manage
models. For package CLI experiments, delete the local Hugging Face model folders
you downloaded manually.

Do not commit:

- model weights,
- `.safetensors` files,
- generated images,
- `Signing.xcconfig`,
- Xcode user data.
